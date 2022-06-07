import json
import logging
import os
import shutil
import typing
from collections import defaultdict
from typing import List, Dict, Text, Any, Optional

import rasa.shared.nlu.constants as rasa_constants
from dialogue_bot.intent_understanding.iu import ExtractedEntity
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLUResult, NLU, \
    create_nlu_train_test_data
from dialogue_bot.intent_understanding.natural_language_understanding.tools.entity_value_mapper import \
    EntityValueMapper
from dialogue_bot.models.utils import RankingScore, complete_ranking, chunk_list
from rasa import model
from rasa.exceptions import ModelNotFound
from rasa.model_training import train_nlu
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

if typing.TYPE_CHECKING:
    from dialogue_bot.models.phrase import Phrase, PhraseEntity
    from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLUTrainingData

logger = logging.getLogger(__name__)

CURR_DIRPATH = os.path.dirname(os.path.abspath(__file__))


def format_rasa_training_data(training_data: 'TrainingData', indent=None):
    def default_format(o):
        if isinstance(o, set):
            return list(o)
        return o.__dict__

    return json.dumps({
        'training_examples': training_data.training_examples,
        'entity_synonyms': training_data.entity_synonyms,
        'regex_features': training_data.regex_features,
        'lookup_tables': training_data.lookup_tables,
    }, default=default_format, indent=indent)


class RasaNLU(NLU):
    """
    This NLU ranks expressions and extracts entities.
    """

    CONVERTED_VALUE_PREFIX = 'CONVERTED_VALUE_'

    def __init__(self, env: 'BotEnv', id: str, confidence_threshold: float = 0.48):
        super().__init__(env, id)
        self._model = None
        self._entity_value_mapper = EntityValueMapper(env)
        self.confidence_threshold = confidence_threshold

    def _value_to_rasa_value(self, value: str) -> str:
        # The problem is that RASA does not really distinguish between "our" set of discrete values and surface patterns
        # In order to distinguish RASA's extracted entity-values from surface patterns, we ensure that "our" values
        # cannot conflict with text surface patterns
        return RasaNLU.CONVERTED_VALUE_PREFIX + value

    def _rasa_value_to_value(self, rasa_value: str) -> Optional[str]:
        if rasa_value.startswith(RasaNLU.CONVERTED_VALUE_PREFIX):
            return rasa_value[len(RasaNLU.CONVERTED_VALUE_PREFIX):]
        return None

    def _entity_to_rasa_entity(self, entity: 'PhraseEntity'):
        return {
            rasa_constants.ENTITY_ATTRIBUTE_TYPE: entity.entity,
            rasa_constants.ENTITY_ATTRIBUTE_VALUE: self._value_to_rasa_value(entity.value),
            rasa_constants.ENTITY_ATTRIBUTE_START: entity.start,
            rasa_constants.ENTITY_ATTRIBUTE_END: entity.end,
        }

    def _phrase_to_rasa_phrase(self, phrase: 'Phrase') -> Message:
        return Message(data={
            rasa_constants.TEXT: phrase.text,
            rasa_constants.INTENT: phrase.expression_id,
            rasa_constants.ENTITIES: [self._entity_to_rasa_entity(e) for e in phrase.entities]
        })

    def _create_rasa_training_examples(self, training_data: 'NLUTrainingData') -> List['Message']:
        phrases = set([])
        for phrase_pattern in training_data.phrase_patterns:
            for phrase in phrase_pattern.generate_phrases(self.env, 15):
                phrases.add(phrase)

        return [self._phrase_to_rasa_phrase(p) for p in phrases]

    def _create_rasa_entity_synonyms(self, training_data: 'NLUTrainingData') -> Dict[Text, Text]:
        """
        Creates a dict that maps text surface forms to values
        """

        # we create synonyms only for values that we actually want to extract
        entities = self.env.entities(intent_filter=training_data.intent_filter)
        values = self.env.values(entity_filter=[e.id for e in entities])

        # map synonyms to values
        synonym_value_mapping = defaultdict(set)  # synonym -> value(s)
        for value in values:
            for synonym in value.synonyms:

                if synonym.entity_context is not None:
                    logger.warning('Entity-contexts in synonyms are not supported: {}'.format(synonym))
                    continue

                # assume that RegexEntityExtractor is used with case_sensitive: False
                if synonym.case_sensitive:
                    logger.warning('Case-sensitive synonyms are not supported: {}'.format(synonym))
                    continue

                synonym_value_mapping[synonym.text].add(value.id)

        # remove ambiguous mappings
        unique_synonym_value_mapping = {}
        for s, vs in synonym_value_mapping.items():
            if len(vs) > 1:
                logger.warning('Ambiguous synonym-value-mappings are not supported: "{}" => {}'.format(s, vs))
            else:
                unique_synonym_value_mapping[s] = list(vs)[0]

        # convert to RASA values
        for s, v in unique_synonym_value_mapping.items():
            unique_synonym_value_mapping[s] = self._value_to_rasa_value(v)

        return unique_synonym_value_mapping

    def _create_rasa_regex_features(self, training_data: 'NLUTrainingData') -> List[Dict[Text, Text]]:
        """
        Creates a list with:
        {
            "name": "number",     # entity-id (when using RegexEntityExtractor) or any other name (when using RegexFeaturizer)
            "pattern": "[0-9]+",  # regex-pattern
            "usage": "intent"     # TODO don't know what this means? or is that's required
        }
        """
        # TODO these regex-patterns are not unique, I don't know if that causes a problem

        # just to not confuse with entity patterns (that should be named exactly like the entity)
        value_feature_pattern_name = lambda value_id: 'value-{}'.format(value_id)

        # For this, we will use all values, not just the ones that are used in the intents (because of regex-feature-extraction)
        values = self.env.values()

        # Create value-patterns
        value_patterns = defaultdict(set)  # value.id => regex-patterns
        for value in values:

            # create regex patterns from synonyms & regex-patterns (== Lookup-Table in RASA-Terminology)
            for regex_pattern in value.all_regex_patterns:
                if regex_pattern.entity_context is not None:
                    logger.warning('Entity-contexts in are not supported in RASA: {}'.format(regex_pattern))
                    continue
                value_patterns[value.id].add(regex_pattern.pattern)

        # Now, create value-regex-patterns for RASA
        res = []
        for value_id, patterns in value_patterns.items():
            for patterns2 in chunk_list(list(patterns),
                                        max_elems=10):  # this is just to avoid very long regular expression patterns
                res.append({
                    'name': value_feature_pattern_name(self._value_to_rasa_value(value_id)),
                    'pattern': '(?:' + '|'.join(patterns2) + ')'
                })

        # Now, create entity-regex-patterns for RASA
        # for this, we only add patterns for entities that are actually used
        entities = self.env.entities(intent_filter=training_data.intent_filter)
        for entity in entities:
            for value in entity.values:
                patterns = value_patterns[value.id]
                for patterns2 in chunk_list(list(patterns),
                                            max_elems=10):  # this is just to avoid very long regular expression patterns
                    res.append({
                        'name': entity.id,
                        'pattern': '(?:' + '|'.join(patterns2) + ')'
                    })

        return res

    def _create_rasa_lookup_tables(self, training_data: 'NLUTrainingData') -> List[Dict[Text, Any]]:
        """
        Creates a list with:
        {
            "name": "city",  # entity-id (when using RegexEntityExtractor) or any other name (when using RegexFeaturizer)
            "elements": ["Berlin", "Amsterdam", "New York", "London"],  # surface-forms (NOT values!)
        }
        """
        """
        From RASA-Documentation: https://rasa.com/docs/rasa/training-data-format#lookup-tables
        "Lookup table regexes are processed identically to the regular expressions directly specified in the training
        data and can be used either with the RegexFeaturizer or with the RegexEntityExtractor.
        The name of the lookup table is subject to the same constraints as the name of a regex feature."
        
        Therefore, we don't use lookup-tables and do everything using regex-features.
        """
        return []

    def _train(self, training_data: Optional['NLUTrainingData'], config_path, nlu_data_path, output_path,
               current_model_path):

        # clear directory
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
        os.makedirs(self.storage_dir)

        if training_data is None:
            training_data, _ = create_nlu_train_test_data(self.env, test_size=0)

        # create rasa training data
        logger.info('Create RASA TrainingData...')
        rasa_training_data = TrainingData()
        rasa_training_data.training_examples = self._create_rasa_training_examples(training_data)
        rasa_training_data.entity_synonyms = self._create_rasa_entity_synonyms(training_data)
        rasa_training_data.regex_features = self._create_rasa_regex_features(training_data)
        rasa_training_data.lookup_tables = self._create_rasa_lookup_tables(training_data)

        # store RASA-training data into file
        rasa_training_data.persist_nlu(nlu_data_path)

        #  copy RASA-config file
        shutil.copyfile(
            os.path.join(CURR_DIRPATH, 'config_{}.yml'.format(self.env.language)),
            config_path
        )

        # train
        logger.info('Train...')
        train_nlu(
            config=config_path,
            nlu_data=nlu_data_path,
            output=output_path,
            train_path=None,
            fixed_model_name=None,
            persist_nlu_training_data=True,
            additional_arguments=None,
            domain=None,
            model_to_finetune=None,
            finetuning_epoch_fraction=1.0,
        )

        # unzip model
        logger.info('Unzip model...')
        zipped_model = model.get_latest_model(output_path)
        model.unpack_model(zipped_model, current_model_path)

        logger.info('Successfully trained RASA model!')

    def _load_model(self, current_model_path):
        _, nlu_model_directory = model.get_model_subdirectories(current_model_path)
        return Interpreter.load(nlu_model_directory)

    def init(self, retrain: bool, training_data: Optional['NLUTrainingData'] = None):
        logger.info('Initializing {} (retrain={}) ...'.format(self.__class__.__name__, retrain))

        # paths
        config_path = os.path.join(self.storage_dir, 'config.yml')
        nlu_data_path = os.path.join(self.storage_dir, 'nlu_training_data.yml')
        output_path = os.path.join(self.storage_dir, 'models')
        current_model_path = os.path.join(self.storage_dir, 'current_model')

        # init used models
        self._entity_value_mapper.init(retrain, training_data=training_data)

        # try to load model
        loaded_model = None
        try:
            loaded_model = self._load_model(current_model_path)
        except ModelNotFound:
            pass

        # train
        if retrain or (loaded_model is None):
            self._model = None
            logger.info('Train {}...'.format(self.id))
            self._train(training_data, config_path, nlu_data_path, output_path, current_model_path)
            loaded_model = self._load_model(current_model_path)
        else:
            logger.info('Loaded {}...'.format(self.id))

        self._model = loaded_model

    def _rasa_entity_value_to_value(self, rasa_entity: str, rasa_value: str, intent_filter: List[str]) -> Optional[str]:
        """ The problem is, that RASA does not necessarily extract the limited set of values that we defined,
            but also include surface patterns.
        """
        # if rasa_value is already one of "our" values
        v = self._rasa_value_to_value(rasa_value)
        if v is not None:
            return v

        # if rasa_value is a surface pattern
        x = self._entity_value_mapper.run(rasa_entity, rasa_value, intent_filter=intent_filter)
        if x is not None:
            return x['value']

        return None

    def run(self, utterance: str, intent_filter: List[str] = None) -> 'NLUResult':
        if intent_filter is None:
            intent_filter = [intent.id for intent in self.env.intents(intent_filter=intent_filter)]
        expression_filter = [expression.id for expression in self.env.nl_expressions(intent_filter=intent_filter)]

        # rasa_res - Format:
        # {
        #    'intent': {'name': 'buy_car', 'confidence': 0.8223017575411612},
        #    'entities': [
        #        {'start': 10, 'end': 14, 'value': 'audi', 'entity': 'car', 'confidence': 0.7951135658737039,
        #         'extractor': 'CRFEntityExtractor'}
        #    ],
        #    'intent_ranking': [
        #        {'name': 'buy_car', 'confidence': 0.8223017575411612},
        #        {'name': 'hello', 'confidence': 0.0946209052126251},
        #        {'name': 'how_are_you', 'confidence': 0.08307733724621377}
        #    ],
        #    'text': 'i need an Audi'
        # }
        #
        # WARNING: the problem is that Rasa's NLU can not be restricted to a subset of intents (here: expressions),
        # therefore, we will use the 'intent_ranking'-attribute and hope that the expressions is in there
        # and assume that the extracted entities are still valid for that trigger

        rasa_res = self._model.parse(utterance)
        logger.debug('RASA-Output: {}'.format(rasa_res))

        # ranking
        expression_ranking = [RankingScore(dct['name'], dct['confidence']) for dct in rasa_res['intent_ranking'] if
                              dct['name'] in expression_filter]

        # complete
        expression_ranking = complete_ranking(expression_ranking, expression_filter)

        # entities
        entities = set([
            ExtractedEntity(
                dct['start'],
                dct['end'],
                dct['entity'],
                self._rasa_entity_value_to_value(dct['entity'], dct['value'], intent_filter),
                utterance[dct['start']:dct['end']],
                dct['confidence'] if 'confidence' in dct else 1.,  # TODO what else?
                self.id
            )
            for dct in rasa_res['entities']])

        return NLUResult(utterance, expression_ranking, self.confidence_threshold, entities)


if __name__ == '__main__':
    from dialogue_bot.example_bot import bot

    nlu = RasaNLU(bot, 'en')
    nlu.init(True)
    res = nlu.run('I live in Germany')
    print(res)
