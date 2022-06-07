import logging
import typing
from typing import List, Optional

from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLU, NLUResult, \
    create_nlu_train_test_data, sort_expression_ranking
from dialogue_bot.models.utils import complete_ranking, RankingScore
from dialogue_bot.nlp.text_scorer.text_scorer import serialize_annotation, deserialize_annotation
from dialogue_bot.utils.lists import list_aggregate_func
from tqdm import tqdm

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLUTrainingData
    from dialogue_bot.nlp.text_scorer.text_scorer import TextScorer

logger = logging.getLogger(__name__)


class PhraseExpressionRanker(NLU):
    """
    This NLU only ranks expressions => no entity extraction.
    This NLU creates a set of phrases from phrase-patterns (from `NLExpression`) which are annotated by `TextScorer`.
    For each incoming user utterance, the similarities from the annotated utterance to all phrases are calculated  by `TextScorer`.
    The expression scores are the aggregated pairwise similarity scores of <user utterance, phrase> for each phrase belonging to that expression:
        score(expression) = aggregate([
            similarity(annotated(utterance), annotated(phrase1)),
            similarity(annotated(utterance), annotated(phrase2)),
            ...
        ])
    """
    def __init__(self, env: 'BotEnv', id: str, phrase_scorer: 'TextScorer',
                 confidence_threshold: float = 0.75,
                 aggregation='max'):
        super().__init__(env, id)
        self.phrase_scorer = phrase_scorer
        self.confidence_threshold = confidence_threshold
        self.aggregation = aggregation
        self._db_collection_name = '{}{}'.format(self.__class__.__name__, id)

    def init(self, retrain: bool, training_data: Optional['NLUTrainingData'] = None):
        logger.info('Initializing {} (retrain={}) ...'.format(self.id, retrain))

        if not self.env.db_handler.exists_collection(self._db_collection_name):
            retrain = True

        # train
        if retrain:
            logger.info('Train {}...'.format(self.id))

            # reset database
            if self.env.db_handler.exists_collection(self._db_collection_name):
                self.env.db_handler.delete_collection(self._db_collection_name)
            self.env.db_handler.ensure_collection(self._db_collection_name)

            if training_data is None:
                training_data, _ = create_nlu_train_test_data(self.env, test_size=0)

            # index training data
            logger.info('Indexing data...')
            for pattern in tqdm(training_data.phrase_patterns):
                for phrase in pattern.generate_phrases(self.env, 50):

                    # we annotate the phrase with `TextScorer` and store the annotation in the database
                    # so that we dont have to do this during runtime
                    self.env.db_handler.index_object(self._db_collection_name, {
                        'expression_id': phrase.expression_id,
                        'text': phrase.text,
                        'annotations': serialize_annotation(self.phrase_scorer.annotate(phrase.text))
                    }, {'expression_id', 'text'})
            self.env.db_handler.commit(self._db_collection_name)
        else:
            # ensure database
            self.env.db_handler.ensure_collection(self._db_collection_name)

            logger.info('Loaded {}...'.format(self.id))

    def run(self, utterance: str, intent_filter: List[str] = None) -> 'NLUResult':
        if intent_filter is None:
            intent_filter = [intent.id for intent in self.env.intents(intent_filter=intent_filter)]
        expression_filter = [expression.id for expression in self.env.nl_expressions(intent_filter=intent_filter)]

        # annotate
        utterance_annotations = self.phrase_scorer.annotate(utterance)

        # prepare query
        query = self.env.db_handler.query_attr_in_value_list('expression_id', expression_filter)

        # query
        expression_scores = {}
        for x in self.env.db_handler.find(self._db_collection_name, query):
            phrase_annotation = deserialize_annotation(x['annotations'])
            similarity = self.phrase_scorer.similarity(utterance_annotations, phrase_annotation)
            expression_scores[x['expression_id']] = expression_scores.get(x['expression_id'], []) + [similarity]

        # aggregate
        for k, vs in expression_scores.items():
            aggregate_func = list_aggregate_func(self.aggregation)
            expression_scores[k] = aggregate_func(vs)
        expression_ranking = [RankingScore(k, s) for k, s in expression_scores.items()]

        # sort
        expression_ranking = sort_expression_ranking(expression_ranking)

        # complete
        expression_ranking = complete_ranking(expression_ranking, expression_filter)

        return NLUResult(utterance, expression_ranking, self.confidence_threshold, set([]))
