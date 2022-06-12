import logging
import typing
from collections import defaultdict
from typing import Set, List, Optional, Dict

from dialogue_bot import logcolor
from dialogue_bot.intent_understanding.iu import IU, IUResult, sort_intent_ranking
from dialogue_bot.intent_understanding.natural_language_understanding.index_er import IndexExpressionRanker
from dialogue_bot.intent_understanding.natural_language_understanding.nlu import create_nlu_train_test_data, NLUTestData
from dialogue_bot.intent_understanding.natural_language_understanding.phrase_er import PhraseExpressionRanker
from dialogue_bot.intent_understanding.natural_language_understanding.pipeline_nlu import PipelineNLU, \
    NLUPipelineElement, ExpressionUsage
from dialogue_bot.intent_understanding.natural_language_understanding.rasa_nlu.rasa_nlu import RasaNLU
from dialogue_bot.intent_understanding.natural_language_understanding.regex_ee import RegexEntityExtractor
from dialogue_bot.intent_understanding.natural_language_understanding.regex_er import RegexExpressionRanker
from dialogue_bot.models.input import UserInput
from dialogue_bot.models.inputs.nl import NLInput
from dialogue_bot.models.intent import PerfectScore, CustomScore, FloatScore
from dialogue_bot.models.state import DialogueState
from dialogue_bot.models.triggers.nl import PhraseNLTrigger, AnyNLTrigger, FallbackNLTrigger
from dialogue_bot.models.utils import RankingScore
from dialogue_bot.nlp.text_scorer.vector import VectorTextScorer
from dialogue_bot.nlp.tokenizer.nlt import NLTKWordTokenizer
from dialogue_bot.nlp.vectorizer.text.agg import AggTextVectorizer
from dialogue_bot.nlp.vectorizer.text.senttrans import SentenceTransformerVectorizer
from dialogue_bot.nlp.vectorizer.word.gensi import GensimVectorizer
from dialogue_bot.utils.rand import uuid_str

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.intent_understanding.iu import ExtractedEntity

logger = logging.getLogger(__name__)


class MyNLUResult(object):
    """ Natural-Language Expression-Understanding Result """

    def __init__(self, utterance: str, intent_ranking: List['RankingScore'], confidence_threshold: float,
                 entities: Set['ExtractedEntity'], nlu_result: 'NLUResult'):
        self.utterance = utterance
        self.intent_ranking = intent_ranking
        self.confidence_threshold = confidence_threshold
        self.entities = set(entities)
        self.nlu_result = nlu_result


class DefaultIU(IU):

    def __init__(self, env: 'BotEnv',
                 default_nlu: str = 'senttran',
                 active_nlus: list = ['senttran']
                 ):
        """
        The Itent-Understanding component.
        :param env: The Bot.
        :param default_nlu: The nlu-id that will be used per default. It has to be one of `active_nlus`
        :param active_nlus: The nlu-ids that can actually be used.
            Since retraining NLUs takes much time, we do not retrain all available NLUs, but only those that are being used.
            For evaluation purposes however, we will consider all NLUs
        """
        self.env = env
        self.default_nlu = default_nlu
        self.active_nlus = active_nlus
        self.nlus = {}

        self.add_nlu(PipelineNLU(env, 'rasa', [
            NLUPipelineElement(RegexExpressionRanker(env, 'rasa.regex'),
                               expression_usage=ExpressionUsage.PERFECT_MATCH),
            NLUPipelineElement(RasaNLU(env, 'rasa.rasa', confidence_threshold=0.48))
        ]))

        self.add_nlu(PipelineNLU(env, 'senttran', [
            NLUPipelineElement(RegexExpressionRanker(env, 'senttran.expression-regex'),
                               expression_usage=ExpressionUsage.PERFECT_MATCH),
            NLUPipelineElement(IndexExpressionRanker(env, 'senttran.index'), expression_usage=ExpressionUsage.FILTER_TOPK, expression_top_k=50),
            NLUPipelineElement(PhraseExpressionRanker(env, 'senttran.phrase',
                                                      VectorTextScorer('senttran.scorer',
                                                                       SentenceTransformerVectorizer(
                                                                           'senttran.senttran'),
                                                                       False
                                                                       ),
                                                      confidence_threshold=0.7
                                                      )
                               ),
            NLUPipelineElement(RegexEntityExtractor(env, 'senttran.entity-regex'), use_expressions=False),
        ]))

    def add_nlu(self, nlu: 'NLU'):
        self.nlus[nlu.id] = nlu

    def get_active_nlus(self) -> dict:
        return {k: v for k, v in self.nlus.items() if k in self.active_nlus}

    def _determine_nlu(self, user_input: 'NLInput', dialogue_state: 'DialogueState') -> str:
        """
        We could use a classifier here that determines the best NLU implementation based on the user utterance.
        But this is not implemented yet.
        """
        return self.default_nlu

    def _process_input(self, user_input: 'UserInput', dialogue_state: 'DialogueState'):
        # e.g. annotate nl-input
        pass

    def _determine_matchable_intents(self, user_input: 'UserInput', dialogue_state: 'DialogueState') -> Set[str]:
        return set([intent.id for intent in self.env.intents()
                    if intent.has_contexts(dialogue_state)
                    and intent.has_trigger(user_input)])

    def _expression_ranking_to_intent_ranking(self, dialogue_state: 'DialogueState', intent_filter,
                                              expression_ranking) -> List['RankingScore']:
        expression_scores = {s.ref_id: s.score for s in expression_ranking}
        intent_scores = {}
        for intent_id in intent_filter:
            intent = self.env.intent(intent_id)
            if (intent.nl_trigger is not None) and (isinstance(intent.nl_trigger, PhraseNLTrigger)) and (
                    intent.nl_trigger.expression.id in expression_scores):
                intent_scores[intent_id] = expression_scores[intent.nl_trigger.expression.id]

        intent_ranking = [RankingScore(i, s) for i, s in intent_scores.items()]
        intent_ranking = sort_intent_ranking(self.env, dialogue_state, intent_ranking)
        return intent_ranking

    def _get_nl_fallback_intent(self, dialogue_state: 'DialogueState', matchable_intents) -> Optional[str]:
        logger.info('Searching NL-Fallback-Intent')

        intent_ids = set([])
        for intent_id in matchable_intents:
            intent = self.env.intent(intent_id)
            if (intent.nl_trigger is not None) and isinstance(intent.nl_trigger, FallbackNLTrigger):
                intent_ids.add(intent_id)
        intent_ids = list(intent_ids)

        logger.info(logcolor('intent', 'NL-Fallback-Intents:'))
        for e in intent_ids[:15]:
            logger.info(logcolor('intent', '\t{}'.format(e)))

        if len(intent_ids) > 0:
            intent_id = intent_ids[0]
            return intent_id

        return None

    def _custom_extract_entities(self, intent: 'Intent', entities: Set['ExtractedEntity'], user_input: 'UserInput',
                                 dialogue_state: 'DialogueState') -> Set['ExtractedEntity']:
        return intent.custom_extract(user_input, dialogue_state, entities)

    def init(self, retrain: bool):
        logger.info('Initializing {} (retrain={}) ...'.format(self.__class__.__name__, retrain))

        # NLU
        for nlu in self.get_active_nlus().values():
            nlu.init(retrain)

    def _run_nlu(self, user_input: 'NLInput', dialogue_state: 'DialogueState', intent_filter) -> 'MyNLUResult':

        logger.info('Determine NLU...')
        nlu_name = self._determine_nlu(user_input, dialogue_state)

        logger.info('Using NLU "{}"'.format(nlu_name))
        nlu = self.get_active_nlus()[nlu_name]

        logger.info('Run NLU...')
        nlu_res = nlu.run(user_input.text, intent_filter)

        # log
        logger.info(logcolor('expression',
                             'NLU-Expression-Ranking (confidence-threshold={}):'.format(nlu_res.confidence_threshold)))
        for e in nlu_res.expression_ranking[:15]:
            logger.info(
                logcolor('expression', '\t{} {}'.format('+' if (e.score >= nlu_res.confidence_threshold) else '-', e)))

        logger.info(logcolor('entity', 'NLU-Entities:'))
        for e in nlu_res.entities:
            logger.info(logcolor('entity', '\t● {}'.format(e)))

        # map expressions to intents
        logger.info('Map Expression-results to intents...')
        nlu_intent_ranking = self._expression_ranking_to_intent_ranking(dialogue_state, intent_filter,
                                                                        nlu_res.expression_ranking)

        logger.info(
            logcolor('intent', 'NLU-Intent-Ranking (confidence-threshold={}):'.format(nlu_res.confidence_threshold)))
        for e in nlu_intent_ranking[:15]:
            logger.info(
                logcolor('intent', '\t{} {}'.format('+' if (e.score >= nlu_res.confidence_threshold) else '-', e)))

        return MyNLUResult(user_input.text, nlu_intent_ranking, nlu_res.confidence_threshold, nlu_res.entities, nlu_res)

    def _score_matchable_intents(self, user_input: 'UserInput', dialogue_state: 'DialogueState',
                                 nlu_res: Optional['MyNLUResult'], matchable_intents: List[str]):
        logger.info('(Re)-Score Intents...')

        confidence_threshold = 0
        nlu_intent_scores = {}

        if nlu_res is not None:
            confidence_threshold = nlu_res.confidence_threshold
            for s in nlu_res.intent_ranking:
                nlu_intent_scores[s.ref_id] = s.score

        intent_scores = {}
        for intent_id in matchable_intents:
            intent = self.env.intent(intent_id)
            score = self._score_matchable_intent(user_input, dialogue_state, intent,
                                                 nlu_intent_scores.get(intent_id, None))
            intent_scores[intent_id] = score

        # check if there are any perfect scores
        perfect_intents = [i for i, s in intent_scores.items() if isinstance(s, PerfectScore)]
        if len(perfect_intents) > 0:
            intent_ranking = [RankingScore(i, 1. if isinstance(s, PerfectScore) else 0.) for i, s in
                              intent_scores.items()]
            confidence_threshold = 1.
        else:
            intent_ranking = [RankingScore(i, s.value) for i, s in intent_scores.items()]
            confidence_threshold = confidence_threshold

        intent_ranking = sort_intent_ranking(self.env, dialogue_state, intent_ranking)

        logger.info(logcolor('intent', 'Intent-Ranking (confidence-threshold={}):'.format(confidence_threshold)))
        for e in intent_ranking[:15]:
            logger.info(logcolor('intent', '\t{} {}'.format('+' if (e.score >= confidence_threshold) else '-', e)))

        return intent_ranking, confidence_threshold

    def _score_matchable_intent(self, user_input: 'UserInput', dialogue_state: 'DialogueState', intent: 'Intent',
                                nlu_score: Optional[float]) -> 'CustomScore':
        if isinstance(user_input, NLInput):
            if (intent.nl_trigger is not None) and isinstance(intent.nl_trigger, AnyNLTrigger):
                score = PerfectScore()
            else:
                score = FloatScore(nlu_score if nlu_score is not None else 0)
        else:
            score = PerfectScore()

        # custom score
        score = intent.custom_score(user_input, dialogue_state, score)

        return score

    def _extract_entities(self, intent_id: Optional[str], nlu_res: Optional, user_input: 'UserInput',
                          dialogue_state: 'DialogueState') -> Set['ExtractedEntity']:

        # use entities from nlu_res
        entities = nlu_res.entities if nlu_res is not None else set([])

        # add custom entities extracted from intent
        if intent_id is not None:
            intent = self.env.intent(intent_id)
            entities = self._custom_extract_entities(intent, entities, user_input, dialogue_state)

        logger.info(logcolor('entity', 'Entities:'))
        for e in entities:
            logger.info(logcolor('entity', '\t● {}'.format(e)))

        return entities

    def run(self, user_input: 'UserInput', dialogue_state: 'DialogueState') -> 'IUResult':
        logger.info('Running {}...'.format(self.__class__.__name__))

        logger.info('Processing Input {}...'.format(user_input))
        self._process_input(user_input, dialogue_state)

        # determine matchable intents
        logger.info('Determine matchable intents...'.format(user_input))
        matchable_intents = list(self._determine_matchable_intents(user_input, dialogue_state))
        logger.info(logcolor('intent', 'There are {} / {} intents to consider'.format(len(matchable_intents),
                                                                                      len(list(self.env.intents())))))
        for intent_id in matchable_intents[:15]:
            logger.info(logcolor('intent', '\t● {}'.format(intent_id)))

        # run NLU
        nlu_res: 'MyNLUResult' = None
        if isinstance(user_input, NLInput):
            nlu_res = self._run_nlu(user_input, dialogue_state, matchable_intents)

        # score intents
        intent_ranking, confidence_threshold = self._score_matchable_intents(user_input, dialogue_state, nlu_res,
                                                                             matchable_intents)

        # determine intent
        top_intent_id = None
        if len(intent_ranking) > 0:
            top_intent_id = intent_ranking[0].ref_id
            logger.info(logcolor('intent', 'Top-Intent found: {}'.format(top_intent_id)))
        else:
            logger.info(logcolor('intent', 'No Top-Intent found'))

        # determine (custom) fallback intent
        fallback_intent_id = None
        if isinstance(user_input, NLInput):
            fallback_intent_id = self._get_nl_fallback_intent(dialogue_state, matchable_intents)
            if fallback_intent_id is not None:
                logger.info(logcolor('intent', 'NL-Fallback-Intent found: "{}"'.format(fallback_intent_id)))
            else:
                logger.info(logcolor('intent', 'No NL-Fallback-Intent found'))

        # determine entities
        entities = self._extract_entities(top_intent_id, nlu_res, user_input, dialogue_state)

        # Return
        return IUResult(user_input, intent_ranking, confidence_threshold, entities,
                        nlu_res.nlu_result if nlu_res is not None else None, fallback_intent_id)

    def update_entities(self, intent_id: str, iu_result: 'IUResult', dialogue_state: 'DialogueState') -> Set[
        'ExtractedEntity']:
        nlu_res = iu_result.nlu_result
        user_input = iu_result.user_input

        # We re-extract entities according to the correct intent
        entities = self._extract_entities(intent_id, nlu_res, user_input, dialogue_state)

        return entities

    def _domain_nl_expressions(self) -> Dict[str, set]:
        """
        For each domain, returns its expression ids
        :return: {domain: expression-ids}
        """
        res = defaultdict(set)
        for intent in self.env.intents():
            if (intent.nl_trigger is None) or (not isinstance(intent.nl_trigger, PhraseNLTrigger)):
                continue
            for domain in intent.domains:
                res[domain].add(intent.nl_trigger.expression.id)
        return res

    def _domain_test_data(self, test_data: 'NLUTestData') -> Dict[str, 'NLUTestData']:
        """
        Subdivides the test_data into multiple test_data for each existing domain
        :return: {domain: 'NLUTestData'}
        """
        domain_nl_expressions = self._domain_nl_expressions()
        res = {}
        for domain, expression_ids in domain_nl_expressions.items():
            domain_phrase_patterns = [p for p in test_data.phrase_patterns if p.expression_id in expression_ids]
            res[domain] = NLUTestData(domain_phrase_patterns)
        return res

    def evaluate_nlus(self, nlu_filter: List[str] = None, intent_filter: List[str] = None):
        if nlu_filter is None:
            nlu_filter = list(self.nlus.keys())

        logger.info('Evaluate NLUs: {}'.format(nlu_filter))
        train_data, test_data = create_nlu_train_test_data(self.env, intent_filter=intent_filter, test_size=0.33)
        evaluation_id = uuid_str(6)

        # subdivide test_data into domain-specific test-data
        domain_test_data = self._domain_test_data(test_data)
        domain_test_data['__all__'] = test_data

        logger.info('Training-data: {}'.format(train_data))
        for domain, t_data in domain_test_data.items():
            logger.info('Test-data for domain "{}":     {}'.format(domain, t_data))

        for nlu_id in nlu_filter:
            self.nlus[nlu_id].evaluate(train_data, domain_test_data, evaluation_id=evaluation_id)
