import logging
import typing
from typing import List, Optional

from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLU, NLUResult, \
    create_nlu_train_test_data, sort_expression_ranking
from dialogue_bot.models.utils import complete_ranking, RankingScore
from dialogue_bot.utils.lists import list_aggregate_func
from tqdm import tqdm
from whoosh.fields import Schema, TEXT, ID
from whoosh.query import And

if typing.TYPE_CHECKING:
    from dialogue_bot.bot_env import BotEnv
    from dialogue_bot.intent_understanding.natural_language_understanding.nlu import NLUTrainingData

logger = logging.getLogger(__name__)


class IndexExpressionRanker(NLU):
    """
    This NLU only ranks expressions => no entity extraction.
    This NLU creates a set of phrases from phrase-patterns (from `NLExpression`) which are stored into the index.
    For each incoming user utterance, the similarities from the annotated utterance to all phrases are calculated by the index.
    """
    def __init__(self, env: 'BotEnv', id: str, aggregation='max'):
        super().__init__(env, id)
        self.aggregation = aggregation
        self._index_name = '{}.{}'.format(self.__class__.__name__, id)
        self._schema = Schema(
            expression=ID(stored=True),
            text=TEXT(stored=True)
        )

    def init(self, retrain: bool, training_data: Optional['NLUTrainingData'] = None):
        logger.info('Initializing {} (retrain={}) ...'.format(self.id, retrain))

        if not self.env.index_handler.exists_index(self._index_name):
            retrain = True

        # train
        if retrain:
            logger.info('Train {}...'.format(self.id))

            # reset index
            if self.env.index_handler.exists_index(self._index_name):
                self.env.index_handler.delete_index(self._index_name)
            self.env.index_handler.ensure_index(self._index_name, self._schema)

            if training_data is None:
                training_data, _ = create_nlu_train_test_data(self.env, test_size=0)

            # index training data
            logger.info('Indexing data...')
            for pattern in tqdm(training_data.phrase_patterns):

                # TODO: problem here is that Whoosh is very much focused on keywords, if only some phrases are generated
                # some entities or entity-values may not appear in the index. Therefore this may lead to some expressions
                # not being recognized
                for phrase in pattern.generate_phrases(self.env, 50):

                    self.env.index_handler.index_object(self._index_name, {
                        'expression': phrase.expression_id,
                        'text': phrase.text,
                    })
            self.env.index_handler.commit(self._index_name)
        else:
            # ensure index
            self.env.index_handler.ensure_index(self._index_name, self._schema)

            logger.info('Loaded {}...'.format(self.id))

    def run(self, utterance: str, intent_filter: List[str] = None) -> 'NLUResult':
        if intent_filter is None:
            intent_filter = [intent.id for intent in self.env.intents(intent_filter=intent_filter)]
        expression_filter = [expression.id for expression in self.env.nl_expressions(intent_filter=intent_filter)]

        # prepare query
        search_query = And([
            self.env.index_handler.query_attr_in_value_list('expression', expression_filter),
            self.env.index_handler.more_like_this_query(self._index_name, self.env.language, 'text', utterance)
        ])

        # query
        expression_scores = {}
        for x in self.env.index_handler.find(self._index_name, search_query=search_query):
            score, obj = x['score'], x['obj']
            expression_scores[obj['expression']] = expression_scores.get(obj['expression'], []) + [score]

        # aggregate
        for k, vs in expression_scores.items():
            aggregate_func = list_aggregate_func(self.aggregation)
            expression_scores[k] = aggregate_func(vs)
        expression_ranking = [RankingScore(k, s) for k, s in expression_scores.items()]

        # sort
        expression_ranking = sort_expression_ranking(expression_ranking)

        # complete
        expression_ranking = complete_ranking(expression_ranking, expression_filter)

        return NLUResult(utterance, expression_ranking, 0., set([]))
