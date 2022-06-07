import typing

if typing.TYPE_CHECKING:
    pass


# class NLUTrainingData(object):
#     def __init__(self,
#                  expression_phrase_patterns: Set['PhrasePattern'] = None,
#                  expression_regex_patterns: Dict[str, Set[str]] = None,
#                  expression_exclude_regex_patterns: Dict[str, Set[str]] = None,
#                  entity_values: Dict[str, Set[str]] = None,
#                  value_synonyms: Dict[str, Set[str]] = None,
#                  value_regex_patterns: Dict[str, Set[str]] = None,
#                  ):
#         """
#         :param trigger_phrases:                    {'name-trigger': [Phrase('Hello, my name is Matthias')]}
#         :param expression_regex_patterns:             {'name-trigger': [r'Hello, my name is .*?']}
#         :param expression_exclude_regex_patterns:     {'name-trigger': [r'Hello']}
#         :param entity_values:                      {'country': ['DE' 'CH'], 'phone_nr': ['TELEPHONE_NR']}
#         :param value_synonyms:                     {'DE': ['Germany'], 'CH': ['Switzerland']}
#         :param value_regex_patterns:               {'TELEPHONE_NR': [r'^\d+']}
#         """
#
#         if trigger_phrases is None:
#             trigger_phrases = []
#         if expression_regex_patterns is None:
#             expression_regex_patterns = {}
#         if expression_exclude_regex_patterns is None:
#             expression_exclude_regex_patterns = {}
#         if entity_values is None:
#             entity_values = {}
#         if value_synonyms is None:
#             value_synonyms = {}
#         if value_regex_patterns is None:
#             value_regex_patterns = {}
#
#         self.trigger_phrases = trigger_phrases
#         self.trigger_regex_patterns = expression_regex_patterns
#         self.trigger_exclude_regex_patterns = expression_exclude_regex_patterns
#         self.entity_values = entity_values
#         self.value_synonyms = value_synonyms
#         self.value_regex_patterns = value_regex_patterns
#
#     @property
#     def entities(self) -> Set[str]:
#         entity_ids = set()
#
#         # collect entities from phrases
#         for phrase in self.trigger_phrases:
#             for entity in phrase.entities:
#                 entity_ids.add(entity.entity)
#
#         # collect entities from entity-values
#         entity_ids.update(self.entity_values.keys())
#
#         return entity_ids
#
#     def __repr__(self):
#         return json.dumps(self, default=lambda o: o.__dict__)
