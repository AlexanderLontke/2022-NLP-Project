from dialogue_bot.models.action import (
    JoinedAction,
    NLAction,
    ContextSetAction,
    SlotSetAction,
    ChoiceAction,
    Choice,
)
from dialogue_bot.static.texts import LANG_TEXTS

utter_invalid_input = lambda env: JoinedAction(
    [
        NLAction(LANG_TEXTS["invalid_input.1"][env.language]),
        NLAction(LANG_TEXTS["invalid_input.2"][env.language]),
    ]
)

utter_intent_not_understand = lambda env: JoinedAction(
    [
        NLAction(LANG_TEXTS["intent_not_understand"][env.language]),
    ]
)

verify_intent = (
    lambda env, intent_scores, iu_result, dialogue_state, user_input: JoinedAction(
        [
            ContextSetAction("verify_intent"),
            SlotSetAction("verify_intent#intents", [s.ref_id for s in intent_scores]),
            SlotSetAction("verify_intent#input", user_input),
            SlotSetAction("verify_intent#iu_result", iu_result),
            SlotSetAction("verify_intent#dialogue_state", dialogue_state),
            SlotSetAction(
                "verify_intent#slots",
                [
                    "verify_intent#intents",
                    "verify_intent#input",
                    "verify_intent#iu_result",
                    "verify_intent#dialogue_state",
                    "verify_intent#slots",
                ],
            ),
            NLAction(LANG_TEXTS["verify_intent_not_understand"][env.language]),
            ChoiceAction(
                [
                    Choice(
                        "intent",
                        i,
                        env.intent(s.ref_id).verify_description,
                        score=s.score,
                    )
                    for i, s in enumerate(intent_scores)
                ]
                + [Choice("none", -1, "None of these")]
            ),
        ]
    )
)

utter_verify_intent_none = lambda env: JoinedAction(
    [
        NLAction(LANG_TEXTS["verify_intent_none.1"][env.language]),
        NLAction(LANG_TEXTS["verify_intent_none.2"][env.language]),
    ]
)

utter_verify_intent_thanks = lambda env: JoinedAction(
    [
        NLAction(LANG_TEXTS["verify_intent_thanks"][env.language]),
    ]
)
