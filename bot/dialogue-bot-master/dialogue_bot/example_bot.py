from dialogue_bot.bot_env import BotEnv
from dialogue_bot.bot_session import BotSession
from dialogue_bot.models.inputs.nl import NLInput

bot = BotEnv("example-bot", "en")

# Values
bot.register_value("COUNTRY_DE", regex_patterns=[r"(?i:Deuts\w+)", r"(?i:Germa\w+)"])
bot.register_value("COUNTRY_FR", synonyms=["Frankreich", "France"])
bot.register_value("COUNTRY_IT", synonyms=["Italien", "Italy"])

# Entities
bot.register_entity(
    "living-country",
    value_refs=["COUNTRY_DE", "COUNTRY_FR", "COUNTRY_IT"],
    questions=["Where do you live?"],
)

bot.register_intent("hello")
bot.register_intent_phrase_patterns(
    "hello",
    [
        "Hi",
        "hello",
        "ahoi",
        "good morning",
        "good evening",
        "good afternoon",
        "hi there",
    ],
)

bot.register_intent("bye")
bot.register_intent_phrase_patterns(
    "bye",
    ["Good bye", "bye", "see you", "until later", "see you later", "I have to go"],
)

bot.register_intent("living")
bot.register_intent_phrase_patterns(
    "living",
    [
        "I live in ((living-country))",
        "I moved to ((living-country))",
        "My home country is ((living-country))",
        "((living-country)) is my home country",
    ],
)

if __name__ == "__main__":
    bot.start(True)

    session = BotSession(bot)
    bot.respond(session, NLInput("I live in Germany"))
