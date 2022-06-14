import logging
import os
import threading
from datetime import datetime

import coloredlogs
from termcolor import colored


def timestamp(milliseconds=False) -> str:
    now = datetime.now()
    if milliseconds:
        return now.strftime("%d/%m/%Y %H:%M:%S.%f")
    else:
        return now.strftime("%d/%m/%Y %H:%M:%S")


_LOGGING_COLORS = {
    "intent": "yellow",
    "entity": "green",
    "value": "blue",
    "expression": "blue",
    "phrase": "blue",
    "domain": "blue",
    "slot": "green",
    "context": "blue",
    "bot-utterance": "blue",
    "user-utterance": "blue",
}


def logcolor(key, text):
    return colored(text, _LOGGING_COLORS.get(key, "blue"))


coloredlogs.install(
    level="DEBUG",
    fmt="{}:{}".format(os.getpid(), threading.get_ident())
    + " %(module)-20s%(levelname)-10s%(message)s",
)
logging.basicConfig(level=logging.DEBUG)
