LANG_TEXTS = {
    'default_entity_questions': {
        'en': lambda entity_name: [
            'What is the value for {}?'.format(entity_name),
            'Please give a value for {}'.format(entity_name)
        ],
        'de': lambda entity_name: [
            'Was ist der Wert für {}?'.format(entity_name),
            'Bitte gebe einen Wert an für {}'.format(entity_name)
        ]
    },
    'invalid_input.1': {
        'en': [
            'Your input is not valid!',
            'Your input is invalid!',
        ],
        'de': [
            'Dein Input ist nicht gültig!'
            'Dein Input ist ungültig!',
        ]
    },
    'invalid_input.2': {
        'en': [
            '(Your input might be too long or contains invalid characters)'
        ],
        'de': [
            '(Dein Input ist möglicherweise zu lang oder beinhaltet ungültige Zeichen)'
        ]
    },
    'intent_not_understand': {
        'en': [
            'Oops, it seems that I did not understand your intent. Can you try to rephrase it for me?',
            'Sorry, but I did not understand your intent. Can you try to rephrase it for me?',
            'Sorry, can you rephrase that for me?'
        ]
    },
    'verify_intent_not_understand': {
        'en': [
            'Can you help me to figure out your intent?',
            'I am trying to figure out what you were saying:',
            'I am not sure if I understood correctly. Here are some suggestions:',
            'Did you mean one of these intents?',
            'What did you mean by that?'
        ]
    },
    'verify_intent_none.1': {
        'en': [
            'Sorry that I could not help',
            'Okay'
        ]
    },
    'verify_intent_none.2': {
        'en': [
            'Maybe you want to rephrase your intent for me',
            'You can try to rephrase your intent for me'
        ]
    },
    'verify_intent_thanks': {
        'en': [
            'Thank you, this really helps me to learn over time!',
            'Great, this helps me to improve over time!',
            'Thanks! I will remember that.',
        ]
    }
}
