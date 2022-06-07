from typing import Iterable

from nltk.corpus import stopwords as ntk_stopwords


def stopwords(lang: str):
    """ Returns stopwords (in lowercase) for the given language"""
    if lang == 'en':
        return ntk_stopwords.words('english')
    else:
        raise NotImplementedError('No stopwords defined for language "{}"'.format(lang))


def word_variations(lang: str, word: str) -> Iterable[str]:
    """ Returns a list of word variations (including word) for the given language """
    if lang == 'en':
        from whoosh.lang.morph_en import variations
        res = variations(word)
        return list(res)
    else:
        raise NotImplementedError('No stopwords defined for language "{}"'.format(lang))


if __name__ == '__main__':
    print(stopwords('en'))
    print(word_variations('en', 'is'))
    print(word_variations('en', 'President'))
