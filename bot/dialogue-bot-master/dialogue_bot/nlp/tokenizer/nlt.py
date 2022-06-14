from dialogue_bot.nlp.tokenizer.tokenizer import WordTokenizer
from nltk import TreebankWordTokenizer, sent_tokenize


def locate_tokens(original_text, tokens):
    res = []
    idx = 0
    for token in tokens:
        if token in original_text[idx:]:
            idx = idx + original_text[idx:].index(token)
            res.append(idx)
            idx += len(token)
        else:
            res.append(None)
    return res


class NLTKWordTokenizer(WordTokenizer):
    word_tokenizer = TreebankWordTokenizer()

    def tokenize_spans(self, text):
        res = []
        sentences = sent_tokenize(
            text
        )  # TreebankWordTokenizer assumes that it operates on a single sentence
        sentence_idxs = locate_tokens(text, sentences)

        for sent, sent_idx in zip(sentences, sentence_idxs):
            spans = [
                (sent_idx + fr, sent_idx + to)
                for fr, to in NLTKWordTokenizer.word_tokenizer.span_tokenize(sent)
            ]
            res.extend(spans)
        return res


if __name__ == "__main__":
    tokenizer = NLTKWordTokenizer("test")
    print(tokenizer.tokenize("This is my text. And this is another one."))
    print(tokenizer.tokenize_spans("This is my text. And this is another one."))
