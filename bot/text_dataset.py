import json
import os
import logging
import torch
from torch.utils.data import Dataset

"""
Dataset used for the training and evaluation process
"""


class InputFeatures(object):
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer):
    # Set maximum characters for natural language and code. Length were definied based on this notebook:
    # https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb

    code_length = 256
    nl_length = 128

    code = ' '.join(js['code_tokens'])
    code_tokens = tokenizer.tokenize(code)[:code_length - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.examples = []
        self.data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                self.data.append(js)

        for js in self.data:
            self.examples.append(convert_examples_to_features(js, tokenizer))

        logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG)
        # Print first three examples
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))
