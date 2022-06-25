import torch
import numpy as np

from tqdm import tqdm
from torch import nn
from abc import ABC, abstractmethod
from transformers import RobertaTokenizer, RobertaModel
from text_dataset import TextDataset
from torch.utils.data import DataLoader, SequentialSampler


class CodeSearch(ABC):
    @abstractmethod
    def find_code_for_query(self, query: str) -> str:
        pass


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            return self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]


class RobertaCodeSearch(CodeSearch):
    def __init__(self, recompute_embeddings: bool = False):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = Model(RobertaModel.from_pretrained(pretrained_model_name_or_path="python_model/"))
        self.query_dataset = TextDataset(self.tokenizer, "codebase.jsonl")

        if recompute_embeddings:
            codebase_file = ""
            query_dataset = TextDataset(self.tokenizer, codebase_file)
            query_sampler = SequentialSampler(query_dataset)
            query_dataloader = DataLoader(
                query_dataset, sampler=query_sampler, batch_size=128, num_workers=4
            )

            code_vecs = []

            for batch in tqdm(query_dataloader):
                code_inputs = batch[0]
                with torch.no_grad():
                    code_vec = self.model(code_inputs=code_inputs)
                    code_vecs.append(code_vec.cpu().numpy())

            # Delete last embedding to have a valid shape
            code_vecs = np.delete(code_vecs, -1)

            vecs = np.vstack(code_vecs)
            self.vecs = torch.from_numpy(vecs)
        else:
            self.vecs = torch.from_numpy(np.load(file="./embeddings.npy"))

    def find_code_for_query(self, query: str) -> str:
        query_vec = self.model(self.tokenizer(query, return_tensors="pt")["input_ids"])
        scores = torch.einsum("ab,cb->ac", query_vec, self.vecs)
        scores = torch.softmax(scores, -1).detach().numpy()
        scores = np.squeeze(scores)
        scores = scores.argsort()[-5:][::-1]
        result = self.query_dataset.data[scores[0]]

        return result["code"]
