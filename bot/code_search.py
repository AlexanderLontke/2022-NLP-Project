import torch
import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import RobertaTokenizer, RobertaModel, TextDataset
from torch.utils.data import DataLoader, SequentialSampler


class CodeSearchResult:
    def __init__(self, code_string: str, doc_string: str):
        self.code_string = code_string
        self.doc_string = doc_string


class CodeSearch(ABC):
    @abstractmethod
    def find_code_for_query(self, query: str) -> CodeSearchResult:
        pass


class RobertaCodeSearch(CodeSearch):
    def __init__(self, recompute_embeddings: bool = False):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("pytorch_model.bin")
        self.query_dataset = TextDataset(self.tokenizer, codebase_file)

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
            self.vecs = np.load(file="./embeddings.npy")

    def find_code_for_query(self, query: str) -> CodeSearchResult:
        query_vec = self.model(self.tokenizer(query, return_tensors="pt")["input_ids"])
        scores = torch.einsum("ab,cb->ac", query_vec, self.vecs)
        scores = torch.softmax(scores, -1)
        scores = np.squeeze(scores)
        scores = scores.argsort()[-5:][::-1]
        index = torch.argmax(scores)
        result = self.query_dataset.data[index]
        return CodeSearchResult(result["code"], result["docstring"])
