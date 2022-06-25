from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docstring_parser import parse


class Paraphraser(ABC):
    @abstractmethod
    def paraphrase(self, input_string: str) -> str:
        pass


# Implementation


class T5Paraphraser(Paraphraser):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    def paraphrase(self, input_string: str) -> str:
        parsed_docstring = parse(input_string)
        parsed_description = ""
        if parsed_docstring.short_description is not None:
            parsed_description += parsed_docstring.short_description
        elif parsed_docstring.long_description is not None:
            parsed_description += parsed_docstring.long_description

        text = "paraphrase: " + parsed_description + " </s>"
        encoding = self.tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"], encoding[
            "attention_mask"
        ]
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1,
        )
        result = ""
        for output in outputs:
            result += self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        return result  # + "\nOriginal docstring: " + input_string
