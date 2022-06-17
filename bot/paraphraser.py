from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
        text = "paraphrase: " + input_string + " </s>"
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
            line = self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            result += line
        return result + "\nOriginal docstring: " + input_string
