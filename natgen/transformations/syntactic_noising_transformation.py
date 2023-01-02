from typing import Union, Tuple

import nltk
import numpy as np

from . import TransformationBase, NoTransformation


def masking(tokens, p):
    new_tokens = []
    for t in tokens:
        if np.random.uniform() < p:
            new_tokens.append("<mask>")
        else:
            new_tokens.append(t)
    return " ".join(new_tokens)


def deletion(tokens, p):
    new_tokens = []
    for t in tokens:
        if np.random.uniform() >= p:
            new_tokens.append(t)
    return " ".join(new_tokens)


def token_infilling(tokens, p):
    new_tokens = []
    max_infilling_len = round(int(p * len(tokens)) / 2.)
    infilling_len = np.random.randint(1, max_infilling_len)
    start_index = np.random.uniform(high=(len(tokens) - infilling_len))
    end_index = start_index + infilling_len
    for i, t in enumerate(tokens):
        if i < start_index or i > end_index:
            new_tokens.append(t)
    return " ".join(new_tokens)


class SyntacticNoisingTransformation(TransformationBase):
    def __init__(self, parser_path: str, language: str, noise_ratio=0.15):
        # super().__init__(parser_path, language)
        self.language = language
        if self.language == "nl":
            self.tokenizer = nltk.word_tokenize
        else:
            self.tokenizer = NoTransformation(parser_path, language)
        self.noise_ratio = noise_ratio

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        if self.language == "nl":
            tokens = self.tokenizer(code)
        else:
            tokenized_code, _ = self.tokenizer.transform_code(code)
            tokens = tokenized_code.split()
        p = np.random.uniform()
        if p < 0.33:
            transformed_code = masking(tokens, self.noise_ratio)
        elif p < 0.66:
            transformed_code = deletion(tokens, self.noise_ratio)
        else:
            transformed_code = token_infilling(tokens, self.noise_ratio)
        return transformed_code, {
            "success": True
        }
