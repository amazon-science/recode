'''
This file includes transformations on docstring.
Functions are customized based on nlaugmenter (https://github.com/GEM-benchmark/NL-Augmenter).
Original Copyright (c) 2021 GEM-benchmark. Licensed under the MIT License.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

import numpy as np
from nlaugmenter.interfaces.SentenceOperation import SentenceOperation
from nlaugmenter.tasks.TaskTypes import TaskType


class SwapCharactersPerturbation(SentenceOperation):
    tasks = [
        TaskType.TEXT_CLASSIFICATION,
        TaskType.TEXT_TO_TEXT_GENERATION,
    ]
    languages = ["All"]

    ''' Start of Amazon Addition '''
    def __init__(self, seed=0):
        super().__init__(seed)
        self.perturb_level = "char-level"
    ''' End of Amazon Addition '''

    def generate(self, sentence: str, prob=0.05):
        # default is 0.05
        pertubed = self.swap_characters(
            text=sentence, prob=prob, seed=self.seed
        )
        return [pertubed]

    def swap_characters(self, text, prob=0.05, seed=0):
        """
        Swaps characters in text, with probability prob for ang given pair.
        Ex: 'apple' -> 'aplpe'
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters swapping. Default: 0.05
            seed (int): random seed
        """
        max_seed = 2 ** 32
        # seed with hash so each text of same length gets different treatment.
        np.random.seed((seed + sum([ord(c) for c in text])) % max_seed)
        # np.random.seed((seed) % max_seed).
        # number of possible characters to swap.
        num_pairs = len(text) - 1
        # if no pairs, do nothing
        if num_pairs < 1:
            return text
        # get indices to swap.
        indices_to_swap = np.argwhere(
            np.random.rand(num_pairs) < prob
        ).reshape(-1)
        # shuffle swapping order, may matter if there are adjacent swaps.
        np.random.shuffle(indices_to_swap)
        # convert to list.
        text = list(text)
        # swap.
        for index in indices_to_swap:
            if text[index].isalnum() and text[index + 1].isalnum():
                text[index], text[index + 1] = text[index + 1], text[index]
        # convert to string.
        text = "".join(text)
        return text
