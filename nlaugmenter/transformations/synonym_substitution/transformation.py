'''
This file includes transformations on docstring.
Functions are customized based on nlaugmenter (https://github.com/GEM-benchmark/NL-Augmenter).
Original Copyright (c) 2021 GEM-benchmark. Licensed under the MIT License.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

import re
import nltk
import numpy as np
import spacy
from nltk.corpus import wordnet

from nlaugmenter.interfaces.SentenceOperation import SentenceOperation
from nlaugmenter.tasks.TaskTypes import TaskType
from nlaugmenter.utils.initialize import spacy_nlp

"""
Base Class for implementing the different input transformations a generation should be robust against.
"""


def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    ref: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py#L28
    """
    text = " ".join(words)
    step1 = (
        text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
    )
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
    step5 = (
        step4.replace(" '", "'")
        .replace(" n't", "n't")
        .replace("can not", "cannot")
    )
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def synonym_substitution(
    text, spacy_pipeline, seed=42, prob=0.5, max_outputs=1
):
    np.random.seed(seed)
    upos_wn_dict = {
        "VERB": "v",
        "NOUN": "n",
        "ADV": "r",
        "ADJ": "s",
    }

    doc = spacy_pipeline(text)
    results = []
    for _ in range(max_outputs):
        result = []
        for token in doc:
            word = token.text
            wn_pos = upos_wn_dict.get(token.pos_)
            if wn_pos is None:
                result.append(word)
            else:
                syns = wordnet.synsets(word, pos=wn_pos)
                syns = [syn.name().split(".")[0] for syn in syns]
                syns = [syn for syn in syns if syn.lower() != word.lower()]
                coin = np.random.random()
                if len(syns) > 0 and coin < prob:
                    choice = np.random.choice(syns).replace("_", " ")
                    result.append(choice)
                    # print(token, len(syns), coin, prob, choice)
                else:
                    result.append(word)
        # import pdb; pdb.set_trace()
        # detokenize sentences
        result = untokenize(result)
        if result not in results:
            # make sure there is no dup in results
            results.append(result)
    return results


"""
Substitute words with synonyms using stanza (for POS) and wordnet via nltk (for synonyms)
"""


class SynonymSubstitution(SentenceOperation):
    tasks = [
        TaskType.TEXT_CLASSIFICATION,
        TaskType.TEXT_TO_TEXT_GENERATION,
    ]
    languages = ["en"]

    def __init__(self, seed=42, prob=0.5, max_outputs=1):
        super().__init__(seed, max_outputs=max_outputs)
        self.nlp = spacy_nlp if spacy_nlp else spacy.load("en_core_web_sm")
        self.prob = prob
        nltk.download("wordnet")
        ''' Start of Amazon Addition '''
        self.perturb_level = "word-level"
        ''' End of Amazon Addition '''

    def generate(self, sentence: str):
        perturbed = synonym_substitution(
            text=sentence,
            spacy_pipeline=self.nlp,
            seed=self.seed,
            prob=self.prob,
            max_outputs=self.max_outputs,
        )
        return perturbed
