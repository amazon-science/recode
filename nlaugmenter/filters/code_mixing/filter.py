import spacy
from ftlid import identify_language

from nlaugmenter.interfaces.SentenceOperation import SentenceOperation
from nlaugmenter.tasks.TaskTypes import TaskType
from nlaugmenter.utils.initialize import spacy_nlp


class CodeMixing(SentenceOperation):
    """This filter is used to identify code-mixed texts in a dataset.
    It checks that there is at least one sentence in the text where there
    are tokens representing at least `k` unique languages (with at least a
    `threshold` level of confidence that the token is of that language).
    """

    tasks = [
        TaskType.E2E_TASK,
        TaskType.TEXT_TAGGING,
        TaskType.TEXT_CLASSIFICATION,
    ]
    keywords = ["model-based", "tokenizer-required"]
    languages = [
        "en",
        "ru",
        "de",
        "fr",
        "it",
        "ja",
        "es",
        "ceb",
        "tr",
        "pt",
        "uk",
        "eo",
        "pl",
        "sv",
        "nl",
        "he",
        "zh",
        "hu",
        "ar",
        "ca",
        "fi",
        "cs",
        "fa",
        "sr",
        "el",
        "vi",
        "bg",
        "ko",
        "no",
        "mk",
        "ro",
        "id",
        "th",
        "hy",
        "da",
        "ta",
        "hi",
        "hr",
        "sh",
        "be",
        "ka",
        "te",
        "kk",
        "war",
        "lt",
        "gl",
        "sk",
        "bn",
        "eu",
        "sl",
        "kn",
        "ml",
        "mr",
        "et",
        "az",
        "ms",
        "sq",
        "la",
        "bs",
        "nn",
        "ur",
        "lv",
        "my",
        "tt",
        "af",
        "oc",
        "nds",
        "ky",
        "ast",
        "tl",
        "is",
        "ia",
        "si",
        "gu",
        "km",
        "br",
        "ba",
        "uz",
        "bo",
        "pa",
        "vo",
        "als",
        "ne",
        "cy",
        "jbo",
        "fy",
        "mn",
        "lb",
        "ce",
        "ug",
        "tg",
        "sco",
        "sa",
        "cv",
        "jv",
        "min",
        "io",
        "or",
        "as",
        "new",
        "ga",
        "mg",
        "an",
        "ckb",
        "sw",
        "bar",
        "lmo",
        "yi",
        "arz",
        "mhr",
        "azb",
        "sah",
        "pnb",
        "su",
        "bpy",
        "pms",
        "ilo",
        "wuu",
        "ku",
        "ps",
        "ie",
        "xmf",
        "yue",
        "gom",
        "li",
        "mwl",
        "kw",
        "sd",
        "hsb",
        "scn",
        "gd",
        "pam",
        "bh",
        "mai",
        "vec",
        "mt",
        "dv",
        "wa",
        "mzn",
        "am",
        "qu",
        "eml",
        "cbk",
        "tk",
        "rm",
        "os",
        "vls",
        "yo",
        "lo",
        "lez",
        "so",
        "myv",
        "diq",
        "mrj",
        "dsb",
        "frr",
        "ht",
        "gn",
        "bxr",
        "kv",
        "sc",
        "nah",
        "krc",
        "bcl",
        "nap",
        "gv",
        "av",
        "rue",
        "xal",
        "pfl",
        "dty",
        "hif",
        "co",
        "lrc",
        "vep",
        "tyv",
    ]

    def __init__(self, k=2, threshold=0.5):
        super().__init__()
        self.nlp = spacy_nlp if spacy_nlp else spacy.load("en_core_web_sm")
        self.k = k
        self.threshold = threshold

    def filter(self, sentence: str) -> bool:
        doc = self.nlp(sentence)
        for sentence in doc.sents:
            languages = set()
            for token in sentence:
                (language,), (prob,) = identify_language(
                    token.text, with_prob=True
                )
                if prob >= self.threshold:
                    languages.add(language)
            if len(languages) >= self.k:
                return True
        return False
