# yaml -> clinlp
# results aggregator
# class for ETL pob filter

import logging
import re
from collections import defaultdict

import spacy
from clinlp.ie import Term
from spacy.lang.nl import stop_words
from spacy.tokens import Doc
from tqdm import tqdm

STOPWORDS = stop_words.STOP_WORDS


from pathlib import Path
from typing import List, Union

import pandas as pd

from argus.utils.config import load_config


class ETLPobFilter:
    """Class for ETL pob filter."""

    def __init__(
        self,
        config_path: Union[str, Path],
        use_stopwords: bool = False,
        use_cleaner: bool = True,
        use_meta_qualifiers: bool = False,
        use_transformers: bool = False,
        batch_size: int = 64,
        n_jobs: int = 4,
        iterative: bool = False,
        logger=None,
    ) -> None:
        """Initialize the ETLPobFilter with a configuration file.

        :param config_path: Path to the configuration file.
        """
        self.config = load_config(config_path)

        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.logger.info("ETLPobFilter initialized with config: %s", self.config)
        patterns = load_config("pob_filter")
        self.clinlp_patterns = patterns["clinlp_patterns"]
        self.regex_patterns = patterns["regex_patterns"]

        extra_stopwords = self.config.get("stopwords", [])
        if extra_stopwords is not None:
            self.STOPWORDS = set(STOPWORDS).union(set(extra_stopwords))
        else:
            self.STOPWORDS = STOPWORDS

        self.use_stopwords = use_stopwords
        self.use_meta_qualifiers = use_meta_qualifiers
        self.use_cleaner = use_cleaner
        self.use_transformers = use_transformers

        self.nlp = None
        self.re_compiled = False
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.iterative = iterative

    def clean_txt(self, text):
        """Basic cleaning, remove excess white space and flattening text
        :param text ..
        """
        if self.re_compiled == False:
            self.re_multispace = re.compile(r"\s{2,}")
            self.re_linebreaks = re.compile("[\r\n]{1,}")
            self.re_compiled = True

        text = self.re_linebreaks.sub(" ", text)
        text = self.re_multispace.sub(" ", text)

        return text

    def stopword_filter(self, x):
        """Filter out stopwords from the input string.

        :param x: Input string to filter.
        :return: String with stopwords removed.
        """
        # Using a list comprehension to filter out stopwords
        # This is more efficient than using a loop and appending to a list
        return " ".join([word for word in x.split() if word not in self.STOPWORDS])

    def prepare_regex_patterns(self) -> None:
        """Prepare the patterns for filtering based on the configuration."""
        self.logger.info("Preparing patterns for filtering.")
        # Implement pattern preparation logic here
        # This could involve compiling regex patterns or loading other necessary data
        # Example: self.patterns = [re.compile(pattern) for pattern in self.config['patterns']]

        # Placeholder for actual pattern preparation logic
        self.regex_patterns = [re.compile(pattern) for pattern in self.regex_patterns["patterns"]]

    def prepare_clinlp(self) -> None:
        """Prepare the clinlp patterns for filtering based on the configuration."""
        self.logger.info("Preparing clinlp for filtering.")

        self.nlp = spacy.blank("clinlp")
        # Normalization
        self.nlp.add_pipe("clinlp_normalizer")
        # Sentences
        self.nlp.add_pipe("clinlp_sentencizer")
        # Negation

        self.logger.info("Processing seed terms into clinlp..")
        self.prepped_clinlp_patterns = defaultdict(Term)
        cTerms = []

        # POSITIVES
        for _term, _param_dict in self.clinlp_patterns["positive_phrases"].items():
            fuzz = _param_dict.get("fuzzy", None)
            prox = _param_dict.get("pos", None)
            case = _param_dict.get("case", False)

            attr = "TEXT" if case else "NORM"
            if fuzz is None:
                if (len(_term) >= 3) & (len(_term) < 8):
                    fuzz = 0
                elif (len(_term) >= 8) and (len(_term) <= 12):
                    fuzz = 2
                elif len(_term) > 12:
                    if len(_term.split()) > 1:
                        fuzz = 2
                    else:
                        fuzz = 3
            if prox is None:
                if len(_term.split()) == 2:
                    prox = 3

            cTerms.append(
                Term(
                    _term,
                    fuzzy=fuzz,
                    proximity=prox,
                    attr=attr,
                    pseudo=False,
                )
            )
        # NEGATIVES
        for _term, _param_dict in self.clinlp_patterns["exclusion_phrases"].items():
            fuzz = _param_dict.get("fuzzy", None)
            prox = _param_dict.get("pos", None)
            case = _param_dict.get("case", False)

            attr = "TEXT" if case else "NORM"
            if fuzz is None:
                if (len(_term) >= 3) & (len(_term) < 8):
                    fuzz = 0
                elif (len(_term) >= 8) and (len(_term) <= 12):
                    fuzz = 2
                elif len(_term) > 12:
                    if len(_term.split()) > 1:
                        fuzz = 2
                    else:
                        fuzz = 3
            if prox is None:
                if len(_term.split()) == 2:
                    prox = 3

            cTerms.append(Term(_term, fuzzy=fuzz, proximity=prox, attr=attr, pseudo=True))

        self.prepped_clinlp_patterns["positive_phrases"] = cTerms

        self.logger.info("...")
        entity_matcher = self.nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"attr": "NORM", "fuzzy": 2})

        self.logger.info("...")
        entity_matcher.add_terms_from_dict(self.prepped_clinlp_patterns)

        self.logger.info("...")
        # TODO: add option for transformers to experiencer/negation detection
        if self.use_meta_qualifiers:
            self.nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})

            if self.use_transformers:
                self.nlp.add_pipe("clinlp_negation_transformer")
            if self.use_transformers:
                self.nlp.add_pipe("clinlp_experiencer_transformer")

        self.logger.info("Clinlp terms loaded!")
        return self

    def doc_extractor(self, text: str) -> Doc:
        """Filter the input data based on the configuration.

        :text data: text to filter.
        :return: SpaCy doc
        """
        if self.nlp is None:
            self.logger.info("Clinlp not initialized yet, initializing...")
            self.prepare_clinlp()

        if self.use_cleaner:
            text = self.clean_txt(text)

        if self.use_stopwords:
            text = self.stopword_filter(text)

        return self.nlp(text)

    def _doc_extractor_bulk_iter(self, texts: List[str]) -> List[Doc]:
        """Using the SpaCy pipe to expedite things..

        :texts: list of documents to process
        """
        if self.nlp is None:
            self.logger.info("Clinlp not initialized yet, initializing...")
            self.prepare_clinlp()

        if self.use_cleaner:
            texts = [self.clean_txt(text) for text in texts]

        if self.use_stopwords:
            texts = [self.stopword_filter(text) for text in texts]

        return self.nlp.pipe(texts, batch_size=self.batch_size, n_process=self.n_jobs)

    def doc_extractor_bulk(self, texts: List[str]) -> List[Doc]:
        """Using the SpaCy pipe to expedite things..

        :texts: list of documents to process
        """
        if self.iterative:
            res = []
            for _res in tqdm(self._doc_extractor_bulk_iter(texts), total=len(texts)):
                res.append(_res)
            return res
        else:
            return list(self._doc_extractor_bulk_iter(texts))

    @staticmethod
    def meta_check(entity):
        if "Presence.Absent" in entity._.qualifiers_str:
            return False
        if "Experiencer.Patient" not in entity._.qualifiers_str:
            return False
        # if ("Temporality.Current" not in entity._.qualifiers_str) & \
        #    ("Temporality.Future" not in entity._.qualifiers_str):  # e.g. : Als PATIENT X doet, dan POB
        #    return False
        return True

    def get_POB(self, text: str) -> bool:
        """Filter the input data based on the configuration.

        :text data: text to filter.
        :return: whether or not PoB was detected
        """
        _doc = self.doc_extractor(text)

        bools = []
        for e in _doc.spans["ents"]:
            if e.label_ == "positive_phrases":
                not_negated = self.meta_check(e) if self.use_meta_qualifiers else True
                if not_negated:
                    bools.append(True)
                else:
                    bools.append(False)
        if len(bools) == 0:
            return None
        else:
            return any(bools)

    def get_POB_bulk(self, texts: List[str]) -> List[bool]:
        """Filter the input data based on the configuration.
         Using SpaCy pipe for bulk processing

        :texts: texts to filter.
        :return: List of booleans, whether or not PoB was detected
        """
        self.logger.info("Extracting Doc's...")
        docs = self.doc_extractor_bulk(texts)

        self.logger.info("Parsing Doc's for matches..")
        res = []
        for _doc in docs:
            _match = False
            for e in _doc.spans["ents"]:
                if e.label_ == "positive_phrases":
                    not_negated = self.meta_check(e) if self.use_meta_qualifiers else True
                    if not_negated:
                        _match = True
                        continue
                    else:
                        _match = False
                        continue
            res.append(_match)
        return res


if __name__ == "__main__":
    test = [
        {"patid": 1, "text": "De patient kwam met een beklemmend gevoel op de poli"},
        {"patid": 2, "text": "Er was sprake van resternale pijn op de borst"},
    ]
    test_df = pd.DataFrame(test)

    pob_filter = ETLPobFilter(use_stopwords=True)
    pob_filter.prepare_clinlp()

    print("EXTRACTING DOCS..")
    test_df["DOCS"] = test_df.text.apply(pob_filter.doc_extractor)

    print("EXTRACTING POB")
    test_df["POB"] = test_df.text.apply(pob_filter.get_POB)

    print(test_df)
    print("\n\n")
