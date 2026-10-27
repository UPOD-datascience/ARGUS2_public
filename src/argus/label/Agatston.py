"""De kalk score volgens Agatston bedraagt 2,6 (11de MESA percentiel)

Voor het MESA percentiel kunnen we ook alleen op percentiel zoeken.

"""

import os
import re

import dotenv
import numpy as np
import pandas as pd
from benedict import benedict
from joblib import Parallel, delayed


def load_config_file(config_file):
    # Load config file
    return dict(benedict(config_file, format="yaml"))


dir_path = os.path.dirname(os.path.realpath(__file__))

# score (licht|sterk|fors|) verhoogd ([0-9])
synoms = [
    r"agatston[-\s\.]?score",
    r"kalk[-\s\.]?score",
    r"calcium[-\s\.]?score",
    r"ca[-\s\.]?score",
    r"cacs",
    r"calc[-\s\.]?score",
    r"agatston",
    r"cac",
    r"((kalk[-\s\.]?score)|(calcium[-\s\.]?score)|(calc[-\s\.]?score)) volgens agatston",
    r"kalk\s?score bedraagt",
    r"calcium\s?score bedraagt",
    r"kalk\s?score volgens agatston",
    r"agatston score bedraagt",
    r"agatston bedraagt",
    r"cacs bedraagt",
    r"agatston\s?score is",
    r"kalk\s?score is",
    r"calcium\s?score is",
    r"agatston\s?score van",
    r"calcium\s?score van",
    r"kalk\s?score van",
    r"cacs is",
    r"((kalk[-\s\.]?score)|(calcium[-\s\.]?score)|(calc[-\s\.]?score)) van",
]

word_replacements = [
    (re.compile(r"[\r\n]+"), " "),
    (re.compile(r"([0-9])\.([0-9]{3})\,[0-9]{1,3}"), r"\1\2"),
    (re.compile(r"([0-9])\,([0-9])"), r"\1.\2"),
    (re.compile(r"\s\w\s"), " "),
    (re.compile(r"[\+\_\^\$\#\@\(\)\{\}\[\]\;]"), " "),
    (re.compile(r"\t+"), " "),
    (re.compile(r"[\W]ong\."), ":"),
    (re.compile(r"[\W]syst\."), ""),
    (re.compile(r"[23][dD]"), ""),
    (re.compile(r"\([45][Cc][Hh]\)"), ""),
    (re.compile(r"meer dan"), ">"),
    (re.compile(r"minder dan"), "<"),
    (re.compile(r"ongeveer"), ":"),
    (re.compile(r"minimaal"), ">"),
    (re.compile(r"maximaal"), "<"),
    (re.compile(r"minimaal"), ">"),
    (re.compile(r"maximaal"), "<"),
    (re.compile(r"ongeveer"), ":"),
    (re.compile(r"circa"), ":"),
    (re.compile(r"[\=]"), ":"),
    (re.compile(r"minimaal"), ">"),
    (re.compile(r"maximaal"), "<"),
    (re.compile(r"\s{2,}"), " "),
]


def _clean(df, n_jobs=1, stopwords=[], reps=[], text_col="text") -> list:
    # remove noise terms (case sensitive)
    cl = cleaner.Cleaner(replacements=reps, lower=False, stopwords=stopwords, n_jobs=n_jobs, text_cols=[text_col])
    cl.fit(df[[text_col]])
    df_new = cl.transform(df[[text_col]])[0]
    return df_new


class CACS:
    def __init__(
        self,
        min_CACS=0,
        max_CACS=10000,
        n_jobs=1,
        sent_reg_string=None,
        text_col="text",
        id_col="pateventid",
        date_col="ONDERZDAT",
        with_cleaning=False,
    ):
        self.min_CACS = min_CACS  # STALE
        self.max_CACS = max_CACS  # STALE
        self.with_cleaning = with_cleaning

        self.n_jobs = n_jobs

        self.id_col = id_col
        self.date_col = date_col
        self.text_col = text_col

        if sent_reg_string is None:
            sent_reg_string = ""

        self.cacs_reg = re.compile(r"(" + "|".join(synoms) + r")\s{0,4}[\>\<\:\;\=]?\s*([0-9]{1,5}[\.\,]?[0-9]{0,2})")
        self.cacs_score = re.compile(r"([0-9]{1,5}[\.]?[0-9]{0,2})")

    def __extract_spect_sentences(self, txt):
        return self.cacs_reg.findall(txt)

    def _get_scores(self, txt, multi_out=False):
        sents = self.__extract_spect_sentences(txt)
        # do stuff with extracted sents
        cacs = []
        for s in sents:
            for _r in s:
                if len(_r) > 0:
                    cacs_res = self.cacs_score.findall(_r)
                    if len(cacs_res) > 0:
                        cacs.append(float(cacs_res[0]))
        if multi_out == True:
            if len(cacs) == 0:
                return pd.Series({"cacs_mean": np.nan, "cacs_min": np.nan, "cacs_max": np.nan})
            else:
                return pd.Series({"cacs_mean": np.mean(cacs), "cacs_min": np.min(cacs), "cacs_max": np.max(cacs)})
        else:
            if len(cacs) == 0:
                return np.nan, np.nan, np.nan
            else:
                return np.mean(cacs), np.min(cacs), np.max(cacs)

    def _get_scores_multi(self, txt_df):
        txt_df.loc[:, ["cacs_mean", "cacs_min", "cacs_max"]] = txt_df[self.text_col].apply(
            self._get_scores, multi_out=True
        )
        return txt_df[[self.id_col, self.date_col, "cacs_mean", "cacs_min", "cacs_max"]]

    def _chunker(self, seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    def get_scores(self, df, extract_labels=False):
        """Get scores for all text fields in dataframe"""
        if self.with_cleaning:
            print("Cleaning..")
            df[self.text_col] = _clean(df.copy(), n_jobs=1, reps=word_replacements, text_col=self.text_col)

        scores = []
        df[self.text_col] = df[self.text_col].str.lower()
        if self.n_jobs == 1:
            for t in df.itertuples():
                st = self._get_scores(eval("t." + self.text_col))
                scores.append(st)
            return pd.DataFrame(data=scores, index=df.index, columns=["cacs_mean", "cacs_min", "cacs_max"])
        else:
            scores = pd.concat(
                Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(
                    delayed(self._get_scores_multi)(chunk)
                    for chunk in self._chunker(
                        df[[self.id_col, self.date_col, self.text_col]], int(len(df) / self.n_jobs)
                    )
                )
            )
            scores.columns = [self.id_col, self.date_col, "cacs_mean", "cacs_min", "cacs_max"]
            return scores


if __name__ == "__main__":
    dotenv.load_dotenv()
    config = load_config_file(os.getenv("argus_config"))
    noise_loc = os.getenv("noise_terms")
    synonym_loc = os.getenv("misspelling_file")

    file_names = [os.getenv(fn) for fn in config["agatston"]["in_files"]]
    text_col = config["agatston"]["text_column"]
    id_col = config["agatston"]["id_column"]
    date_col = config["agatston"]["date_column"]
    n_jobs = 1
    if config["mode"] == "test":
        nrows = 100
    else:
        nrows = None

    res = pd.DataFrame()
    for fn in file_names:
        print(f"Processing: {fn}")
        file_ext = fn.split(".")[-1]
        if file_ext == "csv":
            df = pd.read_csv(fn, sep=";", nrows=nrows, encoding="latin1")
        elif file_ext == "feather":
            df = pd.read_feather(fn)
            print(df.head())

        if config["agatston"]["remove_noise_terms"]:
            noise_terms = pd.read_csv(noise_loc, sep=",", encoding="latin1")["noise_word"].unique().tolist()
            noise_terms_re = (r"\b(" + "|".join(noise_terms) + r")\b", "")
            _reps = word_replacements + [noise_terms_re]
            df = _clean(df, process_par=True, reps=_reps, text_col=text_col)
        else:
            df = _clean(df, process_par=True, reps=word_replacements, text_col=text_col)

        if config["agatston"]["replace_synonyms"]:
            synonym_df = pd.read_csv(synonym_loc, sep=";", encoding="latin1")[["word", "seed"]]
            synonym_df["word"] = synonym_df["word"].str.strip()
            synonym_df[synonym_df.duplicated(subset="word")]
            syn_map = synonym_df.set_index("word")[["seed"]].to_records()
            df = _clean(df, process_par=True, reps=syn_map, text_col=text_col)

        cacs = CACS(
            n_jobs=4,
            text_col=text_col,
            id_col=id_col,
            date_col=date_col,
        )
        res = pd.concat([res, cacs.get_scores(df)], axis=0)

    if config["agatston"]["write_out"]:
        out_path = os.path.join(dir_path, "..", config["agatston"]["write_path"])
        print(out_path)
        res.to_csv(out_path, sep=";", index=False)
