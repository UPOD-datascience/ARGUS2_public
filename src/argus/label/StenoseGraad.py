r"""graad:
[x] (stenosegraad|stenosering|verkalking|stenose|stenosis)(\sis\s|\svan\s|\s)(meer dan|>|minimaal)\s*([0-9]{1,2})\s?\%
(minder dan)\s*([0-9]{1,2})\%\s*(stenosegraad|stenosering|verkalking|stenose|stenosis)

[x] (stenosegraad|stenosering|verkalking|stenose|stenosis)(\sis\s|\svan\s|\s)(minder dan|<|maximaal)\s*([0-9]{1,2})\s?\%
(meer dan)\s*([0-9]{1,2})\%\s*(stenosegraad|stenosering|verkalking|stenose|stenosis)

[x] (stenosegraad|stenosering|verkalking|stenose|stenosis)(\sis\s|\svan\s|\s)(rond|rondom|ongeveer|\~|\+\-|circa|\+\/-)\s*([0-9]{1,2})\s?\%

[ ] (stenosegraad|stenosering|verkalking|stenose|stenosis)\s(tussen)\s*([0-9]{1,2})\s?(en|\-)\s?([0-9]{1,2})\s?\%

[ ] (RCA|CX|LAD|D1|D2|D3|D4|LCA|MO|LM|RM)[\s\;\:\-]{,3}([0-9\.]{1,3})\s?\%

[ ] tot\s?([0-9]{1,2})\s?\%

[ ] diagonaal[\s\:\;\-]?([0-9]{1,2})\s?\%

[ ] ([0-9]{1,2})\s?\%stenos

[ ] stenose[\s\:\;]{0,2}([0-9]{1,2})\s?\%

"""

import os
import re

import dotenv
import numpy as np
import pandas as pd
from benedict import benedict
from joblib import Parallel, delayed

from argus.utils import text_cleaner


def load_config_file(config_file):
    # Load config file
    return dict(benedict(config_file, format="yaml"))


dir_path = os.path.dirname(os.path.realpath(__file__))

synoms = ["timi flow voor de procedure", "timi flow na de procedure"]

mspace = re.compile(r"\s{2,}")


def _clean(df, process_par=False, stopwords=[], reps=[], text_col="text"):
    # remove noise terms (case sensitive)
    cl = text_cleaner.Cleaner(replacements=reps, lower=False, html=False, stopwords=stopwords)
    df_new = df.copy()
    if process_par == False:
        df_new[text_col] = cl.clean_text(df[text_col])
    else:
        df_new[text_col] = cl.par_clean_text(df, num_processes=4, col_to_process=text_col)
    df_new.dropna(subset=[text_col], axis=0, inplace=True)
    df_new[text_col] = df_new[text_col].apply(lambda x: mspace.sub(" ", x))
    return df_new


class StenoseGraad:
    def __init__(self, min_STEN=0, max_STEN=100, n_jobs=1, text_col="text", id_col="pateventid", date_col="ONDERZDAT"):
        self.min_STEN = min_STEN  # STALE
        self.min_STEN = max_STEN  # STALE

        self.n_jobs = n_jobs

        self.id_col = id_col
        self.date_col = date_col
        self.text_col = text_col

        reg_string_meer = r"(stenosegraad|stenosering|verkalking|stenose|stenosis)(\sis\s|\svan\s|\s)(meer dan|>|minimaal)\s*([0-9]{1,2})\%"
        reg_string_minder = r"(stenosegraad|stenosering|verkalking|stenose|stenosis)(\sis\s|\svan\s|\s)(minder dan|<|maximaal)\s*([0-9]{1,2})\%"
        reg_string_ongeveer = r"(stenosegraad|stenosering|verkalking|stenose|stenosis)(\sis\s|\svan\s|\s)(rond|rondom|ongeveer|\~|\+\-|circa|\+\/-)\s*([0-9]{1,2})"

        reg_string_meer_2 = r"(meer dan)\s*([0-9]{1,2})\%\s*(stenosegraad|stenosering|verkalking|stenose|stenosis)"
        reg_string_minder_2 = (
            r"(minder dan|tot)\s*([0-9]{1,2})\%\s*(stenosegraad|stenosering|verkalking|stenose|stenosis)"
        )
        reg_string_tussen = r"(stenosegraad|stenosering|verkalking|stenose|stenosis)\s(tussen)\s*([0-9]{1,2})\s?(en|\-)\s?([0-9]{1,2})\%"
        reg_string_tussen_2 = (
            r"(stenosegraad|stenosering|verkalking|stenose|stenosis)\s*van ([0-9]{1,2})\s?tot\s?([0-9]{1,2})\%"
        )
        reg_string_exact = r"([0-9]{1,2})\s?\% stenose"

        self.sg_cols = dict()
        self.sg_cols["minder"] = ["sg_minder_mean", "sg_minder_min", "sg_minder_max"]
        self.sg_cols["meer"] = ["sg_meer_mean", "sg_meer_min", "sg_meer_max"]
        self.sg_cols["ongeveer"] = ["sg_ongeveer_mean", "sg_ongeveer_min", "sg_ongeveer_max"]
        self.sg_cols["exact"] = ["sg_exact_mean", "sg_exact_min", "sg_exact_max"]
        self.sg_cols["minder_2"] = ["sg_minder_mean", "sg_minder_min", "sg_minder_max"]
        self.sg_cols["meer_2"] = ["sg_meer_mean", "sg_meer_min", "sg_meer_max"]
        self.sg_cols["tussen"] = ["sg_tussen_min_mean", "sg_tussen_max_mean"]
        self.sg_cols["tussen_2"] = ["sg_tussen_min_mean", "sg_tussen_max_mean"]

        self.sg_meer_reg = re.compile(r"" + reg_string_meer)
        self.sg_minder_reg = re.compile(r"" + reg_string_minder)
        self.sg_ongeveer_reg = re.compile(r"" + reg_string_ongeveer)
        self.sg_exact_reg = re.compile(r"" + reg_string_exact)

        self.sg_meer2_reg = re.compile(r"" + reg_string_meer_2)  # [1]
        self.sg_minder2_reg = re.compile(r"" + reg_string_minder_2)  # [1]
        self.sg_tussen_reg = re.compile(r"" + reg_string_tussen)  # [-2:]
        self.sg_tussen_2_reg = re.compile(r"" + reg_string_tussen_2)  # [-2:]

        self.sg_score = re.compile(r"([0-9]{1,2})")

    def __extract_spect_sentences(self, txt, sg_type="meer", sel_index=-1):
        if sg_type == "meer":
            res = self.sg_meer_reg.findall(txt)
        elif sg_type == "minder":
            res = self.sg_minder_reg.findall(txt)
        elif sg_type == "ongeveer":
            res = self.sg_ongeveer_reg.findall(txt)
        elif sg_type == "meer_2":
            res = self.sg_meer2_reg.findall(txt)
        elif sg_type == "minder_2":
            res = self.sg_minder2_reg.findall(txt)
        elif sg_type == "exact":
            res = self.sg_exact_reg.findall(txt)
        elif sg_type == "tussen":
            res = self.sg_tussen_reg.findall(txt)
        elif sg_type == "tussen_2":
            res = self.sg_tussen_2_reg.findall(txt)

        if len(res) > 0:
            return [e[sel_index] for e in res]
        else:
            return []

    def _get_scores(self, txt, multi_out=False, sg_type="meer", sel_index=-1):
        if "tussen" in sg_type:
            if sg_type == "tussen":
                sents_min = self.__extract_spect_sentences(txt, sg_type=sg_type, sel_index=-3)
            else:
                sents_min = self.__extract_spect_sentences(txt, sg_type=sg_type, sel_index=-2)
            sents_max = self.__extract_spect_sentences(txt, sg_type=sg_type, sel_index=-1)
        else:
            sents = self.__extract_spect_sentences(txt, sg_type=sg_type, sel_index=sel_index)

        if "tussen" not in sg_type:
            sgs = []
            for s in sents:
                sgs.append(int(s))
            if multi_out == True:
                if len(sgs) == 0:
                    return pd.Series(
                        {
                            self.sg_cols[sg_type][0]: np.nan,
                            self.sg_cols[sg_type][1]: np.nan,
                            self.sg_cols[sg_type][2]: np.nan,
                        }
                    )
                else:
                    return pd.Series(
                        {
                            self.sg_cols[sg_type][0]: np.mean(sgs),
                            self.sg_cols[sg_type][1]: np.min(sgs),
                            self.sg_cols[sg_type][2]: np.max(sgs),
                        }
                    )
            else:
                if len(sgs) == 0:
                    return np.nan, np.nan, np.nan
                else:
                    return np.mean(sgs), np.min(sgs), np.max(sgs)
        else:
            sgs_min = []
            for s in sents_min:
                sgs_min.append(int(s))
            sgs_max = []
            for s in sents_max:
                sgs_max.append(int(s))

            if multi_out == True:
                if len(sgs_min) == 0:
                    return pd.Series({self.sg_cols[sg_type][0]: np.nan, self.sg_cols[sg_type][1]: np.nan})
                else:
                    return pd.Series(
                        {self.sg_cols[sg_type][0]: np.mean(sgs_min), self.sg_cols[sg_type][1]: np.mean(sgs_max)}
                    )
            else:
                if len(sgs_min) == 0:
                    return np.nan, np.nan
                else:
                    return np.mean(sgs_min), np.mean(sgs_max)

    def _get_scores_multi(self, txt_df):
        txt_df.loc[:, self.sg_cols["minder"]] = txt_df[self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="minder", sel_index=-1
        )
        txt_df.loc[:, self.sg_cols["meer"]] = txt_df[self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="meer", sel_index=-1
        )
        txt_df.loc[:, self.sg_cols["ongeveer"]] = txt_df[self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="ongeveer", sel_index=-1
        )
        txt_df.loc[:, self.sg_cols["exact"]] = txt_df[self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="exact", sel_index=0
        )

        minder_sel_index = txt_df[self.sg_cols["minder"][0]].isna()
        txt_df.loc[minder_sel_index, self.sg_cols["minder"]] = txt_df.loc[minder_sel_index, self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="minder_2", sel_index=1
        )

        meer_sel_index = txt_df[self.sg_cols["meer"][0]].isna()
        txt_df.loc[meer_sel_index, self.sg_cols["meer"]] = txt_df.loc[meer_sel_index, self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="meer_2", sel_index=1
        )

        txt_df.loc[:, self.sg_cols["tussen"]] = txt_df[self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="tussen"
        )
        tussen_sel_index = txt_df[self.sg_cols["tussen"][0]].isna()
        txt_df.loc[tussen_sel_index, self.sg_cols["tussen"]] = txt_df.loc[tussen_sel_index, self.text_col].apply(
            self._get_scores, multi_out=True, sg_type="tussen_2"
        )

        ## ADD CADRADS
        ###############
        cadrads_ints = (
            txt_df[self.text_col]
            .str.extractall(
                r"CAD[\-\s]?RADS[\:\=\s]?([0-9])|GCA[\-\s]?[\:\=\s]?([0-9])|C[\-\s]RADS[\-\:\=\s]([0-9])|CTA[\-\:\=\s]([0-9])|CAD\-?RADS categorie\s?([0-9])",
                re.IGNORECASE,
            )
            .astype(float)
        )
        cadrads_ints = cadrads_ints.reset_index().set_index("level_0").max(axis=1)
        txt_df["CADRADS"] = None
        txt_df.loc[cadrads_ints.index, "CADRADS"] = cadrads_ints

        ## ADD MESA
        ###############
        txt_df["MESA"] = None
        mesa_ints = (
            txt_df[self.text_col]
            .str.extractall(
                r"([0-9]{1,2})\s?ste MESA|([0-9]{1,2})\s?d?e MESA|MESA percentiel van ([0-9]{1,2})|MESA percentiel ([0-9]{1,2})|MESA ([0-9]{1,2})|MESA is dit het ([0-9]{1,2})|MESA is ([0-9]{1,2})|([0-9]{2})\s? MESA|([0-9]{2})de percentiel|([0-9]{2})ste percentiel",
                re.IGNORECASE,
            )
            .astype(float)
        )
        mesa_ints = mesa_ints.reset_index().groupby("level_0").max().max(axis=1).to_frame()
        mesa_ints = mesa_ints.rename(columns={0: "MESA"})
        txt_df.loc[mesa_ints.index, "MESA"] = mesa_ints.MESA.values

        return txt_df[
            [self.id_col, self.date_col]
            + self.sg_cols["minder"]
            + self.sg_cols["meer"]
            + self.sg_cols["ongeveer"]
            + self.sg_cols["exact"]
            + self.sg_cols["tussen"]
            + ["MESA", "CADRADS"]
        ]

    def _chunker(self, seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    def get_scores(self, df):
        """Get scores for all text fields in dataframe"""
        scores = []
        df[self.text_col] = df[self.text_col].str.lower()
        scores = pd.concat(
            Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(
                delayed(self._get_scores_multi)(chunk)
                for chunk in self._chunker(df[[self.id_col, self.date_col, self.text_col]], int(len(df) / self.n_jobs))
            )
        )
        # scores.columns = [self.id_col, self.date_col] +self.sg_cols_minder+self.sg_cols_meer+self.sg_cols_ongeveer
        return scores


if __name__ == "__main__":
    dotenv.load_dotenv()
    config = load_config_file(os.getenv("argus_config"))
    noise_loc = os.getenv("noise_terms")
    synonym_loc = os.getenv("misspelling_file")

    file_names = [os.getenv(fn) for fn in config["stenosegraad"]["in_files"]]
    text_col = config["stenosegraad"]["text_column"]
    id_col = config["stenosegraad"]["id_column"]
    date_col = config["stenosegraad"]["date_column"]
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

        if config["stenosegraad"]["remove_noise_terms"]:
            noise_terms = pd.read_csv(noise_loc, sep=",", encoding="latin1")["noise_word"].unique().tolist()
            noise_terms_re = (r"\b(" + "|".join(noise_terms) + r")\b", "")
            df = _clean(df, process_par=True, reps=noise_terms, text_col=text_col)

        if config["stenosegraad"]["replace_synonyms"]:
            synonym_df = pd.read_csv(synonym_loc, sep=";", encoding="latin1")[["word", "seed"]]
            synonym_df["word"] = synonym_df["word"].str.strip()
            synonym_df[synonym_df.duplicated(subset="word")]
            syn_map = synonym_df.set_index("word")[["seed"]].to_records()
            df = _clean(df, process_par=True, reps=syn_map, text_col=text_col)

        stenosegraad = StenoseGraad(
            n_jobs=4,
            text_col=text_col,
            id_col=id_col,
            date_col=date_col,
        )
        res = pd.concat([res, stenosegraad.get_scores(df)], axis=0)

    if config["stenosegraad"]["write_out"]:
        out_path = os.path.join(dir_path, "..", config["stenosegraad"]["write_path"])
        print(out_path)
        res.to_csv(out_path, sep=";", index=False)
