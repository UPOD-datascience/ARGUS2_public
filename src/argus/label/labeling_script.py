## TEXT MINING RADIO
#  * run old script
#  * get Agatston score,
#  * get MESA, CADRADS and stenosis degree
#  * get population percentiles from Agatston

## PROCEDURES
# * follow-up diagnostics within 365 days
# * follow-up interventions within 365 days
# * follow-up medication within 365 days


# add logger
import logging
import os
import pickle
import re

# add src directory
from collections import defaultdict

import Agatston
import argus2020_extractor
import dotenv
import numpy as np
import pandas as pd
import StenoseGraad

from argus.label.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# set logging level
# load environment variables

dotenv.load_dotenv("../../../.env")

CACS_MAPPER_FEMALE_2001 = os.getenv("CACS_MAPPER_FEMALE_2001")
CACS_MAPPER_MALE_2001 = os.getenv("CACS_MAPPER_MALE_2001")
CACS_MAPPER_FEMALE_2020 = os.getenv("CACS_MAPPER_FEMALE_2020")
CACS_MAPPER_MALE_2020 = os.getenv("CACS_MAPPER_MALE_2020")


def check_synom_dict(report, rlist):
    for rs in rlist:
        if re.search(rs, report):
            return True
    return False


def is_present_simple_positive(x, phrase_type):
    for p in settings.positive_phrases[phrase_type]:
        if p in x:
            return True
    return False


def is_present_regex_positive(x, phrase_type):
    for p in settings.positive_phrases_re[phrase_type]:
        if p.search(x) is not None:
            return True
    return False


def is_present_simple_negative(x, phrase_type):
    for p in settings.negative_phrases[phrase_type]:
        if p in x:
            return True
    return False


def is_present_regex_negative(x, phrase_type):
    for p in settings.negative_phrases_re[phrase_type]:
        if p.search(x) is not None:
            return True
    return False


def load_df(
    loc,
    TEXT_COL: str | None = "ReportText",
    ID_COL: str = "pateventid",
    DATE_COL: str = "date",
):
    assert isinstance(loc, str), "You should provide a string for the location"
    assert os.path.exists(loc), f"The file does not exist:{loc}"
    assert loc.split(".")[-1] in ["parquet", "csv", "tsv", "jsonl"], (
        "File should have the extension parquet/csv/tsv or jsonl"
    )

    if loc.endswith(".csv"):
        df = pd.read_csv(loc, encoding="latin1")
    elif loc.endswith(".tsv"):
        df = pd.read_csv(loc, sep="\t", encoding="latin1")
    elif loc.endswith(".jsonl"):
        df = pd.read_json(loc, lines=True)
    elif loc.endswith(".parquet"):
        df = pd.read_parquet(loc)

    if TEXT_COL is not None:
        assert all([c in df.columns for c in [TEXT_COL, ID_COL, DATE_COL]]), (
            f"File does not have all the columns [{TEXT_COL},{ID_COL},{DATE_COL}]; df contains: {df.columns}"
        )
    else:
        assert all([c in df.columns for c in [ID_COL, DATE_COL]]), (
            f"File does not have all the columns [{ID_COL},{DATE_COL}]; df contains: {df.columns}"
        )

    return df


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Labeling script for ARGUS2024")
    parser.add_argument(
        "--TEXT_COL",
        type=str,
        default="ReportText",
        help="Column name for the text data",
    )
    parser.add_argument(
        "--ID_COL",
        type=str,
        default="pateventid",
        help="Column name for the patient ID",
    )
    parser.add_argument("--DATE_COL", type=str, default="date", help="Column name for the date")
    parser.add_argument("--DF_LOC", type=str, required=True, help="Location of the dataframe file")
    parser.add_argument(
        "--PATSEL_LOC",
        type=str,
        required=True,
        help="Location of the patient selection file",
    )
    parser.add_argument(
        "--OUTPUT_FOLDER",
        type=str,
        required=True,
        help="Output folder to save the labels",
    )
    parser.add_argument("--SAMPLE", action="store_true", default=False)
    parser.add_argument("--PERC_THRESHOLD", type=float, required=True)  # used 65

    return parser.parse_args()


def main(**kwargs):
    if not kwargs:
        args = parse_args()
        TEXT_COL = args.TEXT_COL
        ID_COL = args.ID_COL
        DATE_COL = args.DATE_COL
        DF_LOC = args.DF_LOC
        PATSEL_LOC = args.PATSEL_LOC
        OUTPUT_FOLDER = args.OUTPUT_FOLDER
        SAMPLE = args.SAMPLE
        PERC_THRESHOLD = args.PERC_THRESHOLD
        # print(TEXT_COL, ID_COL, DATE_COL, DF_LOC, PATSEL_LOC, OUTPUT_FOLDER)

    logger.info(f"Loading data from {DF_LOC} and {PATSEL_LOC}")
    df = load_df(DF_LOC, TEXT_COL=TEXT_COL, ID_COL=ID_COL, DATE_COL=DATE_COL)
    patsel = load_df(PATSEL_LOC, TEXT_COL=None, ID_COL=ID_COL, DATE_COL=DATE_COL)

    df[DATE_COL] = df[DATE_COL].dt.date
    patsel[DATE_COL] = pd.to_datetime(patsel[DATE_COL]).dt.date

    assert all([c in patsel.columns for c in ["Age", "gender"]]), "Age and gender not present in patsel:patsel.columns"

    logger.info("Starting labeling process")
    df["stenosis_mentioned"] = df[TEXT_COL].apply(lambda x: check_synom_dict(x, settings.synonyms["stenosis"]))
    df["plaque_mentioned"] = df[TEXT_COL].apply(lambda x: check_synom_dict(x, settings.synonyms["plaque"]))
    df["perfusion_mentioned"] = df[TEXT_COL].apply(lambda x: check_synom_dict(x, settings.synonyms["perfusion"]))
    df["delayed_enhancement_mentioned"] = df[TEXT_COL].apply(
        lambda x: check_synom_dict(x, settings.synonyms["delayed_enhancement"])
    )
    df["hartfailure_mentioned"] = df[TEXT_COL].apply(lambda x: check_synom_dict(x, settings.synonyms["hartfailure"]))
    df["ischemie_mentioned"] = df[TEXT_COL].apply(lambda x: check_synom_dict(x, settings.synonyms["ischemie"]))
    df["congenitaal_mentioned"] = df[TEXT_COL].apply(lambda x: check_synom_dict(x, settings.synonyms["congenitaal"]))

    # run argus2020_extractor -> stenosis, plaque
    logger.info("Running argus2020_extractor for stenosis and plaque extraction")
    kwargs = {
        "TEXT_COL": TEXT_COL,
        "ID_COL": ID_COL,
        "DATE_COL": DATE_COL,
        "OUTPUT_FOLDER": OUTPUT_FOLDER,
        "sample": SAMPLE,
    }

    stenosis_labels = argus2020_extractor.extract_labels(df, target="stenosis", **kwargs)
    plaque_labels = argus2020_extractor.extract_labels(df, target="plaque", **kwargs)
    # ischemie_labels = argus2020_extractor.extract_labels(df, target="ischemie", **kwargs)
    print(stenosis_labels.head())

    logger.info("Running regex refinement on argus2020_extractor results")
    conds1 = df[TEXT_COL].apply(lambda x: is_present_simple_negative(x, "stenose"))
    conds2 = df[TEXT_COL].apply(lambda x: is_present_regex_negative(x, "stenose"))
    conds_negative_stenosis = conds1 | conds2

    negatives_stenosis = df[conds_negative_stenosis][[ID_COL, DATE_COL]]
    negatives_stenosis["check_negative"] = 1

    stenosis_labels = pd.merge(
        stenosis_labels,
        negatives_stenosis,
        left_on=[ID_COL, DATE_COL],
        right_on=[ID_COL, DATE_COL],
        how="left",
    )
    stenosis_labels.loc[stenosis_labels.check_negative == 1, "stenosis"] = 0

    #########################################

    conds1 = df[TEXT_COL].apply(lambda x: is_present_simple_positive(x, "stenose"))
    conds2 = df[TEXT_COL].apply(lambda x: is_present_regex_positive(x, "stenose"))
    conds_positive_stenosis = conds1 | conds2

    positives_stenosis = df[conds_positive_stenosis][[ID_COL, DATE_COL]]
    positives_stenosis["check_positive"] = 1

    stenosis_labels = pd.merge(
        stenosis_labels,
        positives_stenosis,
        left_on=[ID_COL, DATE_COL],
        right_on=[ID_COL, DATE_COL],
        how="left",
    )
    stenosis_labels.loc[stenosis_labels.check_positive == 1, "stenosis"] = 1

    conds1 = df[TEXT_COL].apply(lambda x: is_present_simple_negative(x, "plaque"))
    conds2 = df[TEXT_COL].apply(lambda x: is_present_regex_negative(x, "plaque"))
    conds_negative_plaque = conds1 | conds2

    negatives_plaque = df[conds_negative_plaque][[ID_COL, DATE_COL]]
    negatives_plaque["check_negative"] = 0

    plaque_labels = pd.merge(
        plaque_labels,
        negatives_plaque,
        left_on=[ID_COL, DATE_COL],
        right_on=[ID_COL, DATE_COL],
        how="left",
    )
    plaque_labels.loc[plaque_labels.check_negative == 0, "plaque"] = 0

    #########################################

    conds1 = df[TEXT_COL].apply(lambda x: is_present_simple_positive(x, "plaque"))
    conds2 = df[TEXT_COL].apply(lambda x: is_present_regex_positive(x, "plaque"))
    conds_positive_plaque = conds1 | conds2

    positives_plaque = df[conds_positive_plaque][[ID_COL, DATE_COL]]
    positives_plaque["check_positive"] = 1

    plaque_labels = pd.merge(
        plaque_labels,
        positives_plaque,
        left_on=[ID_COL, DATE_COL],
        right_on=[ID_COL, DATE_COL],
        how="left",
    )
    plaque_labels.loc[plaque_labels.check_positive == 1, "plaque"] = 1

    df = (
        pd.merge(
            df,
            stenosis_labels[[ID_COL, DATE_COL, "stenosis"]],
            left_on=[ID_COL, DATE_COL],
            right_on=[ID_COL, DATE_COL],
            how="left",
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df = (
        pd.merge(
            df,
            plaque_labels[[ID_COL, DATE_COL, "plaque"]],
            left_on=[ID_COL, DATE_COL],
            right_on=[ID_COL, DATE_COL],
            how="left",
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df = df[~df[TEXT_COL].str.contains("geen diagnostisch verslag")]

    df["macro_stenosis"] = None
    df.loc[df.stenosis == 1, "macro_stenosis"] = 1
    df.loc[df.stenosis == 0, "macro_stenosis"] = 0
    ###############################
    df["macro_plaque"] = None
    df.loc[df.plaque == 1, "macro_plaque"] = 1
    df.loc[df.plaque == 0, "macro_plaque"] = 0

    logger.info("Running Agatston extractor")
    CalcScorer = Agatston.CACS(text_col=TEXT_COL, id_col=ID_COL, date_col=DATE_COL, with_cleaning=True)

    scores = CalcScorer.get_scores(df[[ID_COL, DATE_COL, TEXT_COL]])
    df[["cacs_mean", "cacs_min", "cacs_max"]] = scores
    df = df.assign(cacs_mean=df.cacs_mean.clip(0, 5000))

    logger.info("\t Loading CACS percentile mappers")
    percentile_mapper = defaultdict(dict)
    with open(CACS_MAPPER_FEMALE_2001, "rb") as fr:
        percentile_mapper["female"]["2001"] = pickle.load(fr)

    with open(CACS_MAPPER_MALE_2001, "rb") as fr:
        percentile_mapper["male"]["2001"] = pickle.load(fr)

    with open(CACS_MAPPER_FEMALE_2020, "rb") as fr:
        percentile_mapper["female"]["2020"] = pickle.load(fr)

    with open(CACS_MAPPER_MALE_2020, "rb") as fr:
        percentile_mapper["male"]["2020"] = pickle.load(fr)

    df = df.merge(patsel[[ID_COL, "Age", "gender"]], on=ID_COL, how="left")
    # print(df.gender.unique())
    # df['gender'] = df['gender'].map({0: 'male', 1: 'female'}, na_action='ignore')
    df = df.loc[df.gender.isin(["male", "female"])]

    def percentile_agger(x, version="2001", samples=100):
        perc = 0
        for idx, mapper in enumerate(percentile_mapper[x.gender][version]):
            perc += mapper(x.Age, x.cacs_mean)
            if idx > samples:
                break
        return perc / idx

    logger.info("Calculating percentiles for CACS scores")
    df["cacs_percentile_2001"] = df.apply(lambda x: percentile_agger(x, "2001"), axis=1)

    df["cacs_percentile_2020"] = df.apply(lambda x: percentile_agger(x, "2020"), axis=1)

    df.loc[(df.cacs_percentile_2001.isna()) & (df.cacs_mean == 0), "cacs_percentile_2001"] = 0

    df.loc[(df.cacs_percentile_2020.isna()) & (df.cacs_mean == 0), "cacs_percentile_2020"] = 0

    df["macro_percentile_2001"] = None
    df["macro_percentile_2020"] = None

    df.loc[df.cacs_percentile_2001 > PERC_THRESHOLD, "macro_percentile_2001"] = 1
    df.loc[df.cacs_percentile_2001 <= PERC_THRESHOLD, "macro_percentile_2001"] = 0

    df.loc[df.cacs_percentile_2020 > PERC_THRESHOLD, "macro_percentile_2020"] = 1
    df.loc[df.cacs_percentile_2020 <= PERC_THRESHOLD, "macro_percentile_2020"] = 0

    df["macro_cacs400"] = None
    df.loc[df.cacs_mean > 400, "macro_cacs400"] = 1
    df.loc[df.cacs_mean <= 400, "macro_cacs400"] = 0

    logger.info("Calculating MESA and CADRADS scores")
    StenoseGrader = StenoseGraad.StenoseGraad(text_col=TEXT_COL, id_col=ID_COL, date_col=DATE_COL)
    stenose_res = StenoseGrader.get_scores(df)

    stenose_res["macro_stenosegraad"] = None
    stenose_res.loc[(stenose_res.sg_minder_mean < 50), "macro_stenosegraad"] = 0
    stenose_res.loc[(stenose_res.sg_minder_mean >= 50), "macro_stenosegraad"] = 1
    stenose_res.loc[(stenose_res.sg_ongeveer_mean >= 50), "macro_stenosegraad"] = 1
    stenose_res.loc[(stenose_res.sg_meer_mean >= 50), "macro_stenosegraad"] = 1

    stenose_res["macro_cadrads"] = None
    stenose_res.loc[stenose_res.CADRADS > 2, "macro_cadrads"] = 1
    stenose_res.loc[stenose_res.CADRADS <= 2, "macro_cadrads"] = 0

    stenose_res["macro_mesa"] = None
    stenose_res.loc[stenose_res.MESA >= PERC_THRESHOLD, "macro_mesa"] = 1
    stenose_res.loc[stenose_res.MESA < PERC_THRESHOLD, "macro_mesa"] = 0

    df[["macro_stenosegraad", "macro_cadrads", "macro_mesa", "MESA", "CADRADS"]] = stenose_res[
        ["macro_stenosegraad", "macro_cadrads", "macro_mesa", "MESA", "CADRADS"]
    ]

    # collect macro_ labels
    df["MACRO_ESTIMATE"] = df[[c for c in df.columns if c.startswith("macro_")]].sum(axis=1, skipna=True)

    # RADIO LABELS
    LABEL_COLS = [
        "macro_stenosis",
        "macro_plaque",
        "cacs_percentile_2001",
        "cacs_percentile_2020",
        "cacs_mean",
        "macro_percentile_2001",
        "macro_percentile_2020",
        "macro_stenosegraad",
        "macro_cadrads",
        "macro_mesa",
        "MESA",
        "CADRADS",
    ]

    df[LABEL_COLS] = df[LABEL_COLS].fillna(value=np.nan)
    outpath = os.path.join(OUTPUT_FOLDER, "radio_labels.parquet")
    logger.info(f"Saving labels to parquet file:{outpath}")
    df[[ID_COL, DATE_COL] + [TEXT_COL] + LABEL_COLS].reset_index(drop=True).to_parquet(outpath)

    print(df.head())
    print(df.columns)
    print(df.MACRO_ESTIMATE.value_counts())


if __name__ == "__main__":
    main()
