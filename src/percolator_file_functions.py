# some functions for the parsing of percolator files

import csv
from pathlib import Path
import pandas as pd

def parse_percolator_tsv(filename: str):
    """
    Fast parser for Percolator TSV files where rows may have more columns than headers.

    Parameters
    ----------
    filename : str
        Path to the percolator file.

    Returns 
    ------
    pandas.DataFrame
        DataFrame with the parsed rows. The last column contains tab separated protein accessions.
    """
    fpath = Path(filename)
    if not fpath.exists():
        raise FileNotFoundError(f"{filename} not found.")

    with open(fpath, "r", newline="") as fh:
        header_line = fh.readline().rstrip("\r\n")
        headers = header_line.split("\t")
        nr_headers = len(headers)
        last_idx = nr_headers - 1

        reader = csv.reader(fh, delimiter="\t", quotechar='"')

        container = {h: [] for h in headers}

        for row in reader:
            if len(row) <= nr_headers:
                # pad if shorter
                if len(row) < nr_headers:
                    row = row + [""] * (nr_headers - len(row))
                # trailing proteins field -> list (empty string -> empty list)
                proteins_list = [row[last_idx]] if row[last_idx] != "" else []
            else:
                # more fields than headers -> subsume trailing fields into proteins list
                proteins_list = row[last_idx:]

            for i in range(last_idx):
                container[headers[i]].append(row[i])
                
            container[headers[last_idx]].append("\t".join(proteins_list))

        # create the dataframe
        df = pd.DataFrame.from_dict(container)
        return df


def parse_percolator_pin(filename: str) -> pd.DataFrame:
    """
    Reads in the given percolator pin file as a DataFrame
    """
    df_perc = parse_percolator_tsv(filename)

    # rename the columns (tools write them differently / upper- and lowercase)
    df_perc.rename(columns={df_perc.columns[0]: "SpecId"}, inplace=True)
    df_perc.rename(columns={df_perc.columns[1]: "Label"}, inplace=True)
    df_perc.rename(columns={df_perc.columns[2]: "ScanNr"}, inplace=True)
    # set correct index and data types
    df_perc.set_index("SpecId", inplace=True, drop=False)
    df_perc["Label"] = pd.to_numeric(df_perc["Label"], downcast="integer", errors="coerce")
    df_perc["ScanNr"] = pd.to_numeric(df_perc["ScanNr"], downcast="integer", errors="coerce")

    return df_perc


def parse_percolator_pout(filename: str) -> pd.DataFrame:
    """
    Reads in the given percolator pout file as a DataFrame.
    """
    df_perc = parse_percolator_tsv(filename)

    # set correct index and data types
    df_perc.rename(columns={df_perc.columns[0]: "PSMId"}, inplace=True)
    df_perc.set_index("PSMId", inplace=True, drop=False)
    df_perc["score"] = pd.to_numeric(df_perc["score"], errors="coerce", downcast="float")
    df_perc["q-value"] = pd.to_numeric(df_perc["q-value"], errors="coerce", downcast="float")
    df_perc["posterior_error_prob"] = pd.to_numeric(df_perc["posterior_error_prob"], errors="coerce", downcast="float")

    return df_perc


def read_enriched_pout_with_pin_data(pout_file: str, pin_file: str) -> pd.DataFrame:
    """
    Reads in a Percolator pout file and enriches it with the data from the corresponding pin file
    """
    pin_df = parse_percolator_pin(pin_file)
    pout_df = parse_percolator_pout(pout_file)

    merged = pout_df.join(
        pin_df[["ScanNr"]],  # keep only what’s needed from pin
        how="left",
        rsuffix="_pin",
        sort=False
    )
    
    final_cols = [
        "ScanNr",
        "score",
        "q-value",
        "posterior_error_prob",
        "peptide",
        "proteinIds",
    ]

    return merged[final_cols]


def read_and_filter_pout_file(
    pout_file: str,
    pin_file: str,
    qvalue_thr: float = 0.01,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Reads in a Percolator pout file, enriches it with data from the corresponding pin file and filters it by the given q-value threshold.
    After this, the DataFrame is sorted by the q-value and, if selected, spectra with duplicated identifications are removed (keeping only the first/best identification per spectrum).
    """
    perc_df = read_enriched_pout_with_pin_data(pout_file, pin_file)
    perc_df = perc_df[(perc_df["q-value"] < qvalue_thr)]

    perc_df.sort_values(by=["q-value", "score"], ascending=[True, False], inplace=False)

    if remove_duplicates:
        perc_df.drop_duplicates(subset=["ScanNr"], inplace=True, keep="first")

    return perc_df


def write_percolator_pin_file(perc_df: pd.DataFrame, outfile: str):
    """
    Writes the given DataFrame to a Percolator pin file.
    """

    perc_df["Proteins"] = perc_df["Proteins"].apply(lambda x: "\t".join(x))
    perc_df.to_csv(outfile, sep="\t", index=False, header=True)
