import pandas as pd

def calculate_fdr(pin_df: pd.DataFrame, fdr_by: str, lowerscorebetter: bool = False) -> pd.DataFrame:
    """
    Calculate FDR (and q-value) based on the given score
    """
    pin_df[fdr_by] = pd.to_numeric(pin_df[fdr_by])

    pin_df["is_decoy"] = pin_df["Label"].apply(lambda x: True if x == -1 else False)
    pin_df.sort_values(by=[fdr_by, "is_decoy"], ascending=[lowerscorebetter, True], inplace=True)

    nr_targets = (pin_df["is_decoy"] != True).cumsum()
    nr_decoys = pin_df["is_decoy"].cumsum()

    # Calculate FDR
    fdr = nr_decoys / nr_targets

    # q-value
    pin_df['q-value'] = fdr[::-1].cummin()[::-1]
    return pin_df


def calculate_fdp(pin_df: pd.DataFrame, fdp_by: str, entr_fold: int, lowerscorebetter: bool = False) -> pd.DataFrame:
    """
    Calculate FDP (using entrapment after https://doi.org/10.1101/2024.06.01.596967 )
    """
    pin_df["proteinIds"] = pin_df["proteinIds"].apply(
        lambda ids: ids.split("\t") if isinstance(ids, str) else ids
    )
    
    pin_df["is_entrapment"] = pin_df["proteinIds"].apply(
        lambda ids: all(protein.startswith("ENTRAPMENT") or protein.startswith("DECOY") for protein in ids)
    )

    pin_df[fdp_by] = pd.to_numeric(pin_df[fdp_by])
    pin_df.sort_values(by=[fdp_by, "is_entrapment"], ascending=[lowerscorebetter, True], inplace=True)

    nr_targets = (pin_df["is_entrapment"] != True).cumsum()
    nr_entrapments = pin_df["is_entrapment"].cumsum()

    # entrapment estimation after https://doi.org/10.1101/2024.06.01.596967
    entr_ratio = (1 + 1 / entr_fold)
    pin_df['fdp_ub'] = nr_entrapments * entr_ratio / (nr_targets + nr_entrapments)
    pin_df['fdp_lb'] = nr_entrapments / (nr_targets + nr_entrapments)
    return pin_df
