# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Functions for controlling data to identify possible errors
import numpy as np
import pandas as pd

class vaskify:
    """
    Class for data editing.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        id_nr: str,
        verbose: int = 1,
    ) -> None:
        """
        Initialize general data editing object.

        Args:
            data: Data to be controlled/edited. If multiple time periods are in the data, the data should be in a long format.
            id_nr: Name of the variable to identify units with.
            verbose: Whether to show printed output or not. 1 is minimal output, 2 is more output.
        """
        self.data = data
        self.id_nr = id_nr
        self.verbose = verbose

    def change_verbose(self, verbose: int) -> None:
        """Change the verbose print level."""
        self.verbose = verbose

    
    def thousand_error(self, time_var: str, y_var: str, 
                       lower_bound: float = -2.5, 
                       upper_bound: float = 2.5, 
                       flag: str = "flag_thousand",
                       impute: bool = False,
                       output_format: str = "data"):
        """
        """
        # check data - add in
    
        # Find differences by sorting first - not efficient but works
        df = self.data.sort_values(by=[self.id_nr, time_var]).reset_index(drop=True)
        log10_diff = test_data.groupby(self.id_nr)[y_var].transform(
            lambda x: np.log10(x).diff()
        )
    
        # set flag for first periods to NA
        test_data[flag] = 0
        mask_na = log10_diff.isna()
        df.loc[mask_na, flag] = np.nan
    
        # set flag for outlier
        mask_outlier = (log10_diff > upper_bound) | (log10_diff < lower_bound)
        df.loc[mask_outlier, flag] = 1

        # return data if output_format is data
        if output_format == "data":
            return df

        # select outlier units and return only them if output_format is outliers
        if output_format == "units":
            outlier_ids = df.loc[mask_outlier, self.id_nr]
            mask_outlier_units = df[self.id_nr].isin(outlier_ids)
            return df.loc[mask_outlier_units, :]

    
    def accumulation_error(self,
                           time_var: str, y_var: str, 
                           error: float = 0.5,
                           flag: str = "flag_accumulation",
                           impute: bool = False,
                           output_format: str = "data"):
        """
        """
        # check data

        # Sort and get previous period data
        df = self.data.sort_values(by=[self.id_nr, time_var]).reset_index(drop=True)
        expected_turnover = df.groupby(self.id_nr)[y_var].shift(1)

        # Set flag variable and set Nas
        df[flag] = 0
        mask_na = expected_turnover.isna()
        df.loc[mask_na, flag] = np.nan
        
        # set flag variable
        mask_accum = df[y_var] > expected_turnover * (1 + error)
        df.loc[mask_accum, flag] = 1

        if output_format == "data":
            return df
        if output_format == "units":
            flagged_ids=df.groupby(self.id_nr)[flag].apply(lambda x: ((x == 1)|x.isna()).all()).reset_index()
            if self.verbose == 2:
                print(f'Number of units identified withpossible accumulation errors: {flagged_ids[flag].sum()}')
            ids_with_flag_all_periods = flagged_ids[flagged_ids[flag] == True][self.id_nr]
            mask_units = df[self.id_nr].isin(ids_with_flag_all_periods)
            return df.loc[mask_units, :]

