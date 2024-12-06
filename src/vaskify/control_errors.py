# %% [markdown]
# # Functions for controlling data to identify possible errors
# To do:
#
# - Add in data checks
# - Stratification option
# - Impute for accumulative error
# - Documentation

# %%
import logging

import numpy as np
import pandas as pd


# %%
class Detect:
    """Class for data editing."""

    def __init__(
        self,
        data: pd.DataFrame,
        id_nr: str,
        logger_level: str = "warning",
    ) -> None:
        """Initialize general data editing object.

        Args:
            data: Pandas dataframe to be controlled/edited. If multiple time periods are in the data, the data should be in a long format.
            id_nr: String variable for the name of the variable to identify units with.
            logger_level: Detail level for information output. Choose between 'debug','info','warning','error' and 'critical'.
        """
        self.data = data
        self.id_nr = id_nr

        # Set up logging - doesn't need to be self - global
        logger_level = "info"
        logging_dict = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        logger = logging.getLogger("detect")
        logger.setLevel(logging_dict[logger_level])

        # add in console handling
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_dict[logger_level])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    def change_logging_level(self, logger_level: str) -> None:
        """Change the logging print level.

        Args:
            logger_level: Detail level for information output. Choose between 'debug','info','warning','error' and 'critical'.
        """
        logger = logging.getLogger("detect")
        logging_dict = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        logger.setLevel(logging_dict[logger_level])

    def thousand_error(
        self,
        y_var: str,
        time_var: str,
        lower_bound: float = -2.5,
        upper_bound: float = 2.5,
        flag: str = "flag_thousand",
        impute: bool = False,
        impute_var: str = "",
        output_format: str = "data",
    ) -> pd.DataFrame:
        """Detect thousand errors based on a previous period.

        Args:
            y_var: The variable of insterest to check.
            time_var: String variable for indicating the time period. This should be in a standard format: 'YYYY', 'YYYY-Mm', 'YYYY-Kk'.
            lower_bound: Float variable for the lower bound log factor for defining an outlier.
            upper_bound: Float variable for the upper bound log factor for defining an outlier.
            flag: String for the name of the flag variable to add to the data. Default is 'flag_thousand'.
            impute: Boolean for whether to impute the flagged observations. Default is False.
            impute_var: String for the name of the imputed variable.
            output_format: String for whether to return a data frame 'data', or just the identified outlier units 'outliers'.

        Return:
            Data frame with flags or identified units
        """
        logger = logging.getLogger("detect")
        if (not impute_var) and (impute):
            impute_var = f"{y_var}_imputed"
            mes = f"No impute variable given so using {impute_var}"
            logger.info(mes)

        # check data - add in

        # Find differences by sorting first - not efficient but works
        data = self.data.sort_values(by=[self.id_nr, time_var]).reset_index(drop=True)
        log10_diff = data.groupby(self.id_nr)[y_var].transform(
            lambda x: np.log10(x).diff(),
        )

        # set flag for first periods to NA
        data[flag] = 0
        mask_na = log10_diff.isna()
        data.loc[mask_na, flag] = np.nan

        # set flag for outlier
        mask_outlier = (log10_diff > upper_bound) | (log10_diff < lower_bound)
        data.loc[mask_outlier, flag] = 1

        # Impute
        if impute:
            data[impute_var] = data[y_var].copy()
            data.loc[mask_outlier, impute_var] = data.loc[mask_outlier, y_var] / 1000

        # return data if output_format is data
        if output_format == "data":
            output = data

        # select outlier units and return only them if output_format is outliers
        if output_format == "outliers":
            outlier_ids = data.loc[mask_outlier, self.id_nr]
            mask_outlier_units = data[self.id_nr].isin(outlier_ids)
            output = data.loc[mask_outlier_units, :]

        return output

    def accumulation_error(
        self,
        y_var: str,
        time_var: str,
        error: float = 0.5,
        flag: str = "flag_accumulation",
        impute: bool = False,
        impute_var: str = "",
        output_format: str = "data",
    ) -> pd.DataFrame:
        """Detect accumulation errors based on a previous periods.

        Args:
            y_var: The variable of insterest to check.
            time_var: String variable for indicating the time period. This should be in a standard format: 'YYYY', 'YYYY-Mm', 'YYYY-Kk'.
            error: Float for the allowed error factor.
            flag: String for the name of the flag variable to add to the data. Default is 'flag_thousand'.
            impute: Boolean for whether to impute the flagged observations. Default is False. (NOT IMPLEMENTED)
            impute_var: String for the name of the imputed variable.
            output_format: String for whether to return a data frame 'data', or just the identified outlier units 'outliers'.

        Return:
            Data frame with flags or identified units
        """
        logger = logging.getLogger("detect")
        if (not impute_var) and (impute):
            impute_var = f"{y_var}_imputed"
            mes = f"No imputed variable name given so {impute_var} is being used"
            logger.info(mes)

        # check data

        # Sort and get previous period data
        data = self.data.sort_values(by=[self.id_nr, time_var]).reset_index(drop=True)
        expected_turnover = data.groupby(self.id_nr)[y_var].shift(1)

        # Set flag variable and set Nas
        data[flag] = 0
        mask_na = expected_turnover.isna()
        data.loc[mask_na, flag] = np.nan

        # set flag variable
        mask_accum = data[y_var] > expected_turnover * (1 + error)
        data.loc[mask_accum, flag] = 1

        # Impute - not implemented
        if impute:
            mes = "Imputation not implemented for this method."
            logger.error(mes)

        if output_format == "data":
            output = data
        if output_format == "outliers":
            flagged_ids = (
                data.groupby(self.id_nr)[flag]
                .apply(lambda x: ((x == 1) | x.isna()).all())
                .reset_index()
            )
            mes = f"Number of units identified with possible accumulation errors: {flagged_ids[flag].sum()}"
            logger.info(mes)
            ids_with_flag_all_periods = flagged_ids[flagged_ids[flag]][self.id_nr]
            mask_units = data[self.id_nr].isin(ids_with_flag_all_periods)
            output = data.loc[mask_units, :]

        return output

    def hb(
        self,
        y_var: str,
        time_var: str,
        pu: float = 0.5,
        pa: float = 0.05,
        pc: float = 20,
        percentiles: tuple[float, float] = (0.25, 0.75),
        flag: str = "flag_hb",
        output_format: str = "wide",
    ) -> pd.DataFrame:
        """Outlier detection using the Hidiroglou-Berthelot (HB) method.

        Detects possible outliers of a variable in period t by comparing it with values from period t-1.

        Args:
            y_var: String for the name of the variable of interest to check.
            time_var: String variable for indicating the time period. This should be in a standard format: 'YYYY', 'YYYY-Mm', 'YYYY-Kk'.
            pu: Parameter that adjusts for different level of the variables. Default value 0.5.
            pa: Parameter that adjusts for small differences between the median and the 1st or 3rd quartile. Default value 0.05.
            pc: Parameter that controls the width of the confidence interval. Default value 4.
            percentiles: Tuple for percentile values to use.
            flag: String variable name to use to indicate outliers.
            output_format: String for format to return. Can be 'wide','long','outliers'
        Return:
            Dataframe with flags or with identified units
        """
        logger = logging.getLogger("detect")

        # check data ...
        data = self.data.copy()

        # Get time levesl
        time_levels = data[time_var].unique()
        if len(time_levels) != 2:
            mes = "The time variable must have exactly two unique levels."
            logger.error(mes)
        x1 = time_levels[1]  # t
        x2 = time_levels[0]  # t-1

        # Convert to wide
        wide_data = data.pivot_table(
            index=self.id_nr,
            columns=time_var,
            values=y_var,
            aggfunc="first",
        ).reset_index()
        wide_data.columns.name = None

        # Check for valid rows
        valid_rows = wide_data[(wide_data[x1] > 0) & (wide_data[x2] > 0)]
        if valid_rows.empty:
            mes = "No valid rows with y_var > 0 for both time periods."
            logger.error(mes)

        # Calculate the ratio and related metrics
        valid_rows["ratio"] = valid_rows[x1] / valid_rows[x2]
        med_ratio = valid_rows["ratio"].median()
        s_ratio = np.where(
            valid_rows["ratio"] >= med_ratio,
            valid_rows["ratio"] / med_ratio - 1,
            1 - med_ratio / valid_rows["ratio"],
        )

        max_y = valid_rows[[x1, x2]].max(axis=1)
        e_ratio = s_ratio * max_y**pu

        # Compute quantiles for e ratio
        percentiles = (0.25, 0.75)  # Can also be 0.1, 0.9
        e_ratio_q = e_ratio.quantile([percentiles[0], 0.5, percentiles[1]]).to_numpy()
        q1, q2, q3 = e_ratio_q

        if q2 != 0:
            ell = q2 - pc * max(q2 - q1, abs(q2 * pa))
            eul = q2 + pc * max(q3 - q2, abs(q2 * pa))
        else:
            ell = q2 - pc * max(q2 - q1, pa)
            eul = q2 + pc * max(q3 - q2, pa)
        valid_rows["lower_limit"] = med_ratio * max_y**pu / (max_y**pu - ell)
        valid_rows["upper_limit"] = med_ratio * (max_y**pu + eul) / max_y**pu
        valid_rows[flag] = np.where(
            (valid_rows["ratio"] < valid_rows["lower_limit"])
            | (valid_rows["ratio"] > valid_rows["upper_limit"]),
            1,
            0,
        )
        if output_format == "wide":
            output = valid_rows
        if output_format == "outliers":
            mask_units = valid_rows[flag] == 1
            output = valid_rows.loc[mask_units, :]
            if output.shape[0] == 0:
                logger.info("No outliers detected")

        if output_format == "long":
            output = output.melt(
                id_vars=[self.id_nr, "ratio", "lower_limit", "upper_limit", flag],
                value_vars=time_levels,
                var_name=time_var,
                value_name=y_var,
            )
        return output
