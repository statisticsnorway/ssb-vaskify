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
import re

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
        # Check data
        self._check_data(data, id_nr=id_nr)

        # Create self variables
        self.data = data
        self.id_nr = id_nr

        # Start logging
        logging_dict = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        self.logger = logging.getLogger("detect")
        self.logger.setLevel(logging_dict[logger_level])

        # add in console handling
        if not self.logger.handlers:  # Avoid adding multiple handlers
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_dict[logger_level])
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    @staticmethod
    def _is_valid_date_format(date_str: str) -> bool:
        """Check if a date string matches one of the accepted ISO-like formats.

        Supported formats include:
            - YYYY (e.g., 2025)
            - YYYY-MM (e.g., 2025-07)
            - YYYY-MM-DD (e.g., 2025-07-23)
            - YYYY-Q[1-4] (e.g., 2025-Q3)
            - YYYY-Www (ISO week, e.g., 2025-W31)
            - YYYY-DDD (ordinal date, e.g., 2025-204)

        Parameters
        ----------
        date_str : str
            The date string to validate.

        Returns:
        -------
        bool
            True if the date string matches one of the allowed formats, False otherwise.
        """
        year_pattern = re.compile(r"^\d{4}$")
        year_month_pattern = re.compile(r"^\d{4}-\d{2}$")
        year_month_day_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        year_quarter_pattern = re.compile(r"^\d{4}-Q[1-4]$")
        year_week_pattern = re.compile(r"^\d{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$")
        year_ordinal_pattern = re.compile(r"^\d{4}-\d{3}$")

        return any(
            pattern.match(date_str)
            for pattern in [
                year_pattern,
                year_month_pattern,
                year_month_day_pattern,
                year_quarter_pattern,
                year_week_pattern,
                year_ordinal_pattern,
            ]
        )

    def _check_data(
        self,
        data: pd.DataFrame,
        y_var: str = "",
        time_var: str = "",
        id_nr: str = "",
    ) -> None:
        """Check if the data contains the necessary columns, correct data types, and valid date format.

        Args:
            data: The DataFrame to check.
            y_var: The variable of interest to check.
            time_var: String variable for indicating the time period.
            id_nr: String variable for the identifier.

        Raises:
            ValueError: If any of the checks fail.
        """
        required_columns = [y_var, time_var, id_nr]
        for col in required_columns:
            if col and col not in data.columns:
                mes = f"Missing column: {col}"
                raise ValueError(mes)
        if id_nr and not pd.api.types.is_string_dtype(data[id_nr]):
            mes = f"{id_nr} should be a string."
            raise ValueError(mes)

        if y_var and not pd.api.types.is_numeric_dtype(data[y_var]):
            mes = f"{y_var} should be numeric."
            raise ValueError(mes)

        if time_var:
            if not pd.api.types.is_string_dtype(data[time_var]):
                mes = f"{time_var} should be a string."
                raise ValueError(mes)

            if not data[time_var].apply(self._is_valid_date_format).all():
                mes = f"{time_var} should be in the format 'YYYY', 'YYYY-Qq', 'YYYY-MM','YYYY-Www','YYYY-MM-DD', 'YYYY-DDD'."
                raise ValueError(mes)

    def change_logging_level(self, logger_level: str) -> None:
        """Change the logging print level.

        Args:
            logger_level: Detail level for information output. Choose between 'debug','info','warning','error' and 'critical'.
        """
        logging_dict = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        self.logger.setLevel(logging_dict[logger_level])

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
            time_var: String variable for indicating the time period. This should be in a ISO 8601 standard format for example: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or a SSB standard like 'YYYY-Qq'.
            lower_bound: Float variable for the lower bound log factor for defining an outlier.
            upper_bound: Float variable for the upper bound log factor for defining an outlier.
            flag: String for the name of the flag variable to add to the data. Default is 'flag_thousand'.
            impute: Boolean for whether to impute the flagged observations. Default is False.
            impute_var: String for the name of the imputed variable.
            output_format: String for whether to return a data frame 'data', or just the identified outlier units 'outliers'.

        Returns:
            Data frame containing a flag variable for identified outliers or a dataframe containing only the outliers.
        """
        # Check data
        self._check_data(self.data, y_var=y_var, time_var=time_var)

        if (not impute_var) and (impute):
            impute_var = f"{y_var}_imputed"
            mes = f"No impute variable given so using {impute_var}"
            self.logger.info(mes)

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
            output: pd.DataFrame = data

        # select outlier units and return only them if output_format is outliers
        elif output_format == "outliers":
            outlier_ids = data.loc[mask_outlier, self.id_nr]
            mask_outlier_units = data[self.id_nr].isin(outlier_ids)
            output = data.loc[mask_outlier_units, :]
        else:
            output = data
            mes = "output_format is not valid. Use 'data' or 'outliers'. Returning 'data' format."
            self.logger.warning(mes)

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
            time_var: String variable for indicating the time period. This should be in a ISO 8601 standard format for example: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or a SSB standard like 'YYYY-Qq'.
            error: Float for the allowed error factor.
            flag: String for the name of the flag variable to add to the data. Default is 'flag_thousand'.
            impute: Boolean for whether to impute the flagged observations. Default is False. (NOT IMPLEMENTED)
            impute_var: String for the name of the imputed variable.
            output_format: String for whether to return a data frame 'data', or just the identified outlier units 'outliers'.

        Returns:
            Data frame containing a flag variable for identified outliers or a dataframe containing only the outliers.
        """
        # Check data
        self._check_data(self.data, y_var=y_var, time_var=time_var)

        if (not impute_var) and (impute):
            impute_var = f"{y_var}_imputed"
            mes = f"No imputed variable name given so {impute_var} is being used"
            self.logger.info(mes)

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
            self.logger.error(mes)

        if output_format == "data":
            output: pd.DataFrame = data
        elif output_format == "outliers":
            flagged_ids = (
                data.groupby(self.id_nr)[flag]
                .apply(lambda x: ((x == 1) | x.isna()).all())
                .reset_index()
            )
            mes = f"Number of units identified with possible accumulation errors: {flagged_ids[flag].sum()}"
            self.logger.info(mes)
            ids_with_flag_all_periods = flagged_ids[flagged_ids[flag]][self.id_nr]
            mask_units = data[self.id_nr].isin(ids_with_flag_all_periods)
            output = data.loc[mask_units, :]
        else:
            self.logger.warning("output_format is not valid. Use 'data' or 'outliers'")

        return output

    @staticmethod
    def _calculate_hb(
        x1: pd.Series,
        x2: pd.Series,
        pu: float,
        pa: float,
        pc: float,
        percentiles: tuple[float, float],
    ) -> pd.DataFrame:
        """Calculate HB method."""
        rat = x1 / x2
        med_ratio = rat.median()
        s_ratio = np.where(
            rat >= med_ratio,
            rat / med_ratio - 1,
            1 - med_ratio / rat,
        )

        max_y = pd.concat([x1, x2], axis=1).max(axis=1)
        e_ratio = s_ratio * max_y**pu

        e_ratio_q = e_ratio.quantile([percentiles[0], 0.5, percentiles[1]]).to_numpy()
        q1, q2, q3 = e_ratio_q

        if q2 != 0:
            ell = q2 - pc * max(q2 - q1, abs(q2 * pa))
            eul = q2 + pc * max(q3 - q2, abs(q2 * pa))
        else:
            ell = q2 - pc * max(q2 - q1, pa)
            eul = q2 + pc * max(q3 - q2, pa)

        lower_limit = med_ratio * max_y**pu / (max_y**pu - ell)
        upper_limit = med_ratio * (max_y**pu + eul) / max_y**pu

        return pd.DataFrame({"lower_limit": lower_limit, "upper_limit": upper_limit})

    def hb(
        self,
        y_var: str,
        time_var: str,
        time_periods: list[str] | None = None,
        strata_var: str = "",
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
            time_var: String variable for indicating the time period. This should be in a ISO 8601 standard format for example: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or a SSB standard like 'YYYY-Qq'.
            time_periods: List of strings for the two time periods to compare. Default None, in which case it is assumed that the time variable contains exactly two time preiods.
            strata_var: String variable for stratification. Default is blank ("").
            pu: Parameter that adjusts for different level of the variables. Default value 0.5.
            pa: Parameter that adjusts for small differences between the median and the 1st or 3rd quartile. Default value 0.05.
            pc: Parameter that controls the width of the confidence interval. Default value 20.
            percentiles: Tuple for percentile values to use.
            flag: String variable name to use to indicate outliers.
            output_format: String for format to return. Can be 'wide','long','outliers'.

        Returns:
            Dataframe with flags or with identified units
        """
        # Check data
        self._check_data(self.data, y_var=y_var, time_var=time_var)
        data = self.data.copy()

        # Add in check if number of companies in each strata is too low.

        # Filter time periods
        if time_periods:
            if len(time_periods) != 2:
                mes = "Two time periods should be specified."
                self.logger.error(mes)
            data = data.loc[data[time_var].isin(time_periods), :]

        # Get time levels
        time_levels = np.unique(data[time_var])
        if len(time_levels) != 2:
            mes = "The time variable must have exactly two unique levels."
            self.logger.error(mes)
        time1 = time_levels[1]  # t
        time0 = time_levels[0]  # t-1

        # Convert to wide
        wide_index = [self.id_nr, strata_var] if strata_var else self.id_nr
        wide_data = data.pivot_table(
            index=wide_index,
            columns=time_var,
            values=y_var,
            aggfunc="first",
        ).reset_index()
        wide_data.columns.name = None

        # Check for valid rows
        valid_rows = wide_data[(wide_data[time1] > 0) & (wide_data[time0] > 0)]
        if valid_rows.empty:
            mes = "No valid rows with y_var > 0 for both time periods."
            self.logger.error(mes)

        # Add in ratio
        valid_rows["ratio"] = valid_rows[time1] / valid_rows[time0]

        # Apply the HB function to each strata group
        if strata_var:
            limits = (
                valid_rows.groupby(strata_var)
                .apply(
                    lambda group: self._calculate_hb(
                        group[time1],
                        group[time0],
                        pu,
                        pa,
                        pc,
                        percentiles,
                    ),
                )
                .reset_index(level=strata_var, drop=True)
            )
        else:
            limits = self._calculate_hb(
                valid_rows[time1],
                valid_rows[time0],
                pu,
                pa,
                pc,
                percentiles,
            )

        # Merge the limits back into the valid_rows
        valid_rows = valid_rows.merge(
            limits,
            left_index=True,
            right_index=True,
            how="left",
        )

        # Add in flag
        valid_rows[flag] = np.where(
            (valid_rows["ratio"] < valid_rows["lower_limit"])
            | (valid_rows["ratio"] > valid_rows["upper_limit"]),
            1,
            0,
        )

        # Format in correct output format
        if output_format == "wide":
            output: pd.DataFrame = valid_rows
        elif output_format == "outliers":
            mask_units = valid_rows[flag] == 1
            output = valid_rows.loc[mask_units, :]
            if output.shape[0] == 0:
                self.logger.info("No outliers detected")
        elif output_format == "long":
            output = valid_rows.melt(
                id_vars=[self.id_nr, "ratio", "lower_limit", "upper_limit", flag],
                value_vars=time_levels,
                var_name=time_var,
                value_name=y_var,
            )
            mask = output[time_var] == time_levels[0]
            output.loc[mask, ["lower_limit", "upper_limit", flag]] = np.nan
        else:
            mes = "output_format is not valid. Use 'wide', 'outliers' or 'long'. Wide being returned."
            self.logger.warning(mes)
            output = valid_rows

        return output
