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

        Args:
            date_str: The date string to validate.

        Returns:
            bool: True if the date string matches one of the allowed formats, False otherwise.
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
        y_var: str | list[str],
        time_var: str | None = None,
        lower_bound: float = -2.5,
        upper_bound: float = 2.5,
        flag: str = "flag_thousand",
        impute: bool = False,
        impute_var: str = "",
        output_format: str = "data",
    ) -> pd.DataFrame:
        """Detect thousand errors based on a previous period.

        Args:
            y_var: The variable(s) of interest to check. In long format, a single variable name or list of variable names. In wide format, a prefix or list of prefixes (e.g. 'employees') matching columns
            time_var: String variable for indicating the time period. This should be in a ISO 8601 standard format for example: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or a SSB standard like 'YYYY-Qq'. Set to None for wide-format data.
            lower_bound: Float variable for the lower bound log factor for defining an outlier.
            upper_bound: Float variable for the upper bound log factor for defining an outlier.
            flag: String for the name of the flag variable to add to the data. Default is 'flag_thousand'.
            impute: Boolean for whether to impute the flagged observations. Default is False.
            impute_var: String for the name of the imputed variable.
            output_format: String for whether to return a data frame 'data', or just the identified outlier units 'outliers'.

        Returns:
            Data frame containing a flag variable for identified outliers or a dataframe containing only the outliers.
        """
        wide = time_var is None

        # Normalise y_var and flag to a list
        y_vars = [y_var] if isinstance(y_var, str) else y_var
        flag_names = [flag]
        
        # Resolve impute_var names: one per y_var
        if wide:
            impute_vars = [impute_var if impute_var else y_vars[0].split("_")[0] + "_imputed"]
        else:
            impute_vars = [impute_var if impute_var else f"{y_var}_imputed"]
    
        for v in y_vars:
            self._check_data(self.data, y_var=v, time_var=time_var)

        data = self.data.copy()
        combined_outlier_mask = pd.Series(False, index=data.index)

        # Dispatch to wide or long implementation
        if wide:
            data, combined_outlier_mask = self._thousand_error_wide(
                y_vars, flag_names, impute_vars, lower_bound, upper_bound, impute
            )
        else:
            data, combined_outlier_mask = self._thousand_error_long(
                y_vars, flag_names, impute_vars, time_var, lower_bound, upper_bound, impute
            )

        # Apply output format
        if output_format == "data":
            return data
        elif output_format == "outliers":
            outlier_ids = data.loc[combined_outlier_mask, self.id_nr]
            mask_outlier_units = data[self.id_nr].isin(outlier_ids)
            return data.loc[mask_outlier_units, :]
        else:
            self.logger.warning(
                "output_format is not valid. Use 'data' or 'outliers'. Returning 'data' format."
            )
            return data

    def _thousand_error_wide(
        self,
        y_vars: list[str],
        flag_names: list[str],
        impute_vars: list[str],
        lower_bound: float,
        upper_bound: float,
        impute: bool,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Wide-format implementation of thousand error detection.
    
        Args:
            y_vars: List of variable prefixes to check.
            flag_names: List of flag column names, one per y_var.
            impute_vars: List of imputed variable name prefixes, one per y_var.
            lower_bound: Lower bound log factor for defining an outlier.
            upper_bound: Upper bound log factor for defining an outlier.
            impute: Whether to impute flagged observations.
    
        Returns:
            Tuple of (data, combined_outlier_mask).
        """
        data = self.data.copy()
        combined_outlier_mask = pd.Series(False, index=data.index)
    
        for flag_col, imp_col in zip(flag_names, impute_vars, strict=False):
            
            log10_diff = np.log10(data[y_vars]).diff(axis=1).iloc[:, 1:]  # drop first col

            for col in y_vars[1:]:
                period_suffix = col.split("_", 1)[1]
                period_flag = f"{flag_col}_{period_suffix}"
                mask_na = log10_diff[col].isna()
                mask_outlier = ((log10_diff[col] > upper_bound) | (log10_diff[col] < lower_bound))
    
                data[period_flag] = 0
                data.loc[mask_na, period_flag] = np.nan
                data.loc[mask_outlier, period_flag] = 1
    
                if impute:
                    imp_col_wide = f"{imp_col}_{period_suffix}"
                    data[imp_col_wide] = data[col].copy()
                    data.loc[mask_outlier, imp_col_wide] = data.loc[mask_outlier, col] / 1000
    
                combined_outlier_mask |= mask_outlier
    
        return data, combined_outlier_mask

    def _thousand_error_long(
        self,
        y_vars: list[str],
        flag_names: list[str],
        impute_vars: list[str],
        time_var: str,
        lower_bound: float,
        upper_bound: float,
        impute: bool,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Long-format implementation of thousand error detection.
    
        Args:
            y_vars: List of variable names to check.
            flag_names: List of flag column names, one per y_var.
            impute_vars: List of imputed variable names, one per y_var.
            time_var: Column name indicating the time period.
            lower_bound: Lower bound log factor for defining an outlier.
            upper_bound: Upper bound log factor for defining an outlier.
            impute: Whether to impute flagged observations.
    
        Returns:
            Tuple of (data, combined_outlier_mask).
        """
        data = self.data.sort_values(by=[self.id_nr, time_var]).reset_index(drop=True)
        combined_outlier_mask = pd.Series(False, index=data.index)
    
        for v, flag_col, imp_col in zip(y_vars, flag_names, impute_vars, strict=False):
            log10_diff = data.groupby(self.id_nr)[v].transform(
                lambda x: np.log10(x).diff()
            )
    
            mask_na = log10_diff.isna()
            mask_outlier = (log10_diff > upper_bound) | (log10_diff < lower_bound)
    
            data[flag_col] = 0
            data.loc[mask_na, flag_col] = np.nan
            data.loc[mask_outlier, flag_col] = 1
    
            if impute:
                data[imp_col] = data[v].copy()
                data.loc[mask_outlier, imp_col] = data.loc[mask_outlier, v] / 1000
    
            combined_outlier_mask |= mask_outlier
    
        return data, combined_outlier_mask

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
        y_var: str | list[str],
        time_var: str | None = None,
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
            time_var: String variable for indicating the time period. This should be in a ISO 8601 standard format for example: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or a SSB standard like 'YYYY-Qq'. Set to None for wide-format data.
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
        wide = time_var is None

        if wide:
            data, combined_outlier_mask = self._hb_wide(
                y_var, strata_var, pu, pa, pc, percentiles, flag
            )
        else:
            data = self._hb_long(
                y_var, time_var, time_periods, strata_var, pu, pa, pc, percentiles, flag
            )
            combined_outlier_mask = data[flag].eq(1)

        # Format in correct output format
        if output_format == "wide":
            return data
        elif output_format == "outliers":
            print(len(combined_outlier_mask))
            print(data.shape)
            output = data.loc[combined_outlier_mask, :]
            if output.shape[0] == 0:
                self.logger.info("No outliers detected")
        elif output_format == "long":
            if wide:
                self.logger.warning("'long' output format is not available for wide input. Returning 'wide' format.")
                return data
            return data# long format handling done inside _hb_long
        else:
            mes = "output_format is not valid. Use 'wide', 'outliers' or 'long'. Wide being returned."
            self.logger.warning(mes)
            output = valid_rows


    def _hb_long(
        self,
        y_var: str,
        time_var: str,
        time_periods: list[str] | None,
        strata_var: str,
        pu: float,
        pa: float,
        pc: float,
        percentiles: tuple[float, float],
        flag: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Long-format implementation of the HB method.
        """
        self._check_data(self.data, y_var=y_var, time_var=time_var)
        data = self.data.copy()

        if time_periods:
            if len(time_periods) != 2:
                self.logger.error("Two time periods should be specified.")
            data = data.loc[data[time_var].isin(time_periods), :]
            
        time_levels = np.unique(data[time_var])
        if len(time_levels) != 2:
            self.logger.error("The time variable must have exactly two unique levels.")
        time0, time1 = time_levels[0], time_levels[1]

        wide_index = [self.id_nr, strata_var] if strata_var else self.id_nr
        wide_data = data.pivot_table(
            index=wide_index,
            columns=time_var,
            values=y_var,
            aggfunc="first",
        ).reset_index()
        wide_data.columns.name = None

        result, outlier_mask = self._hb_calculate_and_flag(
            wide_data, time0, time1, strata_var, pu, pa, pc, percentiles, flag
        )
    
        # Handle long output format
        mask = result[time_var] if time_var in result.columns else None
        output = result.melt(
            id_vars=[self.id_nr, "ratio", "lower_limit", "upper_limit", flag],
            value_vars=time_levels,
            var_name=time_var,
            value_name=y_var,
        )
        mask = output[time_var] == time_levels[0]
        output.loc[mask, ["lower_limit", "upper_limit", flag]] = np.nan
    
        return output

    def _hb_wide(
        self,
        y_var: list[str],
        strata_var: str,
        pu: float,
        pa: float,
        pc: float,
        percentiles: tuple[float, float],
        flag: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Wide-format implementation of the HB method.
    
        Args:
            y_var: List of exactly two column names [t-1, t].
            strata_var: Optional stratification variable.
            pu: Level adjustment parameter.
            pa: Small difference adjustment parameter.
            pc: Confidence interval width parameter.
            percentiles: Percentile values to use.
            flag: Flag column name.
    
        Returns:
            Tuple of (data, outlier_mask).
        """
        if len(y_var) != 2:
            self.logger.error("y_var must contain exactly two column names in wide format.")
        time0, time1 = y_var[0], y_var[1]
    
        for v in y_var:
            self._check_data(self.data, y_var=v, time_var=None)
    
        data = self.data.copy()
        return self._hb_calculate_and_flag(
            data, time0, time1, strata_var, pu, pa, pc, percentiles, flag
        )

    def _hb_calculate_and_flag(
        self,
        data: pd.DataFrame,
        time0: str,
        time1: str,
        strata_var: str,
        pu: float,
        pa: float,
        pc: float,
        percentiles: tuple[float, float],
        flag: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Shared HB calculation, limit setting, and flagging logic.
        """
        valid_rows = data[(data[time1] > 0) & (data[time0] > 0)].copy()
        if valid_rows.empty:
            self.logger.error("No valid rows with y_var > 0 for both time periods.")
    
        valid_rows["ratio"] = valid_rows[time1] / valid_rows[time0]
    
        if strata_var:
            limits = (
                valid_rows.groupby(strata_var)
                .apply(
                    lambda group: self._calculate_hb(
                        group[time1], group[time0], pu, pa, pc, percentiles
                    )
                )
                .reset_index(level=strata_var, drop=True)
            )
        else:
            limits = self._calculate_hb(
                valid_rows[time1], valid_rows[time0], pu, pa, pc, percentiles
            )
    
        valid_rows = valid_rows.merge(limits, left_index=True, right_index=True, how="left")
    
        valid_rows[flag] = np.where(
            (valid_rows["ratio"] < valid_rows["lower_limit"])
            | (valid_rows["ratio"] > valid_rows["upper_limit"]),
            1,
            0,
        )
    
        outlier_mask = valid_rows[flag] == 1
        return valid_rows, outlier_mask