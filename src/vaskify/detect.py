# %% [markdown]
# # Functions for controlling data to identify possible errors
# To do:
#
# - Impute for accumulative error
# - Documentation

# %%
import logging
import re
from typing import Any

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
        time_var: str | None = "",
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
        output_format: str = "wide",
        output_scope: str = "all",
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
            output_format: String for whether to return a data frame in 'wide' (default) or 'long' format. For 'infer', the function will return the same format as the input data.
            output_scope: String for whether to return all the data ('all') or just the identified outlier units ('outliers').

        Returns:
            Data frame containing a flag variable for identified outliers or a dataframe containing only the outliers (depending on output_scope).
        """
        wide = time_var is None

        # Check output format
        if output_format not in {"wide", "long", "infer"}:
            self.logger.warning(
                msg="output_format is not valid. Use 'infer', 'wide', 'long'. Returning 'wide' format.",
            )

        if output_format == "infer":
            if not wide:
                output_format = "long"
            if wide:
                output_format = "wide"

        if output_format == "long" and wide:
            self.logger.warning(
                "Only wide output format is curently available for data which is inputed as wide format."
            )
            output_format = "wide"

        # Normalise y_var and flag to a list
        y_vars = [y_var] if isinstance(y_var, str) else y_var
        flag_names = [flag]

        # Resolve impute_var names: one per y_var
        if wide:
            impute_vars = [
                impute_var if impute_var else y_vars[0].split("_")[0] + "_imputed"
            ]
        else:
            impute_vars = [impute_var if impute_var else f"{y_var}_imputed"]

        for v in y_vars:
            self._check_data(self.data, y_var=v, time_var=time_var)

        data = self.data.copy()

        # Dispatch to wide or long implementation
        if wide:
            data = self._thousand_error_wide(
                y_vars,
                flag_names,
                impute_vars,
                lower_bound,
                upper_bound,
                impute,
                output_format,
            )
        else:
            data = self._thousand_error_long(
                y_vars,
                flag_names,
                impute_vars,
                time_var,
                lower_bound,
                upper_bound,
                impute,
                output_format,
            )

        # Apply output scope
        if output_scope == "outliers":
            mask_outlier_units = self._identify_outliers(
                data, flag_names, output_format
            )
            data = data.loc[mask_outlier_units, :]

            if data.shape[0] == 0:
                self.logger.info("No outliers detected")

        return data

    def _identify_outliers(
        self, dt: pd.DataFrame, flag_names: list[str], output_format: str
    ) -> Any:
        if output_format == "wide":
            flag_cols = [c for c in dt.columns if flag_names[0] in c]
            outlier_mask = (dt[flag_cols] == 1).any(axis=1)
        else:
            mask = dt[flag_names[0]] == 1
            outlier_ids = dt.loc[mask, self.id_nr]
            outlier_mask = dt[self.id_nr].isin(outlier_ids)
        return outlier_mask

    def _long_to_wide(
        self, df: pd.DataFrame, id_nr: str, time_var: str | None
    ) -> pd.DataFrame:
        """Convert a data frame from long to wide format."""
        if time_var is None:
            self.logger.error(msg="Time variable can't be missing when in long format")

        # Alle kolonner som ikke er id eller time
        other_vars = [c for c in df.columns if c not in [id_nr] + [time_var]]

        # Split into time-varying vs constant columns
        varying_vars = [
            c for c in other_vars if df.groupby(id_nr)[c].nunique().gt(1).any()
        ]
        constant_vars = [c for c in other_vars if c not in varying_vars]

        df_wide = df.pivot_table(
            index=id_nr,
            columns=time_var,
            values=varying_vars,
            aggfunc="first",
        )

        # Flat ut MultiIndex-kolonner til f.eks. "verdi_2021", "verdi_2022"
        if isinstance(df_wide.columns, pd.MultiIndex):
            cols = df_wide.columns.to_flat_index()
        else:
            raise AssertionError("Expected MultiIndex columns after pivot_table")

        df_wide.columns = [f"{val}_{tid}" for val, tid in cols]
        df_wide = df_wide.reset_index()

        # Merge in the constant columns (one row per id)
        if constant_vars:
            df_const = df[[id_nr] + constant_vars].drop_duplicates(subset=id_nr)
            df_wide = df_wide.merge(df_const, on=id_nr, how="left")

            col_order = (
                [id_nr]
                + constant_vars
                + [c for c in df_wide.columns if c not in [id_nr] + constant_vars]
            )
            df_wide = df_wide[col_order]

        return df_wide

    def _thousand_error_wide(
        self,
        y_vars: list[str],
        flag_names: list[str],
        impute_vars: list[str],
        lower_bound: float,
        upper_bound: float,
        impute: bool,
        output_format: str,
    ) -> pd.DataFrame:
        """Wide-format implementation of thousand error detection. Output_format not implemented yet."""
        data = self.data.copy()

        for flag_col, imp_col in zip(flag_names, impute_vars, strict=False):

            # Get log diff and drop first col
            log10_diff = np.log10(data[y_vars]).diff(axis=1).iloc[:, 1:]  # type: ignore

            for col in y_vars[1:]:
                period_suffix = col.split("_", 1)[1]
                period_flag = f"{flag_col}_{period_suffix}"
                mask_na = log10_diff[col].isna()
                mask_outlier = (log10_diff[col] > upper_bound) | (
                    log10_diff[col] < lower_bound
                )

                data[period_flag] = 0
                data.loc[mask_na, period_flag] = np.nan
                data.loc[mask_outlier, period_flag] = 1

                if impute:
                    imp_col_wide = f"{imp_col}_{period_suffix}"
                    data[imp_col_wide] = data[col].copy()
                    data.loc[mask_outlier, imp_col_wide] = (
                        data.loc[mask_outlier, col] / 1000
                    )

        return data

    def _thousand_error_long(
        self,
        y_vars: list[str],
        flag_names: list[str],
        impute_vars: list[str],
        time_var: str | None,
        lower_bound: float,
        upper_bound: float,
        impute: bool,
        output_format: str,
    ) -> pd.DataFrame:
        """Long-format implementation of thousand error detection."""
        if time_var is None:
            self.logger.error(msg="Time variable missing in long format")
        else:
            data = self.data.sort_values(by=[self.id_nr, time_var]).reset_index(
                drop=True
            )

        for v, flag_col, imp_col in zip(y_vars, flag_names, impute_vars, strict=False):
            log10_diff = data.groupby(self.id_nr)[v].transform(
                lambda x: np.log10(x).diff(),
            )

            mask_na = log10_diff.isna()
            mask_outlier = (log10_diff > upper_bound) | (log10_diff < lower_bound)

            data[flag_col] = 0
            data.loc[mask_na, flag_col] = np.nan
            data.loc[mask_outlier, flag_col] = 1

            if impute:
                data[imp_col] = data[v].copy()
                data.loc[mask_outlier, imp_col] = data.loc[mask_outlier, v] / 1000

        if output_format == "wide":
            data = self._long_to_wide(data, self.id_nr, time_var)

        return data

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
        """Detect accumulation errors based on a previous periods (unstable beta method).

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
            flagged_ids = data.groupby(self.id_nr, group_keys=False)[flag].apply(
                lambda x: ((x == 1) | x.isna()).all(),  # type: ignore
                include_groups=False,
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
        y1: pd.Series,
        y2: pd.Series,
        pu: float,
        pa: float,
        pc: float,
        percentiles: tuple[float, float],
    ) -> pd.DataFrame:
        """Calculate HB method."""
        rat = y1 / y2
        med_ratio = rat.median()
        s_ratio = np.where(
            rat >= med_ratio,
            rat / med_ratio - 1,
            1 - med_ratio / rat,
        )

        max_y = pd.concat([y1, y2], axis=1).max(axis=1)
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

        output_dt = pd.DataFrame(
            {
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
                "ratio": rat,
                "median_ratio": med_ratio,
            }
        )
        return output_dt

    def _fix_output_format(self, output_format: str, wide: bool) -> str:

        if output_format not in {"wide", "long", "infer"}:
            self.logger.warning(
                "output_format is not valid. Use 'wide', 'long' or 'infer'. Returning 'wide' format.",
            )

        if output_format == "infer":
            if not wide:
                output_format = "long"
            if wide:
                output_format = "wide"

        return output_format

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
        output_scope: str = "all",
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
            percentiles: Tuple for percentile values to use. Default (0.25, 0.75)
            flag: String variable name to use to indicate outliers.
            output_format: String for data format to return. Can be 'wide' (default), 'long' or 'infer'. For 'infer', the data format returned (wide or long) will be that of the input data.
            output_scope: String for which units to return, either all ('all') or just the outliers ('outliers').

        Returns:
            Dataframe with flags or with identified units depending on output_scope
        """
        wide: bool = time_var is None

        output_format = self._fix_output_format(output_format, wide)

        if output_format == "long" and wide:
            self.logger.warning(
                "Only wide output format is curently available for data which is inputed as wide format."
            )
            output_format = "wide"

        if wide:
            assert isinstance(y_var, list)  # to please mypy
            data = self._hb_wide(
                y_var,
                strata_var,
                pu,
                pa,
                pc,
                percentiles,
                flag,
                output_format,
            )
        else:
            assert isinstance(y_var, str)  # to please mypy
            assert isinstance(time_var, str)  # to please mypy
            data = self._hb_long(
                y_var,
                time_var,
                time_periods,
                strata_var,
                pu,
                pa,
                pc,
                percentiles,
                flag,
                output_format,
            )

        # Apply output format
        if output_scope == "outliers":
            mask_outlier_units = self._identify_outliers(
                dt=data, flag_names=[flag], output_format=output_format
            )
            data = data.loc[mask_outlier_units, :]
            if data.shape[0] == 0:
                self.logger.info("No outliers detected")

        return data

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
        output_format: str,
    ) -> pd.DataFrame:
        """Long-format implementation of the HB method."""
        self._check_data(self.data, y_var=y_var, time_var=time_var)
        data = self.data.copy()

        if time_periods:
            if len(time_periods) != 2:
                self.logger.error("Two time periods should be specified.")
            data = data.loc[data[time_var].isin(time_periods), :]

        time_levels = np.unique(data[time_var])
        if len(time_levels) != 2:
            self.logger.error("The time variable must have exactly two unique levels.")
        time0, time1 = f"{y_var}_{time_levels[0]}", f"{y_var}_{time_levels[1]}"

        wide_data = self._long_to_wide(data, self.id_nr, time_var)  # new

        result = self._hb_calculate_and_flag(
            wide_data,
            time0,
            time1,
            strata_var,
            pu,
            pa,
            pc,
            percentiles,
            flag,
        )

        # Handle long output format
        if output_format == "long":
            mask = result[time_var] if time_var in result.columns else None
            output = result.melt(
                id_vars=[
                    self.id_nr,
                    "ratio",
                    "lower_limit",
                    "upper_limit",
                    "median_ratio",
                    flag,
                ],
                value_vars=[time0, time1],
                var_name=time_var,
                value_name=y_var,
            )
            mask = output[time_var] == time0
            output.loc[
                mask, ["ratio", "median_ratio", "lower_limit", "upper_limit", flag]
            ] = np.nan

            # Extract time variable back
            time_pattern = "|".join(time_levels)
            output[time_var] = output[time_var].str.extract(f"({time_pattern})")
            output = pd.merge(data, output, how="left")
        else:
            output = result

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
        output_format: str,
    ) -> pd.DataFrame:
        """Wide-format implementation of the HB method. Output not implemented yet"""
        if len(y_var) != 2:
            self.logger.error(
                "y_var must contain exactly two column names in wide format."
            )
        time0, time1 = y_var[0], y_var[1]

        for v in y_var:
            self._check_data(self.data, y_var=v, time_var="")

        data = self.data.copy()
        return self._hb_calculate_and_flag(
            data,
            time0,
            time1,
            strata_var,
            pu,
            pa,
            pc,
            percentiles,
            flag,
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
    ) -> pd.DataFrame:
        """Shared HB calculation, limit setting, and flagging logic."""
        valid_rows = data[(data[time1] > 0) & (data[time0] > 0)].copy()
        if valid_rows.empty:
            self.logger.error("No valid rows with y_var > 0 for both time periods.")

        if strata_var:
            limits = valid_rows.groupby(
                strata_var, group_keys=False
            ).apply(  # type: ignore
                lambda group: self._calculate_hb(
                    group[time1],
                    group[time0],
                    pu,
                    pa,
                    pc,
                    percentiles,
                ),
                include_groups=False,
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

        valid_rows = valid_rows.merge(
            limits, left_index=True, right_index=True, how="left"
        )

        valid_rows[flag] = np.where(
            (valid_rows["ratio"] < valid_rows["lower_limit"])
            | (valid_rows["ratio"] > valid_rows["upper_limit"]),
            1,
            0,
        )

        return valid_rows

    def quartile_error(
        self,
        x_var: str | list[str],
        y_var: str | list[str] | None = None,
        time_var: str | None = None,
        time_periods: list[str] | None = None,
        strata_var: str = "",
        pkl: float = 3,
        pku: float = 3,
        percentiles: tuple[float, float] = (0.25, 0.75),
        flag: str = "flag_quartile",
        output_format: str = "infer",
        output_scope: str = "all",
    ) -> pd.DataFrame:
        """Detect and flag potential errors based on quartile ranges.

        This method identifies outliers using a quartile-based approach applied to
        ratios between variables. The method currently supports only *wide-format*
        data. Observations with missing values or non-positive values in the
        required variables are excluded before calculations.

        The method supports both single ratios (one x- and one y-variable) and
        multiple ratios (lists of variables), where lower and upper bounds are
        computed using specified percentiles and scaling parameters.

        Parameters
        ----------
        x_var : str or list of str
            Name(s) of numerator variable(s). If a list is provided, `y_var`
            must also be a list of the same length.
        y_var : str or list of str or None, optional
            Name(s) of denominator variable(s). If `None`, a temporary constant
            denominator is used. Default is None.
        time_var : str or None, optional
            Name of a time variable. Currently not supported; only wide-format
            input is implemented. If provided, an error is logged. Default is None.
        time_periods : list of str or None, optional
            Reserved for future use. Currently not applied.
        strata_var : str, optional
            Optional variable defining strata within which quartiles are calculated.
            Default is an empty string (no stratification).
        pkl : float, optional
            Scaling factor applied to the lower quartile limit. Default is 3.
        pku : float, optional
            Scaling factor applied to the upper quartile limit. Default is 3.
        percentiles : tuple of float, optional
            Lower and upper percentiles used to compute quartiles. Default is
            (0.25, 0.75).
        flag : str, optional
            Name of the output flag variable indicating detected outliers.
            Default is "flag_quartile".
        output_format : str, optional
            Reserved for future use. Currently not applied. Only wide format returned.
        output_scope : {"all", "outliers"}, optional
            Determines whether all observations are returned or only those flagged
            as outliers. Default is "all".

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the original data along with calculated
            quartile limits, ratios, and an indicator flag for quartile-based
            outliers. If `output_scope="outliers"`, only flagged observations
            are returned.

        Notes:
        -----
        - Only wide-format input data is currently supported.
        - The method assumes that `x_var` and `y_var` are either both strings or
        both lists of strings when provided.
        """
        wide = time_var is None
        if not wide:
            self.logger.error(
                "Only wide data input format is currently programmed. Please reformat data and do not use a time variable."
            )

        output_format = self._fix_output_format(output_format, wide)

        dt = self.data

        # If missing y_var set up temp for calculations
        if y_var is None:
            dt["y_var_temp"] = 1
            if isinstance(x_var, list):
                dt["y_var_temp2"] = 1
                y_var = ["y_var_temp", "y_var_temp2"]
            else:
                y_var = "y_var_temp"

        # Check if multiple time periods used in wide format
        if isinstance(x_var, list) and isinstance(y_var, list):
            var_list: list[str] = x_var + y_var
            two_ratios = True
        else:
            assert isinstance(x_var, str)  # to please mypy
            assert isinstance(y_var, str)  # to please mypy
            var_list = [x_var, y_var]
            two_ratios = False

        # Check for valid observations (not na and > 0)
        keep = dt[var_list].notna().all(axis=1) & (self.data[var_list] > 0).all(axis=1)
        dt = dt.loc[keep].copy()

        dt_quartiles = self._calculate_quartiles(
            dt, strata_var, x_var, y_var, percentiles, pkl, pku, flag, two_ratios
        )
        dt_quartiles = dt_quartiles.drop(
            columns=["y_var_temp", "y_var_temp2"], errors="ignore"
        )

        # Apply output scope
        if output_scope == "outliers":
            mask_outlier_units = self._identify_outliers(
                dt_quartiles, flag_names=[flag], output_format="wide"
            )  # only wide implemented
            dt_quartiles = dt_quartiles.loc[mask_outlier_units, :]
            if dt_quartiles.shape[0] == 0:
                self.logger.info("No outliers detected")

        return dt_quartiles

    def _calculate_quartiles(
        self,
        dt: pd.DataFrame,
        strata_var: str,
        x_var: str | list[str],
        y_var: str | list[str],
        percentiles: tuple[float, float],
        pkl: float,
        pku: float,
        flag: str,
        two_ratios: bool,
    ) -> pd.DataFrame:
        """Internal function for calculating quartiles"""
        # Ratios
        if not two_ratios:
            dt["ratio"] = dt[x_var] / dt[y_var]
            dt["ratio2"] = np.nan
        else:

            dt["ratio"] = dt[x_var[0]] / dt[y_var[0]]
            dt["ratio2"] = dt[x_var[1]] / dt[y_var[1]]

        # Global quartiles
        q1, q2, q3 = dt["ratio"].quantile([percentiles[0], 0.50, percentiles[1]])

        ll = q1 - pkl * (q2 - q1)
        ul = q3 + pku * (q3 - q2)

        dt["lower_limit"] = ll
        dt["upper_limit"] = ul

        dt[flag] = ((dt["ratio"] < ll) | (dt["ratio"] > ul)).astype(int)

        # Ranking
        dt["ranking"] = dt["ratio"].rank(method="average")

        # Overall ratio
        if not two_ratios:
            dt["ratio_all"] = dt[x_var].sum() / dt[y_var].sum()
            dt["ratio_all_2"] = np.nan
        else:
            dt["ratio_all"] = dt[x_var[0]].sum() / dt[y_var[0]].sum()
            dt["ratio_all_2"] = dt[x_var[1]].sum() / dt[y_var[1]].sum()

        # Stratification
        if not strata_var:
            dt["strata_variable"] = 1
            dt["ratio_strata"] = np.nan
            dt["ratio_strata_2"] = np.nan
        else:
            g = dt.groupby(strata_var, dropna=False)

            if not two_ratios:
                dt["ratio_strata"] = g[x_var].transform("sum") / g[y_var].transform(
                    "sum"
                )
            else:
                dt["ratio_strata"] = g[x_var[0]].transform("sum") / g[
                    y_var[0]
                ].transform("sum")
                dt["ratio_strata_2"] = g[x_var[1]].transform("sum") / g[
                    y_var[1]
                ].transform("sum")
        return dt
