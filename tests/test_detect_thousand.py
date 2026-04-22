# %% [markdown]
# ## Unittests for Thousand error detection
from tests.conftest import wide_data
from vaskify.detect import Detect
from vaskify.createdata import create_test_data
import pandas as pd
import logging
import pytest


def test_thousand_error(detector_long: Detect) -> None:

    dt_controlled = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
    )
    assert any(
        dt_controlled.columns.str.startswith("flag_thousand")
    ), "Flag variable created"


def test_thousand_error_outliers(detector_long: Detect) -> None:
    outliers = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
        output_format="long",
        output_scope="outliers",
    )
    expected_shape = (0, 6)
    assert (
        outliers.shape == expected_shape
    ), "output_format 'outlier' returns only outliers"


def test_thousand_error_wide(detector_wide: Detect) -> None:
    dt_controlled = detector_wide.thousand_error(
        y_var=["turnover_2020", "turnover_2021"],
    )

    expected_shape = (5, 7)
    assert dt_controlled.shape == expected_shape, "Wide format correct dimensions"


def test_thousand_error_flag_name_custom(detector_long: Detect) -> None:
    result = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
        flag="my_flag",
    )
    assert any(result.columns.str.startswith("my_flag")), "Flag variable created"


def test_thousand_error_invalid_output_format_falls_back_to_wide(
    detector_long: Detect,
) -> None:
    result = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
        output_format="invalid",
    )
    assert "time_period" not in result.columns  # wide format has no time_period column


def test_thousand_error_infer_returns_long_for_long_input(
    detector_long: Detect,
) -> None:
    result = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
        output_format="infer",
    )
    assert "time_period" in result.columns


def test_thousand_error_wide_logs_warning_for_long_format(detector_wide, caplog):

    with caplog.at_level(logging.WARNING):
        result = detector_wide._thousand_error_wide(
            y_vars=["turnover_2020", "turnover_2021"],
            flag_names=["flag_thousand"],
            impute_vars=["turnover_imputed"],
            lower_bound=-2.5,
            upper_bound=2.5,
            impute=False,
            output_format="long",
        )

    assert "long format not implemented" in caplog.text
    assert isinstance(result, pd.DataFrame)


def test_thousand_error_infer_returns_wide_for_wide_input(
    detector_wide: Detect,
) -> None:
    result = detector_wide.thousand_error(
        y_var=["turnover_2020", "turnover_2021"],
        output_format="infer",
    )
    assert "time_period" not in result.columns


def test_thousand_error_returns_dataframe(detector_long: Detect) -> None:
    result = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
    )
    assert isinstance(result, pd.DataFrame)


def test_thousand_error_wide_impute(detector_wide: Detect) -> None:
    dt = create_test_data(
        n=5,
        n_periods=2,
        freq="yearly",
        seed=42,
        wide=True,
    )
    dt.loc[0, "turnover_2021"] = (
        dt.loc[0, "turnover_2021"] * 1000
    )  # use column name, not index
    obs = dt.loc[0, "turnover_2021"].copy()
    detector = Detect(dt, id_nr="id_company")
    result = detector._thousand_error_wide(
        y_vars=["turnover_2020", "turnover_2021"],
        flag_names=["flag_thousand"],
        impute_vars=["turnover_imputed"],
        lower_bound=-2.5,
        upper_bound=2,
        impute=True,
        output_format="wide",
    )

    assert "turnover_imputed_2021" in result.columns
    assert (
        result.loc[1, "turnover_imputed_2021"] == 747544.08
    )  # non-flagged row is unchanged
    assert result.loc[0, "turnover_imputed_2021"] == pytest.approx(
        obs / 1000
    )  # flagged row is divided by 1000


def test_thousand_error_long_impute(detector_wide: Detect) -> None:
    dt = create_test_data(
        n=5,
        n_periods=2,
        freq="yearly",
        seed=42,
        wide=False,
    )
    dt.iloc[7, 4] = dt.iloc[7, 4] * 1000
    detector = Detect(dt, id_nr="id_company")
    result = detector._thousand_error_long(
        y_vars=["turnover"],
        time_var="time_period",
        flag_names=["flag_thousand"],
        impute_vars=["turnover_imputed"],
        lower_bound=-2.5,
        upper_bound=2.5,
        impute=True,
        output_format="long",
    )
    assert "turnover_imputed" in result.columns
    assert (
        result.loc[6, "turnover_imputed"] == 3377791.79
    )  # non-flagged row is unchanged
    assert (
        result.loc[7, "turnover_imputed"] == 2561030.66
    )  # flagged row is divided by 1000
