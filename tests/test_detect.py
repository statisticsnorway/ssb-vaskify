# %%
from typing import Any
import pandas as pd
import logging

from vaskify.createdata import create_test_data
from vaskify.detect import Detect
import vaskify

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


# %%
def test_logger(detector_wide: Detect) -> None:
    logger = logging.getLogger("detect")
    logger_level_observed = logger.getEffectiveLevel()
    logger_level_expected = 30  # "warning"
    assert logger_level_observed == logger_level_expected, "Logger level set correctly"

    detector_wide.change_logging_level("info")
    logger_level_observed = logger.getEffectiveLevel()
    logger_level_expected = 20  # "info"
    assert (
        logger_level_observed == logger_level_expected
    ), "Logger level changed correctly"


# ---------------------------------------------------------------------------
# check data
# ---------------------------------------------------------------------------


def make_checker() -> Any:
    instance = Detect.__new__(Detect)
    instance._is_valid_date_format = lambda x: x.startswith("2020")
    return instance


def make_base_df() -> Any:
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "period": ["2020-01", "2020-02", "2020-03"],
            "value": [100.0, 200.0, 300.0],
        }
    )


def test_passes_with_valid_data() -> None:
    make_checker()._check_data(
        make_base_df(),
        y_var="value",
        time_var="period",
        id_nr="id",
    )


def test_raises_on_missing_column() -> None:
    try:
        make_checker()._check_data(make_base_df(), y_var="nonexistent")
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "Missing column: nonexistent" in str(e)


def test_raises_if_id_not_string() -> None:
    df = make_base_df()
    df["id"] = [1, 2, 3]
    try:
        make_checker()._check_data(df, id_nr="id")
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "id should be a string" in str(e)


def test_raises_if_y_var_not_numeric() -> None:
    df = make_base_df()
    df["value"] = ["a", "b", "c"]
    try:
        make_checker()._check_data(df, y_var="value")
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "value should be numeric" in str(e)


def test_raises_if_time_var_invalid_format() -> None:
    df = make_base_df()
    df["period"] = ["Jan-2020", "Feb-2020", "Mar-2020"]
    try:
        make_checker()._check_data(df, time_var="period")
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "period should be in the format" in str(e)


def test_skips_checks_for_empty_args() -> None:
    make_checker()._check_data(make_base_df())


# ---------------------------------------------------------------------------
# Accumulation error
# ---------------------------------------------------------------------------


# %%
def test_no_impute(caplog) -> None:  # type: ignore[no-untyped-def]
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    detect.accumulation_error(
        y_var="turnover",
        time_var="time_period",
        impute=True,
    )

    # Check that the message was logged
    assert "Imputation not implemented for this method." in caplog.text


# %%
def test_accumulation_error(detector_long: Detect) -> None:
    dt_controlled = detector_long.accumulation_error(
        y_var="turnover",
        time_var="time_period",
    )

    assert any(
        dt_controlled.columns.isin(["flag_accumulation"]),
    ), "Flag variable created"
    expected_value = 1
    assert (
        dt_controlled.flag_accumulation.sum() == expected_value
    ), "Potential errors flagged"
