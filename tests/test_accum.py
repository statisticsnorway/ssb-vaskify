# ---------------------------------------------------------------------------
# Accumulation error
# ---------------------------------------------------------------------------
from vaskify.createdata import create_test_data
from vaskify.detect import Detect
import pandas as pd
import logging


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


def test_accumulation_error_flag_column_created(detector_long):
    result = detector_long.accumulation_error(y_var="turnover", time_var="time_period")
    assert "flag_accumulation" in result.columns


def test_accumulation_error_first_period_is_nan(detector_long):
    result = detector_long.accumulation_error(y_var="turnover", time_var="time_period")
    first_period = result["time_period"].min()
    first_rows = result[result["time_period"] == first_period]
    assert first_rows["flag_accumulation"].isna().all()


def test_accumulation_error_returns_dataframe(detector_long):
    result = detector_long.accumulation_error(y_var="turnover", time_var="time_period")
    assert isinstance(result, pd.DataFrame)


def test_accumulation_error_outliers_scope(detector_long):
    result = detector_long.accumulation_error(
        y_var="turnover",
        time_var="time_period",
        output_format="outliers",
    )
    assert isinstance(result, pd.DataFrame)
    assert "flag_accumulation" in result.columns


def test_accumulation_error_custom_flag_name(detector_long):
    result = detector_long.accumulation_error(
        y_var="turnover",
        time_var="time_period",
        flag="my_flag",
    )
    assert "my_flag" in result.columns


def test_accumulation_error_impute_logs_error(detector_long, caplog):
    with caplog.at_level(logging.ERROR):
        detector_long.accumulation_error(
            y_var="turnover",
            time_var="time_period",
            impute=True,
            impute_var="turnover_imputed",
        )
    assert "Imputation not implemented" in caplog.text
