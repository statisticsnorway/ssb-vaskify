# %%
import logging
import pytest

from vaskify.createdata import create_test_data
from vaskify.detect import Detect

# %%
def test_logger(detector_wide) -> None:
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
def test_accumulation_error(detector_long) -> None:
    dt_controlled = detector_long.accumulation_error(y_var="turnover", time_var="time_period")

    assert any(
        dt_controlled.columns.isin(["flag_accumulation"]),
    ), "Flag variable created"
    expected_value = 1
    assert (
        dt_controlled.flag_accumulation.sum() == expected_value
    ), "Potential errors flagged"