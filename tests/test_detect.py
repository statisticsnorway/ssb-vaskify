# %%
import logging

from vaskify import Detect
from vaskify import create_test_data


# %%
def test_thousand_error() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detection = Detect(dt, id_nr="id_company")
    dt_controlled = detection.thousand_error(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_thousand"])), "Flag variable created"

    outliers = detection.thousand_error(
        y_var="turnover",
        time_var="time_period",
        output_format="outliers",
    )
    expected_shape = (0, 6)
    assert (
        outliers.shape == expected_shape
    ), "output_format 'outlier' returns only outliers"


# %%
def test_accumulation_error() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    dt_controlled = detect.accumulation_error(y_var="turnover", time_var="time_period")

    assert any(
        dt_controlled.columns.isin(["flag_accumulation"]),
    ), "Flag variable created"
    expected_value = 1
    assert (
        dt_controlled.flag_accumulation.sum() == expected_value
    ), "Potential errors flagged"


# %%
def test_hb() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    dt_controlled = detect.hb(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_hb"])), "Flag variable created"
    expected_shape = 5
    assert dt_controlled.shape[0] == expected_shape, "Wide format returned as default"

    detect.change_logging_level("error")
    dt_controlled = detect.hb(
        y_var="turnover",
        time_var="time_period",
        output_format="outliers",
    )
    expected_shape = 0
    assert dt_controlled.shape[0] == expected_shape, "Oulier format returned"

    dt_controlled = detect.hb(
        y_var="turnover",
        time_var="time_period",
        output_format="long",
    )
    expected_shape = 10
    assert dt_controlled.shape[0] == expected_shape, "Long format returned"


# %%
def test_logger() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    logger = logging.getLogger("detect")
    logger_level_observed = logger.getEffectiveLevel()
    logger_level_expected = 30  # "warning"
    assert logger_level_observed == logger_level_expected, "Logger level set correctly"

    detect.change_logging_level("info")
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
