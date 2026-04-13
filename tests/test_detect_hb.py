# %% [markdown]
# ## Unittests for HB method

# %%

from vaskify.createdata import create_test_data
from vaskify.detect import Detect


# %%
def test_hb(detector_long: Detect) -> None:
    dt_controlled = detector_long.hb(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_hb"])), "Flag variable created"
    expected_shape = 5
    assert dt_controlled.shape[0] == expected_shape, "Wide format returned as default"

    detector_long.change_logging_level("error")


def test_hb_outliers(detector_long: Detect) -> None:
    dt_controlled = detector_long.hb(
        y_var="turnover",
        time_var="time_period",
        output_scope="outliers",
    )
    expected_shape = 0
    assert dt_controlled.shape[0] == expected_shape, "Oulier format returned"


def test_hb_output(detector_long: Detect) -> None:
    dt_controlled = detector_long.hb(
        y_var="turnover",
        time_var="time_period",
        output_format="long",
    )
    expected_shape = 10
    assert dt_controlled.shape[0] == expected_shape, "Long format returned"


def test_hb_long_input_wide_output(detector_long: Detect) -> None:
    dt_controlled = detector_long.hb(
        y_var="turnover",
        time_var="time_period",
        output_format="wide",
    )

    expected_shape = 5, 11
    assert dt_controlled.shape == expected_shape, "Wide format returned as default"


def test_hb_wide_output(detector_wide: Detect) -> None:
    dt_controlled = detector_wide.hb(
        y_var=["turnover_2020", "turnover_2021"],
        output_format="wide",
    )

    expected_shape = 5, 11
    assert dt_controlled.shape == expected_shape, "Wide format returned as default"


def test_hb_flag(detector_long: Detect) -> None:
    dt_controlled = detector_long.hb(
        y_var="turnover",
        time_var="time_period",
        flag="outlier_indicator",
    )
    assert any(
        dt_controlled.columns.isin(["outlier_indicator"]),
    ), "Flag variable created"


def test_hb_strata_wide() -> None:
    dt = create_test_data(n=50, seed=10)
    dt2 = dt.loc[dt.time_period.isin(["2020-04", "2020-05"]), :]

    detect = Detect(dt2, id_nr="id_company")
    dt_controlled = detect.hb(
        y_var="turnover",
        time_var="time_period",
        strata_var="nace",
    )

    assert any(dt_controlled.columns.isin(["flag_hb"])), "Flag variable created"
    expected_shape = 50
    assert dt_controlled.shape[0] == expected_shape, "Wide format returned as default"


def test_hb_strata_outliers() -> None:
    dt = create_test_data(n=50, seed=10)
    dt2 = dt.loc[dt.time_period.isin(["2020-04", "2020-05"]), :]
    detect = Detect(dt2, id_nr="id_company")

    dt_controlled = detect.hb(
        y_var="turnover",
        strata_var="nace",
        time_var="time_period",
        output_scope="outliers",
    )
    expected_shape = 2
    assert dt_controlled.shape[0] == expected_shape, "Oulier format returned"
