# %%
from vaskify import Detect
from vaskify import create_test_data


# %%
def test_thousand_error() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    dt_controlled = detect.thousand_error(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_thousand"])), "Flag variable created"


# %%
def test_accumulation_error() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    dt_controlled = detect.accumulation_error(y_var="turnover", time_var="time_period")

    assert any(
        dt_controlled.columns.isin(["flag_accumulation"]),
    ), "Flag variable created"
    expected_value = 2
    assert (
        dt_controlled.flag_accumulation.sum() == expected_value
    ), "Potential errors flagged"


# %%
def test_hb() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed=42)
    detect = Detect(dt, id_nr="id_company")
    dt_controlled = detect.hb(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_hb"])), "Flag variable created"
    expected_value = 5
    assert dt_controlled.shape[0] == expected_value, "Wide format returned as default"


# %%

# %%
