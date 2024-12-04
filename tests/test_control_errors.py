# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from vaskify import create_test_data, Detect


# %%
def test_thousand_error() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed = 42)
    detect = Detect(dt, id_nr = "id_company")
    dt_controlled = detect.thousand_error(y_var="turnover", time_var="time_period")

    assert any(dt_cont.columns.isin(["flag_thousand"])), "Flag variable created"


# %%
def test_accumulation_error() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed = 42)
    detect = Detect(dt, id_nr = "id_company")
    dt_controlled = detect.accumulation_error(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_accumulation"])), "Flag variable created"
    assert dt_controlled.flag_accumulation.sum() == 2, "Potential errors flagged"


# %%
def test_hb() -> None:
    dt = create_test_data(n=5, n_periods=2, freq="monthly", seed = 42)
    detect = Detect(dt, id_nr = "id_company")
    dt_controlled = detect.hb(y_var="turnover", time_var="time_period")

    assert any(dt_controlled.columns.isin(["flag_hb"])), "Flag variable created"
    assert dt_controlled.shape[0] == 5, "Wide format returned as default"
    

# %%

# %%
