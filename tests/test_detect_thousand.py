# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Unittests for Thousand error detection
from vaskify.detect import Detect

# %%
def test_thousand_error(detector_long: Detect) -> None:

    dt_controlled = detector_long.thousand_error(y_var="turnover", time_var="time_period")
    assert any(dt_controlled.columns.isin(["flag_thousand"])), "Flag variable created"

def test_thousand_error_outliers(detector_long: Detect) -> None:
    outliers = detector_long.thousand_error(
        y_var="turnover",
        time_var="time_period",
        output_format="long",
        output_scope = "outliers"
    )
    expected_shape = (0, 6)
    assert (
        outliers.shape == expected_shape
    ), "output_format 'outlier' returns only outliers"


# %%
def test_thousand_error_wide(detector_wide: Detect) -> None:
    dt_controlled = detector_wide.thousand_error(y_var=["turnover_2020", "turnover_2021"])

    expected_shape = (5, 7)
    assert dt_controlled.shape == expected_shape, "Wide format correct dimensions"
