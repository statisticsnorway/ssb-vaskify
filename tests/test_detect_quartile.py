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
# ## Unittests for quartile method

# %%
import logging
import pytest

from vaskify.createdata import create_test_data
from vaskify.detect import Detect

# Do not import fixtures!

# %%
def test_quartile_error_basic_ratio(detector_wide):
    res = detector_wide.quartile_error(
        x_var="employees_2020",
        y_var="employees_2021",
    )

    assert "ratio" in res.columns
    assert res["ratio"].dtype.kind == "f"

    # Ratio should not be constant
    assert res["ratio"].nunique() > 1

    # Quartile bounds should exist
    assert "lower_limit" in res.columns
    assert "upper_limit" in res.columns

# %%
