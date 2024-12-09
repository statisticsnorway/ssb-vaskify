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
# mypy: ignore-errors

# %% [markdown]
# # Test and dev file

# %% [markdown]
# This file is for testing the package locally. It is excluded from linting and type-checking.

# %%
from vaskify import Detect
from vaskify import create_test_data
import logging

# %%
dt = create_test_data(10, n_periods=2, freq="yearly", seed=4)
dt.head()

# %%
det = Detect(dt, id_nr="id_company", logger_level="debug")

# %%
det.change_logging_level("debug")

# %% [markdown]
# ### thousand error

# %%
det.thousand_error(y_var="turnover", time_var="time_period").head()

# %%
det.thousand_error(y_var="turnover", time_var="time_period", output_format="outliers")

# %% [markdown]
# ### Accumulation error

# %%
det.accumulation_error(y_var="turnover", time_var="time_period").head()

# %%
det.thousand_error(
    y_var="turnover",
    time_var="time_period",
    impute=True,
)

# %% [markdown]
# ### HB

# %%
det.hb(y_var="turnover", time_var="time_period")

# %%
det.hb(y_var="turnover", time_var="time_period", output_format="outliers")
