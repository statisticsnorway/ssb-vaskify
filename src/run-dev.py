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

# %%
# mypy: ignore-errors

# %% [markdown]
# # Test and dev file

# %% [markdown]
# This file is for testing the package locally. It is excluded from linting and type-checking.

# %%
import importlib
import vaskify
importlib.reload(vaskify)
from vaskify import Detect, create_test_data

import logging

# %%
dt = create_test_data(10, n_periods=2, freq="yearly", seed=4)
dt2 = create_test_data(10, n_periods=3, freq="yearly", seed=4, wide = True)

det = Detect(dt, id_nr="id_company", logger_level="debug")
det2 = Detect(dt2, id_nr="id_company", logger_level="debug")

det.change_logging_level("debug")

# %% [markdown]
# ### thousand error

# %%
det.thousand_error(y_var="turnover", time_var="time_period", impute = True).head()

# %%
det.thousand_error(y_var="turnover", time_var="time_period", output_format="outliers")

# %%
det2.thousand_error(y_var=["turnover_2020", "turnover_2021", "turnover_2022"], impute = True).head(20)

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

# %%
det2.hb(y_var=["turnover_2020","turnover_2021"], strata_var = "nace")

# %%
