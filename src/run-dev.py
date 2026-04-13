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
dt = create_test_data(5, n_periods=2, freq="yearly", seed=42)
dt2 = create_test_data(5, n_periods=2, freq="yearly", seed=42, wide=True)

det = Detect(dt, id_nr="id_company", logger_level="debug")
det2 = Detect(dt2, id_nr="id_company", logger_level="debug")

det.change_logging_level("debug")

# %% [markdown]
# ### thousand error

# %%
df_long = det.thousand_error(y_var="turnover", time_var="time_period")
df_long.head()

# %%
det.thousand_error(y_var="turnover", time_var="time_period", impute=True)

# %%
det2.thousand_error(y_var=["turnover_2020", "turnover_2021"], impute=True)

# %%
dt = create_test_data(n=5, n_periods=3, freq="yearly", seed=42, wide=True)
detection = Detect(dt, id_nr="id_company")
dt_controlled = detection.thousand_error(
    y_var=["turnover_2020", "turnover_2021", "turnover_2022"]
)

# %%
dt = create_test_data(n=5, n_periods=3, freq="monthly", seed=42)
detection = Detect(dt, id_nr="id_company")
dt_controlled = detection.thousand_error(y_var="turnover", time_var="time_period")
dt_controlled


# %%

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
from vaskify import Detect, create_test_data
import logging

dt = create_test_data(5, n_periods=2, freq="yearly", seed=4)
det = Detect(dt, id_nr="id_company", logger_level="debug")

# %%
det.hb(y_var="employees", time_var="time_period", output_format="long")

# %%
det.hb(y_var="turnover", time_var="time_period", output_format="wide")

# %%
det.hb(y_var="turnover", time_var="time_period")

# %%
dt = create_test_data(n=50, seed=10)
dt2 = dt.loc[dt.time_period.isin(["2020-04", "2020-05"]), :]
detect = Detect(dt2, id_nr="id_company")
dt_controlled = detect.hb(
    y_var="turnover",
    strata_var="nace",
    time_var="time_period",
    output_scope="outliers",
)
dt_controlled

# %% [markdown]
# ### Quartile error

# %%
from vaskify import Detect, create_test_data
import logging

dt2 = create_test_data(10, n_periods=2, freq="yearly", seed=4, wide=True)
det = Detect(dt2, id_nr="id_company", logger_level="debug")

# %%
test = det.quartile_error(
    x_var=["employees_2020", "employees_2021"],
    y_var=["turnover_2020", "turnover_2021"],
    strata_var="nace",
)
test

# %%

# %%
