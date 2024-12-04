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

# %%
import pandas as pd
import numpy as np
from vaskify import create_test_data, Detect

# %%
dt = create_test_data(10, n_periods=2, freq="yearly", seed = 4)
dt.head()

# %%
det = Detect(dt, id_nr="id_company")

# %%
det.thousand_error(y_var="turnover", time_var="time_period").head()

# %%
det.accumulation_error(y_var="turnover", time_var="time_period").head()

# %%
det.accumulation_error(y_var="turnover", time_var="time_period", output_format = "outliers")

# %%
det.hb(y_var="turnover", time_var="time_period")

# %%
det.hb(y_var="turnover", time_var="time_period", output_format = "outliers")

# %%
