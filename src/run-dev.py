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

# %%
dt = create_test_data(10, n_periods=2, freq="yearly", seed=4)
dt.head()

# %%
det = Detect(dt, id_nr="id_company", logger_level="debug")

# %%
det.change_logging_level("debug")

# %%
det.thousand_error(y_var="turnover", time_var="time_period").head()

# %%
det.thousand_error(y_var="turnover", time_var="time_period", output_format="outliers")

# %%
det.accumulation_error(y_var="turnover", time_var="time_period").head()

# %%
det.thousand_error(
    y_var="turnover",
    time_var="time_period",
    impute=True,
)

# %%

# %%
det.hb(y_var="turnover", time_var="time_period")

# %%
det.hb(y_var="turnover", time_var="time_period", output_format="outliers")

# %%
logging.CRITICAL

# %%
logging_dict={"debug":10,"info":20,"warning":30,"error":40,"critical":50}
logging_dict['debug']

# %%
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set logger level to DEBUG
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# %%
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# %%
logger.debug('This is a debug message')

# %%

# %%
logger.info('This is an info message')

# %%
logger.warning('This is a warning message')

# %%
logger.error('This is an error message')

# %%
logger.critical('This is a critical message')

# %%

# %%
