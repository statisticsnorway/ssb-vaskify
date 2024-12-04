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
# Functions to create data

import numpy as np
import pandas as pd


def create_test_data(
    n: int = 5,
    n_periods: int = 5,
    freq: str = "monthly",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate test data with columns: NACE, number of employees, turnover, time period.

    Parameters:
        n (int): Number of unique companies to create.
        num_periods (int): Number of time periods.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Test data in long format.
    """
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(12345) #### work on this

    company_ids = np.array(range(n))

    # Generate unique industry codes (NACE)
    industry_codes = ["B", "C", "F", "G", "H", "J", "M", "N", "S"]
    industries = np.random.choice(industry_codes, size=n, replace=True)

    # Generate time periods
    if freq == "monthly":
        time_periods = pd.date_range(
            start="2020-01-01",
            periods=n_periods,
            freq="ME",
        ).strftime("%Y-M%#m")
    if freq == "quarterly":
        time_periods = pd.date_range(
            start="2020-01-01",
            periods=n_periods,
            freq="QE",
        ).strftime("%Y-Q%q")
    if freq == "yearly":
        time_periods = pd.date_range(
            start="2020-01-01",
            periods=n_periods,
            freq="YE",
        ).strftime("%Y")

    # Create Cartesian product of industries and periods
    data = pd.DataFrame(
        [(id_company, period) for id_company in company_ids for period in time_periods],
        columns=["id_company", "time_period"],
    )

    # Map each company to its NACE code
    nace_mapping = dict(zip(company_ids, industries, strict=False))
    data["nace"] = data["id_company"].map(nace_mapping)

    # Generate random number of employees and turnover
    data["employees"] = np.random.randint(10, 500, size=len(data))

    # Calculate turnover based on number of employees, with some random variation
    data["turnover"] = np.round(data["employees"] * np.random.uniform(20000, 5000), 2)

    return data
