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
        n_periods (int): Number of time periods to create.
        freq (str): Frequency of the time periods: 'monthly', 'quarterly' or 'yearly'.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Test data in long format.
    """
    rng = np.random.default_rng(seed) if seed else np.random.default_rng()

    company_ids = np.array(range(n))

    # Generate unique industry codes (NACE)
    industry_codes = ["B", "C", "F", "G", "H", "J", "M", "N", "S"]
    industries = rng.choice(industry_codes, size=n, replace=True)

    # Generate time periods
    if freq == "monthly":
        time_periods = (
            pd.date_range(
                start="2020-01-01",
                periods=n_periods,
                freq="ME",
            )
            .to_period("M")
            .astype(str)
        )
    if freq == "quarterly":
        time_periods = (
            pd.date_range(
                start="2020-01-01",
                periods=n_periods,
                freq="QE",
            )
            .to_period("Q")
            .astype(str)
        )
    if freq == "yearly":
        time_periods = (
            pd.date_range(
                start="2020-01-01",
                periods=n_periods,
                freq="YE",
            )
            .to_period("Y")
            .astype(str)
        )

    # Create Cartesian product of industries and periods
    data = pd.DataFrame(
        [(id_company, period) for id_company in company_ids for period in time_periods],
        columns=["id_company", "time_period"],
    )

    # Map each company to its NACE code
    nace_mapping = dict(zip(company_ids, industries, strict=False))
    data["nace"] = data["id_company"].map(nace_mapping)

    # Generate random number of employees and turnover
    data["employees"] = rng.integers(10, 500, size=len(data))

    # Calculate turnover based on number of employees, with some random variation
    data["turnover"] = np.round(
        data["employees"] * rng.uniform(5000, 20000),
        2,
    )  # check if all get same random or not...

    data["id_company"] = data["id_company"].astype(str)

    return data
