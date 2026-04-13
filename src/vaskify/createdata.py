# %%
# Functions to create data


import numpy as np
import pandas as pd


def create_test_data(
    n: int = 5,
    n_periods: int = 5,
    freq: str = "monthly",
    seed: int | None = None,
    wide: bool = False,
) -> pd.DataFrame:
    """Generate test data with columns: NACE, number of employees, turnover, time period.

    Args:
        n: Number of unique companies to create.
        n_periods: Number of time periods to create.
        freq: Frequency of the time periods: 'monthly', 'quarterly' or 'yearly'.
        seed: Random seed for reproducibility.
        wide: If True, return data in wide format with time periods as columns.

    Returns:
        pd.DataFrame: Test data in long format.

    Raises:
        ValueError: If `freq` is not one of "monthly", "quarterly", or "yearly".

    """
    rng = np.random.default_rng(seed) if seed else np.random.default_rng()

    company_ids = np.array(range(n))

    # Generate unique industry codes (NACE)
    industry_codes = ["B", "C", "F", "G", "H", "J", "M", "N", "S"]
    industries = rng.choice(industry_codes, size=n, replace=True)

    # Generate time periods as List[str]
    if freq == "monthly":
        periods = pd.period_range(start="2020-01-01", periods=n_periods, freq="M")
        time_periods: list[str] = [f"{p.year}-{p.month:02d}" for p in periods]
    elif freq == "quarterly":
        periods = pd.period_range(start="2020-01-01", periods=n_periods, freq="Q-DEC")
        time_periods = [f"{p.year}-Q{p.quarter}" for p in periods]
    elif freq == "yearly":
        periods = pd.period_range(start="2020-01-01", periods=n_periods, freq="Y")
        time_periods = [f"{p.year}" for p in periods]
    else:
        mes = "freq must be one of: 'monthly', 'quarterly', 'yearly'"
        raise ValueError(mes)

    # Create product of industries and periods
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

    return _to_wide(data) if wide else data


def _to_wide(data: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format test data to wide format.

    Args:
        data: Long-format DataFrame with columns: id_company, time_period, nace,
            employees, turnover.

    Returns:
        pd.DataFrame: Wide-format DataFrame with one row per company and
            separate columns for each metric/period combination,
            prefixed with 'employees_' and 'turnover_' respectively.
    """
    employees_wide = data.pivot(
        index="id_company", columns="time_period", values="employees"
    )
    employees_wide.columns = [f"employees_{col}" for col in employees_wide.columns]

    turnover_wide = data.pivot(
        index="id_company", columns="time_period", values="turnover"
    )
    turnover_wide.columns = [f"turnover_{col}" for col in turnover_wide.columns]

    static = data[["id_company", "nace"]].drop_duplicates().set_index("id_company")

    wide_data = pd.concat([static, employees_wide, turnover_wide], axis=1).reset_index()
    wide_data.columns.name = None

    return wide_data
