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

import pandas as pd
import numpy as np

def create_test_data(num_industries:int=5, num_periods:int =5, freq: str = "monthly", seed:int=None)-> pd.DataFrame:
    """
    Generate test data with columns: NACE, number of employees, turnover, time period.
    
    Parameters:
        num_industries (int): Number of unique industry codes (NACE).
        num_periods (int): Number of time periods.
        seed (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Test data in long format.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate unique industry codes (NACE)
    industry_codes = ['B', 'C', 'F', 'G', 'H', 'J', 'M', 'N', 'S']
    industries = np.random.choice(industry_codes, size=num_industries)
    
    # Generate time periods
    if freq == "monthly":
        time_periods = pd.date_range(start="2020-01-01", periods=num_periods, freq="ME").strftime("%Y-M%#m")
    if freq == "quarterly":
        time_periods = pd.date_range(start="2020-01-01", periods=num_periods, freq="QE").strftime("%Y-Q%q")
    if freq == "yearly":
        time_periods = pd.date_range(start="2020-01-01", periods=num_periods, freq="YE").strftime("%Y")

    # Create Cartesian product of industries and periods
    data = pd.DataFrame(
        [(nace, period) for nace in industry_codes for period in time_periods],
        columns=["nace", "time_period"]
    )
    
    # Generate random number of employees and turnover
    data["employees"] = np.random.randint(10, 500, size=len(data))

    # Calculate turnover based on number of employees, with some random variation
    data["turnover"] = np.round(data["employees"] * np.random.uniform(20000, 5000), 2)

    return data



