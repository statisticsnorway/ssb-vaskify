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
# Test runs
from vaskify.create_data import create_test_data


# %%
def test_create_data() -> None:
    assert create_test_data(num_industries=5, num_periods=2, freq="monthly").shape == (18, 4)
    assert create_test_data(num_industries=5, num_periods=2, freq="yearly").shape == (18, 4)


# %%

# %%

# %%
create_test_data(num_industries=5, num_periods=2, freq="yearly")

# %%
