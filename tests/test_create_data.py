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
from vaskify import create_test_data


# %%
def test_create_data() -> None:
    assert create_test_data(n=5, n_periods=2, freq="monthly").shape == (
        18,
        4,
    )
    assert create_test_data(n=5, n_periods=2, freq="yearly").shape == (
        18,
        4,
    )


# %%

# %%
