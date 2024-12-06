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
