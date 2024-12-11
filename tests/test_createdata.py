# %%
from vaskify.createdata import create_test_data


# %%
def test_create_data() -> None:
    assert create_test_data(n=5, n_periods=2, freq="monthly").shape == (
        10,
        5,
    )
    assert create_test_data(n=5, n_periods=2, freq="yearly").shape == (
        10,
        5,
    )
