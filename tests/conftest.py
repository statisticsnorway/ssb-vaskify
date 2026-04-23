# Fixtures
import pandas as pd
import pytest

from vaskify.createdata import create_test_data
from vaskify.detect import Detect


@pytest.fixture
def wide_data() -> pd.DataFrame:
    return create_test_data(
        n=5,
        n_periods=2,
        freq="yearly",
        seed=42,
        wide=True,
    )


@pytest.fixture
def long_data() -> pd.DataFrame:
    return create_test_data(
        n=5,
        n_periods=2,
        freq="monthly",
        seed=42,
    )


@pytest.fixture
def detector_wide(wide_data: pd.DataFrame) -> Detect:
    return Detect(wide_data, id_nr="id_company")


@pytest.fixture
def detector_long(long_data: pd.DataFrame) -> Detect:
    return Detect(long_data, id_nr="id_company")
