# %%
import pandas as pd
import pytest

from vaskify.createdata import create_test_data

EXPECTED_LONG_COLUMNS = {"id_company", "time_period", "nace", "employees", "turnover"}


# ---------------------------------------------------------------------------
# Generelt
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Schema / structure
# ---------------------------------------------------------------------------


class TestSchema:
    def test_returns_dataframe(self) -> None:
        result = create_test_data()
        assert isinstance(result, pd.DataFrame)

    def test_long_format_columns(self) -> None:
        result = create_test_data()
        assert set(result.columns) == EXPECTED_LONG_COLUMNS

    def test_id_company_is_string(self) -> None:
        result = create_test_data(n=3)
        assert result["id_company"].dtype == object

    def test_employees_are_integers(self) -> None:
        result = create_test_data(n=3)
        assert pd.api.types.is_integer_dtype(result["employees"])

    def test_turnover_is_numeric(self) -> None:
        result = create_test_data(n=3)
        assert pd.api.types.is_float_dtype(result["turnover"])

    def test_nace_within_valid_codes(self) -> None:
        valid_codes = {"B", "C", "F", "G", "H", "J", "M", "N", "S"}
        result = create_test_data(n=20, seed=42)
        assert set(result["nace"].unique()).issubset(valid_codes)


# ---------------------------------------------------------------------------
# Row / shape
# ---------------------------------------------------------------------------


def test_row_count_1_company_1_period():
    assert len(create_test_data(n=1, n_periods=1)) == 1


def test_row_count_5_companies_5_periods():
    assert len(create_test_data(n=5, n_periods=5)) == 25


def test_row_count_10_companies_3_periods():
    assert len(create_test_data(n=10, n_periods=3)) == 30


def test_row_count_3_companies_12_periods():
    assert len(create_test_data(n=3, n_periods=12)) == 36


def test_unique_companies_match_n():
    assert create_test_data(n=7, seed=1)["id_company"].nunique() == 7


def test_unique_periods_match_n_periods():
    assert create_test_data(n_periods=4)["time_period"].nunique() == 4


def test_each_company_appears_in_all_periods():
    result = create_test_data(n=4, n_periods=6, seed=0)
    counts = result.groupby("id_company")["time_period"].nunique()
    assert (counts == 6).all()


# ---------------------------------------------------------------------------
# Time period formats
# ---------------------------------------------------------------------------


class TestTimePeriodFormats:
    def test_monthly_format(self) -> None:
        result = create_test_data(n_periods=3, freq="monthly")
        assert result["time_period"].str.match(r"^\d{4}-\d{2}$").all()

    def test_quarterly_format(self) -> None:
        result = create_test_data(n_periods=4, freq="quarterly")
        assert result["time_period"].str.match(r"^\d{4}-Q[1-4]$").all()

    def test_yearly_format(self) -> None:
        result = create_test_data(n_periods=3, freq="yearly")
        assert result["time_period"].str.match(r"^\d{4}$").all()

    def test_monthly_period_count(self) -> None:
        result = create_test_data(n=2, n_periods=6, freq="monthly")
        assert result["time_period"].nunique() == 6

    def test_quarterly_period_count(self) -> None:
        result = create_test_data(n=2, n_periods=4, freq="quarterly")
        assert result["time_period"].nunique() == 4

    def test_yearly_period_count(self) -> None:
        result = create_test_data(n=2, n_periods=5, freq="yearly")
        assert result["time_period"].nunique() == 5

    def test_invalid_freq_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="freq must be one of"):
            create_test_data(freq="weekly")


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------


class TestValueRanges:
    def test_employees_within_expected_range(self) -> None:
        result = create_test_data(n=20, n_periods=5, seed=0)
        assert result["employees"].between(10, 500).all()

    def test_turnover_is_positive(self) -> None:
        result = create_test_data(n=10, seed=0)
        assert (result["turnover"] > 0).all()

    def test_turnover_rounded_to_2_decimal_places(self) -> None:
        result = create_test_data(n=10, seed=42)
        rounded = result["turnover"].round(2)
        pd.testing.assert_series_equal(result["turnover"], rounded)


# ---------------------------------------------------------------------------
# Reproducibility / seed
# ---------------------------------------------------------------------------


class TestSeed:
    def test_same_seed_produces_identical_output(self) -> None:
        df1 = create_test_data(n=5, n_periods=3, seed=99)
        df2 = create_test_data(n=5, n_periods=3, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_output(self) -> None:
        df1 = create_test_data(n=5, n_periods=3, seed=1)
        df2 = create_test_data(n=5, n_periods=3, seed=2)
        assert not df1["employees"].equals(df2["employees"])

    def test_no_seed_runs_without_error(self) -> None:
        result = create_test_data(n=3, seed=None)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------


class TestWideFormat:
    def test_wide_returns_dataframe(self) -> None:
        result = create_test_data(n=3, n_periods=3, wide=True)
        assert isinstance(result, pd.DataFrame)

    def test_wide_has_no_time_period_column(self) -> None:
        result = create_test_data(n=3, n_periods=3, wide=True)
        assert "time_period" not in result.columns

    def test_wide_has_more_columns_than_long(self) -> None:
        long = create_test_data(n=3, n_periods=4, wide=False)
        wide = create_test_data(n=3, n_periods=4, wide=True)
        assert wide.shape[1] > long.shape[1]

    def test_wide_has_fewer_rows_than_long(self) -> None:
        long = create_test_data(n=5, n_periods=4, wide=False)
        wide = create_test_data(n=5, n_periods=4, wide=True)
        assert wide.shape[0] < long.shape[0]
