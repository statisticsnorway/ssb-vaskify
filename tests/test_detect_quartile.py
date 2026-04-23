# %% [markdown]
# ## Unittests for quartile method

# %%
import pandas as pd

from vaskify.detect import Detect

# Do not import fixtures!


# %%
def test_quartile_error_basic_ratio(detector_wide: Detect) -> None:
    res = detector_wide.quartile_error(
        x_var="employees_2020",
        y_var="employees_2021",
    )

    assert "ratio" in res.columns
    assert res["ratio"].dtype.kind == "f"

    # Ratio should not be constant
    assert res["ratio"].nunique() > 1  # noqa: PD101

    # Quartile bounds should exist
    assert "lower_limit" in res.columns
    assert "upper_limit" in res.columns


# %%
def test_quartile_error_y_var_none(detector_wide: Detect) -> None:
    result = detector_wide.quartile_error(
        x_var="employees_2020",
        y_var=None,
    )

    assert "flag_quartile" in result.columns
    assert "y_var_temp" not in result.columns
    assert "y_var_temp2" not in result.columns


def test_quartile_error_multiple_ratios(detector_wide: Detect) -> None:
    result = detector_wide.quartile_error(
        x_var=["employees_2020", "employees_2021"],
        y_var=["turnover_2020", "turnover_2021"],
    )

    assert isinstance(result, pd.DataFrame)
    assert "flag_quartile" in result.columns


def test_quartile_error_filters_invalid_rows(detector_wide: Detect) -> None:
    detector_wide.data.loc[
        detector_wide.data.index[0],
        "employees_2020",
    ] = -1

    result = detector_wide.quartile_error(
        x_var="employees_2020",
        y_var="employees_2021",
    )

    assert (result["employees_2020"] > 0).all()
    assert result["employees_2021"].notna().all()
