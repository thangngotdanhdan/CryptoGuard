import csv

import numpy as np
import pandas as pd
import pytest

from cryptoguard.data.elliptic_loader import load_elliptic, validate_time_steps


@pytest.mark.parametrize(
    "values",
    [
        [1, 29, 49],
        [1.0, 29.0, 49.0],
        np.array([1, 29, 49], dtype=np.float16),
        ["1", "29", "49"],
        pd.Series([1, 29, 49], dtype="Int64"),
    ],
    ids=["integers", "integral-floats", "float16-array", "numeric-strings", "nullable-integers"],
)
def test_validate_time_steps_keeps_inclusive_bounds_and_returns_int64(values):
    result = validate_time_steps(values)
    np.testing.assert_array_equal(result, [1, 29, 49])
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        pytest.param([1, 29.5], "fractional", id="fractional"),
        pytest.param([None], "missing", id="none"),
        pytest.param(pd.Series([1, pd.NA], dtype="Int64"), "missing", id="nullable-na"),
        pytest.param([np.nan], "missing|non-finite", id="nan"),
        pytest.param(["abc"], "non-numeric|unparseable", id="non-numeric"),
        pytest.param([np.inf], "non-finite", id="infinity"),
        pytest.param([-np.inf], "non-finite", id="negative-infinity"),
        pytest.param(np.array([True, False]), "bool", id="boolean-array"),
        pytest.param(pd.Series([np.bool_(True), 2], dtype=object), "bool", id="object-bool"),
        pytest.param([True, 2], "bool", id="mixed-python-bool-int"),
        pytest.param([0], "out-of-range", id="below-minimum"),
        pytest.param([50], "out-of-range", id="above-maximum"),
        pytest.param([[1], [2]], "1D", id="matrix"),
    ],
)
def test_validate_time_steps_rejects_invalid_values(values, reason):
    with pytest.raises(ValueError, match=reason):
        validate_time_steps(values)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1 + 2j], id="non-real-number"),
        pytest.param(np.array([1, 2], dtype="datetime64[ns]"), id="calendar-timestamps"),
        pytest.param(np.array([1, 2], dtype="timedelta64[ns]"), id="elapsed-durations"),
    ],
)
def test_time_steps_reject_values_coercible_to_unrelated_integers(values):
    with pytest.raises(ValueError):
        validate_time_steps(values)


def test_validate_time_steps_honors_custom_bounds():
    np.testing.assert_array_equal(validate_time_steps([3, 7], min_step=3, max_step=7), [3, 7])
    for outside in [2, 8]:
        with pytest.raises(ValueError, match="out-of-range"):
            validate_time_steps([outside], min_step=3, max_step=7)


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("29.5", id="fractional"),
        pytest.param("29.0000000000000001", id="fraction-rounded-by-float64"),
        pytest.param("", id="missing"),
        pytest.param("not-a-step", id="non-numeric"),
        pytest.param("-inf", id="non-finite"),
        pytest.param("True", id="boolean"),
        pytest.param("0", id="below-minimum"),
        pytest.param("50", id="above-maximum"),
    ],
)
def test_loader_rejects_invalid_time_step_tokens(tmp_path, token):
    with (tmp_path / "elliptic_txs_features.csv").open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([123, token, *([0] * 165)])
    (tmp_path / "elliptic_txs_classes.csv").write_text("txId,class\n123,2\n", encoding="utf-8")
    (tmp_path / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="time_step"):
        load_elliptic(tmp_path)
