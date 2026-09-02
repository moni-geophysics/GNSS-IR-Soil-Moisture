import pandas as pd

from src.data_quality import check_time_series_quality


def test_check_time_series_quality():
    data = pd.DataFrame({
        "datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 01:00:00",
            "2016-01-01 02:00:00",
            "2016-01-01 02:00:00",
        ],
        "value": [0.218, 0.219, None, 0.221],
    })

    report = check_time_series_quality(data)

    assert report["number_of_rows"] == 4
    assert report["number_of_columns"] == 2
    assert report["missing_values"]["value"] == 1
    assert report["duplicate_rows"] == 1