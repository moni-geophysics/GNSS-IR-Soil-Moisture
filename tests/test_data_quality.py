import pandas as pd

from data_quality import (
    check_time_series_quality,
    check_time_continuity,
    check_datetime_validity,
    check_duplicate_timestamps,
    check_soil_moisture_values,
    count_quality_flags,
    count_individual_quality_flags)


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

def test_check_time_continuity():
    data = pd.DataFrame({
        "datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 01:00:00",
            "2016-01-01 03:00:00",
            "2016-01-01 04:00:00",
        ],
        "value": [0.218, 0.219, 0.220, 0.221],
    })

    report = check_time_continuity(data)

    assert report["number_of_gaps"] == 1
    assert report["gap_locations"][0] == pd.Timestamp("2016-01-01 03:00:00")

def test_check_datetime_validity():
    data = pd.DataFrame({
        "datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 01:00:00",
            "not-a-date",
            "2016-01-01 03:00:00",
        ],
        "value": [0.218, 0.219, 0.220, 0.221],
    })

    report = check_datetime_validity(data)

    assert report["number_of_invalid_datetimes"] == 1
    assert report["invalid_rows"] == [2]

def test_check_duplicate_timestamps():
    data = pd.DataFrame({
        "datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 01:00:00",
            "2016-01-01 01:00:00",
            "2016-01-01 02:00:00",
        ],
        "value": [0.218, 0.219, 0.220, 0.221],
    })

    report = check_duplicate_timestamps(data)

    assert report["number_of_duplicate_timestamps"] == 1
    assert report["duplicate_rows"] == [1, 2]

def test_check_soil_moisture_values():
    data = pd.DataFrame({
        "datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 01:00:00",
            "2016-01-01 02:00:00",
            "2016-01-01 03:00:00",
        ],
        "value": [0.218, -0.010, None, 0.221],
    })

    report = check_soil_moisture_values(data)

    assert report["number_of_invalid_values"] == 2
    assert report["invalid_rows"] == [1, 2]

def test_count_quality_flags():
        data = pd.DataFrame({
            "datetime": [
                "2016-01-01 00:00:00",
                "2016-01-01 01:00:00",
                "2016-01-01 02:00:00",
                "2016-01-01 03:00:00",
                "2016-01-01 04:00:00",
            ],
            "ismn_flag": ["G", "G", "D03", "M", "D03"],
        })

        report = count_quality_flags(data)

        assert report["G"] == 2
        assert report["D03"] == 2
        assert report["M"] == 1

def test_count_individual_quality_flags():
        data = pd.DataFrame({
            "ismn_flag": [
                "G",
                "D03",
                "D05,D03",
                "C03,D03",
                "D05,C03",
            ]
        })

        report = count_individual_quality_flags(data)

        assert report["G"] == 1
        assert report["D03"] == 3
        assert report["D05"] == 2
        assert report["C03"] == 2