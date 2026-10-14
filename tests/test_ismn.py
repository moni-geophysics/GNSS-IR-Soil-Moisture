from pathlib import Path

from gnss_ir.ismn import read_ismn_stm


# Path to the ISMN .stm file
ISM_FILE = Path(
    "data/raw/COSMOS_COSMOS_MarshallColorado_sm_0.000000_0.250000_Hydroinnova-CRS-1000B_1_1_20160101_20160801.stm"
)



def test_read_ismn_file():
    header, data = read_ismn_stm(ISM_FILE)

    print("\nHEADER:")
    print(header)

    print("\nFIRST 5 OBSERVATIONS:")
    print(data.head())

    print("\nNUMBER OF OBSERVATIONS:")
    print(len(data))


def test_read_ismn_from_zip_stream():
    """Verify that read_ismn_stm reads directly from a zip file-like stream."""
    import zipfile
    zip_path = Path("E:/MTP Data/P041.zip")
    if not zip_path.exists():
        return

    inner_file = "COSMOS/MarshallColorado/COSMOS_COSMOS_MarshallColorado_sm_0.000000_0.250000_Hydroinnova-CRS-1000B_1_1_20160101_20161231.stm"
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner_file) as stream:
            header, data = read_ismn_stm(stream)

    assert header["station"] == "Marshall_Colorado"
    assert header["sensor"] == "Hydroinnova-CRS-1000B"
    assert len(data) == 8633
    assert "datetime" in data.columns
    assert "soil_moisture" in data.columns
    assert "ismn_flag" in data.columns