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