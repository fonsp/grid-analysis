from pathlib import Path
import json
import re


datadir = Path("C:/dev/grid-analysis/data/armafits")
monthnames = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


months = [datadir/Path(mn) for mn in monthnames]

tablestringlines = []

for month in months:
    solardefault = len([f for f in (month/Path("solar")).iterdir() if re.search("\\d\\.json", f.name) is not None and f.name.endswith(".json")])
    solarMLSANN = len([f for f in (month/Path("solar_ML_SANN")).iterdir() if f.name.endswith("SANN.json")])
    solarMLSANNnoise = len([f for f in (month/Path("solar_ML_SANN")).iterdir() if f.name.endswith("noise.json")])

    winddefault = len([f for f in (month/Path("wind")).iterdir() if re.search("\\d\\.json", f.name) is not None and f.name.endswith(".json")])
    windMLSANN = winddefault
    windMLSANNnoise = len([f for f in (month/Path("wind")).iterdir() if f.name.endswith("noise.json")])

    tablestringlines.append(" & ".join([solardefault, solarMLSANN, solarMLSANNnoise, winddefault, windMLSANN, windMLSANNnoise]))

print(tablestringlines)
