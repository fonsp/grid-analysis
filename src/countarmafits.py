from pathlib import Path
import json
import re


datadir = Path("C:/dev/grid-analysis/data/armafits")
monthnames = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


months = [datadir/Path(mn) for mn in monthnames]

tablestringlines = []


def contents(f):
    with f.open() as ff:
        return json.JSONDecoder().decode(ff.read())


totals = 0
totalw = 0

for month in months:
    solardefault = len([f for f in (month/Path("solar")).iterdir() if re.search("\\d\\.json", f.name) is not None and f.name.endswith(".json")])
    solarMLSANN = len([f for f in (month/Path("solar_ML_SANN")).iterdir() if f.name.endswith("SANN.json")])
    solarMLSANNnoise = len([f for f in (month/Path("solar_ML_SANN")).iterdir() if f.name.endswith("noise.json")])

    solarMLSANNnoisefiles = [f for f in (month/Path("solar_ML_SANN")).iterdir() if f.name.endswith("noise.json")]

    solarMLSANNnoise1p = sum(contents(f)["noise"][0] == 0.01 for f in solarMLSANNnoisefiles)
    solarMLSANNnoise2p = sum(contents(f)["noise"][0] == 0.02 for f in solarMLSANNnoisefiles)
    assert(solarMLSANNnoise == solarMLSANNnoise1p + solarMLSANNnoise2p)

    winddefault = len([f for f in (month/Path("wind")).iterdir() if re.search("\\d\\.json", f.name) is not None and f.name.endswith(".json")])
    windMLSANN = winddefault
    windMLSANNnoise = len([f for f in (month/Path("wind")).iterdir() if f.name.endswith("noise.json")])

    windMLSANNnoisefiles = [f for f in (month/Path("wind")).iterdir() if f.name.endswith("noise.json")]

    windMLSANNnoise1p = sum(contents(f)["noise"][0] == 0.01 for f in windMLSANNnoisefiles)
    windMLSANNnoise2p = sum(contents(f)["noise"][0] == 0.02 for f in windMLSANNnoisefiles)
    #windMLSANNnoisedict = dict((contents(f)["noise"][0], f) for f in windMLSANNnoisefiles)
    assert(windMLSANNnoise == windMLSANNnoise1p + windMLSANNnoise2p)

    monthdisplay = str.upper(month.name[0])+month.name[1:]

    tt = lambda i: "\\texttt{"+str(i)+"}"
    bf = lambda i: "\\textbf{"+str(i)+"}"

    tablestringlines.append(" & ".join([monthdisplay, tt(solardefault), tt(bf(solarMLSANN)), tt(bf(solarMLSANNnoise1p)), tt(bf(solarMLSANNnoise2p)), tt(bf(winddefault)), tt(windMLSANN), tt(bf(windMLSANNnoise1p)), tt(bf(windMLSANNnoise2p))]) + " \\\\\n")

    totals += solarMLSANNnoise
    totalw += windMLSANNnoise

#print("".join(tablestringlines))
print(totals / totalw)
