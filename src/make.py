import sys
from . import scigridnetwork
from . import armafitloader
from . import globals
import numpy as np
import logging
from tqdm import tqdm
import json
from pathlib import Path

pypsapath = "C:/dev/py/PyPSA/"
if sys.path[0] != pypsapath:
    sys.path.insert(0, pypsapath)

import pypsa

pretty = True
highres = True

#rcParams["figure.dpi"]=300


def make_export_renewable_covariance():
    logging.info("Loading SciGRID network")
    sgn = scigridnetwork.SciGRID_network()

    logging.info("Loading monthly ARMA fits and exporting covariance matrices")
    for mi in tqdm(range(1)):
        armafits = armafitloader.ARMAfit_loader(sgn, mi)
        armafits.compute_covariances(save_csv=True, save_npy=True)

    logging.info("Exporting wind & solar capacity")
    np.savetxt(globals.data_path/"processed"/"wind_capacity.csv", sgn.wind_capacity, delimiter=",")
    np.savetxt(globals.data_path/"processed"/"solar_capacity.csv", sgn.solar_capacity, delimiter=",")
    np.save(globals.data_path/"processed"/"wind_capacity.npy", sgn.wind_capacity, allow_pickle=False)
    np.save(globals.data_path/"processed"/"solar_capacity.npy", sgn.solar_capacity, allow_pickle=False)


def make_export_flow_matrix():
    logging.info("Loading SciGRID network")
    sgn = scigridnetwork.SciGRID_network()

    logging.info("Writing flow matrix")
    with open(globals.data_path / "processed" / "flow.json", 'w') as f:
        f.write(json.dumps(sgn.F.tolist()))

    logging.info("Writing node properties")
    lines = sgn.network.lines
    x = sgn.network.buses.loc[sgn.new_nodes].x
    y = sgn.network.buses.loc[sgn.new_nodes].y

    data_to_export = {'x': list(x),
                      'y': list(y),
                      'bus0': list(lines.bus0.apply(sgn.node_index)),
                      'bus1': list(lines.bus1.apply(sgn.node_index)),
                      'capacity': list(sgn.line_capacity)}

    lines = []
    for name, d in data_to_export.items():
        lines.append(name+" = " + json.dumps(d))

    with open(globals.data_path / "processed" / "nodeproperties.js", 'w') as f:
        f.write("\n".join(lines))


def make_list():
    print(makeable)


makeable = [s[5:] for s in dir() if s.startswith("make_")]

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    for funcname in sys.argv[1:]:
        if funcname in makeable:
            print(" == " + funcname + " == ")
            eval("make_" + funcname)()
            print(" ===" + "="*len(funcname) + "=== ")
