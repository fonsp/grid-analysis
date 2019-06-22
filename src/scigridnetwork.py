import sys
from . import globals
import numpy as np
import pandas as pd
import scipy as sp
import logging

from pathlib import Path

pypsapath = "C:/dev/py/PyPSA/"
if sys.path[0] != pypsapath:
    sys.path.insert(0, pypsapath)

import pypsa
from tqdm import tqdm


class SciGRID_network():
    def __init__(self, data_path=None):
        data_path = globals.data_path / "raw" / "scigrid-with-load-gen-trafos" if data_path is None else data_path
        csv_folder_name = Path(data_path)

        logging.info("Using SciGRID csv path: {0}".format(csv_folder_name))

        if not Path(csv_folder_name).exists():
            raise Warning("SciGRID data not found. Make sure that you have downloaded the dataset corretly, and that the path is set correctly in this script.")

        self.network = pypsa.Network(import_name=csv_folder_name.as_posix())
        #self.network.generators.carrier = self.network.generators.source
        self.contingency_factor_scaled = False

        self.reload()

    def reload(self):
        self.bus_names = list(self.network.buses.index)
        self.bus_names_220_suffix = [n for n in self.bus_names if n[-6:] == "_220kV"]
        self.suffix_removed = [n[:-6] for n in self.bus_names_220_suffix]
        self.pairs = list(zip(self.suffix_removed, self.bus_names_220_suffix))

        generator_buses = self.network.generators.bus.unique()

        self.n = len(generator_buses)
        self.m = len(self.network.lines)

        # N:
        self.new_nodes = sorted(list(generator_buses))

        # L:
        self.new_lines = [(self.node_index(a), self.node_index(b)) for _, a, b in self.network.lines[['bus0', 'bus1']].itertuples()]

        self.operating_voltage_lines = self.network.lines.voltage.values

        # %% Flow matrix
        self.C = SciGRID_network.edge_vertex_incidence_matrix(self.new_nodes, self.new_lines)
        self.CV = sp.sparse.diags(self.operating_voltage_lines) * self.C

        pypsa.pf.apply_line_types(self.network)

        X = self.network.lines.x.values
        R = self.network.lines.r.values
        minusB = X / (X*X + R*R)
        self.i_times_lineadmittance = minusB# * operating_voltage_lines

        # TODO: contingency factor
        self.line_threshold = self.network.lines.s_nom.values.copy()# * 1e6 / operating_voltage_lines
        # There are some infeasibilities without small extensions
        for line_name in ["316", "527", "602"]:
            line_index = next(i for i, s in enumerate(self.network.lines.index) if s == line_name)
            self.line_threshold[line_index] = 1200
        if self.contingency_factor_scaled:
            self.line_threshold = self.network.lines.s_nom.values / .7

        self.L = self.CV.T.dot(sp.sparse.diags(self.i_times_lineadmittance)).dot(self.CV)
        self.Linv = np.array(np.linalg.pinv(self.L.todense()))

        self.F = np.multiply((self.operating_voltage_lines * self.i_times_lineadmittance)[:, np.newaxis], self.CV.dot(self.Linv))

        self.locations = pd.DataFrame(self.network.buses.loc[self.new_nodes][["x", "y"]])
        self.locations.index = range(self.n)

        # %% Renewable generation series

        self.T = len(self.network.generators_t.p_max_pu)

        self.solar_capacity = np.zeros(self.n)
        self.solar_generation = np.zeros((self.n, self.T))
        self.wind_capacity = np.zeros(self.n)
        self.wind_generation = np.zeros((self.n, self.T))

        self.offshorebuses = set()

        for gen_name, series in self.network.generators_t.p_max_pu.iteritems():
            carrier = self.network.generators.carrier[gen_name]
            if carrier == "":
                carrier = self.network.generators.source[gen_name]
            busname = self.network.generators.bus[gen_name]
            capacity = self.network.generators.p_nom[gen_name]

            nodeindex = self.node_index(busname)

            if carrier == "Solar":
                self.solar_generation[nodeindex, :] += series.values * capacity
                self.solar_capacity[nodeindex] += capacity
            elif carrier[0:4] == "Wind":
                self.wind_generation[nodeindex, :] += series.values * capacity
                self.wind_capacity[nodeindex] += capacity
                if carrier == "Wind Offshore":
                    self.offshorebuses.add(nodeindex)

        self.isdaylighthour = np.sum(self.solar_generation, axis=0) > 0.0

        self.t = self.network.generators_t.p_max_pu.index

        print(pypsa.__version__, pypsa.__path__)

    def dist_squared(self, node_a, node_b):
        s = np.square

        def lon2km(deg):
            return np.cos(50 * np.pi/180) * 40000 * (deg / 360)

        def lat2km(deg):
            return 40000 * (deg / 360)

        return s(lon2km(self.locations.x[node_a] - self.locations.x[node_b])) + s(lat2km(self.locations.y[node_a] - self.locations.y[node_b]))

    def dist(self, node_a, node_b):
        return np.sqrt(self.dist_squared(node_a, node_b))

    def get_dist_squared_matrix(self):
        xi, yi = np.meshgrid(range(self.n), range(self.n))
        return np.vectorize(self.dist_squared)(xi, yi)

    def get_dist_matrix(self):
        return np.sqrt(self.get_dist_squared_matrix())

    def run_lopf_jan1(self):
        # From the PyPSA example: https://github.com/PyPSA/PyPSA/blob/master/examples/scigrid-de/scigrid-lopf-then-pf.py
        contingency_factor = 0.7
        if not self.contingency_factor_scaled:
            self.network.lines.s_nom *= contingency_factor
            self.contingency_factor_scaled = True
        # There are some infeasibilities without small extensions
        for line_name in ["316", "527", "602"]:
            self.network.lines.loc[line_name, "s_nom"] = 1200

        # the lines to extend to resolve infeasibilities can
        # be found by
        # uncommenting the lines below to allow the network to be extended

        # network.lines["s_nom_original"] = network.lines.s_nom
        #
        # network.lines.s_nom_extendable = True
        # network.lines.s_nom_min = network.lines.s_nom

        group_size = 4

        solver_name = "glpk"

        print("Performing linear OPF for one day, {} snapshots at a time:".format(group_size))

        self.network.storage_units.state_of_charge_initial = 0.

        for i in tqdm(range(int(24/group_size))):
            # set the initial state of charge based on previous round
            if i > 0:
                self.network.storage_units.state_of_charge_initial = self.network.storage_units_t.state_of_charge.loc[self.network.snapshots[group_size*i-1]]
            self.network.lopf(self.network.snapshots[group_size*i:group_size*i+group_size],
                              solver_name=solver_name,
                              keep_files=False)
            self.network.lines.s_nom = self.network.lines.s_nom_opt

        self.network.lines.s_nom /= contingency_factor
        self.contingency_factor_scaled = False

        self.injection_total = pd.DataFrame(0.0, columns=range(self.n), index=self.network.generators_t.p.index[:24])

        for gen_name, series in self.network.generators_t.p.iteritems():
            i = self.node_index(self.network.generators.bus[gen_name])
            self.injection_total[i] += series[:24]
        for bus_name, series in self.network.loads_t.p_set.iteritems():
            i = self.node_index(bus_name)
            self.injection_total[i] -= series[:24]

        self.line_flow_linear = pd.DataFrame(0.0, columns=range(self.m), index=self.network.generators_t.p.index[:24])
        self.line_saturation_linear = pd.DataFrame(0.0, columns=range(self.m), index=self.network.generators_t.p.index[:24])
        for t, p in self.injection_total.iterrows():
            self.line_flow_linear.loc[t, :] += self.F @ p.values
            self.line_saturation_linear.loc[t, :] += self.line_flow_linear.loc[t, :] / self.line_threshold

        self.line_flow_nonlinear = self.network.lines_t.p0.iloc[:24, :]
        self.line_saturation_nonlinear = self.line_flow_nonlinear / self.line_threshold


    @staticmethod
    def edge_vertex_incidence_matrix(vertices, edges):
        row_indices = np.arange(len(edges)).repeat(2)
        column_indices = np.concatenate(edges)
        data = np.repeat([[1, -1]], repeats=len(edges), axis=0).flatten()
        return sp.sparse.csr_matrix((data, (row_indices, column_indices)), shape=(len(edges), len(vertices)))

    def node_index(self, bus_name):
        if bus_name in self.suffix_removed:
            return self.node_index(bus_name + "_220kV")
        return self.new_nodes.index(bus_name)

    def random_bus_pair(self):
        i = np.random.randint(self.n)
        j = np.random.randint(self.n - 1)
        j += j >= i
        return i, j

    @staticmethod
    def lon2km(deg):
        return np.cos(50 * np.pi/180) * 40000 * (deg / 360)

    @staticmethod
    def lat2km(deg):
        return 40000 * (deg / 360)

    def node_distance(self, a, b):
        def s(z):
            return z*z

        return np.sqrt(s(SciGRID_network.lon2km(self.locations.x[a] - self.locations.x[b]))+s(SciGRID_network.lat2km(self.locations.y[a] - self.locations.y[b])))
