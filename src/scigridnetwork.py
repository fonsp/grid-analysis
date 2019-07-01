import sys
from . import globals
import numpy as np
import pandas as pd
import scipy as sp
import logging
import json
from pathlib import Path

import src.globals

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

        self.operating_voltage_lines = self.network.lines.voltage.values
        pypsa.pf.apply_line_types(self.network)

        X = self.network.lines.x.values
        R = self.network.lines.r.values
        minusB = X / (X*X + R*R)
        self.i_times_lineadmittance_old = minusB * self.operating_voltage_lines * self.operating_voltage_lines

        self.reload()

    def reload(self):
        self.bus_names = list(self.network.buses.index)
        self.bus_names_220_suffix = [n for n in self.bus_names if n[-6:] == "_220kV"]
        self.suffix_removed = [n[:-6] for n in self.bus_names_220_suffix]
        self.pairs = list(zip(self.suffix_removed, self.bus_names_220_suffix))

        generator_buses = self.network.generators.bus.unique()

        self.n = len(generator_buses)

        # N:
        self.new_nodes = sorted(list(generator_buses))

        # L:
        self.old_lines = [(self.node_index(a), self.node_index(b)) for _, a, b in self.network.lines[['bus0', 'bus1']].itertuples()]

        # There are duplicate lines!
        # We will combine them by adding admittances

        self.old_line_indices = {l: [i for i, l_old in enumerate(self.old_lines) if l_old == l] for l in self.old_lines}
        self.scigrid_line_indices = [[self.network.lines.index.values[i] for i in indices] for indices in self.old_line_indices.values()]

        self.new_lines = list(self.old_line_indices)
        self.m = len(self.new_lines)

        self.old_index_to_new = [self.new_lines.index(old) for old in self.old_lines]
        self.new_index_to_old = [indices[0] for indices in self.old_line_indices.values()]

        self.line_lengths = [self.network.lines.length[i] for i in self.new_index_to_old]

        self.i_times_lineadmittance_new = self.map_to_new_line_array(self.i_times_lineadmittance_old)

        X = self.network.lines.x.values / self.operating_voltage_lines
        R = self.network.lines.r.values / self.operating_voltage_lines
        X_new = self.map_to_new_line_array(X)
        R_new = self.map_to_new_line_array(R)
        minusB_new = X_new / (X_new*X_new + R_new*R_new)
        self.i_times_lineadmittance_new = minusB_new

        logging.info("Combined {} parallel lines".format(len(self.network.lines) - self.m))

        # %% Flow matrix
        self.C = SciGRID_network.edge_vertex_incidence_matrix(self.new_nodes, self.new_lines)

        self.line_threshold = self.network.lines.s_nom.values.copy()
        self.line_threshold = self.map_to_new_line_array(self.line_threshold)

        # There are some infeasibilities without small extensions
        for line_name in ["316", "527", "602"]:
            line_index = next(i for i, s in enumerate(self.network.lines.index) if s == line_name)
            self.line_threshold[self.old_index_to_new[line_index]] = 1200
        # if self.contingency_factor_scaled:
        #     self.line_threshold = self.network.lines.s_nom.values / .7

        self.L = self.C.T @ sp.sparse.diags(self.i_times_lineadmittance_new) @ self.C
        self.Linv = np.array(np.linalg.pinv(self.L.todense()))

        self.F = np.multiply((self.i_times_lineadmittance_new)[:, np.newaxis], self.C @ self.Linv)

        self.M = self.F @ self.C.T - np.identity(self.m)

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

        self.line_flow_nonlinear = self.map_to_new_line_df(self.network.lines_t.p0.iloc[:24, :])

        self.line_saturation_nonlinear = self.line_flow_nonlinear / self.line_threshold

    def get_line_ratings(self, bus_covariance, time=None):
        if time is None:
            time = self.network.generators_t.p.index[11]

        f = self.line_saturation_nonlinear.loc[time]
        line_cov = self.F @ bus_covariance @ self.F.T

        def true_prob(mu_l, sigma_l):
            return 1.0 - (sp.stats.norm.cdf(1.0, loc=mu_l, scale=sigma_l) - sp.stats.norm.cdf(-1.0, loc=mu_l, scale=sigma_l))

        def rate(mu_l, sigma_l):
            return np.square(1-np.abs(mu_l))/(2.0*np.square(sigma_l))

        line_ratings = pd.DataFrame({"l": self.scigrid_line_indices,
                                     "f": f,
                                     "σ": np.sqrt(np.diagonal(line_cov)) / self.line_threshold})

        ε_Σ = 1
        line_ratings["P>1"] = true_prob(line_ratings.f, line_ratings.σ*ε_Σ)
        line_ratings["rate"] = rate(line_ratings.f, line_ratings.σ*ε_Σ)

        # line_ratings["Pnorm"] = line_ratings["P>1"] / np.max(line_ratings["P>1"])

        return line_ratings

    def most_likely_power_injection_given_line_failure(self, first_failure, bus_covariance, time=None):
        if time is None:
            time = self.network.generators_t.p.index[11]
        mup = self.injection_total.loc[time]
        muf = self.line_saturation_nonlinear.loc[time] # should be line_saturation_linear to be mathematically correct

        sigmap = bus_covariance
        line_cov = self.F @ bus_covariance @ self.F.T
        sigmaf = (line_cov * (1.0/self.line_threshold)) * ((1.0/self.line_threshold)[:, np.newaxis])

        def sign_mod(x):
            """The regular sign function, with the modification that `sign_mod(0.0)==1.0`, instead of zero."""
            return 1.0 if x >= 0.0 else -1.0

        return mup + (sign_mod(muf[first_failure]) - muf[first_failure]) / sigmaf[first_failure, first_failure] * (sigmap @ (self.F[first_failure] / self.line_threshold[first_failure]))

    def line_outage_flow_difference(self, failed_lines, nominal_flow, verbose_rank_loss=False):
        # Mzeroed = self.M.copy()
        # Mzeroed[np.abs(Mzeroed) < 1e-10] = 0.0
        # MN = Mzeroed[:, failed_lines]
        MN = self.M[:, failed_lines]
        NtMN = MN[failed_lines, :]
        Ntf = nominal_flow[failed_lines]

        NtMNinvNtf, residuals, NtMNrank, _ = np.linalg.lstsq(NtMN, Ntf, rcond=None)
        # print(NtMNinvNtf)
        if verbose_rank_loss and NtMNrank < len(failed_lines):
            print("Loss of rank: {}".format(len(failed_lines) - NtMNrank))
        return np.dot(-MN, NtMNinvNtf)

    def simulate_cascade(self, first_failure, bus_covariance, time=None, linearity_correction=True, iter_lim=50):
        if time is None:
            time = self.network.generators_t.p.index[11]
        failed_lines = {first_failure}
        assumed_injection = self.most_likely_power_injection_given_line_failure(first_failure, bus_covariance=bus_covariance, time=time)
        base_flow = self.F @ assumed_injection

        if linearity_correction:
            base_flow += self.line_flow_nonlinear.loc[time] - self.line_flow_linear.loc[time]

        for _i in range(iter_lim):
            fl_list = list(failed_lines)
            new_flow = base_flow + self.line_outage_flow_difference(fl_list, base_flow)

            # Uncomment to check validity:
            #assert np.max(np.abs(new_flow[fl_list])) < 1e-10

            yield (fl_list, new_flow)
            new_failures = {l for l in range(self.m) if new_flow[l] >= self.line_threshold[l]} - failed_lines
            if not new_failures:
                return
            failed_lines |= new_failures
        logging.warning("Cascade simulation: iteration limit {} reached. First failure: {}".format(iter_lim, first_failure))

    def export_cascades_to_json(self, bus_covariance, filename=None, first_failures=None, time=None, linearity_correction=True, iter_lim=50):
        cascades = dict()
        cascades_approx = dict()
        if first_failures is None:
            first_failures = range(self.m)

        for l in tqdm(first_failures):
            try:
                casc = list(self.simulate_cascade(l, bus_covariance=bus_covariance, time=time, linearity_correction=linearity_correction, iter_lim=iter_lim))
                cascades[l] = [(failed, list(flow)) for failed, flow in casc]

                casc_approx = []
                for failed, flow in casc:
                    sat_approx = np.clip(np.abs(flow) / self.line_threshold, 0.0, 1.0)
                    out = "".join('#' if i in failed else chr(48 + int(el * 64)) for i, el in enumerate(sat_approx))
                    casc_approx.append(out)
                cascades_approx[l] = casc_approx
            except np.linalg.LinAlgError:
                print("Simulating cascade of line {} failed with LinAlgError".format(l))

        if filename is None:
            filename = "simulated_cascades"

        with open(src.globals.data_path / "processed" / (filename + ".json"), "w") as f:
            print("Writing to {}".format(f.name))
            json.dump(cascades, f, separators=(',', ':'))

        with open(src.globals.data_path / "processed" / (filename + "_interactive.json"), "w") as f:
            print("Writing to {}".format(f.name))
            json.dump(cascades_approx, f, separators=(',', ':'))

    @staticmethod
    def edge_vertex_incidence_matrix(vertices, edges):
        row_indices = np.arange(len(edges)).repeat(2)
        column_indices = np.concatenate(edges)
        data = np.repeat([[1, -1]], repeats=len(edges), axis=0).flatten()
        # data = np.array([[1, -1] if a < b else [-1, 1] for a, b in edges]).flatten()
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

    def map_to_old_line_array(self, x):
        x_mapped = [0.0]*len(self.network.lines)
        for val, i in enumerate(x):
            x_mapped[self.new_index_to_old[i]] = val
        if type(x) is np.ndarray:
            return np.array(x_mapped)
        return x_mapped

    def map_to_new_line_array(self, x):
        x_mapped = [sum(x[l] for l in indices) for indices in self.old_line_indices.values()]
        if type(x) is np.ndarray:
            return np.array(x_mapped)
        return x_mapped

    def map_to_new_line_df(self, d):
        return pd.DataFrame.from_dict(dict((i, self.map_to_new_line_array(r)) for i, r in d.iterrows()), orient="index")

    def scigrid_name_to_new_line_number(self, name):
        return next(i for i, names in enumerate(self.scigrid_line_indices) if name in names)

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

    def bus_array_to_plot(self, x=None):
        bus_sizes = [0]*len(self.network.buses)
        if x is None:
            return bus_sizes, ['#00000000']*len(self.network.buses)
        bus_colors = ['g']*len(self.network.buses)

        for i, bus_name in enumerate(self.network.buses.index):
            if bus_name in self.new_nodes:
                val = x[self.node_index(bus_name)]
                bus_sizes[i] = np.abs(val) / 10
                bus_colors[i] = '#6fd08c' if val > 0 else '#7b9ea8'
        return bus_sizes, bus_colors

    def line_array_to_plot(self, color=None, width=None):
        if color is None:
            color = np.zeros(self.m)
        if width is None:
            width = np.ones(self.m)
        line_colors = [0]*len(self.network.lines)
        line_widths = [0]*len(self.network.lines)

        for i, (c, w) in enumerate(zip(color, width)):
            old_i = self.new_index_to_old[i]
            line_colors[old_i] = color[i]
            line_widths[old_i] = width[i]
        return line_colors, line_widths
