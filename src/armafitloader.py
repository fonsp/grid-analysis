import sys
from . import globals
import numpy as np
import pandas as pd
import json
import re
import logging

from pathlib import Path


class ARMAfit_loader():
    monthnames = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    def __init__(self, sgn, month_index=0, armafit_path=None):
        self.month_index = month_index
        self.sgn = sgn
        self.armafit_path = globals.data_path / "armafits" if armafit_path is None else armafit_path

        self.months = [self.armafit_path / Path(mn) for mn in ARMAfit_loader.monthnames]

        self.monthlengths = [len(self.jsoncontents((m/Path("wind")).iterdir().__next__())["residuals"]) for m in self.months]
        self.monthstarts = [sum(self.monthlengths[:i]) for i in range(13)]

        self.t = sgn.network.generators_t.p_max_pu.index

        self.load_month()

    def load_month(self):
        self.solar_monthseries = ARMAfit_loader.seriesofmonth(self.months[self.month_index], "solar_ML_SANN")
        self.wind_monthseries = ARMAfit_loader.seriesofmonth(self.months[self.month_index], "wind")

        self.solar_noised = {i for i in self.solar_monthseries if "noise" in self.solar_monthseries[i]}
        self.wind_noised = {i for i in self.wind_monthseries if "noise" in self.wind_monthseries[i]}

        if self.sgn.solar_generation.shape[1] < self.monthstarts[self.month_index+1]:
            logging.warn("Not enough generation data in the selected month.")
        self.solar_generationmonth = self.sgn.solar_generation[:, self.monthstarts[self.month_index]:self.monthstarts[self.month_index+1]]
        self.wind_generationmonth = self.sgn.wind_generation[:, self.monthstarts[self.month_index]:self.monthstarts[self.month_index+1]]

        self.tmonth = self.sgn.t[self.monthstarts[self.month_index]:self.monthstarts[self.month_index+1]]

        self.isdaylighthourmonth = self.sgn.isdaylighthour[self.monthstarts[self.month_index]:self.monthstarts[self.month_index+1]]

    def compute_covariances(self, save_csv=True, save_npy=True, save_path=None, append_month_dir=True):
        """When save_path is None, `globals.data_path / 'processed' / 'covariance'` is used."""
        if save_path is None:
            save_path = globals.data_path / "processed" / "covariance"
        if append_month_dir:
            save_path = save_path / ARMAfit_loader.monthnames[self.month_index]
        for parent in (list(save_path.parents)[::-1] + [save_path]):
            if not parent.exists():
                parent.mkdir()

        self.solar_res = np.array([self.solar_monthseries[i]["residuals"] if i in self.solar_monthseries else np.zeros(self.monthlengths[self.month_index]) for i in range(self.sgn.n)])
        self.solar_gen = self.solar_generationmonth
        self.wind_res = np.array([self.wind_monthseries[i]["residuals"] if i in self.wind_monthseries else np.zeros(self.monthlengths[self.month_index]) for i in range(self.sgn.n)])
        self.wind_gen = self.wind_generationmonth

        with np.errstate(divide='ignore'):
            solar_invcapacity = 1/self.sgn.solar_capacity
            solar_invcapacity[self.sgn.solar_capacity == 0.0] = 0.0
            wind_invcapacity = 1/self.sgn.wind_capacity
            wind_invcapacity[self.sgn.wind_capacity == 0.0] = 0.0

        self.solar_rescov = np.cov(self.solar_res)
        self.solar_gencov = np.cov(self.solar_gen)
        self.solar_gencovnorm = np.cov(solar_invcapacity[:, np.newaxis]*self.solar_gen)
        self.solar_rescovnorm = np.cov(solar_invcapacity[:, np.newaxis]*self.solar_res)
        self.solar_resdaylightcov = np.cov(self.solar_res[:, self.isdaylighthourmonth])
        self.solar_gendaylightcov = np.cov(self.solar_gen[:, self.isdaylighthourmonth])
        self.solar_resdaylightcovnorm = np.cov((solar_invcapacity[:, np.newaxis]*self.solar_res)[:, self.isdaylighthourmonth])
        self.solar_gendaylightcovnorm = np.cov((solar_invcapacity[:, np.newaxis]*self.solar_gen)[:, self.isdaylighthourmonth])

        self.wind_rescov = np.cov(self.wind_res)
        self.wind_gencov = np.cov(self.wind_gen)
        self.wind_gencovnorm = np.cov(wind_invcapacity[:, np.newaxis]*self.wind_gen)
        self.wind_rescovnorm = np.cov(wind_invcapacity[:, np.newaxis]*self.wind_res)

        solar_gen_diff = self.solar_gen[:, 1:] - self.solar_gen[:, :-1]
        wind_gen_diff = self.wind_gen[:, 1:] - self.wind_gen[:, :-1]

        isdaylighthourmonth_diff = np.logical_or(self.isdaylighthourmonth[:-1], self.isdaylighthourmonth[1:])

        self.solar_diffcov = np.cov(solar_gen_diff)
        self.solar_diffcovnorm = np.cov(solar_invcapacity[:, np.newaxis]*solar_gen_diff)
        self.solar_diffdaylightcov = np.cov(solar_gen_diff[:, isdaylighthourmonth_diff])
        self.solar_diffdaylightcovnorm = np.cov((solar_invcapacity[:, np.newaxis]*solar_gen_diff)[:, isdaylighthourmonth_diff])

        self.wind_diffcov = np.cov(wind_gen_diff)
        self.wind_diffcovnorm = np.cov(wind_invcapacity[:, np.newaxis]*wind_gen_diff)

        if save_csv:
            np.savetxt(save_path / "solar_rescov.csv", self.solar_rescov, delimiter=", ")
            np.savetxt(save_path / "solar_gencov.csv", self.solar_gencov, delimiter=", ")
            np.savetxt(save_path / "solar_diffcov.csv", self.solar_diffcov, delimiter=", ")
            np.savetxt(save_path / "solar_rescovnorm.csv", self.solar_rescovnorm, delimiter=", ")
            np.savetxt(save_path / "solar_gencovnorm.csv", self.solar_gencovnorm, delimiter=", ")
            np.savetxt(save_path / "solar_diffcovnorm.csv", self.solar_diffcovnorm, delimiter=", ")
            np.savetxt(save_path / "solar_resdaylightcov.csv", self.solar_resdaylightcov, delimiter=", ")
            np.savetxt(save_path / "solar_gendaylightcov.csv", self.solar_gendaylightcov, delimiter=", ")
            np.savetxt(save_path / "solar_diffdaylightcov.csv", self.solar_diffdaylightcov, delimiter=", ")
            np.savetxt(save_path / "solar_resdaylightcovnorm.csv", self.solar_resdaylightcovnorm, delimiter=", ")
            np.savetxt(save_path / "solar_gendaylightcovnorm.csv", self.solar_gendaylightcovnorm, delimiter=", ")
            np.savetxt(save_path / "solar_diffdaylightcovnorm.csv", self.solar_diffdaylightcovnorm, delimiter=", ")

            np.savetxt(save_path / "wind_rescov.csv", self.wind_rescov, delimiter=", ")
            np.savetxt(save_path / "wind_gencov.csv", self.wind_gencov, delimiter=", ")
            np.savetxt(save_path / "wind_diffcov.csv", self.wind_diffcov, delimiter=", ")
            np.savetxt(save_path / "wind_rescovnorm.csv", self.wind_rescovnorm, delimiter=", ")
            np.savetxt(save_path / "wind_gencovnorm.csv", self.wind_gencovnorm, delimiter=", ")
            np.savetxt(save_path / "wind_diffcovnorm.csv", self.wind_diffcovnorm, delimiter=", ")

        if save_npy:
            np.save(save_path / "solar_rescov.npy", self.solar_rescov)
            np.save(save_path / "solar_gencov.npy", self.solar_gencov)
            np.save(save_path / "solar_diffcov.npy", self.solar_diffcov)
            np.save(save_path / "solar_rescovnorm.npy", self.solar_rescovnorm)
            np.save(save_path / "solar_gencovnorm.npy", self.solar_gencovnorm)
            np.save(save_path / "solar_diffcovnorm.npy", self.solar_diffcovnorm)
            np.save(save_path / "solar_resdaylightcov.npy", self.solar_resdaylightcov)
            np.save(save_path / "solar_gendaylightcov.npy", self.solar_gendaylightcov)
            np.save(save_path / "solar_diffdaylightcov.npy", self.solar_diffdaylightcov)
            np.save(save_path / "solar_resdaylightcovnorm.npy", self.solar_resdaylightcovnorm)
            np.save(save_path / "solar_gendaylightcovnorm.npy", self.solar_gendaylightcovnorm)
            np.save(save_path / "solar_diffdaylightcovnorm.npy", self.solar_diffdaylightcovnorm)

            np.save(save_path / "wind_rescov.npy", self.wind_rescov)
            np.save(save_path / "wind_gencov.npy", self.wind_gencov)
            np.save(save_path / "wind_diffcov.npy", self.wind_diffcov)
            np.save(save_path / "wind_rescovnorm.npy", self.wind_rescovnorm)
            np.save(save_path / "wind_gencovnorm.npy", self.wind_gencovnorm)
            np.save(save_path / "wind_diffcovnorm.npy", self.wind_diffcovnorm)

    @staticmethod
    def jsoncontents(f):
        with f.open() as ff:
            return json.JSONDecoder().decode(ff.read())

    @staticmethod
    def seriesofmonth(month_path, carrier_name):
        def validfilename(s):
            return s.endswith("noise.json") or s.endswith("SANN.json") or (s.endswith(".json") and re.search("\\d\\.json", s) is not None)
        files = (f for f in (month_path/Path(carrier_name)).iterdir() if validfilename(f.name))

        interpreted = [ARMAfit_loader.jsoncontents(f) for f in files]
        # In Python, bus indices start at 0
        return dict((result["bus"][0] - 1, result) for result in interpreted)
