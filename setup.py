"""
This script constructs an object that returns objects for comparing norp and respy

"""
import numpy as np
import pandas as pd

from norpy import simulate
from norpy.simulate.simulate import simulate_compare, return_simulated_shocks
from norpy.model_spec import get_random_model_specification, get_model_obj
from respy_smm.auxiliary_depreciation import respy_ini_old_to_new
from python.ov_simulation import ov_simulation, ov_simulation_alt
from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy.pre_processing.model_processing import write_init_file, read_init_file, convert_init_dict_to_attr_dict
from respy import clsRespy
from comparison_auxiliary import respy_obj_from_new_init




def simulate_models_no_shocks(constr, init_path):
    norpy_obj, respy_obj, shocks_norpy, shocks_respy = test_setup(constr,init_path)

    sim_norpy = simulate_compare(norpy_obj,shocks_norpy)
    sim_respy =ov_simulation_alt(respy_obj,shocks_respy)
    return sim_norpy, sim_respy, norpy_obj, respy_obj

def simulate_models(constr, init_path):
    norpy_obj, respy_obj = test_setup(constr, init_path)
    sim_norpy = simulate(norpy_obj)
    sim_respy = ov_simulation(respy_obj)
    return sim_norpy, sim_respy


def test_setup(constr,init_path):
    """
    This function returns relevant model_objects

    """
    #Get an init dict for norpy
    norpy_init = get_random_model_specification(constr=constr)
    #Somehow we fixe that to zero for now
    norpy_init["coeffs_work"][7] = 0
    #Get respy_init
    respy_prelim_init = read_init_file(init_path)
    respy_init = _norpy_to_respy_spec(norpy_init,respy_prelim_init)
    #Get model objects
    respy_obj = respy_obj_from_new_init(respy_init)
    norpy_obj = get_model_obj(norpy_init)
    shocks_norpy, shocks_respy = get_no_shocks(norpy_init)
    return norpy_obj, respy_obj, shocks_norpy, shocks_respy

def get_no_shocks(norpy_init):
    shocks_respy = dict()
    shocks_norpy = dict()
    # shocks["emax"] = return_simulated_shocks(norpy_obj)
    # shocks["simulation"] = return_simulated_shocks(norpy_obj, True)
    shocks_respy["emax"] = np.zeros(
        norpy_init["num_draws_emax"] * 4 * norpy_init["num_periods"]).reshape(
        norpy_init["num_periods"], norpy_init["num_draws_emax"], 4)
    shocks_respy["simulation"] = np.zeros(
        norpy_init["num_agents_sim"] * 4 * norpy_init["num_periods"]).reshape(
        norpy_init["num_periods"], norpy_init["num_agents_sim"], 4)

    shocks_norpy["emax"] = np.zeros(
        norpy_init["num_draws_emax"] * 3 * norpy_init["num_periods"]).reshape(
        norpy_init["num_periods"], norpy_init["num_draws_emax"], 3)
    shocks_norpy["simulation"] = np.zeros(
        norpy_init["num_agents_sim"] * 3 * norpy_init["num_periods"]).reshape(
        norpy_init["num_periods"], norpy_init["num_agents_sim"], 3)
    return shocks_norpy,shocks_respy

def _norpy_to_respy_spec(norpy_init, respy_init):
    """
    We want to translate a norpy init dict to a respy init dict.
    We assume that we have three types for now

    ARGS:
        norpy_init: dict containing norpy specs
        respy_init: template that already specifies all boilerplate args


    """
    out = respy_init
    # Basic stuff
    out["BASICS"]["periods"] = norpy_init["num_periods"]
    out["BASICS"]["coeffs"] = np.array([norpy_init["delta"]])

    # Common coeffs
    out["COMMON"]["coeffs"] = norpy_init["coeffs_common"]

    # WOrk
    out["OCCUPATION A"]["coeffs"][:4] = norpy_init["coeffs_work"][:4]
    out["OCCUPATION A"]["coeffs"][6:] = norpy_init["coeffs_work"][4:]
    out["OCCUPATION A"]["coeffs"][9] = 0

    # Education
    out["EDUCATION"]["coeffs"] = norpy_init["coeffs_edu"]

    out["EDUCATION"]["start"] = norpy_init["edu_range_start"]
    out["EDUCATION"]["max"] = norpy_init["edu_max"]

    # Home

    out["HOME"]["coeffs"] = norpy_init["coeffs_home"]


    # Shocks
    # For now nehmen wir einfach die identity
    out["SHOCKS"]["coeffs"][0] = 1
    out["SHOCKS"]["coeffs"][2:4] = np.ones(2)
    out["SHOCKS"]["coeffs"][5] = 0
    out["SHOCKS"]["coeffs"][7] = 0
    out["SHOCKS"]["coeffs"][9] = 0

    # Type SHARES
    out["TYPE SHARES"]["coeffs"][0] = norpy_init["type_prob_cond_schooling"][0, 1]
    out["TYPE SHARES"]["coeffs"][1] = norpy_init["type_prob_cond_schooling"][0, 2]

    # Type Shifts
    out["TYPE SHIFTS"]["coeffs"][0] = norpy_init["type_spec_shifts"][1][0]
    out["TYPE SHIFTS"]["coeffs"][2:4] = norpy_init["type_spec_shifts"][1][1:3]
    out["TYPE SHIFTS"]["coeffs"][4] = norpy_init["type_spec_shifts"][2][0]
    out["TYPE SHIFTS"]["coeffs"][5:7] = norpy_init["type_spec_shifts"][2][1:3]

    # Sol details
    out["SOLUTION"]["seed"] = norpy_init["seed_emax"]
    out["SOLUTION"]["draws"] = norpy_init["num_draws_emax"]

    # Simulation Details
    out["SIMULATION"]["agents"] = norpy_init["num_agents_sim"]
    out["SIMULATION"]["seed"] = norpy_init["seed_sim"]

    return out




