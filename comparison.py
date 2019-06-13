"""
Implement a function that returns equivalent specs for respy and norpy.
Then simulate each and compare the results.
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
from respy_smm.SimulationBasedEstimation import SimulationBasedEstimationCls
from comparison_auxiliary import respy_obj_from_new_init

init_path = "/home/moritz/OpenSourceEconomics/dev_norpy/ressources/model.respy.ini"

#Compare init dicts
respy_example_dict = read_init_file(init_path)

#Norpy example dict. We start witrh constrained values
constr = {"num_types":3,
          "num_edu_start":1,
          "edu_range_start":np.array([9]),
          "intial_lagged_schooling_prob":float(1),
          "type_spec_shifts":np.zeros(9).reshape(3,3),
          "shocks_cov":np.identity(3),
          "num_periods":10,
          "num_agents_sim":1000}
norpy_example_dict = get_random_model_specification(constr=constr)


def norpy_to_respy_spec(norpy_init,respy_init):
    """
    We want to translate a norpy init dict to a respy init dict.
    We assume that we have three types for now

    ARGS:
        norpy_init: dict containing norpy specs
        respy_init: template that already specifies all boilerplate args


    """
    out = respy_init
    #Basic stuff
    out["BASICS"]["periods"] = norpy_init["num_periods"]
    out["BASICS"]["coeffs"] = np.array([norpy_init["delta"]])

    #Common coeffs
    out["COMMON"]["coeffs"] = norpy_init["coeffs_common"]

    #WOrk
    out["OCCUPATION A"]["coeffs"][:3] = norpy_init["coeffs_work"][:4]
    out["OCCUPATION A"]["coeffs"][3:8] = norpy_init["coeffs_work"][4:7]
    out["OCCUPATION A"]["coeffs"][9:] = norpy_init["coeffs_work"][7:]


    #Education
    out["EDUCATION"]["coeffs"] = norpy_init["coeffs_edu"]


    out["EDUCATION"]["start"] = norpy_init["edu_range_start"]
    out["EDUCATION"]["max"] = norpy_init["edu_max"]

    #Home

    out["HOME"]["coeffs"][0] = norpy_init["coeffs_home"][0]
    out["HOME"]["coeffs"][2] = norpy_init["coeffs_home"][1]

    #Shocks
    #For now nehmen wir einfach die identity
    out["SHOCKS"]["coeffs"][0] = 1
    out["SHOCKS"]["coeffs"][2:4] = np.ones(2)
    out["SHOCKS"]["coeffs"][5] = 0
    out["SHOCKS"]["coeffs"][7] = 0
    out["SHOCKS"]["coeffs"][9] = 0

    #Type SHARES
    out["TYPE SHARES"]["coeffs"][0] = norpy_init["type_prob_cond_schooling"][0,1]
    out["TYPE SHARES"]["coeffs"][1] = norpy_init["type_prob_cond_schooling"][0,2]

    #Type Shifts
    out["TYPE SHIFTS"]["coeffs"][0] = norpy_init["type_spec_shifts"][1][0]
    out["TYPE SHIFTS"]["coeffs"][2:4] = norpy_init["type_spec_shifts"][1][1:3]
    out["TYPE SHIFTS"]["coeffs"][4] = norpy_init["type_spec_shifts"][2][0]
    out["TYPE SHIFTS"]["coeffs"][5:7] = norpy_init["type_spec_shifts"][2][1:3]

    #Sol details
    out["SOLUTION"]["seed"] = norpy_init["seed_emax"]
    out["SOLUTION"]["draws"] = norpy_init["num_draws_emax"]

    #Simulation Details
    out["SIMULATION"]["agents"] = norpy_init["num_agents_sim"]
    out["SIMULATION"]["seed"] = norpy_init["seed_sim"]

    return out


#Change the dict
a=norpy_to_respy_spec(norpy_example_dict,respy_example_dict)

#write new init file
respy_obj = respy_obj_from_new_init(a)
norpy_obj = get_model_obj(norpy_example_dict)

#Now run simulations
sim_respy = ov_simulation(respy_obj)

sim_norpy = simulate(get_model_obj(norpy_example_dict))

#Get a dict that compares the most impoirtant moments
decision_norpy= pd.Series(sim_norpy[:,2]).value_counts()
decision_respy= pd.Series(sim_respy[:,2]).value_counts()


#Up next generate auxiliary shocks that are equal and pass them to the functions !
shocks = dict()
#shocks["emax"] = return_simulated_shocks(norpy_obj)
#shocks["simulation"] = return_simulated_shocks(norpy_obj, True)
shocks["emax"] = np.zeros(norpy_example_dict["num_draws_emax"])
shocks["simulation"] = np.zeros(norpy_example_dict["num_agents_sim"])

sim_norpy_shocks = simulate_compare(norpy_obj,shocks)
sim_respy_shocks = ov_simulation_alt(respy_obj,shocks)

decision_norpy_shocks= pd.Series(sim_norpy_shocks[:,2]).value_counts()
decision_respy_shocks= pd.Series(sim_respy_shocks[:,2]).value_counts()
