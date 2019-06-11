"""
Implement a function that returns equivalent specs for respy and norpy.
Then simulate each and compare the results.
"""
import numpy as np

from norpy import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj
from python.ov_simulation import ov_simulation
from respy_smm.auxiliary_depreciation import respy_spec_old_to_new
from respy.pre_processing.model_processing import write_init_file, read_init_file
from respy import clsRespy

init_path = "/home/moritz/OpenSourceEconomics/dev_norpy/ressources/model.respy.ini"

#Compare init dicts
respy_example_dict = read_init_file(init_path)

#Norpy example dict. We start witrh constrained values
constr = {"num_types":3,
          "intial_lagged_schooling_prob":1}
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
    out["BASICS"]["coeffs"] = norpy_init["delta"]

    #Common coeffs
    out["COMMON"]["coeffs"] = norpy_init["coeffs_common"]

    #WOrk
    out["OCCUPATION A"]["coeffs"][:3] = norpy_init["coeffs_work"][:4]
    out["OCCUPATION A"]["coeffs"][3:8] = norpy_init["coeffs_work"][4:7]
    out["OCCUPATION A"]["coeffs"][9:] = norpy_init["coeffs_work"][7:]


    #Education
    out["EDUCATION"]["coeff"] = norpy_init["coeffs_edu"]
    out["EDUCATION"]["start"] = norpy_init["edu_range_start"]
    out["EDUCATION"]["share"] = np.array([1/len(out["EDUCATION"]["start"])]*len(out["EDUCATION"]["start"]))
    out["EDUCATION"]["lagged"] = norpy_init["intial_lagged_schooling_prob"] #do we even need probabilities here ??
    out["EDUCATION"]["max"] = norpy_init["edu_max"]

    #Home
    out["HOME"]["coeffs"] = np.zeros(3)
    out["HOME"]["coeffs"][0] = norpy_init["coeffs_home"][0]
    out["HOME"]["coeffs"][2] = norpy_init["coeffs_home"][1]

    #Shocks
    cholesky = np.linalg.cholesky(norpy_init["shocks_cov"])#Wie teile ich die auf die covs auf ?
    out["SHOCKS"]["coeffs"][0:3] = cholesky[0][0:3]
    out["SHOCKS"]["coeffs"][3:5] = cholesky[1][2:4]
    out["SHOCKS"]["coeffs"][5] = cholesky[2][2]


    #Type SHARES
    out["TYPE SHARES"]["coeffs"][0] = norpy_init["type_prob_cond_schooling"][0,1]
    out["TYPE SHARES"]["coeffs"][1] = norpy_init["type_prob_cond_schooling"][0,2]

    #Type Shifts
    out["TYPE SHIFTS"]["coeffs"][0] = norpy_init["type_spec_shifts"][0][0]
    out["TYPE SHIFTS"]["coeffs"][2:4] = norpy_init["type_spec_shifts"][0][1:3]
    out["TYPE SHIFTS"]["coeffs"][4] = norpy_init["type_spec_shifts"][1][0]
    out["TYPE SHIFTS"]["coeffs"][5:7] = norpy_init["type_spec_shifts"][1][1:3]

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
respy_obj = clsRespy(a)
#Now run simulations
sim_respy = ov_simulation(a)

sim_norpy = simulate(get_model_obj(norpy_example_dict))
