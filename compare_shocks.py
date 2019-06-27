"""
Compare the both with the adequate shocks
"""
import numpy as np
import pandas as pd
import matplotlib as mp


from norpy import simulate
from norpy.simulate.simulate import simulate_compare, return_simulated_shocks
from norpy.model_spec import get_random_model_specification, get_model_obj
from setup import test_setup,simulate_models, simulate_models_det_shocks, test_func, simulate_models_semi_det_shocks
#from python.ov_simulation import ov_simulation, ov_simulation_alt
from respy_smm.auxiliary import smm_sample_f2py, get_initial_conditions
from respy.python.shared.shared_auxiliary import dist_class_attributes

#specify variables
init_path = "/home/moritz/OpenSourceEconomics/dev_norpy/ressources/model.respy.ini"

constr = {"num_types":3,
          "num_edu_start":1,
          "edu_range_start":np.array([9]),
          "intial_lagged_schooling_prob":float(1),
          "type_spec_shifts":np.zeros(9).reshape(3,3),
          "shocks_cov":np.identity(3)*2,
          "num_agents_sim":1000}

norpy_sim,respy_sim,norpy_init,respy_init = simulate_models_det_shocks(constr, init_path,shocks="base")
np.testing.assert_array_almost_equal(norpy_sim[:, 9], respy_sim[:,11])

def test_setup_no_shock_irw():
    df_norpy, df_respy, respy_obj, norpy_obj = simulate_models_det_shocks(constr, init_path, shocks="base")
    np.testing.assert_array_almost_equal(df_norpy[:, 11], df_respy[:, 13])
    np.testing.assert_array_almost_equal(df_norpy[:, 9], df_respy[:, 11])
    np.testing.assert_array_almost_equal(df_norpy[:, 10], df_respy[:, 12])
    np.testing.assert_array_almost_equal(df_norpy[:, 8], df_respy[:, 9])



for x in range(10):
    test_setup_no_shock_irw()
    print("test_irw_{}_worked".format(x))

constr = {"num_types":3,
          "type_spec_shifts":np.zeros(9).reshape(3,3),
          "shocks_cov":np.identity(3),
          "num_agents_sim":1000}

norpy_sim,respy_sim,norpy_init,respy_init = simulate_models_semi_det_shocks(constr, init_path,shocks="base")
np.testing.assert_array_almost_equal(norpy_sim[:, 9], respy_sim[:,11])

def test_setup_no_shock_irw():
    df_norpy, df_respy, respy_obj, norpy_obj = simulate_models_semi_det_shocks(constr, init_path, shocks="base")
    np.testing.assert_array_almost_equal(df_norpy[:, 11], df_respy[:, 13])
    np.testing.assert_array_almost_equal(df_norpy[:, 9], df_respy[:, 11])
    np.testing.assert_array_almost_equal(df_norpy[:, 10], df_respy[:, 12])
    np.testing.assert_array_almost_equal(df_norpy[:, 8], df_respy[:, 9])



for x in range(10):
    test_setup_no_shock_irw()
    print("test_irw_{}_worked".format(x))