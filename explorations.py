import numpy as np
import pandas as pd
import pybobyqa

from norpy import simulate
from norpy import get_random_model_specification, get_model_obj
from smm_prep import get_moments, get_weigthing_matrix
from norpy.adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from optimizers.auxiliray_pyogba import wrapper_pybobyqa

# Define which paramteres to optimize over!
# Which one do we start with anyways
# Shall we specify a full contraint vector ????????
optim_paras = {
    "coeffs_common": slice(0, 2),
    "coeffs_home": slice(2, 6),
    "coeffs_edu": slice(6, 14),
    "coeffs_work": slice(14, 27),
}
###We want to generate a set of obeservations to be simulated. WHich decsions do we need
num_agents_obs = 100
num_boots = 100
num_periods = 20
max_evals = 100000
decision_array = np.random.randint(1, 4, size=2000)
period_array = np.array([np.arange(1, 21)] * 100).reshape(2000)
identifier_array = np.zeros(2000)
for x in range(2000):
    identifier_array[x] = int(np.floor(x / 10) + 1)

wage_array = np.random.randint(1000, 10000, size=2000)

# Put the observations into the right format
sim_df = pd.DataFrame(
    {
        "identifier": identifier_array,
        "period": period_array,
        "wages": wage_array,
        "choice": decision_array,
    }
)


moment_obs = get_moments(sim_df)

weighting_matrix = get_weigthing_matrix(sim_df, num_agents_obs, num_boots)

initialization_object = get_random_model_specification()

# Now we start with the optimization
args = (
    initialization_object,
    moment_obs,
    weighting_matrix,
    get_moments,
    optim_paras,
    max_evals,
)
adapter_smm = SimulationBasedEstimationCls(*args)


kwargs = dict()
kwargs["scaling_within_bounds"] = False
kwargs["objfun_has_noise"] = True
kwargs["maxfun"] = 10e6

rslt = pybobyqa.solve(adapter_smm.evaluate, adapter_smm.free_params, **kwargs)
