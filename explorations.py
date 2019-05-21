import numpy as np
import pandas as pd

from norpy import simulate
from norpy import get_random_model_specification, get_model_obj
from smm_prep import get_moments,get_weigthing_matrix
from norpy.adapter.SimulationBasedEstimation import SimulationBasedEstimationCls

###We want to generate a set of obeservations to be simulated. WHich decsions do we need
num_agents_obs = 100
num_boots=100
num_periods = 20
decision_array = np.random.randint(1, 4, size=2000)
period_array = np.array([np.arange(1, 21)] * 100).reshape(2000)
identifier_array = np.zeros(2000)
for x in range(2000):
    identifier_array[x] = int(np.floor(x/10)+1)

wage_array = np.random.randint(1000,10000,size=2000)

#Put the observations into the right format
sim_df = pd.DataFrame({"identifier":identifier_array,"period":period_array,"wages":wage_array,"choice":decision_array})

moment_obs = get_moments(sim_df)

weighting_matrix = get_weigthing_matrix(sim_df,num_agents_obs,num_boots)

initialization_object = get_random_model_specification()

#Now we need to get an optimizer












