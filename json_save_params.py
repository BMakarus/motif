import json
import numpy as np


tmp = np.array([[3/8,1/8,2/8,2/8], [1/10,2/10,3/10,4/10], [1/7,2/7,1/7,3/7]])
theta = tmp.T
theta_b = np.array([1/4,1/4,1/4,1/4])

params = {
    "w" : 3,
    "alpha" : 0.5,
    "k" : 10,
    "theta" : theta.tolist(),
    "theta_b" : theta_b.tolist()
    }


with open('params_set1.json', 'w') as outfile:
    json.dump(params, outfile)
