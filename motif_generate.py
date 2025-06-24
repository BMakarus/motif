import json
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False,
                        help='File with parameters (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False,
                        help='File to save generated data (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output

param_file, output_file = parse_arguments()

with open(param_file, 'r') as input_file:
    params = json.load(input_file)


w = params['w']
k = params['k']
alpha = params['alpha']
theta = np.asarray(params['Theta'])
theta_b = np.asarray(params['ThetaB'])


Sigma = np.arange(1, 5, 1)
z = np.random.binomial(1, alpha, size=k)
X = np.empty((k, w))
for i, value in enumerate(z):
    if value == 0:
        X[i, ] = np.random.choice(Sigma, size=w, p=theta_b)
    else:
        for j in range(w):
            X[i, j] = np.random.choice(Sigma, p=theta[:, j])


gen_data = {
    "alpha": alpha,
    "X": X.tolist()
}

with open(output_file, 'w') as output_file:
    json.dump(gen_data, output_file)
