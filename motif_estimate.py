import json
import numpy as np
import argparse 
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False,
                        help='File with input data (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False,
                        help='File where the estimated parameters will be saved (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Should alpha be estimated or not? (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha

input_file, output_file, estimate_alpha = parse_arguments()

with open(input_file, 'r') as input_file:
    data = json.load(input_file)


alpha = data['alpha']
X = np.asarray(data['X'], dtype=np.int64)
k, w = X.shape


Sigma = np.arange(1, 5, 1)


def e_step(x, alpha, theta, theta_b):
    q = np.empty((k, 2))
    for i in range(x.shape[0]):
        prob_0 = np.prod(theta_b[x[i, ]-1])
        prob_1 = np.prod(np.array([theta[j, i] for i, j in enumerate(x[0, ]-1)]))
        prob = prob_0+prob_1

        q[i, ] = np.array([1-alpha, alpha])*np.array([prob_0, prob_1])/prob
    return q


def m_step(x, q):
    counts = np.array([[np.sum(x[i, ] == j) for i in range(x.shape[0])] for j in Sigma]).T
    lam_0 = X.shape[1]*np.sum(q[:, 0])
    lam_1 = np.sum(q[:, 1])

    theta_b = np.matmul(q[:, 0], counts)/lam_0
    theta = np.array([[np.sum(q[X[:, j] == i, 1])/lam_1 for i in Sigma] for j in range(x.shape[1])]).T

    return theta, theta_b


def d_tv(orig, orig_b, estim, estim_b):
    return (np.sum(np.absolute(estim_b-orig_b))/2+np.sum(np.absolute(estim-orig))/2)/(estim.shape[1]+1)


theta_b = np.array([np.sum(X == i)/(k*w) for i in Sigma])
theta = np.array([[np.sum(X[:, i] == j)/k for j in Sigma] for i in range(w)]).T

diff_history = []
n_epochs = 10
for epoch in range(n_epochs):
    q_tmp = e_step(X, alpha, theta, theta_b)
    theta_tmp, theta_b_tmp = m_step(X, q_tmp)
    diff = d_tv(theta, theta_b, theta_tmp, theta_b_tmp)
    diff = (np.sum(np.absolute(theta_b_tmp-theta_b))/2+np.sum(np.absolute(theta_tmp-theta))/2)/(theta.shape[1]+1)
    diff_history.append(diff)
    theta, theta_b = theta_tmp, theta_b_tmp

diff_history = []
diff = 1
t = 0
while diff>0.0001:
    if t>1000:
        break
    t+=1
    q_tmp = e_step(X, alpha, theta, theta_b)
    theta_tmp, theta_b_tmp = m_step(X, q_tmp)
    diff = d_tv(theta, theta_b, theta_tmp, theta_b_tmp)
    diff_history.append(diff)
    theta, theta_b = theta_tmp, theta_b_tmp


estimated_params = {
    "alpha" : alpha,            
    "Theta" : theta.tolist(),   
    "ThetaB" : theta_b.tolist()  
    }

with open(output_file, 'w') as output_file:
    json.dump(estimated_params, output_file)
