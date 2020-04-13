from structural_generation import *
from notears import *
import argparse



parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--n_data', default= 100, type = int, help='number of examples')

args = parser.parse_args()

X, W_true = generate_structure(args.height, args.n_data)
W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
print(W_est.shape)
