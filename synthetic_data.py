# import numpy as np
#
# def generate_synthetic_data(num_samples, input_dim, output_dim):
#     X = np.random.rand(num_samples, input_dim)
#     y = (np.sum(X, axis=1, keepdims=True) > (input_dim / 2)).astype(int)
#     return X, y



import numpy as np

def generate_synthetic_data(num_samples, input_dim, output_dim):
    X = np.random.rand(num_samples, input_dim)
    y = (np.sum(X, axis=1, keepdims=True) > (input_dim / 2)).astype(int)
    return X, y
