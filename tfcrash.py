# import bopt
# import numpy as np
# bopt.compute_optimized_kernel_tf(np.array([[1.0]]), np.array([.1]), .1)

import tensorflow as tf
tf.enable_eager_execution()
tf.linalg.solve([[.1, .0], [.0, .1]], [[.1], [.1]])
tf.linalg.solve([[.1, .0], [.0, .1]], [.1, .1])
