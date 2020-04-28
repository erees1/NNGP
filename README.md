# NNGP

Implementation of *"[Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)"* by J.Lee, Y.Bahri et al. in Tensorflow 2.x

To see a recreation of some of the results from the paper see the notebook [here](https://nbviewer.jupyter.org/github/erees1/NNGP/blob/master/nngp_experiments.ipynb). Note this uses nbviewer as I used Bokeh for my plots which do not render in Github.

## Project Structure

There is one main module: [`nngp.py`](./nngp.py) which contains the code for creating kernels and running Gaussian process regression. The only other module is [`neural_net.py`](./neural_net.py) which has the code to build the nerual network that approximates to the Gaussian process.

## How to use

The following snippet shows how to obtain a predicted mean and variance of the first 100 items of MNIST using a gaussian process with a specified kernel. The gaussian process is implemented as a regression using the cholesky decompostion, see *"[Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)"* by C. E. Rasmussen & C. K. I. Williams,  pg.19 for details.

Note preprocessing is to ensure outputs are zero-mean regression targets.

When instantiating a `GeneralKernel()` it checks in the `save_loc` folder to see if a pre computed grid is available - in this repo I have saved the results for `relu` and `tanh` with  `n_g=401`, `n_v=400`, `n_c=400`, `u_max=10` and `s_max=100`.

```python
from NNGP import nngp
import tensorflow as tf
from tensorflow import keras
import numpy as np

def prep_data(X, Y, dtype=tf.float64):
    X_flat = tf.convert_to_tensor(X.reshape(-1, 28*28)/255, dtype=dtype)
    Y_cat = keras.utils.to_categorical(Y)
    Y_cat = Y_cat - 0.1
    Y_cat = tf.convert_to_tensor(Y_cat, dtype=dtype)
    return X_flat, Y_cat


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

X_train_flat, Y_train_reg = prep_data(X_train, Y_train)
X_test_flat, Y_test_reg = prep_data(X_test, Y_test)


act = tf.nn.relu
sigma_b = 0.1**0.5
sigma_w = 1.6**0.5
n_layers = 3
n_data = 100


general_kernel = nngp.GeneralKernel(act,
                                    L=n_layers,
                                    n_g=401,
                                    n_v=400,
                                    n_c=400,
                                    u_max=10,
                                    s_max=100,
                                    sigma_b=sigma_b,
                                    sigma_w=sigma_w,
                                    save_loc='NNGP/kernel_grids')

mu_bar, K_bar = nngp.GP_cholesky(X_train_flat[:n_data], Y_train_reg[:n_data], X_test_flat[:n_data], general_kernel.K)

predictions = tf.argmax(mu_bar, axis=-1)
acc = np.equal(predictions.numpy(), Y_test[:n_data]).sum()/len(Y_test[:n_data])
print(acc)

```
