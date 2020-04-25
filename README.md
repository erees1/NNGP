# NNGP

Implementation of *"[Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)"* by J.Lee, Y.Bahri et al. in Tensorflow 2.x

To see a recreation of some of the results from the paper see the notebook [here](https://nbviewer.jupyter.org/github/erees1/NNGP/blob/master/nngp_experiments.ipynb). Note this uses nbviewer as I used Bokeh for my plots which do not render in Github.

## How to use

The following snippet shows how to obtain a predicted mean and variance of the first 1000 items of MNIST using a gaussian process with a specified kernel. The gaussian process is implemented as a regression using the cholesky decompostion, see *"[Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)"* by C. E. Rasmussen & C. K. I. Williams,  pg.19 for details.

Note preprocessing is to ensure outputs are zero-mean regression targets.

```python
def prep_data(X, Y, dtype=tf.float64):
    X_flat = tf.convert_to_tensor(X.reshape(-1, 28*28)/255, dtype=dtype)
    Y_cat = keras.utils.to_categorical(Y)
    Y_cat = Y_cat - 0.1
    Y_cat = tf.convert_to_tensor(Y_cat, dtype=dtype)
    return X_flat, Y_cat


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

X_train_flat, Y_train_reg = prep_data(X_train, Y_train)
X_test_flat, Y_test_reg = prep_data(X_test, Y_test)


act = tf.nn.tanh

general_kernel = nngp.GeneralKernel(act,
                                    L=n_layers,
                                    n_g=401,
                                    n_v=400,
                                    n_c=400,
                                    u_max=10,
                                    s_max=100,
                                    sigma_b=sigma_b,
                                    sigma_w=sigma_w)

mu_bar, K_bar = nngp.GP_cholesky(X_train_flat[:1000], Y_train[:1000], X_test[:1000], K)

```