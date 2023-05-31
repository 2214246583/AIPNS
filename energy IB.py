import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.io as sio
import scipy.constants as C
from scipy.interpolate import griddata
from pyDOE import lhs
# from plotting import newfig,savefig
# from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# sys.path.insert(0, 'Utilities/')

np.random.seed(1234)
tf.set_random_seed(1234)


class NN:
    # Initialize the class
    def __init__(self, X_u, E, Ne, Te, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]
        self.E = E
        self.Ne = Ne
        self.Te = Te

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.loss1 = 0

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))  # 自动选择运行设备，记录设备指派情况

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])

        self.E_tf = tf.placeholder(tf.float32, shape=[None, self.E.shape[1]])
        self.Ne_tf = tf.placeholder(tf.float32, shape=[None, self.Ne.shape[1]])
        self.Te_tf = tf.placeholder(tf.float32, shape=[None, self.Te.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.E_pred, self.N_e_pred, self.T_e_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.E_tf - self.E_pred)) + tf.reduce_mean(tf.square(self.Ne_tf - self.N_e_pred)) + \
                    tf.reduce_mean(tf.square(self.Te_tf - self.T_e_pred)) + tf.reduce_mean(tf.square(self.f))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method='L-BFGS-B',
                options={'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol': 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        E = u[:, 0:1]
        Ne = u[:, 1:2]
        Te = u[:, 2:3]
        return E, Ne, Te

    def net_f(self, x, t):

        ue = 394.7
        De = 1579
        KB = C.k
        e = C.e

        E_pred_nor, Ne_pred, Te_pred = self.net_u(x, t)

        E_pred = E_pred_nor * 1000

        E_pred_x = tf.gradients(E_pred, x)[0]
        Ne_pred_t = tf.gradients(Ne_pred, t)[0]
        Ne_pred_x = tf.gradients(Ne_pred, x)[0]
        Ne_pred_xx = tf.gradients(Ne_pred_x, x)[0]
        Te_pred_t = tf.gradients(Te_pred, t)[0]
        Te_pred_x = tf.gradients(Te_pred, x)[0]
        Te_pred_xx = tf.gradients(Te_pred_x, x)[0]

        Lambda_e = 5/2 * KB * De * Ne_pred
        Nabla_Lambda_e = 5/2 * KB * De * Ne_pred_x
        Gamma_e = -ue * E_pred * Ne_pred - De * Ne_pred_x
        Nabla_Gamma_e = - ue * (E_pred_x * Ne_pred + E_pred * Ne_pred_x) - De * Ne_pred_xx
        Nabla_W = -Nabla_Lambda_e * Te_pred_x - Lambda_e * Te_pred_xx + 5/2 * KB * Te_pred_x * Gamma_e + 5/2 * KB * Te_pred * Nabla_Gamma_e

        f = 3/2 * KB * (Ne_pred * Te_pred_t + Te_pred * Ne_pred_t) + Nabla_W + e * Gamma_e * E_pred

        return f

    def callback(self, loss):
        print('Loss:', loss)
        self.loss1 = np.vstack((self.loss1, loss))
        sio.savemat('lossenergy_pnor.mat', {'loss': self.loss1})

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.E_tf: self.E, self.Ne_tf: self.Ne, self.Te_tf: self.Te, self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss],
                                 loss_callback=self.callback)
    def predict(self, X_star):

        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        E = self.sess.run(self.E_pred, tf_dict)
        Ne = self.sess.run(self.N_e_pred, tf_dict)
        Te = self.sess.run(self.T_e_pred, tf_dict)

        return E, Ne, Te

if __name__ == "__main__":

    N_u = 1280
    N_f = 10000
    layers = [2, 130, 130, 130, 130, 130, 130, 130, 130, 3]

    data = scipy.io.loadmat('energynor.mat')

    t = data['t_p'].flatten()[:, None]
    # t = t * 10**8
    x = data['x_p'].flatten()[:, None]
    # x = x * 1000
    E = data['Eg_p']
    Ne = data['Ne_p']
    Te = data['Te_p']

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    E_star = E.flatten()[:, None]
    Ne_star = Ne.flatten()[:, None]
    Te_star = Te.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    ii1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    E_i1 = E[0:1, :].T
    Ne_i1 = Ne[0:1, :].T
    Te_i1 = Te[0:1, :].T
    bb1 = np.hstack((X[:, 0:1], T[:, 0:1]))
    E_b1 = E[:, 0:1]
    Ne_b1 = Ne[:, 0:1]
    Te_b1 = Te[:, 0:1]
    bb2 = np.hstack((X[:, -1:], T[:, -1:]))
    E_b2 = E[:, -1:]
    Ne_b2 = Ne[:, -1:]
    Te_b2 = Te[:, -1:]

    X_u_train = np.vstack([ii1, bb1, bb2])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    E_train = np.vstack([E_i1, E_b1, E_b2])
    Ne_train = np.vstack([Ne_i1, Ne_b1, Ne_b2])
    Te_train = np.vstack([Te_i1, Te_b1, Te_b2])

    idx_u = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx_u, :]
    E_train = E_train[idx_u, :]
    Ne_train = Ne_train[idx_u, :]
    Te_train = Te_train[idx_u, :]

    model = NN(X_u_train, E_train, Ne_train, Te_train, X_f_train, layers, lb, ub)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    E, Ne, Te = model.predict(X_star)

    error_E = np.linalg.norm(E_star - E, 2) / np.linalg.norm(E_star, 2)
    error_Ne = np.linalg.norm(Ne_star - Ne, 2) / np.linalg.norm(Ne_star, 2)
    error_Te = np.linalg.norm(Te_star - Te, 2) / np.linalg.norm(Te_star, 2)
    print('Error E: %e' % (error_E))
    print('Error Ne: %e' % (error_Ne))
    print('Error Te: %e' % (error_Te))

    EG = griddata(X_star, E.flatten(), (X, T), method='cubic')
    NE = griddata(X_star, Ne.flatten(), (X, T), method='cubic')
    TE = griddata(X_star, Te.flatten(), (X, T), method='cubic')

    sio.savemat('energy_pnor.mat', {'E_pred': EG, 'Ne_pred': NE, 'Te_pred': TE})