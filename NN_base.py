import os, sys
import numpy as np
import pandas as pd
from collections import OrderedDict

#data_treatment

train = pd.read_csv("../../../input/lish-moa/train_features.csv")
test = pd.read_csv("../../../input/lish-moa/test_features.csv")
train_targets_scored = pd.read_csv("../../../input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("../../../input/lish-moa/train_targets_nonscored.csv")

train_cp24_D1 = train.query('cp_type == "trt_cp" & cp_time == 24 & cp_dose == "D1"')
train_cp24_D2 = train.query('cp_type == "trt_cp" & cp_time == 24 & cp_dose == "D2"')
train_cp48_D1 = train.query('cp_type == "trt_cp" & cp_time == 48 & cp_dose == "D1"')
train_cp48_D2 = train.query('cp_type == "trt_cp" & cp_time == 48 & cp_dose == "D2"')
train_cp72_D1 = train.query('cp_type == "trt_cp" & cp_time == 72 & cp_dose == "D1"')
train_cp72_D2 = train.query('cp_type == "trt_cp" & cp_time == 72 & cp_dose == "D2"')

test_cp24_D1 = test.query('cp_time == 24 & cp_dose == "D1"')
test_cp24_D2 = test.query('cp_time == 24 & cp_dose == "D2"')
test_cp48_D1 = test.query('cp_time == 48 & cp_dose == "D1"')
test_cp48_D2 = test.query('cp_time == 48 & cp_dose == "D2"')
test_cp72_D1 = test.query('cp_time == 72 & cp_dose == "D1"')
test_cp72_D2 = test.query('cp_time == 72 & cp_dose == "D2"')

X_train_cp24_D1 = train_cp24_D1.loc[:,'g-0':'c-99']
X_train_cp24_D2 = train_cp24_D2.loc[:,'g-0':'c-99']
X_train_cp48_D1 = train_cp48_D1.loc[:,'g-0':'c-99']
X_train_cp48_D2 = train_cp48_D2.loc[:,'g-0':'c-99']
X_train_cp72_D1 = train_cp72_D1.loc[:,'g-0':'c-99']
X_train_cp72_D2 = train_cp72_D2.loc[:,'g-0':'c-99']

X_test_cp24_D1 = test_cp24_D1.loc[:,'g-0':'c-99']
X_test_cp24_D2 = test_cp24_D2.loc[:,'g-0':'c-99']
X_test_cp48_D1 = test_cp48_D1.loc[:,'g-0':'c-99']
X_test_cp48_D2 = test_cp48_D2.loc[:,'g-0':'c-99']
X_test_cp72_D1 = test_cp72_D1.loc[:,'g-0':'c-99']
X_test_cp72_D2 = test_cp72_D2.loc[:,'g-0':'c-99']

index_X_train_cp24_D1 = X_train_cp24_D1.index.values
index_X_train_cp24_D2 = X_train_cp24_D2.index.values
index_X_train_cp48_D1 = X_train_cp48_D1.index.values
index_X_train_cp48_D2 = X_train_cp48_D2.index.values
index_X_train_cp72_D1 = X_train_cp72_D1.index.values
index_X_train_cp72_D2 = X_train_cp72_D2.index.values

index_X_test_cp24_D1 = X_test_cp24_D1.index.values
index_X_test_cp24_D2 = X_test_cp24_D2.index.values
index_X_test_cp48_D1 = X_test_cp48_D1.index.values
index_X_test_cp48_D2 = X_test_cp48_D2.index.values
index_X_test_cp72_D1 = X_test_cp72_D1.index.values
index_X_test_cp72_D2 = X_test_cp72_D2.index.values

train_scored_cp24_D1 = train_targets_scored.iloc[index_X_train_cp24_D1]
train_scored_cp24_D2 = train_targets_scored.iloc[index_X_train_cp24_D2]
train_scored_cp48_D1 = train_targets_scored.iloc[index_X_train_cp48_D1]
train_scored_cp48_D2 = train_targets_scored.iloc[index_X_train_cp48_D2]
train_scored_cp72_D1 = train_targets_scored.iloc[index_X_train_cp72_D1]
train_scored_cp72_D2 = train_targets_scored.iloc[index_X_train_cp72_D2]

Y_train_scored_cp24_D1 = train_scored_cp24_D1.loc[:, '5-alpha_reductase_inhibitor':'wnt_inhibitor']
Y_train_scored_cp24_D2 = train_scored_cp24_D2.loc[:, '5-alpha_reductase_inhibitor':'wnt_inhibitor']
Y_train_scored_cp48_D1 = train_scored_cp48_D1.loc[:, '5-alpha_reductase_inhibitor':'wnt_inhibitor']
Y_train_scored_cp48_D2 = train_scored_cp48_D2.loc[:, '5-alpha_reductase_inhibitor':'wnt_inhibitor']
Y_train_scored_cp72_D1 = train_scored_cp72_D1.loc[:, '5-alpha_reductase_inhibitor':'wnt_inhibitor']
Y_train_scored_cp72_D2 = train_scored_cp72_D2.loc[:, '5-alpha_reductase_inhibitor':'wnt_inhibitor']

columns = Y_train_scored_cp24_D1.columns.values
ids = test['sig_id']

train_x = [X_train_cp24_D1, X_train_cp24_D2, X_train_cp48_D1, X_train_cp48_D2, X_train_cp72_D1, X_train_cp72_D2]
train_t = [Y_train_scored_cp24_D1, Y_train_scored_cp24_D2, Y_train_scored_cp48_D1, Y_train_scored_cp48_D2, Y_train_scored_cp72_D1, Y_train_scored_cp72_D2]
test_x = [X_test_cp24_D1, X_test_cp24_D2, X_test_cp48_D1, X_test_cp48_D2, X_test_cp72_D1, X_test_cp72_D2]
train_index = [index_X_train_cp24_D1, index_X_train_cp24_D2, index_X_train_cp48_D1, index_X_train_cp48_D2, index_X_train_cp72_D1, index_X_train_cp72_D2]
test_index = [index_X_test_cp24_D1, index_X_test_cp24_D2, index_X_test_cp48_D1, index_X_test_cp48_D2, index_X_test_cp72_D1, index_X_test_cp72_D2]
names= ['cp24_D1', 'cp24_D2', 'cp48_D1', 'cp48_D2', 'cp72_D1', 'cp72_D2']

df_results = {}
df_results_train ={} #del

#common_functions

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-45)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す

    return grad

#common_layers

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class Selu:
    def __init__(self):
        self.mask = None
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x):
        scale = self.scale
        alpha = self.alpha
        f1 = scale * (alpha * (np.exp(x) - 1))
        f2 = scale * x
        out = np.where(x <= 0, f1, f2)
        return out

    def backward(self, dout):
        scale = self.scale
        alpha = self.alpha
        b1 = scale * (alpha * (np.exp(dout) - 1))
        b2 = scale * dout
        dx = np.where(dout <= 0, b1, b2)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx

#neural_network

class MultiLayerNetExtend:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='selu', weight_init_std='selu', weight_decay_lambda=0,
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'selu': Selu, 'relu': Relu,'sigmoid': Sigmoid}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('selu', 'relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def score(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        n = x.shape[0]
        m = 206
        y_min = np.minimum(y,1-10**(-15))
        y_max = np.maximum(y_min,10**(-15))
        c1 = t * np.log(y_max)
        c2 = (1-t) * np.log(1-y_max)
        c = c1 + c2
        s = np.sum(c)
        score = -1 * s / 206 / n
        return score

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads

#optimizers
class SGD:

    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:

    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]

class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

#train

class Trainer:
    def __init__(self, network, x_train, t_train, x_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_score_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            train_score = self.network.score(x_train, t_train)
            print(train_score)
            self.train_score_list.append(train_score)

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

#MoA_train_predict

for idx in range(1,4):
    df_result = pd.DataFrame()
    for i in range(6):

        x_train = train_x[i].values
        t_train = train_t[i].values
        x_test = test_x[i].values
        train_size = x_train.shape[0]
        index = test_index[i].flatten()

        # Dropuoutの有無、割り合いの設定 ========================
        use_dropout = True  # Dropoutなしのときの場合はFalseに
        dropout_ratio = 0.2
        # ====================================================

        network = MultiLayerNetExtend(input_size=872, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                      output_size=206, use_dropout=use_dropout, dropout_ration=dropout_ratio)
        trainer = Trainer(network, x_train, t_train, x_test,
                          epochs=30, mini_batch_size=100,
                          optimizer='adagrad', optimizer_param={'lr': 0.01})

        trainer.train()
        print('en' + str(en_idx) + ' : [' + names[i] + '] is Finished!!')

        result = network.predict(x_test, train_flg=False)

        df = pd.DataFrame(data=result, index=index, columns=columns, dtype='float')
        df_result = pd.concat([df_result, df], axis=0)

    df_fin = df_result.sort_index()
    df_results['df_result_en' + str(idx)] = df_fin

else:
    den1 = df_results['df_result_en1'].values
    den2 = df_results['df_result_en2'].values
    den3 = df_results['df_result_en3'].values

    den_all = (np.array(den1) + np.array(den2) + np.array(den3)) / 3

    index_test = test.index.values

    df_all = pd.DataFrame(data=den_all, index=index_test, columns=columns, dtype='float')
    df_all.insert(0, 'sig_id', ids)

    df_all.to_csv('submission.csv', sep=',', index=False)
