# original https://arxiv.org/pdf/2303.06349.pdf
# https://github.com/NicolasZucchet/minimal-LRU/


import tensorflow as tf
import tensorflow_probability as tfp

from keras.initializers.initializers_v2 import VarianceScaling

from lru_unofficial.tf.geglu import GEGLU


# Parallel scan operations
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j

    return A_j * A_i, A_j * b_i + b_j



class HalfGlorotNormal(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(
            scale=1 / 2, mode="fan_avg", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


class ComplexGlorotNormal(tf.keras.initializers.Initializer):

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        r = tf.keras.initializers.GlorotNormal(self.seed)(shape, dtype=tf.float32) / tf.sqrt(2.)
        i = tf.keras.initializers.GlorotNormal(self.seed)(shape, dtype=tf.float32) / tf.sqrt(2.)
        return tf.dtypes.complex(r, i)

    def get_config(self):  # To support serialization
        return {'seed': self.seed}


class InitFromTensor(tf.keras.initializers.Initializer):

    def __init__(self, tensor):
        self.tensor = tensor.numpy()

    def __call__(self, shape, dtype=None):
        return self.tensor

    def get_config(self):  # To support serialization
        return {'tensor': self.tensor}


class LinearRecurrentUnitCell(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, d_hidden=None,
                 rmax=.99, rmin=.4, reduced_phase=True, locked_gamma=False, **kwargs):
        super().__init__(**kwargs)
        if d_hidden is None:
            d_hidden = 2 * num_neurons

        self.init_args = dict(num_neurons=num_neurons, rmax=rmax, rmin=rmin, d_hidden=d_hidden,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase)
        self.__dict__.update(self.init_args)

        self.state_size = (d_hidden, d_hidden)

    def build(self, input_shape):

        n_in = input_shape[-1]
        n_rec = self.num_neurons

        self.C_re = self.add_weight(shape=(self.d_hidden, n_rec), initializer=HalfGlorotNormal(), name='C_re')
        self.B_re = self.add_weight(shape=(n_rec, self.d_hidden), initializer=HalfGlorotNormal(), name='B_re')
        self.C_im = self.add_weight(shape=(self.d_hidden, n_rec), initializer=HalfGlorotNormal(), name='C_im')
        self.B_im = self.add_weight(shape=(n_rec, self.d_hidden), initializer=HalfGlorotNormal(), name='B_im')
        self.D = self.add_weight(shape=(n_rec,), initializer=tf.keras.initializers.RandomNormal(stddev=1), name='D')

        numax = tf.math.log(-tf.math.log(self.rmin))
        numin = tf.math.log(-tf.math.log(self.rmax))
        nuinit = tf.keras.initializers.RandomUniform(minval=numin, maxval=numax, seed=None)
        self.nu = self.add_weight(shape=(self.d_hidden,), initializer=nuinit, name='lambda_nu')

        if self.reduced_phase:
            theta_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=3.14 / 10, seed=None)
        else:
            theta_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=2 * 3.14, seed=None)

        self.theta = self.add_weight(shape=(self.d_hidden,), initializer=theta_initializer, name='lambda_theta')

        # Normalization
        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(self.nu), self.theta))
        gamma_log = tf.math.log(tf.sqrt(1 - tf.abs(lambda_) ** 2))
        if self.locked_gamma:
            self.gamma_log = gamma_log
        else:
            gamma_initializer = InitFromTensor(gamma_log)
            self.gamma_log = self.add_weight(shape=(self.d_hidden,), initializer=gamma_initializer, name='gamma_log')

        if input_shape[-1] != self.num_neurons:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')
            self.adapter.build(input_shape)
        else:
            self.adapter = lambda x: x

        self.built = True

    def call(self, inputs, states, **kwargs):
        # self.call_simpler(inputs, states, **kwargs)

        if not kwargs['training'] is None:
            tf.keras.backend.set_learning_phase(kwargs['training'])

        u = self.adapter(inputs)
        x = tf.dtypes.complex(states[0], states[1])

        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(self.nu), self.theta))
        Lambda = tf.linalg.diag(lambda_)

        # turning floats to complex
        u_ = tf.cast(u, tf.complex64)
        gamma_ = tf.cast(tf.exp(self.gamma_log), tf.complex64)
        B = tf.dtypes.complex(self.B_re, self.B_im)
        C = tf.dtypes.complex(self.C_re, self.C_im)

        # rnn operations
        new_u = gamma_ * (u_ @ B)

        new_x_ = tf.einsum('bi,ij->bj', x, Lambda)

        x_ = new_x_ + new_u
        y = tf.math.real(x_ @ C) + self.D * u
        output = y
        new_state = [tf.math.real(x_), tf.math.imag(x_)]
        return output, new_state


class ResLRUCell(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, d_hidden=None,
                 rmax=.99, rmin=.4, reduced_phase=True, dop=.1, locked_gamma=False, linear_input=False, **kwargs):
        super().__init__(**kwargs)

        if d_hidden is None:
            d_hidden = 2 * num_neurons

        self.init_args = dict(num_neurons=num_neurons, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase, dop=dop,
                              linear_input=linear_input)
        self.__dict__.update(self.init_args)

        self.state_size = (d_hidden, d_hidden)
        self.lru = LinearRecurrentUnitCell(
            num_neurons=num_neurons, rmax=rmax, rmin=rmin, reduced_phase=reduced_phase, locked_gamma=locked_gamma
        )

        self.norm = tf.keras.layers.LayerNormalization()
        # self.norm = lambda x: x
        self.glu = GEGLU(num_neurons, num_neurons, activation='sigmoid', comments='onlyglu')
        self.gelu = tf.keras.layers.Activation('gelu')
        self.dropout_1 = tf.keras.layers.Dropout(dop)
        self.dropout_2 = tf.keras.layers.Dropout(dop)
        # self.dropout_1 = lambda x: x
        # self.dropout_2 = lambda x: x
        if linear_input:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')

    def build(self, input_shape):

        new_input_shape = input_shape[:-1] + (self.num_neurons,)
        self.lru.build(new_input_shape)
        self.norm.build(new_input_shape)
        self.glu.build(new_input_shape)
        self.glu.w_1.build(new_input_shape)
        self.glu.w_3.build(new_input_shape)

        if input_shape[-1] != self.num_neurons:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')
            self.adapter.build(input_shape)
        else:
            self.adapter = lambda x: x

        self.built = True

    def call(self, inputs, states, **kwargs):
        if not kwargs['training'] is None:
            tf.keras.backend.set_learning_phase(kwargs['training'])

        adapted = self.adapter(inputs)
        u = self.norm(adapted)

        y, new_states = self.lru.call(u, states, **kwargs)
        y = self.gelu(y)
        y = self.dropout_1(y)
        y = self.glu(y)
        y = self.dropout_2(y)

        output = y + adapted
        return output, new_states


# FFN version of LinearRecurrentUnitCell
class LinearRecurrentUnitFFN(tf.keras.layers.Layer):
    # 6.2x faster than the recurrent cell on a sequence of length 10K on my laptop

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, d_hidden=None,
                 rmax=.99, rmin=.4, reduced_phase=True, locked_gamma=False, **kwargs):
        super().__init__(**kwargs)

        if d_hidden is None:
            d_hidden = 2 * num_neurons
        self.init_args = dict(num_neurons=num_neurons, d_hidden=d_hidden, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase)
        self.__dict__.update(self.init_args)

        self.state_size = (d_hidden, d_hidden)

    def build(self, input_shape):
        n_in = input_shape[-1]
        n_rec = self.num_neurons

        self.C_re = self.add_weight(shape=(self.d_hidden, n_rec), initializer=HalfGlorotNormal(), name='C_re')
        self.B_re = self.add_weight(shape=(n_rec, self.d_hidden), initializer=HalfGlorotNormal(), name='B_re')
        self.C_im = self.add_weight(shape=(self.d_hidden, n_rec), initializer=HalfGlorotNormal(), name='C_im')
        self.B_im = self.add_weight(shape=(n_rec, self.d_hidden), initializer=HalfGlorotNormal(), name='B_im')
        self.D = self.add_weight(shape=(n_rec,), initializer=tf.keras.initializers.RandomNormal(stddev=1), name='D')

        numax = tf.math.log(-tf.math.log(self.rmin))
        numin = tf.math.log(-tf.math.log(self.rmax))
        nuinit = tf.keras.initializers.RandomUniform(minval=numin, maxval=numax, seed=None)
        self.nu = self.add_weight(shape=(self.d_hidden,), initializer=nuinit, name='lambda_nu')

        if self.reduced_phase:
            theta_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=3.14 / 10, seed=None)
        else:
            theta_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=2 * 3.14, seed=None)

        self.theta = self.add_weight(shape=(self.d_hidden,), initializer=theta_initializer, name='lambda_theta')

        # Normalization
        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(self.nu), self.theta))
        gamma_log = tf.math.log(tf.sqrt(1 - tf.abs(lambda_) ** 2))
        if self.locked_gamma:
            self.gamma_log = gamma_log
        else:
            gamma_initializer = InitFromTensor(gamma_log)
            self.gamma_log = self.add_weight(shape=(self.d_hidden,), initializer=gamma_initializer, name='gamma_log')

        if input_shape[-1] != self.num_neurons:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')
            self.adapter.build(input_shape)
        else:
            self.adapter = lambda x: x

        self.built = True

    def call(self, inputs, training=None):

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        u = self.adapter(inputs)

        lambda_ = tf.dtypes.complex(-tf.exp(self.nu), self.theta)
        lambda_ = tf.repeat(tf.expand_dims(lambda_, axis=0), tf.shape(u)[1], axis=0)

        # turning floats to complex
        u_ = tf.cast(u, tf.complex64)
        gamma_ = tf.cast(tf.exp(self.gamma_log), tf.complex64)
        B = tf.dtypes.complex(self.B_re, self.B_im)
        C = tf.dtypes.complex(self.C_re, self.C_im)

        # rnn operations
        new_u = gamma_ * (u_ @ B)

        lambda_scan = tf.expand_dims(tf.exp(lambda_), axis=0)
        _, x_ = tfp.math.scan_associative(binary_operator_diag, (lambda_scan, new_u), axis=1)

        y = tf.math.real(x_ @ C) + self.D * u
        output = y
        return output



class ResLRUFFN(tf.keras.layers.Layer):
    #  90.62/0.24x=377x faster than the recurrent cell on a sequence of length 10K on a GPU

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, d_hidden=None,
                 rmax=.99, rmin=.4, reduced_phase=True, dop=.1, locked_gamma=False, linear_input=True, **kwargs):
        super().__init__(**kwargs)

        if d_hidden is None:
            d_hidden = 2 * num_neurons

        self.init_args = dict(num_neurons=num_neurons, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase, dop=dop,
                              linear_input=linear_input)
        self.__dict__.update(self.init_args)

        self.state_size = (d_hidden, d_hidden)
        self.lru = LinearRecurrentUnitFFN(
            num_neurons=num_neurons, rmax=rmax, rmin=rmin, reduced_phase=reduced_phase, locked_gamma=locked_gamma
        )

        self.norm = tf.keras.layers.LayerNormalization()

        self.glu = GEGLU(num_neurons, num_neurons, activation='sigmoid', comments='onlyglu')
        self.gelu = tf.keras.layers.Activation('gelu')
        self.dropout_1 = tf.keras.layers.Dropout(dop)
        self.dropout_2 = tf.keras.layers.Dropout(dop)

        if linear_input:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')

    def build(self, input_shape):

        new_input_shape = input_shape[:-1] + (self.num_neurons,)
        self.lru.build(new_input_shape)
        self.norm.build(new_input_shape)
        self.glu.build(new_input_shape)
        self.glu.w_1.build(new_input_shape)
        self.glu.w_3.build(new_input_shape)

        if input_shape[-1] != self.num_neurons:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')
            self.adapter.build(input_shape)
        else:
            self.adapter = lambda x: x

        self.built = True

    def call(self, inputs, **kwargs):
        if not kwargs['training'] is None:
            tf.keras.backend.set_learning_phase(kwargs['training'])

        adapted = self.adapter(inputs)
        u = self.norm(adapted)

        y = self.lru(u)
        y = self.gelu(y)
        y = self.dropout_1(y)
        y = self.glu(y)
        y = self.dropout_2(y)

        output = y + adapted
        return output



