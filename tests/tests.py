import tensorflow as tf
import tensorflow_probability as tfp

from lru_unofficial.src.lru_unofficial.tf.linear_recurrent_unit import LinearRecurrentUnitCell, ResLRUCell, LinearRecurrentUnitFFN, \
    ResLRUFFN, binary_operator_diag


def test_1():
    import time
    # set all seeds

    num_neurons = 210
    time_steps = 10000  # 10000
    batch_size = 32

    # num_neurons = 2
    # time_steps = 3
    # batch_size = 1

    test_forward_pass = False
    test_rnn_is_ffn = False
    test_res_rnn_is_ffn = True
    test_long_time = False
    test_reslru = False

    if test_forward_pass:
        input_tensor = tf.random.normal((batch_size, num_neurons))
        init_state = tf.random.normal((batch_size, num_neurons))
        lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
        out = lru(input_tensor, (init_state,))
        print(out[0].shape)

        reslru = ResLRUCell(num_neurons=num_neurons)
        out = reslru(input_tensor, (init_state,))
        print(out[0].shape)

        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons))
        lruffn = LinearRecurrentUnitFFN(num_neurons=num_neurons)
        out = lruffn(input_tensor)
        print(out.shape)

        lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
        lrurnn = tf.keras.layers.RNN(lru, return_sequences=True)
        out = lrurnn(input_tensor)
        print(out.shape)

    if test_rnn_is_ffn:
        # move parameters from lruffn to lrurnn
        print('=-.-=' * 100)
        print('hey!')
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons)) / 1

        lruffn = LinearRecurrentUnitFFN(num_neurons=num_neurons)
        lrucell = LinearRecurrentUnitCell(num_neurons=num_neurons)
        lrurnn = tf.keras.layers.RNN(lrucell, return_sequences=True)

        lruffn.build(input_tensor.shape)
        lrurnn.build(input_tensor.shape)

        print('-' * 100)

        lrurnn.set_weights(lruffn.get_weights())

        start_time = time.time()
        outrnn = lrurnn(input_tensor)
        rnn_time = time.time() - start_time
        print('outrnn')
        print(outrnn)

        start_time = time.time()
        outffn = lruffn(input_tensor)
        ffn_time = time.time() - start_time
        print('outffn')
        print(outffn)

        print('rnn time: ', rnn_time)
        print('ffn time: ', ffn_time)

        print(tf.reduce_sum(tf.square(outffn - outrnn) / tf.square(outffn)))


    if test_res_rnn_is_ffn:
        # move parameters from lruffn to lrurnn
        print('=-.-=' * 100)
        print('hey!')
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons)) / 1

        lruffn = ResLRUFFN(num_neurons=num_neurons, dop=0.)
        lrucell = ResLRUCell(num_neurons=num_neurons, dop=0.)
        lrurnn = tf.keras.layers.RNN(lrucell, return_sequences=True)

        lruffn.build(input_tensor.shape)
        lrurnn.build(input_tensor.shape)

        print('-' * 100)

        lrurnn.set_weights(lruffn.get_weights())

        start_time = time.time()
        outrnn = lrurnn(input_tensor)
        rnn_time = time.time() - start_time
        print('outrnn')
        print(outrnn)

        start_time = time.time()
        outffn = lruffn(input_tensor)
        ffn_time = time.time() - start_time
        print('outffn')
        print(outffn)

        print('rnn time: ', rnn_time)
        print('ffn time: ', ffn_time)

        print(tf.reduce_sum(tf.square(outffn - outrnn) / tf.square(outffn)))

    if test_long_time:
        # move parameters from lruffn to lrurnn
        print('=-.-=' * 100)
        print('hey!')
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons)) / .1

        lruffn = LinearRecurrentUnitFFN(num_neurons=num_neurons)

        outffn = lruffn(input_tensor)
        print(outffn)

    if test_reslru:
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons))
        reslru = ResLRUCell(num_neurons=num_neurons)
        # lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
        lrurnn = tf.keras.layers.RNN(reslru, return_sequences=True)
        out = lrurnn(input_tensor)
        print(out.shape)

    # ResLRUCell


def test_2():
    rec = 2
    batch_shape = 1
    rand = lambda shape=(rec,): tf.random.normal(shape)

    C_re = tf.Variable(rand((rec, rec)))
    B_re = tf.Variable(rand((rec, rec)))
    C_im = tf.Variable(rand((rec, rec)))
    B_im = tf.Variable(rand((rec, rec)))
    D = tf.Variable(rand())

    nu = tf.Variable(rand())
    theta = tf.Variable(rand())

    # Normalization
    gamma = tf.Variable(rand())

    inputs = tf.Variable(rand((batch_shape, rec)))
    states = [tf.Variable(rand((batch_shape, rec))), tf.Variable(rand((batch_shape, rec)))]

    with tf.GradientTape(persistent=True) as t:
        u = inputs
        x = tf.dtypes.complex(states[0], states[1])

        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(nu), theta))
        Lambda = tf.linalg.diag(lambda_)

        # turning floats to complex
        u_ = tf.cast(u, tf.complex64)
        x_ = tf.cast(x, tf.complex64)
        gamma_ = tf.cast(gamma, tf.complex64)
        B = tf.dtypes.complex(B_re, B_im)
        C = tf.dtypes.complex(C_re, C_im)

        # rnn operations
        new_u = gamma_ * u_ @ B
        new_x_ = tf.einsum('bi,ij->bj', x_, Lambda)

        x_ = new_x_ + new_u

        y = tf.math.real(x_ @ C) + D * u
        new_state = [tf.math.real(x_), tf.math.imag(x_)]

    grad = t.gradient(y, {'x': inputs, 'y': states})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])

    grad = t.gradient(x_, {'x': inputs, 'y': states})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])

    print('Derivative of new state vs inputs and states')
    grad = t.gradient(new_state, {'x': inputs, 'y': states})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])

    print('jacobian')
    # hs = [t.batch_jacobian(inputs, ns) for ns in new_state]
    hs = [t.batch_jacobian(ns, x) for ns in new_state]
    # hs = t.batch_jacobian(new_x_, x_)
    # hs = t.batch_jacobian(x_, new_x_)
    print(hs)


def test_speeds_scan():
    import operator, time

    # compare tfp.math.scan_associative() with cumsum
    # https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative
    # time
    random_tensor = tf.random.normal((200, 1000))
    start_time = time.time()
    a = tfp.math.scan_associative(operator.add, random_tensor)
    # print(a)
    print('Associative scan took:', time.time() - start_time)

    start_time = time.time()
    b = tf.cumsum(random_tensor)
    # print(b)
    # print(time.time() - start_time)
    print('Cumsum took:', time.time() - start_time)

    # are a == b?
    print(tf.reduce_mean(tf.square(a - b) / tf.square(a)))


def test_4():
    rec = 2
    batch_shape = 1
    time_steps = 3
    rand = lambda shape=(rec,): tf.random.normal(shape)

    lru = LinearRecurrentUnitCell(num_neurons=2)
    rnn = tf.keras.layers.RNN(lru, return_sequences=True, return_state=True)

    lru_2 = LinearRecurrentUnitCell(num_neurons=2)
    rnn_2 = tf.keras.layers.RNN(lru_2, return_sequences=True, return_state=True)

    inputs = tf.Variable(rand((batch_shape, time_steps, rec)))
    states = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]
    states_2 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]

    with tf.GradientTape(persistent=True) as t:
        all_outs = rnn(inputs, states)
        output, new_states = all_outs[0], all_outs[1:]

        all_outs = rnn_2(output, states_2)
        output_2, new_states_2 = all_outs[0], all_outs[1:]

        print(len(all_outs))

    grad = t.gradient(output, {'x': inputs, 'y': states})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])

    grad = t.gradient(new_states, {'x': inputs, 'y': states})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])

    print('jacobian')
    hs = [t.batch_jacobian(ns, inputs) for ns in new_states]
    print(hs)
    hs = [t.batch_jacobian(ns, s) for ns in new_states for s in states]
    print(hs)

    print('jacobian 2')
    hs = [t.batch_jacobian(ns, output) for ns in new_states_2]
    print(hs)
    hs = [t.batch_jacobian(ns, s) for ns in new_states_2 for s in states_2]
    print(hs)


def test_conv2scan():
    neurons = 2
    batch_shape = 1
    time_steps = 3
    rand = lambda shape=(neurons,): tf.random.normal(shape)
    rand = lambda shape=(neurons,): tf.ones(shape)

    u = rand((batch_shape, time_steps, neurons))
    lambda_ = rand()
    lambda_ = tf.repeat(tf.expand_dims(lambda_, axis=0), tf.shape(u)[1], axis=0)

    # repeat on axis 0
    time = tf.range(tf.shape(u)[1], dtype=tf.float32)
    time = tf.expand_dims(time, axis=-1)

    # exponentiate by time
    lambda_pow = lambda_ * time
    lambda_pow = tf.exp(lambda_pow)
    Lambda_pow = tf.linalg.diag(lambda_pow)

    lambda_minpow = -lambda_ * time
    lambda_minpow = tf.exp(lambda_minpow)
    Lambda_minpow = tf.linalg.diag(lambda_minpow)

    x_ = tf.einsum('bti,tij->btj', u, Lambda_minpow)
    x_ = tf.cumsum(x_, axis=1)
    x_conv = tf.einsum('bti,tij->btj', x_, Lambda_pow)

    # scan version
    lambda_scan = tf.expand_dims(tf.exp(lambda_), axis=0)
    _, x_scan = tfp.math.scan_associative(binary_operator_diag, (lambda_scan, u), axis=1)

    print('x_conv')
    print(x_conv)

    print('x_scan')
    print(x_scan)


if __name__ == '__main__':
    test_1()
    # test_lru_scan()
    # test_conv2scan()
