# encoding=utf-8
import tensorflow as tf

def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
        outputs = gamma * normalized + beta
    return outputs

def multi_head_attention(keys, values, querys, num_units, num_heads, scope='multihead_attention', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.nn.relu(
            tf.layers.dense(querys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        K = tf.nn.relu(
            tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        V = tf.nn.relu(
            tf.layers.dense(values, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(querys)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs)

        query_masks = tf.sign(tf.abs(tf.reduce_sum(querys, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += querys
        outputs = normalize(outputs)

        u1 = tf.layers.dense(outputs, num_units, use_bias=True)  # (N, T_q, C)
        u2 = tf.nn.relu(u1)
        outputs = tf.layers.dense(u2, num_units, use_bias=True)

    return outputs

def GCN(input_pic_pp_image, k, pp_adj, picture_length, picture_dimension, random_base, name):
    input_pic_pp_image = tf.reshape(input_pic_pp_image, [-1, picture_dimension])
    weight = tf.get_variable(
        name='w_gcn_' + name,
        shape=[picture_dimension, picture_dimension],
        initializer=tf.random_uniform_initializer(-random_base, random_base)
    )
    bisa = tf.get_variable(
        name='b_gcn_' + name,
        shape=[picture_dimension],
        initializer=tf.random_uniform_initializer(-random_base, random_base)
    )
    input_pic_pp_image = tf.matmul(input_pic_pp_image, weight)
    input_pic_pp_image = tf.reshape(input_pic_pp_image, [-1, k, picture_length * picture_dimension])
    output = tf.matmul(pp_adj, input_pic_pp_image)
    output = tf.reshape(output, [-1, picture_length, picture_dimension])
    output = output + bisa
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, k, picture_dimension])
    return output
        