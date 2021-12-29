import tensorflow as tf
# final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
#                                                      time_major=False,
#                                                      return_alphas=True)
def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False,reuse=False):
    with tf.variable_scope('SimpleAttLayer', reuse=reuse) as scope:
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        temp=inputs * tf.expand_dims(alphas, -1)

        output = tf.reduce_sum(temp, 1)
        # output=tf.concat([temp[i,:,:] for i in range(2)],1)
        if not return_alphas:
            return output
        else:
            return output, alphas