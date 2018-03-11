import tensorflow as tf

def model(input):
    """Model function for CNN."""

    conv1 = tf.layers.conv2d(
        inputs=input,
        filters=16,
        kernel_size=[5, 5],
        strides=[4,4],
        padding="same",
        kernel_initializer=tf.initializers.truncated_normal(),
        bias_initializer=tf.initializers.constant(value=0.1, dtype=tf.float32),
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[5, 5],
        strides=[4,4],
        padding="same",
        kernel_initializer=tf.initializers.truncated_normal(),
        bias_initializer=tf.initializers.constant(value=0.1, dtype=tf.float32),
        activation=tf.nn.relu
    )

    reshape = tf.layers.flatten(conv2)

    dense = tf.layers.dense(
        inputs=reshape,
        units=1024,
        kernel_initializer=tf.initializers.truncated_normal(),
        bias_initializer=tf.initializers.constant(value=0.1, dtype=tf.float32),
        activation=tf.nn.relu
    )

    output = tf.layers.dense(
        inputs=dense,
        units=3,
        kernel_initializer=tf.initializers.truncated_normal(),
        bias_initializer=tf.initializers.constant(value=0.1, dtype=tf.float32),
        activation=tf.nn.sigmoid
    )

    return output
