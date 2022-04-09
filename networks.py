import tensorflow as tf


## Pixel normalization layer
class PixelNorm(tf.keras.layers.Layer):
    """"
       Implements the Pixel Normalization layer as opposed to BatchNormalization layer
       """""

    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def call(self, x, **kwargs):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)


# Function to fade in the new layers in progression
def fade_in(alpha, a, b):
    return alpha * a + (1.0 - alpha)


## The Loss function
def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)


## The Minibatch Standard Deviation layer
def minibatch_std(input_tensor, epsilon=1e-8):
    n, h, w, c = tf.shape(input_tensor)
    group_size = tf.minimum(4, n)
    x = tf.reshape(input_tensor, [group_size, -1, h, w, c])
    group_mean, group_var = tf.nn.moments(x, axes=(0), keepdims=False)
    group_std = tf.sqrt(group_var + epsilon)
    avg_std = tf.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
    x = tf.tile(avg_std, [group_size, h, w, 1])
    return tf.concat([input_tensor, x], axis=-1)


## Scaled Conv Layer for Equalized Learning rate
class EqualizedConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel=3, gain=2, **kwargs):
        super(EqualizedConv, self).__init__(**kwargs)
        self.w = None
        self.b = None
        self.scale = None
        self.in_channels = None
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel != 1

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.w = self.add_weight(
            shape=[self.kernel, self.kernel, self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.out_channels,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.kernel * self.kernel * self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs, **kwargs):
        if self.pad:
            x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        else:
            x = inputs
        output = (
                tf.nn.conv2d(x, self.scale * self.w, strides=1, padding="VALID") + self.b
        )
        return output


## Dense layer with Equalized Learning rates
class EqualizedDense(tf.keras.layers.Layer):
    def __init__(self, units, gain=2, learning_rate_multiplier: float=1, **kwargs):
        super(EqualizedDense, self).__init__(**kwargs)
        self.in_channels = None
        self.w = None
        self.b = None
        self.scale = None
        self.units = units
        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0 / self.learning_rate_multiplier
        )
        self.w = self.add_weight(
            shape=[self.in_channels, self.units],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs, **kwargs):
        output = tf.add(tf.matmul(inputs, self.scale * self.w), self.b)
        return output * self.learning_rate_multiplier


class AddNoise(tf.keras.layers.Layer):
    def __init__(self):
        super(AddNoise, self).__init__()
        self.b = None

    def build(self, input_shape):
        n, h, w, c = input_shape[0]
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(
            shape=[1, 1, 1, c], initializer=initializer, trainable=True, name="kernel"
        )

    def call(self, inputs, **kwargs):
        x, noise = inputs
        output = x + self.b * noise
        return output


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, gain=1, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.gain = gain
        self.w_channels = None
        self.x_channels = None
        self.dense_1 = None
        self.dense_2 = None

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = EqualizedDense(self.x_channels, gain=1)
        self.dense_2 = EqualizedDense(self.x_channels, gain=1)

    def call(self, inputs, **kwargs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb


def Mapping(num_stages, input_shape=512):
    z = tf.keras.layers.Input(shape=(input_shape))
    w = PixelNorm()(z)
    for i in range(8):
        w = EqualizedDense(512, learning_rate_multiplier=0.01)(w)
        w = tf.keras.layers.LeakyReLU(0.2)(w)
    w = tf.tile(tf.expand_dims(w, 1), (1, num_stages, 1))
    return tf.keras.Model(z, w, name="mapping")