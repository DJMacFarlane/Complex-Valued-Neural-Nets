import tensorflow as tf


class ComplexDense(tf.keras.layers.Layer):
  """
  A complex dense layer that takes complex/real inputs and outputs complex outputs.
  """
  def __init__(self, units=32, activation=None, use_bias=True, **kwargs):
      super(ComplexDense, self).__init__(**kwargs)
      self.units = units
      self.activation = tf.keras.activations.get(activation)
      self.use_bias = use_bias

  def build(self, input_shape):
      w_real_init = tf.random_normal_initializer()
      w_imag_init = tf.random_normal_initializer()
      
      w_real = w_real_init(shape=(input_shape[-1], self.units), dtype='float32')
      w_imag = w_imag_init(shape=(input_shape[-1], self.units), dtype='float32')
      
      self.w = tf.Variable(tf.complex(w_real, w_imag), trainable=True)

      if self.use_bias:
          b_real_init = tf.zeros_initializer()
          b_imag_init = tf.zeros_initializer()
          
          b_real = b_real_init(shape=(self.units,), dtype='float32')
          b_imag = b_imag_init(shape=(self.units,), dtype='float32')
          
          self.b = tf.Variable(tf.complex(b_real, b_imag), trainable=True)
      else:
          self.b = None

  def call(self, inputs): 
      if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
          inputs = tf.complex(inputs, tf.zeros_like(inputs))

      outputs = tf.matmul(inputs, self.w)

      if self.use_bias:
          outputs = outputs + self.b

      if self.activation is not None:
          outputs = self.activation(outputs)

      return outputs


class ComplexConv2D(tf.keras.layers.Layer):
    """
    A complex 2D convolution layer that takes complex/real inputs and outputs complex outputs.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        kernel_real_init = tf.random_normal_initializer()
        kernel_imag_init = tf.random_normal_initializer()
        
        self.kernel_real = self.add_weight(shape=(*self.kernel_size, input_shape[-1], self.filters),
                                           initializer=kernel_real_init,
                                           trainable=True)
        self.kernel_imag = self.add_weight(shape=(*self.kernel_size, input_shape[-1], self.filters),
                                           initializer=kernel_imag_init,
                                           trainable=True)

        if self.use_bias:
            bias_real_init = tf.zeros_initializer()
            bias_imag_init = tf.zeros_initializer()

            self.bias_real = self.add_weight(shape=(self.filters,),
                                             initializer=bias_real_init,
                                             trainable=True)
            self.bias_imag = self.add_weight(shape=(self.filters,),
                                             initializer=bias_imag_init,
                                             trainable=True)
        else:
            self.bias_real, self.bias_imag = None, None

    def call(self, inputs):
        if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
            inputs = tf.complex(inputs, tf.zeros_like(inputs))

        input_real = tf.math.real(inputs)
        input_imag = tf.math.imag(inputs)

        real_conv_real = tf.nn.conv2d(input_real, self.kernel_real, strides=self.strides, padding=self.padding)
        imag_conv_imag = tf.nn.conv2d(input_imag, self.kernel_imag, strides=self.strides, padding=self.padding)
        real_conv_imag = tf.nn.conv2d(input_real, self.kernel_imag, strides=self.strides, padding=self.padding)
        imag_conv_real = tf.nn.conv2d(input_imag, self.kernel_real, strides=self.strides, padding=self.padding)

        conv_real = real_conv_real - imag_conv_imag
        conv_imag = real_conv_imag + imag_conv_real

        outputs = tf.complex(conv_real, conv_imag)

        if self.use_bias:
            outputs = outputs + tf.complex(self.bias_real, self.bias_imag)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class ComplexConv1D(tf.keras.layers.Layer):
    """
    A complex 1D convolution layer that takes complex/real inputs and outputs complex outputs.
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=True, **kwargs):
        super(ComplexConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        kernel_real_init = tf.random_normal_initializer()
        kernel_imag_init = tf.random_normal_initializer()
        
        self.kernel_real = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.filters),
                                           initializer=kernel_real_init,
                                           trainable=True)
        self.kernel_imag = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.filters),
                                           initializer=kernel_imag_init,
                                           trainable=True)

        if self.use_bias:
            bias_real_init = tf.zeros_initializer()
            bias_imag_init = tf.zeros_initializer()

            self.bias_real = self.add_weight(shape=(self.filters,),
                                             initializer=bias_real_init,
                                             trainable=True)
            self.bias_imag = self.add_weight(shape=(self.filters,),
                                             initializer=bias_imag_init,
                                             trainable=True)
        else:
            self.bias_real, self.bias_imag = None, None

    def call(self, inputs):
        if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
            inputs = tf.complex(inputs, tf.zeros_like(inputs))

        input_real = tf.math.real(inputs)
        input_imag = tf.math.imag(inputs)

        real_conv_real = tf.nn.conv1d(input_real, self.kernel_real, stride=self.strides, padding=self.padding)
        imag_conv_imag = tf.nn.conv1d(input_imag, self.kernel_imag, stride=self.strides, padding=self.padding)
        real_conv_imag = tf.nn.conv1d(input_real, self.kernel_imag, stride=self.strides, padding=self.padding)
        imag_conv_real = tf.nn.conv1d(input_imag, self.kernel_real, stride=self.strides, padding=self.padding)

        conv_real = real_conv_real - imag_conv_imag
        conv_imag = real_conv_imag + imag_conv_real

        outputs = tf.complex(conv_real, conv_imag)

        if self.use_bias:
            outputs = outputs + tf.complex(self.bias_real, self.bias_imag)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class ComplexDropout(tf.keras.layers.Layer):
    """
    A complex dropout layer that takes complex inputs and performs dropout separately on the real and imaginary parts.
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(ComplexDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            input_shape = tf.shape(inputs)
            noise_shape = self._get_noise_shape(input_shape)

            real_part = tf.math.real(inputs)
            imag_part = tf.math.imag(inputs)

            dropped_real = tf.nn.dropout(real_part, rate=self.rate, noise_shape=noise_shape, seed=self.seed)
            dropped_imag = tf.nn.dropout(imag_part, rate=self.rate, noise_shape=noise_shape, seed=self.seed)

            outputs = tf.complex(dropped_real, dropped_imag)
            return outputs
        else:
            return inputs

    def _get_noise_shape(self, input_shape):
        if self.noise_shape is None:
            return self.noise_shape

        concrete_input_shape = input_shape.numpy()
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_input_shape[i] if value is None else value)

        return tuple(noise_shape)


# OFTEN MAKES THINGS WORSE - NEED TO FIGURE OUT WHY
class ComplexMaxPool2D(tf.keras.layers.Layer):
    """
    A complex max pooling layer that takes complex inputs and outputs complex outputs.
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(ComplexMaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.upper()

    def call(self, inputs):
        if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
            inputs = tf.complex(inputs, tf.zeros_like(inputs))

        # Calculate the magnitude of complex numbers
        magnitude = tf.math.abs(inputs)

        outputs, argmax = tf.nn.max_pool_with_argmax(magnitude, ksize=self.pool_size, strides=self.strides, padding=self.padding)

        shape = tf.shape(outputs)
        outputs = tf.reshape(tf.gather(tf.reshape(inputs, [-1]), argmax), shape)

        return outputs


# Coplex average pooling layer
class ComplexAvgPool2D(tf.keras.layers.Layer):
    """
    A complex average pooling layer that takes complex inputs and outputs complex outputs.
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(ComplexAvgPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.upper()

    def call(self, inputs):
        if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
            inputs = tf.complex(inputs, tf.zeros_like(inputs))

        # Calculate the real and imaginary parts of the inputs
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)

        # Calculate the average of the real and imaginary parts
        real_avg = tf.nn.avg_pool2d(real, ksize=self.pool_size, strides=self.strides, padding=self.padding)
        imag_avg = tf.nn.avg_pool2d(imag, ksize=self.pool_size, strides=self.strides, padding=self.padding)

        outputs = tf.complex(real_avg, imag_avg)

        return outputs


class ComplexLayerNormalization(tf.keras.layers.Layer):
    """
    A complex layer normalization layer that takes complex inputs and outputs complex outputs.
    """
    def __init__(self, axis=-1, epsilon=1e-12, **kwargs):
        super(ComplexLayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)

    def call(self, inputs):
        if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
            inputs = tf.complex(inputs, tf.zeros_like(inputs))

        mean = tf.math.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=self.axis, keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std

        outputs = outputs * self.gamma + self.beta

        return outputs


class ComplexUpSampling2D(tf.keras.layers.Layer):
    """
    A complex upsampling layer that takes complex inputs and outputs complex outputs.
    """
    def __init__(self, size=(2, 2), **kwargs):
        super(ComplexUpSampling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        if inputs.dtype in [tf.float16, tf.float32, tf.float64]:
            inputs = tf.complex(inputs, tf.zeros_like(inputs))

        # Calculate the real and imaginary parts of the inputs
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)

        # Upsample the real and imaginary parts
        real_upsampled = tf.keras.layers.UpSampling2D(size=self.size)(real)
        imag_upsampled = tf.keras.layers.UpSampling2D(size=self.size)(imag)

        outputs = tf.complex(real_upsampled, imag_upsampled)

        return outputs