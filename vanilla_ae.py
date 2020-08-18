'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-11
  email        : bao.salirong@gmail.com
  Task         : Vanilla AutoEncoder based on Keras Model
'''

import tensorflow as tf

class Vanilla_Encoder(tf.keras.Model):
    def __init__(self, units = 32, name = "bao_encoder"):
        super(Vanilla_Encoder, self).__init__(name = name)

        self.flatten= tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units = units, activation = "relu")

    def call(self, inputs):
        x = inputs
        x = self.flatten(x)
        x = self.dense1(x)
        return x


class Vanilla_Decoder(tf.keras.Model):
    def __init__(self, input_shape, units = 32, name = "bao_decoder"):
        super(Vanilla_Decoder, self).__init__(name = name)

        h = input_shape[1]
        w = input_shape[2]
        c = input_shape[3]

        self.dense1 = tf.keras.layers.Dense(units = units, activation = "relu")
        self.dense2 = tf.keras.layers.Dense(units = h * w * c, activation="relu")
        self.reshape= tf.keras.layers.Reshape((w, h, c), name = "de_main_out")

    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return x


class Vanilla_ae(tf.keras.Model):
    def __init__(self, latent = 100, units = 32, input_shape = None):
        super(Vanilla_ae, self).__init__()

        self.encoder = Vanilla_Encoder(units = units)
        self.la_dense= tf.keras.layers.Dense(units = latent, activation="relu")
        self.decoder = Vanilla_Decoder(input_shape = input_shape, units = units)

    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.la_dense(x)
        x = self.decoder(x)
        return x

#------------------------------------------------------------------------------
def Vanilla_Ae(input_shape, latent, units):
    model = Vanilla_ae(latent = latent, units = units, input_shape = input_shape)
    model.build(input_shape = input_shape)
    return model
