# -*- coding: utf-8 -*-

from collections import namedtuple
import os

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from PIL import Image


LayerStructure = namedtuple('LayerStructure', ['kernel', 'stride', 'channel'])

#ENCODER_LAYER_STRUCTS = [
#    LayerStructure(3, 2, 32),
#    LayerStructure(3, 2, 64),
#    LayerStructure(3, 2, 128),
#    LayerStructure(3, 2, 256)
#]
#DECODER_LAYER_STRUCTS = [
#    LayerStructure(3, 2, 128),
#    LayerStructure(3, 2, 64),
#    LayerStructure(3, 2, 32),
#    LayerStructure(3, 2, 3)
#]
#ENCODER_LAYER_STRUCTS = [
#    LayerStructure(3, 2, 128),
#    LayerStructure(3, 2, 256),
#]
#DECODER_LAYER_STRUCTS = [
#    LayerStructure(3, 2, 128),
#    LayerStructure(3, 2, 3)
#]
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01
DECAY = 0.99
IN_IMG_WIDTH = 144


def show_image(np_3d_array):
    img = Image.fromarray(np.uint8(np.clip(
        np_3d_array, a_max=1.0, a_min=0.0)*255))
    img.show()


def main():
    data_root = os.path.join('data', 'gen')
    train_img_list = [fn for fn in os.listdir(data_root) if fn.endswith('.png')]
    x_data = []
    for train_fn in train_img_list:
        img = image.load_img(os.path.join(data_root, train_fn))
        x = image.img_to_array(img) / 255.0
        x_data.append(x)
    x_data = np.array(x_data)
    print(x_data.shape)

    x_input = Input(shape=(IN_IMG_WIDTH, IN_IMG_WIDTH, 3))
    x = x_input

    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
               activation='relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = Conv2D(filters=8, kernel_size=3, strides=1, padding='same',
               activation='relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = Conv2D(filters=4, kernel_size=3, strides=1, padding='same',
               activation='relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)

    mid_layer = x
    mid_shape = K.int_shape(x)
    print(mid_shape)

    encoder_model = Model(x_input, mid_layer, name='encoder')
    encoder_model.summary()

    decoder_input = Input(shape=(mid_shape[1], mid_shape[2], mid_shape[3]))
    x = decoder_input

    x = Conv2DTranspose(filters=4, kernel_size=3, strides=1, padding='same',
                        activation='relu')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(filters=8, kernel_size=3, strides=1, padding='same',
                        activation='relu')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same',
                        activation='relu')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same',
                        activation='relu')(x)
    outputs = x

    decoder_model = Model(decoder_input, outputs, name='decoder')
    decoder_model.summary()

    ae_model = Model(x_input, decoder_model(encoder_model(x_input)),
                     name='auto_encoder')
    ae_model.summary()

    ae_model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE, decay=DECAY))
    ae_model.fit(x_data, x_data, batch_size=BATCH_SIZE, epochs=EPOCHS)

    x_decoded = ae_model.predict(x_data)
    show_image(x_data[0])
    show_image(x_decoded[0])
    print(x_decoded[0][1,1])


if __name__ == '__main__':
    main()
