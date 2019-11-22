import keras.backend as K

from keras.models import Model
from keras.layers import Input, Reshape, merge, Lambda, Activation, LeakyReLU
from keras.layers import Convolution3D, MaxPooling3D, Convolution2D
from keras.utils.data_utils import get_file

from custom_layers import BilinearUpsampling


C3D_WEIGHTS_URL = 'http://imagelab.ing.unimore.it/files/c3d_weights/w_up2_conv4_new.h5'

K.set_image_dim_ordering('th')

def CoarseSaliencyModel(input_shape, pretrained, branch=''):
    c, fr, h, w = input_shape
    assert h % 8 == 0 and w % 8 == 0, 'I think input shape should be divisible by 8. Should it?'

    # input_layers
    model_in = Input(shape=input_shape, name='input')
    # encoding net
    H = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(model_in)
    H = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(H)
    H = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(H)
    H = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(H)
    H = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(H)
    H = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(H)
    H = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1))(H)
    H = MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1), border_mode='valid', name='pool4')(H)

    H = Reshape(target_shape=(512, h // 8, w // 8))(H)  # squeeze out temporal dimension

    model_out = BilinearUpsampling(upsampling=8, name='{}_8x_upsampling'.format(branch))(H)
    model = Model(input=model_in, output=model_out, name='{}_coarse_model'.format(branch))

    if pretrained:
        weights_path = get_file('w_up2_conv4_new.h5', C3D_WEIGHTS_URL, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model


def SaliencyBranch(input_shape, c3d_pretrained, branch=''):
    c, fr, h, w = input_shape

    coarse_predictor = CoarseSaliencyModel(input_shape=(c, fr, h // 4, w // 4), pretrained=c3d_pretrained, branch=branch)

    ff_in = Input(shape=(c, 1, h, w), name='{}_input_ff'.format(branch))
    small_in = Input(shape=(c, fr, h // 4, w // 4), name='{}_input_small'.format(branch))
    crop_in = Input(shape=(c, fr, h // 4, w // 4), name='{}_input_crop'.format(branch))

    # coarse + refinement
    ff_last_frame = Reshape(target_shape=(c, h, w))(ff_in)  # remove singleton dimension
    coarse_h = coarse_predictor(small_in)
    coarse_h = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(coarse_h)
    coarse_h = BilinearUpsampling(upsampling=4, name='{}_4x_upsampling'.format(branch))(coarse_h)

    fine_h = merge([coarse_h, ff_last_frame], mode='concat', concat_axis=1, name='{}_full_frame_concat'.format(branch))
    fine_h = Convolution2D(32, 3, 3, border_mode='same', init='he_normal', name='{}_refine_conv1'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(16, 3, 3, border_mode='same', init='he_normal', name='{}_refine_conv2'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(8, 3, 3, border_mode='same', init='he_normal', name='{}_refine_conv3'.format(branch))(fine_h)
    fine_h = LeakyReLU(alpha=.001)(fine_h)
    fine_h = Convolution2D(1, 3, 3, border_mode='same', init='glorot_uniform', name='{}_refine_conv4'.format(branch))(fine_h)
    fine_out = Activation('relu')(fine_h)

    fine_out = Activation('linear', name='prediction_fine')(fine_out)

    # coarse on crop
    crop_h = coarse_predictor(crop_in)
    crop_h = Convolution2D(1, 3, 3, border_mode='same', init='glorot_uniform', name='{}_crop_final_conv'.format(branch))(crop_h)
    crop_out = Activation('relu', name='prediction_crop')(crop_h)

    model = Model(input=[ff_in, small_in, crop_in], output=[fine_out, crop_out],
                  name='{}_saliency_branch'.format(branch))

    return model