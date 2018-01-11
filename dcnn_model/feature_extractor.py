from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.data_utils import get_file



def ExtractFeature():
    
    # Determine proper input shape
    input_shape = None
    
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=True)

    
    img_input = Input(shape=input_shape)
    

    # Block 1
    dcnn = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    dcnn = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(dcnn)
    dcnn = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(dcnn)

    # Block 2
    dcnn = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(dcnn)
    dcnn = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(dcnn)
    dcnn = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(dcnn)

    # Block 3
    dcnn = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(dcnn)
    dcnn = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(dcnn)
    dcnn = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(dcnn)
    dcnn = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(dcnn)

    # Block 4
    dcnn = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(dcnn)
    dcnn= Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(dcnn)
    dcnn = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(dcnn)
    dcnn = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(dcnn)

    # Block 5
    dcnn = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(dcnn)
    dcnn = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(dcnn)
    dcnn = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(dcnn)
    dcnn = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(dcnn)


    # Classification block
    dcnn = Flatten(name='flatten')(dcnn)
    dcnn = Dense(4096, name='fc6')(dcnn)
    dcnn = Activation('relu', name='fc6/relu')(dcnn)
    dcnn = Dense(4096, name='fc7')(dcnn)
    dcnn = Activation('relu', name='fc7/relu')(dcnn)
    dcnn = Dense(2622, name='fc8')(dcnn)
    dcnn = Activation('softmax', name='fc8/softmax')(dcnn)

   
    inputs = img_input
    
    # Create model
    model = Model(inputs, dcnn)  
    
    # load weights
    weights_path = get_file('vgg16_weights.h5','~/.keras',
                             cache_subdir='models')
        
    model.load_weights(weights_path, by_name=True)
       
    return model
