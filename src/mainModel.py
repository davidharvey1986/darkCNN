from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from .globalVariables import *

from .model_helpers import *

def simpleModel( nClasses , dropout=0., finalLayer=256, momentum=0.9, learning_rate=3e-6, name='CNN', nAttributes=0 ):

    
    mainCNN = tf.keras.Sequential([
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten()])        


    if nAttributes > 0:
        #Not yet tested,
        atrtribute_layers= tf.keras.Sequential([
                layers.Dense(finalLayer, activation='relu'),
                layers.MaxPooling1D(pool_size=2, padding='same'),
                layers.Flatten() ])
        
   
  

    finalLayers = tf.keras.Sequential([
          layers.Dense(128, activation='relu'),
          layers.Dense(nClasses) ])
    

        
        
    finalModel = tf.keras.Sequential([
        mainCNN, finalLayers])
        

    finalModel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return finalModel
    

    
def DIBARE(input_shape=(256,256,1),classes=10,num_layersA=4,num_layersB=7,num_layersC=3,feature_dropout=.2,bn_momentum=.99,leak=0.,FC1=0,FC2=0,FC1_dropout=0.,FC2_dropout=0.,bn=True):

    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,bn_momentum=bn_momentum,leak=leak,selection='DI-Bare',bn=bn)
    #Stack of InceptionA layers
    for index in range(0,num_layersA):
        x = InceptionA(x,bn_momentum=bn_momentum,leak=leak,bn=bn,selection='DI-Bare')

    x = ReductionA(x,bn_momentum=bn_momentum,leak=leak,bn=bn,selection='DI-Bare')

    #Stack of InceptionB layers
    for index in range(0,num_layersB):
        x = InceptionB(x,bn_momentum=bn_momentum,leak=leak,bn=bn,selection='DI-Bare')

    x = ReductionB(x,bn_momentum=bn_momentum,leak=leak,bn=bn,selection='DI-Bare')

    #Stack of InceptionC layers
    for index in range(0,num_layersC):
        x = InceptionC(x,bn_momentum=bn_momentum,leak=leak,bn=bn,selection='DI-Bare')

    #Top layer Inception-style, global averaging
    x = GlobalAveragePooling2D()(x)

    #Dropout and softmax producing the predictions
    x = Dropout(feature_dropout)(x)

    #Fully connected top if wanted
    if(FC1 > 0):
        x = Dense(FC1)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC1_dropout)(x)
    if(FC2 > 0):
        x = Dense(FC2)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC2_dropout)(x)

    #Softmax
    x = Dense(classes, activation='softmax')(x)

    #Constructing model and returning it
    model = Model(img_input,x,name='DIBARE')
    return model


def InceptionV4(input_shape=(256,256,1),classes=10,num_layersA=4,num_layersB=7,num_layersC=3,feature_dropout=.2,bn_momentum=.99,leak=0.,FC1=0,FC2=0,FC1_dropout=0.,FC2_dropout=0.,final_softmax=True):
    
    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,bn_momentum=bn_momentum,leak=leak)

    #Stack of Inception-A layers
    for index in range(0,num_layersA):
        x = InceptionA(x,bn_momentum=bn_momentum,leak=leak)

    #Reduction-A layer
    x = ReductionA(x,bn_momentum=bn_momentum,leak=leak)

    #Stack of Inception-B layers
    for index in range(0,num_layersB):
        x = InceptionB(x,bn_momentum=bn_momentum,leak=leak)

    #Reduction-B layer
    x = ReductionB(x,bn_momentum=bn_momentum,leak=leak)

    #Stack of Inception-C layers
    for index in range(0,num_layersC):
        x = InceptionC(x,bn_momentum=bn_momentum,leak=leak)

    #Top layer Inception-style, global averaging
    x = GlobalAveragePooling2D()(x)

    #Dropout and softmax producing the predictions
    x = Dropout(feature_dropout)(x)

    #Fully connected top if wanted
    if(FC1 > 0):
        x = Dense(FC1)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC1_dropout)(x)
    if(FC2 > 0):
        x = Dense(FC2)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC2_dropout)(x)

    #Softmax
    if(final_softmax):
        x = Dense(classes, activation='softmax')(x)
    else:
        x = Dense(classes)(x)

    #Constructing model and returning it
    model = Model(img_input,x,name='Inception-v4mod')
    return model

def InceptionV4small(input_shape=(256,256,1),classes=10,num_layersA=4,num_layersB=7,num_layersC=3,feature_dropout=.2,bn_momentum=.99,leak=0.,FC1=0,FC2=0,FC1_dropout=0.,FC2_dropout=0.):
    
    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-small')

    #Stack of Inception-A layers
    for index in range(0,num_layersA):
        x = InceptionA(x,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-small')

    #Reduction-A layer
    x = ReductionA(x,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-small')

    #Stack of Inception-B layers
    for index in range(0,num_layersB):
        x = InceptionB(x,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-small')

    #Reduction-B layer
    x = ReductionB(x,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-small')

    #Stack of Inception-C layers
    for index in range(0,num_layersC):
        x = InceptionC(x,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-small')

    #Top layer Inception-style, global averaging
    x = GlobalAveragePooling2D()(x)

    #Dropout and softmax producing the predictions
    x = Dropout(feature_dropout)(x)

    #Fully connected top if wanted
    if(FC1 > 0):
        x = Dense(FC1)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC1_dropout)(x)
    if(FC2 > 0):
        x = Dense(FC2)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC2_dropout)(x)

    #Softmax
    x = Dense(classes, activation='softmax')(x)

    #Constructing model and returning it
    model = Model(img_input,x,name='Inception-v4small')
    return model

def InceptionV4wide(input_shape=(256,256,1),classes=10,num_layersA=4,num_layersB=7,num_layersC=3,feature_dropout=.2,bn_momentum=.99,leak=0.,FC1=0,FC2=0,FC1_dropout=0.,FC2_dropout=0.):
    
    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,selection='Inception-v4-wide',bn_momentum=bn_momentum,leak=leak)

    #Stack of Inception-A layers
    for index in range(0,num_layersA):
        x = InceptionA(x,bn_momentum=bn_momentum,leak=leak)

    #Reduction-A layer
    x = ReductionA(x,bn_momentum=bn_momentum,leak=leak)

    #Stack of Inception-B layers
    for index in range(0,num_layersB):
        x = InceptionB(x,bn_momentum=bn_momentum,leak=leak)

    #Reduction-B layer
    x = ReductionB(x,bn_momentum=bn_momentum,leak=leak)

    #Stack of Inception-C layers
    for index in range(0,num_layersC):
        x = InceptionC(x,bn_momentum=bn_momentum,leak=leak,selection='Inception-v4-wide')

    #Top layer Inception-style, global averaging
    x = GlobalAveragePooling2D()(x)

    #Dropout and softmax producing the predictions
    x = Dropout(feature_dropout)(x)

    #Fully connected top if wanted
    if(FC1 > 0):
        x = Dense(FC1)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC1_dropout)(x)
    if(FC2 > 0):
        x = Dense(FC2)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC2_dropout)(x)

    #Softmax
    x = Dense(classes, activation='softmax')(x)

    #Constructing model and returning it
    model = Model(img_input,x,name='Inception-v4mod')
    return model


def InceptionResNetV1(input_shape=(256,256,1),classes=10,num_layersA=5,num_layersB=10,num_layersC=5,drop_out_strength=.2,bn_momentum=.99, leak=0.):
    
    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,selection='Inception-ResNet-v1',bn_momentum=bn_momentum,leak = leak)

    #Stack of Inception-A layers
    for index in range(0,num_layersA-1):
        x = InceptionA(x,selection='Inception-ResNet-v1',bn_momentum=bn_momentum, leak = leak)
    x = InceptionA(x,selection='Inception-ResNet-v1',leak=-1.,bn_momentum=bn_momentum)

    #Reduction-A layer
    x = ReductionA(x,selection='Inception-ResNet-v1',bn_momentum=bn_momentum, leak = leak)

    #Stack of Inception-B layers
    for index in range(0,num_layersB-1):
        x = InceptionB(x,selection='Inception-ResNet-v1',bn_momentum=bn_momentum, leak = leak)
    x = InceptionB(x,selection='Inception-ResNet-v1',leak=-1.,bn_momentum=bn_momentum)

    #Reduction-B layer
    x = ReductionB(x,selection='Inception-ResNet-v1',bn_momentum=bn_momentum)

    #Stack of Inception-C layers
    for index in range(0,num_layersC-1):
        x = InceptionC(x,selection='Inception-ResNet-v1',bn_momentum=bn_momentum)
    x = InceptionC(x,selection='Inception-ResNet-v1',leak=-1.,bn_momentum=bn_momentum)

    #Top layer Inception-style, global averaging
    x = GlobalAveragePooling2D()(x)

    #Dropout and softmax producing the predictions
    x = Dropout(drop_out_strength)(x)
    x = Dense(classes, activation='softmax')(x)

    #Constructing model and returning it
    model = Model(img_input,x,name='Inception-ResNet-v1')
    return model

def InceptionResNetV2(input_shape=(256,256,1),classes=10,num_layersA=5,num_layersB=10,num_layersC=5,drop_out_strength=.2,bn_momentum=.99, leak = 0.):
    
    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,bn_momentum=bn_momentum, leak = leak)

    #Stack of Inception-A layers
    for index in range(0,num_layersA-1):
        x = InceptionA(x,selection='Inception-ResNet-v2',bn_momentum=bn_momentum, leak = leak)
    x = InceptionA(x,selection='Inception-ResNet-v2',leak=-1.,bn_momentum=bn_momentum)

    #Reduction-A layer
    x = ReductionA(x,selection='Inception-ResNet-v2',bn_momentum=bn_momentum, leak = leak)

    #Stack of Inception-B layers
    for index in range(0,num_layersB-1):
        x = InceptionB(x,selection='Inception-ResNet-v2',bn_momentum=bn_momentum, leak = leak)
    x = InceptionB(x,selection='Inception-ResNet-v2',leak=-1.,bn_momentum=bn_momentum)

    #Reduction-B layer
    x = ReductionB(x,selection='Inception-ResNet-v2',bn_momentum=bn_momentum, leak = leak)

    #Stack of Inception-C layers
    for index in range(0,num_layersC-1):
        x = InceptionC(x,selection='Inception-ResNet-v2',bn_momentum=bn_momentum, leak = leak)
    x = InceptionC(x,selection='Inception-ResNet-v2',leak=-1.,bn_momentum=bn_momentum)

    #Top layer Inception-style, global averaging
    x = GlobalAveragePooling2D()(x)

    #Dropout and softmax producing the predictions
    x = Dropout(drop_out_strength)(x)
    x = Dense(classes, activation='softmax')(x)

    #Constructing model and returning it
    model = Model(img_input,x,name='Inception-ResNet-v2')
    return model

def dark_inception(input_shape=(256,256,1),classes=4,first_layer_channels=4,second_layer_channels=12,third_layer_channels=32,first_fc_layer=1024,second_fc_layer=256,feature_dropout=.5,first_fc_dropout=.5,second_fc_dropout=.5,bn=False,bn_momentum=.99):

    #Input layer
    img_input = Input(shape=input_shape)

###---------------------------------START INCEPTION LAYER 0

    #pooling branch
    branch_pool = AveragePooling2D((4, 4), strides=(1, 1), padding='same')(img_input)
    
    #1x1 branch
    branch1x1 = Conv2D(first_layer_channels, (1,1),padding='same')(img_input)
    if(bn):
        branch1x1 = BatchNormalization(momentum=bn_momentum,scale=False)(branch1x1)
    branch1x1 = LeakyReLU(0.03)(branch1x1)

    #3x3 branch
    branch3x3 = Conv2D(first_layer_channels, (3,3),padding='same')(img_input)
    if(bn):
        branch3x3 = BatchNormalization(momentum=bn_momentum,scale=False)(branch3x3)
    branch3x3 = LeakyReLU(0.03)(branch3x3)
    
    #put together
    x = concatenate([branch_pool, branch1x1, branch3x3])

###-----------------------------------END INCEPTION LAYER 0
    
    #average pooling

    x = AveragePooling2D((4,4))(x)

###-----------------------------------END INCEPTION LAYER 1

    #Pooling branch
    branch_pool = AveragePooling2D((4,4),strides=(1,1),padding='same')(x)

    #3x3dbl branch
    branch3x3dbl = Conv2D(second_layer_channels, (3,3),padding='same')(x)
    if(bn):
        branch3x3dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch3x3dbl)
    branch3x3dbl = LeakyReLU(0.03)(branch3x3dbl)
    branch3x3dbl = Conv2D(second_layer_channels, (3,3),padding='same')(branch3x3dbl)
    if(bn):
        branch3x3dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch3x3dbl)
    branch3x3dbl = LeakyReLU(0.03)(branch3x3dbl)
    
    #5x5 branch
    branch5x5dbl = Conv2D(second_layer_channels, (5,5),padding='same')(x)
    if(bn):
        branch5x5dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch5x5dbl)
    branch5x5dbl = LeakyReLU(0.03)(branch5x5dbl)
    branch5x5dbl = Conv2D(second_layer_channels, (5,5),padding='same')(branch5x5dbl)
    if(bn):
        branch5x5dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch5x5dbl)
    branch5x5dbl = LeakyReLU(0.03)(branch5x5dbl)
    
    #7x7 branch
    branch7x7 = Conv2D(second_layer_channels, (7,7),padding='same')(x)
    if(bn):
        branch7x7 = BatchNormalization(momentum=bn_momentum,scale=False)(branch7x7)
    branch7x7 = LeakyReLU(0.03)(branch7x7)

    #put together
    x = concatenate([branch_pool, branch3x3dbl, branch5x5dbl,branch7x7])

###-----------------------------------END INCEPTION LAYER 1

    #average pooling
    x = AveragePooling2D((2,2))(x)

###-----------------------------------START INCEPTION LAYER 2

    #Pooling branch
    branch_pool = AveragePooling2D((3,3),padding='same')(x)

    #3x3dbl branch
    branch1x13x3dbl = Conv2D(third_layer_channels, (1,1),padding='same')(x)
    if(bn):
        branch1x13x3dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch1x13x3dbl)
    branch1x13x3dbl = LeakyReLU(0.03)(branch1x13x3dbl)
    branch1x13x3dbl = Conv2D(third_layer_channels, (3,3),strides =(3,3),padding='same')(branch1x13x3dbl)
    if(bn):
        branch1x13x3dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch1x13x3dbl)
    branch1x13x3dbl = LeakyReLU(0.03)(branch1x13x3dbl)


    #3x3dbl branch
    branch3x3dbl = Conv2D(third_layer_channels, (3,3),padding='same')(x)
    if(bn):
        branch3x3dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch3x3dbl)
    branch3x3dbl = LeakyReLU(0.03)(branch3x3dbl)
    branch3x3dbl = Conv2D(third_layer_channels, (3,3),strides =(3,3),padding='same')(branch3x3dbl)
    if(bn):
        branch3x3dbl = BatchNormalization(momentum=bn_momentum,scale=False)(branch3x3dbl)
    branch3x3dbl = LeakyReLU(0.03)(branch3x3dbl)
    
    #put together
    x = concatenate([branch_pool, branch1x13x3dbl, branch3x3dbl])

###-----------------------------------END INCEPTION LAYER 2

    #Flatten and dropout

    x = Flatten()(x)
    x = Dropout(feature_dropout)(x)


    #FC top
    x = Dense(first_fc_layer)(x)
    if(bn):
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(first_fc_dropout)(x)
    x = Dense(second_fc_layer)(x)
    if(bn):
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(second_fc_dropout)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(img_input,x,name='dark_inception')

    return model

def DarkInceptionResNet(input_shape=(256,256,1),classes=10,num_layersA=3,num_layersB=5,FC1 = 256,FC2=0,leak = .03,feature_dropout =.2,FC_dropout =.2,bn_momentum=.99):

    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,selection='Dark-Inception-ResNet',leak=leak,bn_momentum=bn_momentum)

    #Stack of InceptionA layers
    for index in range(0,num_layersA-1):
        x = InceptionA(x,selection='Dark-Inception-ResNet',leak=leak,bn_momentum=bn_momentum)
    x = InceptionA(x,selection='Dark-Inception-ResNet',leak=-1.,bn_momentum=bn_momentum)

    #Reduction layer
    x = ReductionA(x,selection='Dark-Inception-ResNet',leak=leak,bn_momentum=bn_momentum)

    #Stack of InceptionB layers
    for index in range(0,num_layersB-1):
        x = InceptionB(x,selection='Dark-Inception-ResNet',leak=leak,bn_momentum=bn_momentum)
    x = InceptionB(x,selection='Dark-Inception-ResNet',leak=-1.,bn_momentum=bn_momentum)

    #Further reduction and flattening
    x = ReductionB(x,selection='Dark-Inception-ResNet',leak=leak,bn_momentum=bn_momentum)
    x = Flatten()(x)

    #Feature dropout
    x = Dropout(feature_dropout)(x)

    #Fully connected top if wanted
    if(FC1 > 0):
        x = Dense(FC1)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC1_dropout)(x)
    if(FC2 > 0):
        x = Dense(FC2)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC2_dropout)(x)

    #Softmax
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input,x,name='DarkInception')

    return model

def DarkInception(input_shape=(256,256,1),classes=10,num_layersA=3,num_layersB=5,FC1 = 256,FC2=0,leak = .03,feature_dropout =.2,FC_dropout =.2,bn_momentum=.99):

    #Input layer
    img_input = Input(shape=input_shape)

    #Stem layer
    x = Stem(img_input,selection='Dark-Inception',leak=leak,bn_momentum=bn_momentum)

    #Stack of InceptionA layers
    for index in range(0,num_layersA-1):
        x = InceptionA(x,selection='Dark-Inception',leak=leak,bn_momentum=bn_momentum)
    x = InceptionA(x,selection='Dark-Inception',leak=-1.,bn_momentum=bn_momentum)

    #Reduction layer
    x = ReductionA(x,selection='Dark-Inception',leak=leak,bn_momentum=bn_momentum)

    #Stack of InceptionB layers
    for index in range(0,num_layersB-1):
        x = InceptionB(x,selection='Dark-Inception',leak=leak,bn_momentum=bn_momentum)
    x = InceptionB(x,selection='Dark-Inception',leak=-1.,bn_momentum=bn_momentum)

    #Further reduction and flattening
    x = ReductionB(x,selection='Dark-Inception',leak=leak,bn_momentum=bn_momentum)
    x = Flatten()(x)

    #Feature dropout
    x = Dropout(feature_dropout)(x)

    #Fully connected top if wanted
    if(FC1 > 0):
        x = Dense(FC1)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC1_dropout)(x)
    if(FC2 > 0):
        x = Dense(FC2)(x)
        x = BatchNormalization(momentum=bn_momentum,scale=False)(x)
        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        else:
            x = Activation('relu')(x)
        x = Dropout(FC2_dropout)(x)

    #Softmax
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input,x,name='DarkInception')

    return model

def columbia(input_shape=(256,256,1),classes=4,first_layer_channels=4,second_layer_channels=12,third_layer_channels=32,first_fc_layer=1024,second_fc_layer=256,first_fc_dropout=.5,second_fc_dropout=.5,third_fc_dropout=.5,bn=False,bn_momentum=.99):

    #input layer
    img_input = Input(shape=input_shape,name='input')

    #Conv layer0
    x = Conv2D(first_layer_channels, (3,3),name='conv0')(img_input)
    if(bn):
        x = BatchNormalization(name='bn0',momentum=bn_momentum)(x)
    x = LeakyReLU(0.03)(x)

    #Pooling layer0
    x = AveragePooling2D((2,2),name='pooling0')(x)

    #Conv layer1
    x = Conv2D(second_layer_channels, (3,3),name='conv1_0')(x)
    if(bn):
        x = BatchNormalization(name='bn1_0',momentum=bn_momentum)(x)
    x = LeakyReLU(0.03)(x)

    x = Conv2D(second_layer_channels, (3,3),name='conv1_1')(x)
    if(bn):
        x = BatchNormalization(name='bn1_1',momentum=bn_momentum)(x)
    x = LeakyReLU(0.03)(x)

    #Pooling layer1
    x = AveragePooling2D((2,2),name='pooling1')(x)

    #Intermediate conv&pool layers
    x = Conv2D(third_layer_channels, (3,3),name='conv_i_0')(x)
    if(bn):
        x = BatchNormalization(name='bn_i_0',momentum=bn_momentum)(x)    
    x = AveragePooling2D((2,2),name='pooling_i_0')(x)
    x = Conv2D(2*third_layer_channels, (3,3),name='conv_i_1')(x)
    if(bn):
        x = BatchNormalization(name='bn_i_1',momentum=bn_momentum)(x)    
    x = AveragePooling2D((2,2),name='pooling_i_1')(x)

    #Further reduction layers
    x = AveragePooling2D((2,2),name='pooling2')(x)
    x = Flatten(name='flatten')(x)

    #Fully connected top
    x = Dense(first_fc_layer,name='fc0')(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(first_fc_dropout)(x)
    x = Dense(second_fc_layer,name='fc1')(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(second_fc_dropout)(x)
    x = Dense(10,name='fc2')(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(third_fc_dropout)(x)    
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input,x,name='columbia')

    return model

def cmu(input_shape=(256,256,1),classes=4,first_layer_channels=4,second_layer_channels=12,third_layer_channels=32,first_fc_layer=1024,second_fc_layer=256,first_fc_dropout=.5,second_fc_dropout=.5,bn=False,bn_momentum=.99):

    #input layer
    img_input = Input(shape=input_shape,name='input')

    #Conv layer0
    x = Conv2D(first_layer_channels, (3,3),name='conv0')(img_input)
    if(bn):
        x = BatchNormalization(name='bn0',momentum=bn_momentum)(x)
    x = LeakyReLU(0.03)(x)

    #Pooling layer0
    x = AveragePooling2D((2,2),name='pooling0')(x)

    #Conv layer1
    x = Conv2D(second_layer_channels, (3,3),name='conv1_0')(x)
    if(bn):
        x = BatchNormalization(name='bn1_0',momentum=bn_momentum)(x)
    x = LeakyReLU(0.03)(x)

    x = Conv2D(second_layer_channels, (3,3),name='conv1_1')(x)
    if(bn):
        x = BatchNormalization(name='bn1_1',momentum=bn_momentum)(x)
    x = LeakyReLU(0.03)(x)

    #Pooling layer1
    x = AveragePooling2D((2,2),name='pooling1')(x)

    #Intermediate conv&pool layers
    x = Conv2D(third_layer_channels, (3,3),strides=2,name='conv_i_0')(x)
    if(bn):
        x = BatchNormalization(name='bn_i_0',momentum=bn_momentum)(x)    
    x = LeakyReLU(0.03)(x)

    x = Conv2D(2*third_layer_channels, (3,3),strides=2,name='conv_i_1')(x)
    if(bn):
        x = BatchNormalization(name='bn_i_1',momentum=bn_momentum)(x)    
    x = LeakyReLU(0.03)(x)

    #Fully connected top
    x = Flatten(name='flatten')(x)
    x = Dense(first_fc_layer,name='fc0')(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(first_fc_dropout)(x)
    x = Dense(second_fc_layer,name='fc1')(x)
    x = LeakyReLU(0.03)(x)
    x = Dropout(second_fc_dropout)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input,x,name='cmu')

    return model
