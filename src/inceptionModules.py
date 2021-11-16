from .globalVariables import *




def stemInception( model ):
    
    '''
    The first stem incpetion
    
    
    '''
    
    #Inception max pool and convolutin
    
    tower_1 = layers.Conv2D(96, (3,3), strides=(2,2), padding='valid')(model)
    tower_2 = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(model)

    model = layers.concatenate([tower_1, tower_2], axis = 3)
    
    tower_1 = layers.Conv2D(64, (1,1), strides=(1,1), padding='valid')(model)    
    tower_1 = layers.Conv2D(96, (3,3), strides=(1,1), padding='same')(tower_1)
    
    tower_2 = layers.Conv2D(64, (1,1), strides=(1,1), padding='same')(model)    
    tower_2 = layers.Conv2D(64, (7,1), strides=(1,1), padding='same')(tower_2)
    tower_2 = layers.Conv2D(64, (1,7), strides=(1,1), padding='same')(tower_2)
    tower_2 = layers.Conv2D(96, (3,3), strides=(1,1), padding='valid')(tower_2)
    
    model = concatenate([tower_1, tower_2], axis=3)

    
    tower_1 = layers.Conv2D(96, (3,3), strides=(2,2), padding='valid') (model)
    tower_2 = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(model)  
    
    model = concatenate([tower_1, tower_2], axis=3)

    
    return model



def inceptionA( model ):
    '''
    The inception A layer 
    
    '''
    
    tower_1 = layers.AveragePooling2D((2,2), padding='same',strides=(1,1))(model)
    tower_1 = layers.Conv2D(96, (1,1), padding='same', strides=(1,1))(tower_1)

    tower_2 = layers.Conv2D(96, (1,1), padding='same', strides=(1,1))(model)
    
    tower_3 = layers.Conv2D(64, (1,1), padding='same', strides=(1,1))(model)
    tower_3 = layers.Conv2D(96, (3,3), padding='same', strides=(1,1))(tower_3)

    tower_4 = layers.Conv2D(64, (1,1), padding='same', strides=(1,1))(model)
    tower_4 = layers.Conv2D(96, (3,3), padding='same', strides=(1,1))(tower_4)
    tower_4 = layers.Conv2D(96, (3,3), padding='same', strides=(1,1))(tower_4)

    
    model = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

    return model


def reductionA( model ):
    '''
    The inception B layer
    '''
    
    tower_1 = layers.MaxPooling2D((3,3), padding='valid', strides=(2,2))(model)


    tower_2 = layers.Conv2D(384, (3,3), padding='valid', strides=(2,2))(model)
    
    tower_3 = layers.Conv2D(192, (1,1), padding='same', strides=(1,1))(model)
    tower_3 = layers.Conv2D(224, (3,3), padding='same', strides=(1,1))(tower_3)    
    tower_3 = layers.Conv2D(256, (3,3), padding='valid', strides=(2,2))(tower_3)
    
    model = layers.concatenate([tower_1, tower_2, tower_3], axis=3)

    return model


def inceptionB( model ):
    '''
    The inception A layer 
    
    '''
    
    tower_1 = layers.AveragePooling2D((2,2), padding='same', strides=(1,1))(model)
    tower_1 = layers.Conv2D(128, (2,2), padding='same',strides=(1,1))(tower_1)

    tower_2 = layers.Conv2D(384, (1,1), padding='same',strides=(1,1))(model)
    
    tower_3 = layers.Conv2D(192, (1,1), padding='same',strides=(1,1))(model)
    tower_3 = layers.Conv2D(224, (1,7), padding='same',strides=(1,1))(tower_3)
    tower_3 = layers.Conv2D(256, (7,1), padding='same',strides=(1,1))(tower_3)

    tower_4 = layers.Conv2D(192, (1,1), padding='same',strides=(1,1))(model)
    tower_4 = layers.Conv2D(192, (1,7), padding='same',strides=(1,1))(tower_4)
    tower_4 = layers.Conv2D(224, (7,1), padding='same',strides=(1,1))(tower_4)
    tower_4 = layers.Conv2D(224, (1,7), padding='same',strides=(1,1))(tower_4)
    tower_4 = layers.Conv2D(256, (7,1), padding='same',strides=(1,1))(tower_4)

    
    model = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

    return model


def reductionB( model ):
    '''
    The inception B layer
    '''
    
    tower_1 = layers.MaxPooling2D((3,3), padding='valid', strides=(2,2))(model)


    tower_2 = layers.Conv2D(192, (1,1), padding='valid', strides=(1,1))(model)
    tower_2 = layers.Conv2D(192, (3,3), padding='valid', strides=(2,2))(tower_2)
  
    tower_3 = layers.Conv2D(256, (1,1), padding='same', strides=(1,1))(model)
    tower_3 = layers.Conv2D(256, (1,7), padding='same', strides=(1,1))(tower_3)    
    tower_3 = layers.Conv2D(320, (7,1), padding='same', strides=(1,1))(tower_3)
    tower_3 = layers.Conv2D(320, (3,3), padding='valid', strides=(2,2))(tower_3)
   
    model = layers.concatenate([tower_1, tower_2, tower_3], axis=3)

    return model


def inceptionC( model ):
    '''
    The inception C layer 
    
    '''
    
    tower_1 = layers.Conv2D(384,(1,1), padding='same', strides=(1,1))(model)
    
    tower_1a = layers.Conv2D(256, (1,3), padding='same', strides=(1,1))(tower_1)
    tower_1b = layers.Conv2D(256, (3,1), padding='same', strides=(1,1))(tower_1)

    tower_2 = layers.Conv2D(256, (1,1), padding='same', strides=(1,1))(model)
    
    tower_3 = layers.AveragePooling2D((2,2), padding='same', strides=(1,1))(model)
    tower_3 = layers.Conv2D(256, (1,1), padding='same', strides=(1,1))(tower_3)

    tower_4 = layers.Conv2D(284, (1,1), padding='same', strides=(1,1))(model)
    tower_4 = layers.Conv2D(448, (1,3), padding='same', strides=(1,1))(tower_4)
    tower_4 = layers.Conv2D(512, (3,1), padding='same', strides=(1,1))(tower_4)
    
    tower_4a = layers.Conv2D(256, (1,3), padding='same', strides=(1,1))(tower_4)
    tower_4b = layers.Conv2D(256, (3,1), padding='same', strides=(1,1))(tower_4)

    
    model = layers.concatenate([tower_1a, tower_1b,\
                         tower_2, tower_3, \
                         tower_4a, tower_4b], axis=3)

    return model
