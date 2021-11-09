from globalVariables import *
import inceptionModules


def mainModel(imageShape, dropout=0.2, momentum=0.9, nClasses=4, learning_rate=5e-6):
    '''
    This is the main model that is based on the architecture of Mertens et al 2015
    
    Conv(3,3,2,2,v,32)
    AN
    Activation ReLU( 0.03 )
    BN
    Conv(3,3,1,1,v,32)
    ReLU( 0.03 )
    Conv(3,3,1,1,s,64)
    BN
    StemInception
    BN
    InceptionA
    BN
    ReductionA
    BN
    InceptionB
    BN
    ReductionB
    BN
    InceptionC
    BN
    GlobalAvgPool
    Dropout (0.3)
    FC (9)
    Softmax
    
    Defaults : Dropout == 0.33, momentum = 0.99
    
    '''
    
    
    
    inputLayer = Input(shape=imageShape)
    
    #Layer 1
    model = layers.Conv2D(32, (3, 3), activation='relu', strides=(2,2), padding='valid')(inputLayer)
    #Normalise
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Layer 2
    model = layers.Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='valid')(model)
    #Normalise
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Layer 3
    model = layers.Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same')(model)
    #Normalise
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Inception A
    model  = inceptionModules.inceptionA( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Reducion A
    model  = inceptionModules.reductionA( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Inception B
    model  = inceptionModules.inceptionB( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Reducion B
    model  = inceptionModules.reductionB( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Inception C
    model  = inceptionModules.inceptionC( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #GlobalAvg
    model = layers.GlobalAveragePooling2D( )( model )
    #DRH: Add another dense layer here.
    model = layers.Dense(256, activation='relu')(model)
    #Dropout
    model = layers.Dropout( dropout )( model)
    #Dense, fully connected layer
    model = layers.Dense( nClasses )(model)
    
    finalModel = Model(inputLayer, model, name='mainModel')
    optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate)

    finalModel.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    
    
    return finalModel

    
    
    
    
