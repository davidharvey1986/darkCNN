from keras.layers import Conv2D, Dense, Lambda, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,Flatten, BatchNormalization, Activation, Input, Dropout, concatenate, LeakyReLU

from keras import backend as K



def myConv2D(x,filters,kernel_size,strides=1,padding='same',use_bias=None,bn_momentum=.99,leak=.0,bn=True,data_format='channels_last'):

    x = Conv2D(filters,kernel_size,strides=strides,padding=padding,use_bias=use_bias,data_format=data_format)(x)
    if(bn):
        x = BatchNormalization(scale=False,momentum=bn_momentum)(x)
    if(leak > 0.):
        x = LeakyReLU(leak)(x)
    else:
        x = Activation('relu')(x)

    return x


def Stem(x,selection='Inception-v4',leak=0.,bn_momentum=.99,bn=True):

    if(selection=='Inception-v4' or selection=='Inception-ResNet-v2'):
        x = myConv2D(x,32,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,32,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,64,3,leak=leak,bn_momentum=bn_momentum)
    
        branch1_1 =  MaxPooling2D(3, strides=2)(x)
        branch1_2 = myConv2D(x,96,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch1_1,branch1_2])

        branch2_1 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch2_1 = myConv2D(branch2_1,96,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,64,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,64,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,96,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch2_1,branch2_2])
        
        branch3_1 = myConv2D(x,192,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        branch3_2 =  MaxPooling2D(3, strides=2)(x)
        x = concatenate([branch3_1,branch3_2])

    elif(selection=='Inception-v4-small'):
        x = myConv2D(x,16,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,16,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,32,3,leak=leak,bn_momentum=bn_momentum)
    
        branch1_1 =  MaxPooling2D(3, strides=2)(x)
        branch1_2 = myConv2D(x,48,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch1_1,branch1_2])

        branch2_1 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch2_1 = myConv2D(branch2_1,48,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,32,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,32,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,48,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch2_1,branch2_2])
        
        branch3_1 = myConv2D(x,96,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        branch3_2 =  MaxPooling2D(3, strides=2)(x)
        x = concatenate([branch3_1,branch3_2])

    elif(selection=='Inception-v4-wide'):
        x = myConv2D(x,32,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,32,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,64,3,leak=leak,bn_momentum=bn_momentum)
    
        branch1_1 =  MaxPooling2D(3, strides=2)(x)
        branch1_2 = myConv2D(x,96,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch1_1,branch1_2])

        branch2_1 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch2_1 = myConv2D(branch2_1,96,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,64,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,64,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,96,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch2_1,branch2_2])

    elif(selection=='Inception-ResNet-v1'):
        x = myConv2D(x,32,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,32,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,64,3,leak=leak,bn_momentum=bn_momentum)
        x = MaxPooling2D(3, strides=2)(x)
        x = myConv2D(x,80,1,leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,192,3,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,256,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

    elif(selection=='Dark-Inception-ResNet' or selection=='Dark-Inception'):
        x = myConv2D(x,32,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)
        x = myConv2D(x,32,3,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch1_1 = myConv2D(x,64,3,strides=4,padding='valid',leak=leak,bn_momentum=bn_momentum)
        branch1_2 = AveragePooling2D(4,padding='valid')(x)
        x = concatenate([branch1_1,branch1_2])

        branch2_1 = myConv2D(x,16,1,leak=leak,bn_momentum=bn_momentum)
        branch2_1 = myConv2D(branch2_1,16,3,padding='same',leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(x,16,1,leak=leak,bn_momentum=bn_momentum)
        branch2_2 = myConv2D(branch2_2,16,5,padding='same',leak=leak,bn_momentum=bn_momentum)
        branch2_3 = myConv2D(x,16,1,leak=leak,bn_momentum=bn_momentum)
        branch2_3 = myConv2D(branch2_3,16,11,padding='same',leak=leak,bn_momentum=bn_momentum)
        x = concatenate([branch2_1,branch2_2,branch2_3])

    elif(selection=='DI-Bare'):
        x = myConv2D(x,32,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch1_1 = myConv2D(x,16,3,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch1_2 = myConv2D(x,16,5,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch1_3 = myConv2D(x,16,7,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn)
        x = concatenate([branch1_1,branch1_2,branch1_3])

        x = myConv2D(x,32,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch2_1 = myConv2D(x,64,3,padding='valid',leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch2_2 = myConv2D(x,64,(7,1),leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch2_2 = myConv2D(branch2_2,64,(1,7),leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch2_2 = myConv2D(branch2_2,64,3,padding='valid',leak=leak,bn_momentum=bn_momentum,bn=bn)
        x = concatenate([branch2_1,branch2_2])

    return x    

def InceptionA(x,selection='Inception-v4',leak=.0,scale=.1,bn_momentum=.99,bn=True):

    if(selection=='Inception-v4'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,96,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,96,3,leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,96,3,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,96,3,leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='DI-Bare'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,48,1,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch2 = MaxPooling2D(1,padding='same')(x)
        branch2 = myConv2D(branch2,48,1,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch3 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch3 = myConv2D(branch3,48,3,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch4 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch4 = myConv2D(branch4,48,5,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch5 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch5 = myConv2D(branch4,48,(1,7),leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch5 = myConv2D(branch4,48,(7,1),leak=leak,bn_momentum=bn_momentum,bn=bn)

        x = concatenate([branch1,branch2,branch3,branch4,branch5])



    elif(selection=='Inception-v4-small'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,48,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,48,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,48,3,leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,48,3,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,48,3,leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='Inception-ResNet-v1'):
        branch1 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,32,3,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,32,3,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,32,3,leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2,branch3])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    elif(selection=='Inception-ResNet-v2'):
        branch1 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,32,3,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,48,3,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,64,3,leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2,branch3])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    elif(selection=='Dark-Inception-ResNet'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,64,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(x,16,3,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(x,16,3,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,16,(1,5),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,16,(5,1),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,16,(1,5),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,16,(5,1),leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,32,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,16,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,16,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,16,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,16,(7,1),leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2,branch3,branch4])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    elif(selection=='Dark-Inception'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,64,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(x,32,3,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(x,32,3,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,32,(1,5),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,32,(5,1),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,32,(1,5),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,32,(5,1),leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,32,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,32,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,32,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,32,(7,1),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])
        

    else:
        raise ValueError('Invalid model choice to build InceptionA')

 
    
    return x

def InceptionB(x,selection='Inception-v4',leak=.0,scale=.1,bn_momentum=.99,bn=True):

    if(selection=='Inception-v4'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,128,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,384,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,224,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,256,(1,7),leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,192,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,224,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,224,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,256,(7,1),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='DI-Bare'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,128,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,128,(7,1),leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(7,1),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='Inception-v4-small'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,64,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,128,(7,1),leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,(7,1),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='Inception-ResNet-v1'):
        branch1 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,128,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,128,(7,1),leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 


    elif(selection=='Inception-ResNet-v2'):
        branch1 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,160,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,192,(7,1),leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    elif(selection=='Dark-Inception-ResNet'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,64,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,64,3,leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,64,3,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,64,3,leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2,branch3])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    elif(selection=='Dark-Inception'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,128,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,64,3,leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,3,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,128,3,leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3])

    else:
        raise ValueError('Invalid model choice to build InceptionB')

    
    return x

def InceptionC(x,selection='Inception-v4',leak=.0,scale=.1,bn_momentum=.99,bn=True):
    
    if(selection=='Inception-v4'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,256,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,384,1,leak=leak,bn_momentum=bn_momentum)
        branch3_1 = myConv2D(branch3,256,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch3_2 = myConv2D(branch3,256,(3,1),leak=leak,bn_momentum=bn_momentum)


        branch4 = myConv2D(x,384,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,448,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,512,(3,1),leak=leak,bn_momentum=bn_momentum)
        branch4_1 = myConv2D(branch4,256,(3,1),leak=leak,bn_momentum=bn_momentum)
        branch4_2 = myConv2D(branch4,256,(1,3),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3_1,branch3_2,branch4_1,branch4_2])

    elif(selection=='DI-Bare'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,256,1,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch2 = MaxPooling2D(1,padding='same')(x)
        branch2 = myConv2D(branch2,256,1,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch3 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch3 = myConv2D(branch3,256,3,leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch3 = myConv2D(branch3,256,3,leak=leak,bn_momentum=bn_momentum,bn=bn)

        branch4 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch4 = myConv2D(branch4,256,(1,3),leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch4 = myConv2D(branch4,256,(3,1),leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch4_1 = myConv2D(branch4,128,(3,1),leak=leak,bn_momentum=bn_momentum,bn=bn)
        branch4_2 = myConv2D(branch4,128,(1,3),leak=leak,bn_momentum=bn_momentum,bn=bn)

        x = concatenate([branch1,branch2,branch3,branch4_1,branch4_2])

    elif(selection=='Inception-v4-small'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,96,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)
        branch3_1 = myConv2D(branch3,96,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch3_2 = myConv2D(branch3,96,(3,1),leak=leak,bn_momentum=bn_momentum)


        branch4 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,96,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,96,(3,1),leak=leak,bn_momentum=bn_momentum)
        branch4_1 = myConv2D(branch4,96,(3,1),leak=leak,bn_momentum=bn_momentum)
        branch4_2 = myConv2D(branch4,96,(1,3),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3_1,branch3_2,branch4_1,branch4_2])

    elif(selection=='Inception-v4-wide'):
        branch1 = AveragePooling2D(1,padding='same')(x)
        branch1 = myConv2D(branch1,384,1,leak=leak,bn_momentum=bn_momentum)

        branch1w = MaxPooling2D(1,padding='same')(x)
        branch1w = myConv2D(branch1,384,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,384,1,leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,384,1,leak=leak,bn_momentum=bn_momentum)
        branch3_1 = myConv2D(branch3,384,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch3_2 = myConv2D(branch3,384,(3,1),leak=leak,bn_momentum=bn_momentum)


        branch4 = myConv2D(x,384,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,448,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,512,(3,1),leak=leak,bn_momentum=bn_momentum)
        branch4_1 = myConv2D(branch4,384,(3,1),leak=leak,bn_momentum=bn_momentum)
        branch4_2 = myConv2D(branch4,384,(1,3),leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch1w,branch2,branch3_1,branch3_2,branch4_1,branch4_2])

    elif(selection=='Inception-ResNet-v1'):
        branch1 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,192,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,192,(3,1),leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    elif(selection=='Inception-ResNet-v2'):
        branch1 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)

        branch2 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,224,(1,3),leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,256,(3,1),leak=leak,bn_momentum=bn_momentum)

        mixed = concatenate([branch1,branch2])
        up = Conv2D(K.int_shape(x)[-1],1,use_bias=False)(mixed)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={'scale': scale})([x, up])

        if(leak > 0.):
            x = LeakyReLU(leak)(x)
        elif(leak==.0):
            x = Activation('relu')(x)
        else:
            pass 

    else:
        raise ValueError('Invalid model choice to build InceptionC')
    
    return x

def ReductionA(x,selection='Inception-v4',leak=.0,bn_momentum=.99,bn=True):

    if(selection=='Dark-Inception-ResNet' or selection=='Dark-Inception'):

        branch1 = myConv2D(x,128,1,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch1 = MaxPooling2D(3,strides=2,padding='same')(branch1)

        branch2 = myConv2D(x,128,1,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch2 = AveragePooling2D(3,strides=2,padding='same')(branch2)

        branch3 = myConv2D(x,128,1,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch3 = myConv2D(x,128,3,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch3 = myConv2D(branch3,128,3,strides=2,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 

        x = concatenate([branch1,branch2,branch3])

    elif(selection=='DI-Bare'):

        branch1 = myConv2D(x,128,1,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch1 = MaxPooling2D(3,strides=2,padding='valid')(branch1)
        
        branch2 = myConv2D(x,128,1,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch2 = AveragePooling2D(3,strides=2,padding='valid')(branch2)

        branch3 = myConv2D(x,64,1,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch3 = myConv2D(branch3,128,3,padding='same',leak=leak,bn_momentum=bn_momentum,bn=bn) 
        branch3 = myConv2D(branch3,128,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum,bn=bn) 

        x = concatenate([branch1,branch2,branch3])

    else:

        if(selection=='Inception-v4'):
            n = 384
            k = 192
            l = 224
            m = 256
        elif(selection=='Inception-v4-small'):
            n = 128
            k = 128
            l = 128
            m = 128

        elif(selection=='Inception-ResNet-v1'):
            n = 384
            k = 192
            l = 192
            m = 256

        elif(selection=='Inception-ResNet-v2'):
            n = 384
            k = 256
            l = 256
            m = 384

        else:
            raise ValueError('Invalid model choice to build ReductionA')

        branch1 = MaxPooling2D(3,strides=2)(x)
    
        branch2 = myConv2D(x,n,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,k,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,l,3,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,m,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3])

    return x

def ReductionB(x,selection='Inception-v4',leak=.0,bn_momentum=.99,bn=True):

    if(selection=='Inception-v4'):
        branch1 = MaxPooling2D(3,strides=2)(x)

        branch2 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,192,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,256,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,320,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,320,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3])

    elif(selection=='DI-Bare'):
        branch1 = myConv2D(x,192,1,leak=leak,bn_momentum=bn_momentum)
        branch1 = MaxPooling2D(3,strides=2)(branch1)

        branch2 = myConv2D(x,128,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,192,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,64,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,192,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,192,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,192,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3])

    elif(selection=='Inception-v4-small'):
        branch1 = MaxPooling2D(3,strides=2)(x)

        branch2 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,96,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,96,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,96,(1,7),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,96,(7,1),leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,96,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3])


    elif(selection=='Inception-ResNet-v1'):
        branch1 = MaxPooling2D(3,strides=2)(x)

        branch2 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,384,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,256,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,256,3,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,256,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='Inception-ResNet-v2'):
        branch1 = MaxPooling2D(3,strides=2)(x)

        branch2 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch2 = myConv2D(branch2,384,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch3 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch3 = myConv2D(branch3,288,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        branch4 = myConv2D(x,256,1,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,288,3,leak=leak,bn_momentum=bn_momentum)
        branch4 = myConv2D(branch4,320,3,strides=2,padding='valid',leak=leak,bn_momentum=bn_momentum)

        x = concatenate([branch1,branch2,branch3,branch4])

    elif(selection=='Dark-Inception-ResNet' or selection=='Dark-Inception'):

        branch1 = MaxPooling2D(4,padding='same')(x)

        branch2 = AveragePooling2D(4,padding='same')(x)

        branch3 = myConv2D(x,128,1,padding='same',leak=leak,bn_momentum=bn_momentum) 
        branch3 = myConv2D(branch3,64,3,strides=4,padding='same',leak=leak,bn_momentum=bn_momentum) 

        x = concatenate([branch1,branch2,branch3])



    else:
        raise ValueError('Invalid model choice to build ReductionB')

    return x

