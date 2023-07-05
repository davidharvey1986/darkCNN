from keras.callbacks import CSVLogger
from .getSIDMdata import get_tf_DataSet as getData
from .mainModel import simpleModel, InceptionV4
from .globalVariables import *
from .augmentData import augmentData


def main( nEpochs=60, train_split=0.8,\
          fileRootName=None, database=None, attributes=['redshift','mass'], \
          nMonteCarlo=5, dropout=0.2, channels=['total','stellar']):
    '''
    The main function that trains the model 
    
    OPTIONAL ARGUMENTS :
    ---------------------
    
    - nEpochs : integer : the number of epochs the will train the algorithm
    - testTrainSplit : float : the ratio of test to training split 
    - fileRootName : string : root name for the pickled training sample of clusters
    - nMonteCarlo : the number of times the algorithms monte carlos the test - train split to understand
                    the distribution of estimates.
    - dropout : integer : the dropout rate of neurons in the network to avoid overfitting
    - nChannels : float : the number of channels of data to go in to the CNN. 
                        (1, 2, 3: total, xray, baryonic matter)
    - database : string : the database of test and train samples.
    
    '''
    
    #Check for the directory "pickles"
    if not os.path.isdir("pickles"):
        os.system("mkdir pickles")
    
    if fileRootName is None:
        #fileRootName = "pickles/augmentedTrain_%i_channel_noAtt_dropout_%0.1f_testSplit_%0.3f" % \
        #(nChannels,dropout,testTrainSplit)
        
        fileRootName = "pickles/merten_arch"
        
        
        
    print("All files saved to %s" % fileRootName)  
    print("Using attributes: ", attributes)
    for iMonteCarlo in range(1, nMonteCarlo+1):
        
        modelFile = "%s_%i.h5" % (fileRootName, iMonteCarlo)
        csv_file = '%s_%i.csv' % (fileRootName, iMonteCarlo)
        checkpoint_path = '%s_%i.ckpt' % (fileRootName, iMonteCarlo)
        
        
        #Get the training and test labels that will be required.
        
   
        trainingSet, testSet = getData( augment_data=True, channels=channels, 
                            attributes=attributes,  random_state=41+iMonteCarlo, 
                                     train_split=train_split, allDataFile=database)
        if len(attributes) > 0:
            train_dataset_to_numpy = list(trainingSet.as_numpy_iterator())[0]
            test_dataset_to_numpy = list(testSet.as_numpy_iterator())[0]

        else:
            train_dataset_to_numpy = list(trainingSet.as_numpy_iterator())
            test_dataset_to_numpy = list(testSet.as_numpy_iterator())

        inputShape = train_dataset_to_numpy[0][0].shape[1:]
        nTrain = train_dataset_to_numpy[0][0].shape[0]
        nTest = list(test_dataset_to_numpy)[0][0].shape[0]
        print("Using %i training samples and %i test samples" % (nTrain, nTest))
        #Check to see if a previous model exists to continue training off
        print("Input tensor shape is ", inputShape)
        
        if os.path.isfile(modelFile):
            print("FOUND PREVIOUS MODEL, LOADING...")
            mertensModel = models.load_model(modelFile)
        else:
            mertensModel = InceptionV4(input_shape=test_set[0][0].shape, bn_momentum=.0,classes=num_classes,
                                               feature_dropout=.33,num_layersA=1,num_layersB=1,num_layersC=1,leak=.03)
            
        mertensModel.summary()
        
        #Set up some logging for the model
        #This is a checkpoint to save along the way
    
        #This is a csv to log the history of the training
    
        csv_logger = CSVLogger(csv_file, append=True)

        #Check to see if the previous csv logger exists, since i will want to continue
        #Number the training from the previous state
        if os.path.isfile( csv_file ):
            try:
                previousEpochs = np.loadtxt( csv_file, delimiter=',',skiprows=1 )
                initial_epoch = previousEpochs.shape[0]
            except:
                #this means the file was created but nothing is in there
                initial_epoch = 0
        else:
            initial_epoch = 0
        
    
        print("Starting from epoch %i" % initial_epoch)
        
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
        
        # Train the model with the new callback
        inceptionHistory = mertensModel.fit(x=trainingSet,
              epochs=nEpochs,
              validation_data=testSet,
              initial_epoch=initial_epoch,
              callbacks=[cp_callback, csv_logger])  # Pass callback to training

        mertensModel.save(modelFile)
    
if __name__ == '__main__':
    main()
    
