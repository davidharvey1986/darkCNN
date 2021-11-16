from keras.callbacks import CSVLogger

from augmentData import augmentData

def main( nEpochs=20, testTrainSplit=0.15,\
          fileRootName=None,\
          nMonteCarlo=5, dropout=0.2, nChannels=3):
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
    
    '''
    
    
    if fileRootName is None:
        #fileRootName = "pickles/augmentedTrain_%i_channel_noAtt_dropout_%0.1f_testSplit_%0.3f" % \
        #(nChannels,dropout,testTrainSplit)
        
        fileRootName = "pickles/simpleModel_%i_channel_noAtt_dropout_%0.1f_testSplit_%0.3f" % \
            (nChannels,dropout,testTrainSplit)
        
        
        
    print("All files saved to %s" % fileRootName)  
    
    for iMonteCarlo in range(1, nMonteCarlo):
        
        modelFile = "%s_%i.h5" % (fileRootName, iMonteCarlo)
        csv_file = '%s_%i.csv' % (fileRootName, iMonteCarlo)
        checkpoint_path = '%s_%i.ckpt' % (fileRootName, iMonteCarlo)
        
        
        #Get the training and test labels that will be required.
        #nTest is teh number of tets PER CROSS SECTION
        trainingSet, testSet = \
            getData( binning=20, testTrainSplit=testTrainSplit,  \
                    indexFileRoot='pickles/testIndexes_%i' % (iMonteCarlo), \
                     nChannels=nChannels)
        
        print("Number of channels is %i" % trainingSet["images"].shape[-1])
           
        augmentedTrain, augmentedLabels = augmentData( trainingSet['images'], trainingSet['label'])
    
        #Check to see if a previous model exists to continue training off
        
        if os.path.isfile(modelFile):
            print("FOUND PREVIOUS MODEL, LOADING...")
            mertensModel = models.load_model(modelFile)
        else:
            mertensModel = simpleModel( augmentedTrain[0].shape, dropout=dropout )
            
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
        inceptionHistory = mertensModel.fit(augmentedTrain, 
              augmentedLabels,  
              epochs=nEpochs,
              validation_data=(testSet["images"], testSet["label"]),
              initial_epoch=initial_epoch,
              callbacks=[cp_callback, csv_logger])  # Pass callback to training

        mertensModel.save(modelFile)
    
if __name__ == '__main__':
    main()
    
