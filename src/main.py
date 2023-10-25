from .getSIDMdata import get_tf_DataSet as getData
from .tools import get_best_model
from .globalVariables import *


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
        train, test, params = getData(augment_data=True, simulationNames=simulationNames, 
                                                      channels=channels, train_split=0.01,
                                                      allDataFile='pickles/binned_data_20.pkl', 
                                                      random_state=iMonteCarlo, return_test_params=True, 
                                                      meta_data=['bcg_e'], add_noise=these_noise)
         
        checkpoint_filepath = '%s_%i' % (fileRootName, iMonteCarlo)
  
        model_name = get_best_model( train, test, base_cnn_input_shape=test[0][0][0].shape, \
                                    model_name=model_name, 
                                   checkpoint_filepath=checkpoint_filepath, \
                                    epochs=epochs, meta=['bcg_e'])    
    
        accuracy.append(model.evaluate(test[0],test[1])[1])
        
if __name__ == '__main__':
    main()
    
