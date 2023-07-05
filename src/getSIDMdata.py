from .globalVariables import *
from astropy.io import fits
import glob
from scipy.signal import correlate2d
import tqdm
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


       
def get_tf_DataSet( train_split = 0.8, binning = 20, allDataFile = None, \
            attributes = [], massCut=0,            \
            channels = ['total'],                                           \
            simulationNames = ['CDM','SIDM0.1','SIDM0.3','SIDM1'],\
            random_state=42, augment_data=False, shuffle_data=False, batch_size=32, 
            crop_data=False, zoom=False, contrast=False, correlations=None, rescale=1., 
            resize=None, renorm='max', DMO=False, verbose=False, data_labels=None, return_test_params=False ):
    
    '''
    OBJECTIVE
    ----------
    Get the data from the simulations.
    
    NOTES
    -----
    The CNN struggles with images that are too large. 
    
    Moreover, if i am ever to apply this to data, there is no point having a 1kpc/h  resolution.
    
    For the CNN, i will use weak lensing, which will be a map of shears.
    But for now lets use density and move to multi channels later.
    
    So to do this, i want a typical pixel size of a galaxy separation.
    For space based this would be HST, which is 100 g/arcmin2, 10 g / arcmin, 0.04 g / kpc
    So if the field of view is 1Mpc, then the pixel size will be 20 kpc
    
    So, bin the image by 20 pixels.
    
    OPTIONAL ARGUMENTS
    ------------------
    
    - testTrainSplit : the proportion of models that are in the test and training samples
    - binning : integer : the binning of the maps from the raw data to be used. This only goes
                           in to the string of the filename that has already analysed the maps
    - allDataFile : string : the data file with all the clusters pre-reduced in. If None, use the default

    - channels : list : can be any combination of the following ['total','xray','stellar']
    - attributes : a list of M attributes to be returned with the training and test sets e.g. ['redshift', 'mass']
    - simulationNames : a list of the names of simulations I want to return
    
    returns : tuple : Tuple of the (N x training images, numpy array of N x M attritues, N training labels),
                dictionary of test sets for each dark matter model, with J test clusters with their attrbutes and 
                labels
    '''
    np.random.seed(random_state)
    nChannels = len(channels)
    nAttributes = len(attributes)
    
    if DMO:
        if 'xray' in channels:
            raise ValueError("Cannot get X-ray for DMO files")
        channel_idxs = [ np.where( i == np.array(['total','stellar']))[0][0] for i in channels]    
    else:
        channel_idxs = [ np.where( i == np.array(['total','xray','stellar']))[0][0] for i in channels]    
    
    if allDataFile is None:
        allDataFile = 'pickles/allSimData_binning_20.pkl'
    
    if not os.path.isfile( allDataFile ):
        raise ValueError( "Cant find data file %s, run rebinAllData.py" % allDataFile)
        
    allDataParams, allImages = pkl.load(open(allDataFile, 'rb'))
    dataParamsKeys = list(allDataParams.keys())
    dataParamsKeys.append("images")            

    
    
    
    #Check the sims
    data_class_names = np.unique( allDataParams['sim'])
    falsehoods = []
    for i in simulationNames:
        falsehoods.append( ~np.any([i in j for j in data_class_names]) )
    if np.any(falsehoods):
        raise ValueError("Simulation %s not recognised, should be of %s" % \
                         ( ', '.join(np.array(simulationNames)[falsehoods]), ', '.join(data_class_names)))
    
    
    
    
    #images is a list so make it an array
    allImages = np.array(allImages)*rescale
    
    if resize is not None:
        print("rescaling with nearestr")
        allImages = tf.image.resize(allImages, resize, method='nearest')
       
    images = np.stack([ allImages[:,:,:,i] for i in channel_idxs], axis=-1)
    
    if renorm == 'mean':
        print(allImages.shape)
        allImages = np.stack([ allImages[ i, :, :, j] - np.mean(allImages[ i, :, :, j]) 
                           for i in np.arange(allImages.shape[0]) 
                           for j in np.arange(allImages.shape[-1]) ])
    
    
    for i in allDataParams.keys():
        if i != 'lensing_norm':
            allDataParams[i] = np.array(allDataParams[i])
    
    selectGalaxy = np.zeros(len(allDataParams['label']))
    
    selectGalaxy[ (allDataParams['mass'] > massCut) ] = 1

    for i in simulationNames:

        if DMO:
            modelMatch = np.array( [ i == iSim for iSim in allDataParams['sim']])
        else:
            modelMatch = np.array([ i+"+baryons" == iSim for iSim in allDataParams['sim']])
        selectGalaxy[ modelMatch ] += 1
    
    images = images[ selectGalaxy==2, :, :, :]

    #[selectGalaxy==2]
    for i in allDataParams.keys():
        if np.array(allDataParams[i]).shape[0] == 0:
            continue
        allDataParams[i] = np.array(allDataParams[i])[ selectGalaxy == 2 ]
        
    labels = np.array(allDataParams['sim'])
    labelClasses =  np.unique(labels)
    newLabels = np.zeros(labels.shape[0])
    if data_labels is None:
        data_labels = np.arange(len(labelClasses))
    else:
        if len(data_labels) != len(labelClasses):
            raise ValueError("Input data labels is not hte same length as the number of classes")
            
    for i, iClass in enumerate(labelClasses):
        if verbose:
            print("Number of %s (%i) class : %i" % (iClass, i, len(newLabels[ labels == iClass ])))
        newLabels[ labels == iClass ] = data_labels[i]
        

        
    if correlations is not None:
        all_corrs = []
        for iCorr in correlations:
            corr1_idx = np.where( iCorr[0] == np.array(['total','xray','stellar']))[0][0] 
            corr2_idx = np.where( iCorr[1] == np.array(['total','xray','stellar']))[0][0] 

            corr_fct = []
            im_idxs = np.arange(allImages.shape[0])[selectGalaxy==2]
            
            for i in tqdm.tqdm(im_idxs):
                
                corr_fct.append(correlate2d( allImages[i,:,:,corr1_idx],allImages[i,:,:,corr2_idx], mode='same')/100.)
                
            all_corrs.append( np.stack([ corr_fct ], axis=-1) )
        corr_arr = np.concatenate( all_corrs, axis=-1)
        
        images = np.concatenate([images, corr_arr], axis=-1)
    if augment_data:
        gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         rotation_range = 360, fill_mode='reflect',interpolation_order=3 )
    else:
        gen = ImageDataGenerator(horizontal_flip = False,
                         vertical_flip = False )
        

    X_train, X_val, y_train, y_val = train_test_split(images, newLabels, test_size=1.-train_split, stratify=newLabels, random_state=random_state)  
    
    train_gen = gen.flow( X_train, y_train, batch_size=batch_size )

    
    
    if return_test_params:
        index_train, index_val, _, _ =  train_test_split(np.arange(images.shape[0]), newLabels, test_size=1.-train_split, stratify=newLabels, random_state=random_state)  
        
        for i in allDataParams.keys():
            allDataParams[i] = allDataParams[i][ index_val ]
            
        return train_gen, ( X_val,    y_val), allDataParams
      
    else:
        return train_gen, ( X_val,    y_val)
    
    
