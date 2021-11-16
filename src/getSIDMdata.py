from .globalVariables import *
from astropy.io import fits
import glob

from tools import rebin

def getData( testTrainSplit = 0.3, binning = 20, allDataFile = None, \
            attributes = ['redshift', 'mass'], massCut=0,            \
            indexFileRoot = 'pickles/testIndexes',                   \
            nChannels = 1,                                           \
            simulationNames = ['CDM','SIDM0.1','SIDM0.3','SIDM1']):
    
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
    - indexFileRoot : i will need to store the indexes for the test set so when i loadthe data and models,
                      it remembers which clusters were train and which were test.
    - nChannels :  float : number of channels to be read, 1, 2, 3 : total matter, xray, baryonic
    - attributes : a list of M attributes to be returned with the training and test sets
    - simulationNames : a list of the names of simulations I want to return
    
    returns : tuple : Tuple of the (N x training images, numpy array of N x M attritues, N training labels),
                dictionary of test sets for each dark matter model, with J test clusters with their attrbutes and 
                labels
    '''
    if allDataFile is None:
        allDataFile = '../../examples/pickles/allSimData_binning_%i.pkl' % binning
    
    if not os.path.isfile( allDataFile ):
        raise ValueError( "Cant find data file %s, run rebinAllData.py" % allDataFile)
        
    allDataParams, images = pkl.load(open(allDataFile, 'rb'))
    dataParamsKeys = list(allDataParams.keys())
    dataParamsKeys.append("images")            
                         
    #images is a list so make it an array
    images = np.array(images)
    
    for i in allDataParams.keys():
        allDataParams[i] = np.array(allDataParams[i])
    
    selectGalaxy = np.zeros(len(allDataParams['label']))
    
    selectGalaxy[ (allDataParams['mass'] > massCut) ] = 1

    for i in simulationNames:
        modelMatch = np.array([ i in iSim for iSim in allDataParams['sim']])
        selectGalaxy[ modelMatch ] += 1
    
    images = images[ selectGalaxy==2, :, :, :]

    for i in allDataParams.keys():
        allDataParams[i] = allDataParams[i][ selectGalaxy == 2 ]
        
    labels = np.array(allDataParams['label'])
    labelClasses =  np.unique(labels)
    newLabels = np.zeros(labels.shape[0])
    
    for i, iClass in enumerate(labelClasses):
        newLabels[ labels == iClass ] = i
        

    #I need a training set that has nTest taken from each class so i can train one model and
    #test over each scenario
    testSet = {}
    #initialise the different keys of the dict
    for iKey in dataParamsKeys:
        testSet[iKey] = np.array([])
    
    
    allTestIndexes = np.array([])
    allIndexes = np.arange(images.shape[0]) 
    
    for labelIndex, iLabel in enumerate(labelClasses):
        
        getLabelIndex = np.where(  labels == iLabel )
                
        nTest = np.int(testTrainSplit*len(getLabelIndex[0]))
        
        indexFile = "%s_%0.3f_%s_%i.pkl" % (indexFileRoot, testTrainSplit, iLabel, binning)

        
        if os.path.isfile(indexFile):
            testIndexes = pkl.load( open( indexFile, 'rb'))
        else:
            testIndexes = np.random.choice( getLabelIndex[0], replace=False, size=np.int(nTest) )
            pkl.dump( testIndexes, open( indexFile, 'wb'))

        for iKey in dataParamsKeys:
            if iKey == 'images':
                if labelIndex == 0:
                    testSet[iKey] =  images[testIndexes, :, :, :nChannels]
                else:
                    testSet[iKey] = np.vstack((testSet[iKey], images[testIndexes, :, :, :nChannels]))

            else:  
                testSet[iKey] = np.append(testSet[iKey], allDataParams[iKey][testIndexes])
        
        allTestIndexes = np.append(allTestIndexes, testIndexes)
        
    print("Number of Samples in the Test Set is %i" % testSet["label"].shape)

    trainingSet = {}
    #all the training indexes are those that are not in the test indexes
    trainIndexes = np.array( [ i for i in allIndexes if i not in allTestIndexes ])
    for iAtt in dataParamsKeys:
        if iAtt == 'images':
            trainingSet[iAtt] = images[trainIndexes, :, :, :nChannels]    
        else:
            trainingSet[iAtt] = allDataParams[iAtt][trainIndexes]
        
    #Add an axis to the labels to conform to tensorflow.
    trainingSet['label'] = trainingSet['label'][:, np.newaxis]
    testSet['label'] = testSet['label'][:, np.newaxis]
                 
    return trainingSet, testSet    
    
       
        
def getCrossSection( simName  ):
    '''
    Get the equivalent self-interaction cross-section
    for a given simulation run
    
    '''
    crossSections = \
        {'CDM_low':0.,
         'CDM_hi':0.,
         'CDM':0., \
         'SIDM1':1.,\
         'SIDM0.1':0.1,\
         'SIDM0.3':0.3 }
    
    return crossSections[simName]


       
        
