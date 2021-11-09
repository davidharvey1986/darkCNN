from globalVariables import *
from astropy.io import fits
import glob

from tools import rebin

def getData( testTrainSplit = 0.3, binning = 20, allDataFile = None, \
            dynamic='all', xrayConcLim=0.2,                          \
            attributes = ['redshift', 'mass'], massCut=0,            \
            indexFileRoot = 'pickles/testIndexes',                   \
            nChannels = 1):
    
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
    
    - dynamic : string : 'all', 'relaxed', 'merging' -> defines the state of the cluster i want to return
    - xrayConcLim : float : the limit separation for the dynamiic state of a cluster
    - nChannels :  float : number of channels to be read, 1, 2, 3 : total matter, xray, baryonic
    - attributes : a list of M attributes to be returned with the training and test sets
    
    returns : tuple : Tuple of the (N x training images, numpy array of N x M attritues, N training labels),
                dictionary of test sets for each dark matter model, with J test clusters with their attrbutes and 
                labels
    '''
    if allDataFile is None:
        allDataFile = '../examples/pickles/allSimData_binning_%i.pkl' % binning
    
    if not os.path.isfile( allDataFile ):
        raise ValueError( "cant find data file, run rebinAllData.py")
        
    allDataParams, images = pkl.load(open(allDataFile, 'rb'))

    #images is a list so make it an array
    images = np.array(images)
    
    for i in allDataParams.keys():
        allDataParams[i] = np.array(allDataParams[i])
    
    selectGalaxy = np.zeros(len(allDataParams['label']))
    
    for i in redshifts:
        selectGalaxy[ (i == allDataParams['redshift']) & (allDataParams['mass'] > massCut) ] = 1

    for i in models:
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
    testSets = {}
    allTestIndexes = np.array([])
    allIndexes = np.arange(images.shape[0]) 
    
    for iLabel in labelClasses:
        
        getLabelIndex = np.where(  labels == iLabel )
        
        print(len(getLabelIndex[0]))
        
        nTest = np.int(testTrainSplit*len(getLabelIndex[0]))
        
        indexFile = "%s_%0.3f_%s_%i.pkl" % (indexFileRoot,testTrainSplit,iLabel,binning)

        print("nTests is %i" % nTest)
        
        if os.path.isfile(indexFile):
            testIndexes = pkl.load( open( indexFile, 'rb'))
        else:
            testIndexes = np.random.choice( getLabelIndex[0], replace=False, size=np.int(nTest) )
            pkl.dump( testIndexes, open( indexFile, 'wb'))

        if dynamic == 'relaxed':
            testIndexes = np.array([ i for i in testIndexes if allDataParams['xrayConc'][i] > xrayConcLim ])
        elif dynamic == 'merging':
            testIndexes = np.array([ i for i in testIndexes if allDataParams['xrayConc'][i] < xrayConcLim ])

        
        testSets[iLabel] = {}
        testSets[ iLabel ]['labels'] = newLabels[testIndexes][:,np.newaxis]
        testSets[ iLabel ]['images'] = images[testIndexes, :, :, :nChannels]
        testSets[ iLabel ]['xrayConc'] = allDataParams['xrayConc'][testIndexes]
        testSets[ iLabel ]['clusterID'] = allDataParams['clusterID'][testIndexes, np.newaxis]
        testSets[ iLabel ]['redshift'] = allDataParams['redshift'][testIndexes]
        testSets[ iLabel ]['mass'] = allDataParams['mass'][testIndexes]

        
        allTestIndexes = np.append(allTestIndexes, testIndexes)
                
    if dynamic == 'relaxed':
        trainIndexes = np.array( [ i for i in allIndexes if i not in allTestIndexes if allDataParams['xrayConc'][i] > xrayConcLim ])
    elif dynamic == 'merging':
        trainIndexes = np.array( [ i for i in allIndexes if i not in allTestIndexes if allDataParams['xrayConc'][i] < xrayConcLim  ])
    else:
        trainIndexes = np.array( [ i for i in allIndexes if i not in allTestIndexes ])
    
    
    attributes = np.array([allDataParams[i][trainIndexes] for i in attributes]).T
    
    return (images[trainIndexes, :, :, :nChannels], attributes, newLabels[trainIndexes][:,np.newaxis]), testSets
        
    
       
        
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


       
        