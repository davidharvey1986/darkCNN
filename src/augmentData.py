from .globalVariables import *

from scipy.ndimage import rotate

def augmentData( train_images, train_labels, flip=True, nRotations=10, fixedRotation=None):
    '''
    OBJECTIVE
    ---------
    To artifically increase the training size I can augment the data. I do this in two ways
    1. I randomly flip the image
    2. I rotate the image by some angle (and crop to the same size)
        
    Notes
    ------
    -> I do not rescale, translate or alter the brightness as in the galaxy zoo paper since
    seems to have no impact.
    -> More than 10 nRotations crashes the computer as this requires too much memory, plus 
    it seems that more than 10 is irrelavant.
    
    INPUTS
    ------
    - train_images : a numpy array of N images x (image dimensions)
    - train_labels : a numpy array of N labels 
    
    OPTIONAL ARGUMENTS
    ------------------
    flip : If true randomly flip the image before rotating
    nRotations : the number of random rotations of an image I carry out
    fixedRotation : None or float : if None then use a random rotation, otherwise use float
    
    RETURNS
    -------
    allRotatedImages :  a numpy array of N * nRotations x (image dimensions)
    newLabels : a numpy array of N * nRotations long
    
    '''
    
    
    #Get the image dimensions
    imageSize = train_images.shape[1]
    #Set up the required lists to be returned
    allRotatedImages = []
    allRotatedLabels = []
    
    #Loop through each image and start rotating
    for iImage in range(train_images.shape[0]):
            
        
        for iRotation in range(nRotations):
            if fixedRotation is None:
                rotAngle = np.random.uniform(0, 360)
            else:
                rotAngle = fixedRotation
                
            if flip:
                if np.random.uniform(0,1) > 0.5:
                    flippedImage = train_images[iImage, :,:, :]
                else:
                    flippedImage = train_images[iImage, ::-1,:, :]
            else:
                flippedImage = train_images[iImage, :,:, :]

                    
            for iChannel in range(flippedImage.shape[-1]):
                    
                rotatedImage = rotate(flippedImage[:,:,iChannel], rotAngle)
                centralPix = rotatedImage.shape[0]//2
                croppedImage = \
                    rotatedImage[centralPix - imageSize // 2 : centralPix + imageSize // 2,\
                                 centralPix - imageSize // 2 : centralPix + imageSize // 2 ]
                        
                if croppedImage.shape != (imageSize,imageSize):
                    raise ValueError("Shape not correct (%i,%i) from rotation %0.3f (%i, %i)" %\
                                         ( croppedImage.shape[0], croppedImage.shape[0],rotAngle, imageSize, imageSize))
                        
                if iChannel == 0:
                    finalImage = croppedImage
                else:
                    finalImage = np.dstack( (finalImage, croppedImage))
                        
            allRotatedImages.append( finalImage )
            allRotatedLabels.append( train_labels[iImage,0])  
            
            
            
    allRotatedImages = np.array(allRotatedImages)
    allRotatedLabels = np.array(allRotatedLabels)
    
    newLabels = allRotatedLabels[:,np.newaxis]
    
    
    #if only one channel, then it will remove it
    if train_images.shape[-1] == 1:
        allRotatedImages = allRotatedImages[:,:,:,np.newaxis]
        
        
    return allRotatedImages, newLabels
