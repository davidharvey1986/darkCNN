import unittest
from darkCNN.augmentData import augmentData
import numpy as np

class TestDarkCNNMethods(unittest.TestCase):

    def test_rotateData(self):
        #Test that the image is correctly rotated 90 degrees
        
        testImage, yImage = np.meshgrid( np.arange(10), np.ones(10))
        testImage = testImage[np.newaxis, :, :, np.newaxis]
        testLabel = np.array(['test'])[:, np.newaxis]
        
        rotatedData, rotatedLabel = \
            augmentData( testImage, testLabel, fixedRotation=90., nRotations=1, flip=False)
        
        self.assertTrue( np.allclose(rotatedData[0,0,:,0], testImage[0,:,-1,0]))

    def test_augmentDataShape(self):
        #Test that the image is correctly rotated 90 degrees
        
        testImage, yImage = np.meshgrid( np.arange(10), np.ones(10))
        testImage = testImage[np.newaxis, :, :, np.newaxis]
        testLabel = np.array(['test'])[:, np.newaxis]
        
        rotatedData, rotatedLabel = \
            augmentData( testImage, testLabel, fixedRotation=90., nRotations=10, flip=False)
        
        self.assertEquals( rotatedData.shape[0], 10)

if __name__ == '__main__':
    unittest.main()