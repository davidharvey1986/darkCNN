# darkCNN
A convolutional neural network to constrain dark matter


The objective of this algorithm is to take in simulated data, augment it and then train a convolutional neural network with multiple channels to estimate the model of dark matter it came from.

This has been run and tested in a virtual environment, which it is highly recommended carried out.

Pre-reduced data that has been extracted from simulations consist of 2 dimensional maps, each 2x2 Mpc, projected to 10 Mpc made up of three "channels": 
1. Total projected mass maps
2. Xray lumonisity maps
3. Baryonic Matter maps

There are four dark matter models:
1. CDM
2. SIDM0.1 (SIDM with a cross-section of 0.1 cm2/g )
3. SIDM0.3
4. SIDM1

Each with four redshift slices: z = 0., 0.125, 0.250, 0.375 

This data cube is binned in to 10 kpc resolution pixels, resulting in maps of 200 x 200 pixels. Each map comes with auxillary data or "attributes", consisting of
1. Redshift
2. Cluster mass
3. Xray concentration ( equal to S(<100 kpc) / S(<400 kpc) )

Dark matter self-interactions are rotationally symmetric so augmentations rotating and flipping can artificially increase the sample. This is carried out with
10 rotations and a random flipping of the image. 

To Run
-------
Make sure the python interpreter in main.py is correct. 
./main.py

Python Modules
--------------
augmentData.py : augment input data to increase sample size
getSIDMdata.py : get training and tests with labels
globalVariables.py : standarised modules required
inceptionModules.py : the inception layers for the main model
main.py : the main modules that trains and saves the models
mainModel.py : contains two models : mainModel - from the Merten paper, and a simple model
tools.py

Notebooks
---------
testModel.ipynb : a


Requirements
--------------
Python == 3.7
Tensorflow == 2.4.1
Keras  == 2.4.0
astropy == 4.0
pickle
numpy == 1.19.5
matplotlib == 3.3.3
os
scipy == 1.4.1

Example
--------
Make sure that jupyter-lab is installed on the venv via
pip install jupyterlab
To run the example, enter the directory and simply type
darkCNN

Then to run the jupyter-lab notebook ensure to add the virtual environment kernel via
ipython kernel install --name "darkCNN" --user






