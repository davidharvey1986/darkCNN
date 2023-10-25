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

For this example, the data cube is binned in to 40 kpc resolution pixels, resulting in maps of 50 x 50 pixels. Each map comes with auxillary data or "attributes", consisting of
1. Redshift
2. Cluster mass
3. Xray concentration ( equal to S(<100 kpc) / S(<400 kpc) )

Dark matter self-interactions are rotationally symmetric so augmentations rotating and flipping can artificially increase the sample. This is carried out with
10 rotations and a random flipping of the image. 

Note that the example database that has been packaged with this is a subset of al the data.

To Install - Tested on Fedora 33, Mac OS X Big Sur with python 3.7
----------------------------------------------------------------------
#setup a virtual environment
>> virtualenv darkCNN -p /usr/local/bin/python3.7 
>> 
>> cd darkCNN
>> 
>> source bin/activate
>> 
>> git clone https://github.com/davidharvey1986/darkCNN.git
>> 
>> cd darkCNN
>> 
>> python setup.py install

To run the unit tests run
>> python setup.py test

To Run
-------
>> darkCNN -h #to bring up the help

Example (included in package)
--------------------------------
Make sure that jupyter-lab is installed on the venv via
>> pip install jupyterlab
>> 
To run the example, enter the "example" directory and simply type
>> 
>> darkCNN

Then to run the jupyter-lab notebook ensure to add the virtual environment kernel via
ipython kernel install --name "darkCNN" --user

Python Modules
--------------
augmentData.py : augment input data to increase sample size
getSIDMdata.py : get training and tests with labels
globalVariables.py : standarised modules required
inceptionModules.py : the inception layers for the main model
main.py : the main modules that trains and saves the models
mainModel.py : contains two models : mainModel - from the Merten paper, and a simple model
tools.py : a distribution of tools used in the suite of code

Requirements
--------------
Python == 3.7
tensorFlow
astropy==4.0
matplotlib==3.3.3
OyQt5
keras
scipy


A few words on the data
-----------------------

>> params, images = pkl.load(open(data, "rb"))


params is a dict
images is a list

Meta Data information:
- lensing_norm : numpy array of length N : the maximum value of the total matter map before normalisation
- label : numpy array of length N floats : The Value of the cross-section that cluster was simulated with, can be 0, 0.1, 0.3, 1.0, all velocity independent.
- redshift: numpy array of length N floats: redshift of the cluster in the simulation , can be 0, 0.125, 0.25 or 0.375.
- clusterID:  numpy array of length N strings / int : the Friends of Friends label of the cluster. Do not necessarily match between simulations
- mass: numpy array of length N floats : virial mass of the cluster
- galaxy_catalogue : a list of N elements, with each element a recarray containing a catalogue of the galaxies that are members of the cluster in question (each with differing number of galaxies ) , where the fields in the recarray are
- fof : string -> the clusterID
- x : float -> the relative position of the galaxy in the cluster in kpc
- y : float -> the relative position of the galaxy in the cluster in kpc
- z :float -> the relative position of the galaxy in the cluster in Mpc
- m200 : float : log total mass of the galaxy
- mstar_100 : float : log total stellar mass within 100 kpc of the galaxy
- BCG_stellar_conc : the ratio between total stellar mass < 30kpc / < 100kpc
- BCG_e1 & BCG_e2 : numpy array of length N : the two components of ellipticity of the central BCG, where total ellipticity = sqrt(e1^2+e2^2)
- xrayConc : numpy array of length N : the ratio between total xray emmissivity < 100kpc / < 400kpc -> where greater than 0.2 = relaxed, less than 0.2 = merging.
- xray_norm : numpy array of length N : the maximum value of the Xray emission map before normalisation





