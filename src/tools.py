from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import os

from .globalVariables import *
from .mainModel import simpleModel, InceptionV4, DIBARE
from keras.callbacks import CSVLogger
from scipy.ndimage import rotate
from tqdm import tqdm
import pickle as pkl
from scipy.stats import norm
import sys
from matplotlib.gridspec import GridSpec
import scienceplots
plt.style.use(["science", "grid"])


def rebin(a, shape):
    '''
    Re bin an array in to a new shape, and take the average
    '''
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def plot_confusion_matrix( actuals, predictions, labels=None, ax=None, ylabel=True, cbar=False ):
    if labels is None:
        labels = [ str(i) for i in np.unique(actuals)]
    
    cm = confusion_matrix(actuals, predictions)
    if ax is None:
        ax = plt.gca()
    cax = ax.matshow(cm, cmap = 'Blues')
    if cbar:
        plt.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.xaxis.set_ticks_position('default')
    ax.set_xlabel('Predicted')
    if ylabel:
        ax.set_ylabel('True')
        ax.set_yticklabels([''] + labels)
    else:
        ax.set_yticklabels([''])

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.annotate(round(cm[i,j]/cm[i].sum(),2),xy=\
                    (j,i),horizontalalignment='center',verticalalignment='center',size=15,color='black',xycoords='data')
    #plt.show()
    
    
def get_best_model( image_generator, test_set, base_cnn_input_shape=None, model_name='Inception', 
                   checkpoint_filepath="models", meta=None, epochs=50):
    
    if os.path.isdir(checkpoint_filepath):
        print("Found Model")
        return tf.keras.models.load_model(checkpoint_filepath)
        
    if base_cnn_input_shape is None:
        base_cnn_input_shape = test_set[0][0].shape
        
    num_classes = len(np.unique(test_set[1]))
    
    #get the base cnn
    if model_name.lower() == 'simple':
        model = simpleModel(num_classes)
    elif model_name.lower() == 'dibare':
        model = DIBARE(input_shape=base_cnn_input_shape, classes=num_classes, bn_momentum=.0, 
                                FC1=0,FC2=0,feature_dropout=.33,num_layersA=1,num_layersB=1,num_layersC=1,leak=.03)
    elif model_name.lower() == 'inception':
        model = InceptionV4(input_shape=base_cnn_input_shape, bn_momentum=.0,classes=num_classes,
                                               feature_dropout=.33,num_layersA=1,num_layersB=1,num_layersC=1,leak=.03)
    else:
        raise ValueError("Model name not recognised")
    
    #Now concatenate any meta information
    if meta is not None:
        shape = test_set[0][1][0].shape
        input_layer = Input( shape=shape )
        galaxy_model = tf.keras.layers.Flatten( )( input_layer )
        last_layer = model.get_layer(model.layers[-1].name)

        galaxy_model = tf.keras.layers.Dense(last_layer.output_shape[1])(galaxy_model)
  
        concatenated_model = tf.keras.layers.concatenate( [last_layer.output, galaxy_model], axis=1)
    
        combine = tf.keras.layers.Dense(128)(concatenated_model)
        
        out = tf.keras.layers.Dense(num_classes)(combine)
    
        final_model = Model([model.input, input_layer], out)  
    
    else:
        final_model = model
    
    optimizer = tf.keras.optimizers.Adam( learning_rate=1e-3 ) 
    final_model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    
    history = final_model.fit(
      x=image_generator,
      validation_data=test_set,
      epochs=epochs, verbose=1, callbacks=[model_checkpoint_callback])
    
    return final_model
    
def get_predictions_per_subset( probability, n_samples_per_subset, cross_sections = [0.,0.1,1.0], return_weights=False):
    
    cross_sections = np.array(cross_sections)
    nMonte_carlo = probability.shape[0]
    nClusters = probability.shape[1]
    nDM_Models =   probability.shape[2]

    nSubSets = nClusters//n_samples_per_subset
    subset_means, subset_stds, prediction, prediction_err = [], [], [], []
    
    for iSubSet in range(nSubSets):
        this_subset = np.arange(iSubSet*n_samples_per_subset, min([(iSubSet+1)*n_samples_per_subset, nClusters]))
        final_probs = np.ones( (nMonte_carlo, nDM_Models))

        for iMonteCarlo in range(nMonte_carlo):

            final_probs_per_cluster_per_MC = probability[iMonteCarlo, this_subset,: ]
            #normalise this probability
            final_probs_per_cluster_per_MC = final_probs_per_cluster_per_MC / np.sum(final_probs_per_cluster_per_MC,axis=1)[:,np.newaxis]

            for iCluster in range(final_probs_per_cluster_per_MC.shape[0]):
                final_probs[iMonteCarlo] *= final_probs_per_cluster_per_MC[iCluster]
                final_probs[iMonteCarlo] /= np.sum(final_probs[iMonteCarlo])
                
        final_probs_all = np.mean(final_probs, axis=0)
        
        
        #newprob = np.sum(probability[:, this_subset,: ],axis=0)
        #newprob /= np.sum(newprob,axis=1)[:,np.newaxis]
        
        #cluster_means = np.sum(newprob*[0.,0.1,1.0],axis=1)/np.sum(newprob,axis=1)
        #cluster_stds = np.sqrt( np.sum(newprob*( np.array([cross_sections - i for i in cluster_means]) )**2, axis=1))
        
        
        #subset_means.append(np.mean(cluster_means))
        #subset_stds.append(np.sqrt(np.sum(cluster_stds**2))/len(np.sqrt(cluster_stds)))
        
    #return np.array(subset_means), np.array(subset_stds)
        
    
        #prediction.append(np.exp(np.average([np.log(0.01),np.log(0.01),np.log(0.01),np.log(0.1),0.], weights=final_probs_all
        if return_weights:
            prediction.append( final_probs )
        else:
            pred = np.average(cross_sections, weights=final_probs_all)

            prediction.append(pred)
            prediction_err.append( np.sqrt(np.sum(final_probs_all*(cross_sections-pred)**2)/np.sum(final_probs_all)/n_samples_per_subset))
                
                
    return np.array(prediction), np.array(prediction_err)