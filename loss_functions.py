import numpy as np
import keras as k
import math



# Dice function loss optimizer
def dice_loss(y_true, y_pred):

#     szp = k.get_variable_shape(y_pred)
    szp = k.backend.int_shape(y_pred)
    img_len = szp[1]*szp[2]*szp[3]

    y_true = k.backend.reshape(y_true,(-1,img_len))
    y_pred = k.backend.reshape(y_pred,(-1,img_len))

    ovlp = k.backend.sum(y_true*y_pred,axis=-1)

    mu = k.backend.epsilon()
    dice = (2.0 * ovlp + mu) / (k.backend.sum(y_true,axis=-1) + k.backend.sum(y_pred,axis=-1) + mu)
    loss = -1*dice

    return loss

# Dice function loss optimizer
# During test time since it includes a discontinuity
# def dice_loss_test(y_true, y_pred, thresh = 0.5):
    
#     recon = np.squeeze(y_true)
#     y_pred = np.squeeze(y_pred)
#     y_pred = (y_pred > thresh)*y_pred

#     szp = y_pred.shape
#     img_len = szp[1]*szp[2]*szp[3]

#     y_true = np.reshape(y_true,(-1,img_len))
#     y_pred = np.reshape(y_pred,(-1,img_len))

#     ovlp = np.sum(y_true*y_pred,axis=-1)

#     mu = k.backend.epsilon()
#     dice = (2.0 * ovlp + mu) / (np.sum(y_true,axis=-1) + np.sum(y_pred,axis=-1) + mu)

#     return dice

def dice_loss_test(y_true, y_pred, thresh = 0.5):
    
    recon = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_pred = (y_pred > thresh)*y_pred

    szp = y_pred.shape
    img_len = np.product(szp[1:])

    y_true = np.reshape(y_true,(-1,img_len))
    y_pred = np.reshape(y_pred,(-1,img_len))

    ovlp = np.sum(y_true*y_pred,axis=-1)

    mu = k.backend.epsilon()
    dice = (2.0 * ovlp + mu) / (np.sum(y_true,axis=-1) + np.sum(y_pred,axis=-1) + mu)
    

    return np.mean(dice)

def dice_loss_test_volume(y_true, y_pred, thresh = 0.5):
    # Designed to work with one volume, not a batch of volumes
    
    g = np.squeeze(y_true)
    p = np.squeeze(y_pred)
    p = (y_pred > thresh)*1#*y_pred
    
    g = g.flatten()
    p = p.flatten()

    ovlp = np.sum(g*p)

    mu = k.backend.epsilon()
    dice = (2.0 * ovlp + mu) / (np.sum(g) + np.sum(p) + mu)

    return dice

def jaccard(vol1, vol2):
    return np.sum(np.logical_and(vol1,vol2))/np.sum((np.logical_or(vol1,vol2)))

def coefficient_of_variation(y_true, y_pred):
    return ((np.mean((y_pred-y_true)**2))**.5) /  np.mean(y_true)    



def rohan_loss(y_true, y_pred):


#     szp = k.get_variable_shape(y_pred)
    szp = k.backend.int_shape(y_pred)
    img_len = szp[1]*szp[2]*szp[3]

    y_true = k.backend.reshape(y_true,(-1,img_len))
    y_pred = k.backend.reshape(y_pred,(-1,img_len))
    loss = (.9 * np.mean(-1*np.log(y_pred[y_true == 1]))) + (.1*np.mean(-1*np.log(1 - y_pred[y_true == 0])))


    return loss

def weighted_binary_crossentropy(y_true, y_pred):
    
    szp = k.backend.int_shape(y_pred)
    img_len = szp[1]*szp[2]*szp[3]

    y_true = k.backend.reshape(y_true,(-1,img_len))
    y_pred = k.backend.reshape(y_pred,(-1,img_len))

    # Original binary crossentropy (see losses.py):
    # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    # Calculate the binary crossentropy
    b_ce = k.backend.binary_crossentropy(y_true, y_pred)

    # Apply the weights
    weight_vector = y_true * .9999 + (1. - y_true) * .0001
    weighted_b_ce = weight_vector * b_ce

    # Return the mean error
    return k.backend.mean(weighted_b_ce)

