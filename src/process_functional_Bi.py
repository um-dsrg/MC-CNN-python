"""
    processing functions used in stereo matching
"""
from datetime import datetime

import numpy as np
import tensorflow as tf

import LibMccnn.util
from LibMccnn.model import NET

import sys
from memory_profiler import profile
from skimage.transform import warp
from skimage.transform import SimilarityTransform

def disparity_selection(cost_volume, disp_list):
    # We are going to use the subpixel refinement adopted in this paper
    # https://ieeexplore.ieee.org/document/6130280
    	
    # Derive the resolution of the image being considered
    ndisp, height, width = cost_volume.shape
    # Initialize the disparity map
    disparity_map = np.ones([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            # Derive the index of the minimum cost
            idx_min = np.argmin(cost_volume[:,h,w])	
            
            # Get the integer disparity which has the smallest value
            d  = disp_list[idx_min]
            
            if idx_min > 0 and idx_min < ndisp-1:
                # Derive the neighbouring and current cost volumes
                C2dp = cost_volume[idx_min-1,h,w]
                C2dm = cost_volume[idx_min+1,h,w]
                C2d  = cost_volume[idx_min,h,w]
                # Compute the sub-pixel disparity map
                disparity_map[h,w] = d - (C2dp - C2dm)/(2*(C2dp + C2dm - 2*C2d))
            else:
                disparity_map[h,w] = d
            
            disparity_map[h,w] = d

    #return disparity_map
    return -disparity_map
 
def left_right_consistency(left_disparity_map, right_disparity_map):
    print("Doing left-right consistency check...")
       
    # Derive the width and height of the disparity
    height, width = left_disparity_map.shape
    
    # Initialize the right disparity map
    right_disparity_map_aligned = np.full(left_disparity_map.shape, np.nan)
    
    for h in range(height):
        for w in range(width):
            if w - int(left_disparity_map[h,w]) >= 0 and w - int(left_disparity_map[h,w]) < width:
                # Align the right disparity map with the left disparity map
                right_disparity_map_aligned[h,w] = right_disparity_map[h,w - int(left_disparity_map[h,w])]    			
    
    # Derive a mask where the difference between the left and aligned right is smaller than 1
    mask_valid = np.abs(left_disparity_map - right_disparity_map_aligned) <= 1
    
    # Initialize the disparity map
    disparity_map = np.full(left_disparity_map.shape, np.nan)
    
    # Derive the disparity map
    disparity_map[mask_valid] = (left_disparity_map[mask_valid] + right_disparity_map_aligned[mask_valid])/2
    
#def left_right_consistency(left_disparity_map, right_disparity_map):
    #print("Doing left-right consistency check...")
    # Derive the width and height of the disparity
    #height, width = left_disparity_map.shape
    
    
    
    
    # Initialize the consistency_map
    #disparity_map = np.zeros([height, width], dtype=np.float32)
    
    '''
    for h in range(height):
        for w in range(width):
			# Get the left disparity pixel and convert it to type int
            left_disparity = left_disparity_map[h, w]
                       
            # disparities that point outside the range
            if w -int(left_disparity) < 0 or w -int(left_disparity) >= width:
                disparity_map[h,w] = np.nan
            else:
                right_disparity = right_disparity_map[h, w-int(left_disparity)]
                if abs(left_disparity - right_disparity) <= 1:
                    # This is a match
                    disparity_map[h,w] = left_disparity
                else:
                    # The rest are marked as occlusion
                    disparity_map[h,w] = np.nan
    '''             
    return disparity_map   
def compute_features(left_image, right_image, left_lbp, right_lbp,patch_height, patch_width, checkpoint,nchannels):
	# Determine the width and height of an image
    height, width = left_image.shape[:2]
    
    # pad images to make the final feature map size = (height, width..)
    auged_left_image = np.zeros([1, height+patch_height-1, width+patch_width-1, 1], dtype=np.float32)
    auged_right_image = np.zeros([1, height+patch_height-1, width+patch_width-1, 1], dtype=np.float32)
    
    if nchannels == 2:
		# Initialize the augmented lbp feature
        auged_left_lbp = np.zeros([1, height+patch_height-1, width+patch_width-1, 1], dtype=np.float32)
        auged_right_lbp = np.zeros([1, height+patch_height-1, width+patch_width-1, 1], dtype=np.float32)
     
    
    # derive the top-left corner of the augmented (or padded) left and right images 
    row_start = int((patch_height - 1)/2)
    col_start = int((patch_width - 1)/2)
    # Center the left image within the augmented left image. The left (or right) images therefore was padded by (patch_size-1)/2
    # row and column pixels
    auged_left_image[0, row_start: row_start+height, col_start: col_start+width] = left_image
    auged_right_image[0, row_start: row_start+height, col_start: col_start+width] = right_image
    
    if nchannels == 2:
        auged_left_lbp[0, row_start: row_start+height, col_start: col_start+width] = left_lbp
        auged_right_lbp[0, row_start: row_start+height, col_start: col_start+width] = right_lbp	

    # TF placeholder for graph input with the same dimensions as the augmented left and right images - to be inputted to the network
    x = tf.compat.v1.placeholder(tf.float32, shape=[1, height+patch_height-1, width+patch_width-1, nchannels])  

    # Initialize model by specifying the size of the patch and setting the batch size to 1
    #model = NET(x, input_patch_size = patch_height, batch_size=1)
    model = NET(x, input_patch_size = patch_height, batch_size=1,nchannels=nchannels)
    saver = tf.compat.v1.train.Saver(max_to_keep=10)
    
    features = model.features

    # Initialize the left and right features    
    features_left = np.ndarray([auged_left_image.shape[0], auged_left_image.shape[1], auged_left_image.shape[2], nchannels], dtype=np.float32)
    features_right = np.ndarray([auged_right_image.shape[0], auged_right_image.shape[1], auged_right_image.shape[2], nchannels], dtype=np.float32)

    # Put grayscale image in the first channel
    features_left[:,:,:,0] = auged_left_image[:,:,:,0]
    features_right[:,:,:,0] = auged_right_image[:,:,:,0]
    if nchannels == 2:
		# Put the lbp features as the second channel
        features_left[:,:,:,1] = auged_left_lbp[:,:,:,0]
        features_right[:,:,:,1] = auged_right_lbp[:,:,:,0]
    # compute features on both images
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                        log_device_placement=False, \
                        allow_soft_placement=True, \
                        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))) as sess:
        # Set the model to point to the checkpoint as stored after training
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        
        #print("Restoring from {}...".format(ckpt.model_checkpoint_path))
        # Restore the model as derived during training
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Pass the left and right images through the network to extract the features
        #featuresl = sess.run(features, feed_dict = {x: auged_left_image}) 
        #featuresr = sess.run(features, feed_dict = {x: auged_right_image})
        featuresl = sess.run(features, feed_dict = {x: features_left})  
        featuresr = sess.run(features, feed_dict = {x: features_right})
        
        # Squeeze the featuresfeatures of dimensions [height,width,64] 
        # - before it was [1,height,width,64]
        featuresl = np.squeeze(featuresl, axis=0)
        featuresr = np.squeeze(featuresr, axis=0)
        
        #print("{}: features computed done...".format(datetime.now()))

    # clear the used gpu memory
    tf.compat.v1.reset_default_graph()

    return featuresl, featuresr

def compute_cost_volume(featuresl, featuresr, dmin, dmax):
    print("Computing cost_volume ...".format(datetime.now()))
    # Derive the number of disparities
    ndisp = dmax - dmin + 1
    
    # Derive the dimensions of the image
    height, width = featuresl.shape[:2]
    # Initialize the cost volume using the left image as reference
    left_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)
    # Initialize the cost volume using the right image as reference
    right_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)
    
    for i, d in enumerate(range(dmin,dmax+1)):
        #print("{}: disparity {} index {}...".format(datetime.now(), d,i))
        
        # Print the disparity
        featuresr_d = np.zeros(featuresr.shape)
        featuresl_d = np.zeros(featuresl.shape)
        
        # Shift the features by d
        for f in range(featuresl.shape[2]):
            # Compute the translation of the feature f
            featuresr_d[:,:,f] = warp(featuresr[:,:,f],SimilarityTransform(translation=(d, 0)))
            featuresl_d[:,:,f] = warp(featuresl[:,:,f],SimilarityTransform(translation=(-d, 0)))
        
        # Shift the features of the right image by d
        #featuresr_d[:,np.max([0,d]):np.min([width+d,width]),:] = featuresr[:,np.max([-d,0]):np.min([width,width-d]),:]
        #featuresl_d[:,np.max([0,d]):np.min([width+d,width]),:] = featuresl[:,np.max([-d,0]):np.min([width,width-d]),:]

        # Compute the dot product
        left_cost_volume[i,:,:] = np.sum(np.multiply(featuresl, featuresr_d), axis=2)
        right_cost_volume[i,:,:] = np.sum(np.multiply(featuresr, featuresl_d), axis=2)
    print("Cost_volume for right image computed...")
    # convert from matching score to cost
    # match score larger = cost smaller
    left_cost_volume =  -1. * left_cost_volume
    right_cost_volume = -1. * right_cost_volume
    return left_cost_volume, right_cost_volume

# disparity prediction
# simple "Winner-take-All"
def disparity_prediction(left_cost_volume, right_cost_volume,ndisp):

    print("Disparity map computation...")
    _, height, width = left_cost_volume.shape
    left_disparity_map = np.ones([height, width], dtype=np.float32)
    right_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            # Find the index of providng the minimum disparity
            left_disparity_map[h, w]  = np.argmin(left_cost_volume[:,h,w]) - ndisp
            right_disparity_map[h, w] = np.argmin(right_cost_volume[:,h,w]) - ndisp

    return left_disparity_map, -right_disparity_map # Return negative of right disparity

   
# interpolation is combining the left and right disparities
def interpolation(left_disparity_map, right_disparity_map, ndisp):
    print("Doing left-right consistency check...")
    # Derive the width and height of the disparity
    height, width = left_disparity_map.shape
    # Initialize the consistency_map
    disparity_map = np.zeros([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
			# Get the left disparity pixel and convert it to type int
            left_disparity = left_disparity_map[h, w]
                       
            # disparities that point outside the range
            if w -int(left_disparity) < 0 or w -int(left_disparity) >= width:
                disparity_map[h,w] = np.nan
            else:
                right_disparity = right_disparity_map[h, w-int(left_disparity)]
                if abs(left_disparity - right_disparity) <= 1:
                    # This is a match
                    disparity_map[h,w] = left_disparity
                else:
                    # The rest are marked as occlusion
                    disparity_map[h,w] = np.nan
                    
    print("Interpolation done...")
    return disparity_map   
