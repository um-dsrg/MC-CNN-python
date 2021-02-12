"""
    data generator class
"""
import os
import numpy as np
import cv2
import copy
from LibMccnn.util import readPfm
import random
from tensorflow import expand_dims
#import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

class ImageDataGenerator:
    """
        input image patch pairs generator
    """
    def __init__(self, left_image_list_file, shuffle=False, 
                 patch_size=(11, 11),
                 in_left_suffix='im0.png',
                 in_right_suffix='im1.png',
                 gtX_suffix='disp0GT.pfm',
                 # tunable hyperparameters
                 # see origin paper for details
                 dataset_neg_low=1.5, dataset_neg_high=6,
                 dataset_pos=0.5,nchannels=1
                 ):
        """
            left_image_list_file: path to text file containing training set left image PATHS, one path per line
            list of left image paths are formed directly by reading lines from file 
            list of corresponding right image and ground truth disparity image paths are 
            formed by replacing in_left_suffix with in_right_suffix and gt_suffix from every left image path
        """
                
        # Init params
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.in_left_suffix = in_left_suffix
        self.in_right_suffix = in_right_suffix
        self.gtX_suffix = gtX_suffix
        self.dataset_neg_low = dataset_neg_low
        self.dataset_neg_high = dataset_neg_high
        self.dataset_pos = dataset_pos
        self.nchannels = nchannels
        #self.radius = radius

        # the pointer indicates which image are next to be used
        # a mini-batch is fully constructed using one image(pair)
        self.pointer = 0

        self.read_image_list(left_image_list_file)
        self.prefetch()
        if self.shuffle:
            self.shuffle_data()

    def read_image_list(self, image_list):
        """
            form lists of left, right & ground truth paths
        """
        #with open(image_list) as f:
        #    print(f)

        self.left_paths = []
        self.right_paths = []
        self.gtX_paths = []

        for l in image_list:
            sl = os.path.join(l.strip(),self.in_left_suffix)
            self.left_paths.append(sl)
            self.right_paths.append(sl.replace(self.in_left_suffix, self.in_right_suffix))
            self.gtX_paths.append(sl.replace(self.in_left_suffix, self.gtX_suffix))
        # store total number of data
        self.data_size = len(self.left_paths)
        print("total image num in file is {}".format(self.data_size))

    def prefetch(self):
        """
            prefetch all images
            generally dataset for stereo matching contains small number of images
            so prefetch would not consume too much RAM
        """
        self.left_images = []
        self.right_images = []
        self.gtX_images = []
        # Create an empty list of lbps
        if self.nchannels == 2:
            self.left_lbps = []
            self.right_lbps = []

        for _ in range(self.data_size):

            # NOTE: read image as grayscale as the origin paper suggested
            #left_image = cv2.imread(self.left_paths[_], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            #right_image = cv2.imread(self.right_paths[_], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            left_image = cv2.imdecode(np.fromfile(self.left_paths[_],dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            right_image = cv2.imdecode(np.fromfile(self.right_paths[_], dtype=np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

            # preprocess images by subtracting the mean and dividing by the standard deviation
            # as the paper described
            left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
            right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
            if self.nchannels == 2:
                # Load the lbp data
                left_lbp = np.load(self.left_lbp_paths[_]).astype(np.float32)
                right_lbp = np.load(self.right_lbp_paths[_]).astype(np.float32)
            
                # Normalize using zero mean and unit variance
                left_lbp = (left_lbp - np.mean(left_lbp,axis=(0,1))) / np.std(left_lbp, axis=(0,1))
                right_lbp = (right_lbp - np.mean(right_lbp, axis=(0,1))) / np.std(right_lbp, axis=(0,1)) 
            
            # Put the pre-processed images within the lists
            self.left_images.append(left_image)
            self.right_images.append(right_image)
            self.gtX_images.append(readPfm(self.gtX_paths[_]))
            if self.nchannels == 2:
                self.left_lbps.append(left_lbp)
                self.right_lbps.append(right_lbp)
                
        print("prefetch done")

    def shuffle_data(self):
        """
            Random shuffle the images and labels
        """
        # Copy the paths
        left_paths = copy.deepcopy(self.left_paths)
        right_paths = copy.deepcopy(self.right_paths)
        gtX_paths = copy.deepcopy(self.gtX_paths)
        if self.nchannels == 2:
            left_lbp_paths = copy.deepcopy(self.left_lbp_paths)
            right_lbp_paths = copy.deepcopy(self.right_lbp_paths)
            		
        # Copy the images
        left_images = copy.deepcopy(self.left_images)
        right_images = copy.deepcopy(self.right_images)
        gtX_images = copy.deepcopy(self.gtX_images)
        if self.nchannels == 2:
            left_lbps = copy.deepcopy(self.left_lbps)
            right_lbps = copy.deepcopy(self.right_lbps)			
            
        # Reinitialize the paths  
        self.left_paths = []
        self.right_paths = []
        self.gtX_paths = []
        if self.nchannels == 2:
            self.left_lbp_paths = []
            self.right_lbp_paths = []
        # Reinitialize the images    
        self.left_images = []
        self.right_images = []
        self.gtX_images = []
        if self.nchannels == 2:
            self.left_lbps = []
            self.right_lbps = [] 

        # create list of permutated index and shuffle data accordingly
        idx = np.random.permutation(self.data_size)
        for i in idx:
			# Permute the paths
            self.left_paths.append(left_paths[i])
            self.right_paths.append(right_paths[i])
            self.gtX_paths.append(gtX_paths[i])
            if self.nchannels == 2:
                self.left_lbp_paths.append(left_lbp_paths[i])
                self.right_lbp_paths.append(right_lbp_paths[i])
            # Permute the images
            self.left_images.append(left_images[i])
            self.right_images.append(right_images[i])
            self.gtX_images.append(gtX_images[i])
            if self.nchannels == 2:
                self.left_lbps.append(left_lbps[i])
                self.right_lbps.append(right_lbps[i])
                
    def reset_pointer(self):
        """
            reset pointer to beginning of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
            This function reads the next left, right and gt images, 
            and random pick batch_size patch pairs from these images to 
            construct the next batch of training data

            NOTE: one batch consists of 1 left image patch, and 2 right image patches,
            which consists of 1 positive sample and 1 negative sample
            NOTE: in the origin MC-CNN paper, the authors propose to use various data augmentation strategies 
            to enhance the model generalization. Here I do not implement those strategis but I believe it's no
            difficult to do that.
        """
        # Get next batch of image (path) and labels
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gtX_path = self.gtX_paths[self.pointer]
        if self.nchannels == 2:
            left_lbp_path = self.left_lbp_paths[self.pointer]
            right_lbp_path = self.right_lbp_paths[self.pointer]

        left_image = self.left_images[self.pointer]
        right_image = self.right_images[self.pointer]
        gtX_image = self.gtX_images[self.pointer]
        if self.nchannels == 2:
            left_lbp = self.left_lbps[self.pointer]
            right_lbp = self.right_lbps[self.pointer]
        
        assert left_image.shape == right_image.shape
        assert left_image.shape[0:2] == gtX_image.shape
        height, width = left_image.shape[0:2]

        # random choose pixels around which to pick image patchs
        rows = np.random.permutation(height)[0:batch_size]
        cols = np.random.permutation(width)[0:batch_size]

        # rule out those pixels with disparity inf and occlusion
        for _ in range(batch_size):
            while gtX_image[rows[_], cols[_]] == float('inf') or \
                  int(gtX_image[rows[_], cols[_]]) > cols[_]:
                # random pick another pixel
                rows[_] = random.randint(0, height-1)
                cols[_] = random.randint(0, width-1)

        # augment raw image with zero paddings 
        # this prevents potential indexing error occurring near boundaries
        auged_left_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 1], dtype=np.float32)
        auged_right_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 1], dtype=np.float32)
        if self.nchannels == 2:
            auged_left_lbp = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 1], dtype=np.float32)
            auged_right_lbp = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 1], dtype=np.float32)

        # NOTE: patch size should always be odd
        rows_auged = int((self.patch_size[0] - 1)/2)
        cols_auged = int((self.patch_size[1] - 1)/2)
        auged_left_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width, 0] = left_image
        auged_right_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width, 0] = right_image
        if self.nchannels == 2:
            auged_left_lbp[rows_auged: rows_auged+height, cols_auged: cols_auged+width, 0] = left_lbp
            auged_right_lbp[rows_auged: rows_auged+height, cols_auged: cols_auged+width, 0] = right_lbp

        # pick patches
        patches_left = np.ndarray([batch_size, self.patch_size[0], self.patch_size[1], self.nchannels], dtype=np.float32)
        patches_right_pos = np.ndarray([batch_size, self.patch_size[0], self.patch_size[1], self.nchannels], dtype=np.float32)
        patches_right_neg = np.ndarray([batch_size, self.patch_size[0], self.patch_size[1], self.nchannels], dtype=np.float32)
             			 
        for _ in range(batch_size):
            row = rows[_]
            col = cols[_]
            # Get the left patch
            patch_left = auged_left_image[row:row + self.patch_size[0], col:col+self.patch_size[1]]
            
            if self.nchannels == 2:
                patch_lbp_left   = auged_left_lbp[row:row + self.patch_size[0], col:col+self.patch_size[1]]
            # Put the channels as input for the left network
            patches_left[_,:,:,0] = patch_left.reshape((self.patch_size[0],self.patch_size[1]))
            if self.nchannels == 2:
                patches_left[_,:,:,1] = patch_lbp_left.reshape((self.patch_size[0],self.patch_size[1]))
            
            right_col = col - int(gtX_image[row, col])
 
            # postive example
            # small random deviation added
            #pos_col = right_col
            
            pos_col = -1
            pos_row = -1
            while pos_col < 0 or pos_col >= width:
                pos_col = int(right_col + np.random.uniform(-1*self.dataset_pos, self.dataset_pos))
            # Get the positive right patch
            patch_right_pos = auged_right_image[row:row+self.patch_size[0], pos_col:pos_col+self.patch_size[1]]
            if self.nchannels == 2:
                patch_lbp_right_pos = auged_right_lbp[row:row+self.patch_size[0], pos_col:pos_col+self.patch_size[1]]
            
            # Put the channels as input for the right positive network
            patches_right_pos[_,:,:,0] = patch_right_pos.reshape((self.patch_size[0],self.patch_size[1]))
            if self.nchannels == 2:
                patches_right_pos[_,:,:,1] = patch_lbp_right_pos.reshape((self.patch_size[0],self.patch_size[1]))

            # negative example
            # large random deviation added
            neg_col = -1
            while neg_col < 0 or neg_col >= width:
                neg_dev = np.random.uniform(self.dataset_neg_low, self.dataset_neg_high)
                if np.random.randint(-1, 1) == -1:
                    neg_dev = -1 * neg_dev
                neg_col = int(right_col + neg_dev)
            
            # Get the negative right patch
            patch_right_neg = auged_right_image[row:row+self.patch_size[0], neg_col:neg_col+self.patch_size[1]]
            if self.nchannels == 2:
                patch_lbp_right_neg = auged_right_lbp[row:row+self.patch_size[0], neg_col:neg_col+self.patch_size[1]]
 
            # Put the channels as input for the right negative network
            patches_right_neg[_,:,:,0] = patch_right_neg.reshape((self.patch_size[0],self.patch_size[1]))
            if self.nchannels == 2:
                patches_right_neg[_,:,:,1] = patch_lbp_right_neg.reshape((self.patch_size[0],self.patch_size[1]))

            if False:
                fig = plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(np.squeeze(patch_left))
                plt.title('Anchor'.format(_))

                plt.subplot(2,2,2)
                plt.imshow(np.squeeze(patch_right_pos))
                plt.title('Positive')

                plt.subplot(2,2,4)
                plt.imshow(np.squeeze(patch_right_neg))
                plt.title('Negitive')

                plt.show()

        #update pointer
        self.pointer += 1
        return patches_left, patches_right_pos, patches_right_neg
    

    def next_pair(self):
        # Get next images 
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gtX_path = self.gtX_paths[self.pointer]

        # Read images
        left_image = self.left_images[self.pointer]
        right_image = self.right_images[self.pointer]
        gtX_image = self.gtX_images[self.pointer]
        assert left_image.shape == right_image.shape
        assert left_image.shape[0:2] == gtX_image.shape

        #update pointer
        self.pointer += 1

        return left_image, right_image, gtX_image
    
    def test_mk(self, path):
        if os.path.exists(path):
            return
        else:
            os.mkdir(path)

# just used for debug
if __name__ == "__main__" :
    dg = ImageDataGenerator("/scratch/xz/MC-CNN-python/data/list/train.txt")
    patches_left, patches_right_pos, patches_right_neg = dg.next_batch(128)
    print(patches_left.shape)
    print(patches_right_pos.shape)
    print(patches_right_neg.shape)

