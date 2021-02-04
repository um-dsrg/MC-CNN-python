import argparse
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# This script will be used to generate a set of lbps at different scale for each trainning and validation stereo pairs

def update_filenames (train_file, dataset_foldername):
    with open(train_file, 'r') as reader:
        lines = reader.read().splitlines()
        reader.close()
    filenames = []   
    for line in lines:
        # Derive the new filename
        filenames.append(os.path.join(dataset_foldername, line))
        
    return filenames

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Generate LBPs at different scales")
parser.add_argument("--train_file", type=str, required=True, help="path to file containing training  \
                    left_image_list_file s, should be list_dir/train.txt(val.txt)")
parser.add_argument("--val_file", type=str, required=True, help="path to file containing validation \
                    left_image_list_file s, should be list_dir/train.txt(val.txt)")
parser.add_argument("--dataset_foldername",type=str,required=True,help="folder path where the dataset is stored")


def main():
    args = parser.parse_args()
    
    # Get the training and validation files
    train_file = args.train_file
    val_file = args.val_file
    
    # Update the training files and val files to make them computer independent
    train_filenames = update_filenames(train_file,args.dataset_foldername)
    val_filenames   = update_filenames(val_file,args.dataset_foldername)
    
    # Concatenate the two lists
    filenames = train_filenames + val_filenames
    
    for filename in tqdm(filenames):
		# Derive the filenames of the laft and right image
        left_path = filename
        right_path = filename.replace('im0','im1')
        
        # Load the left and right image
        left_image  = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        # Derive the path where they will be stored
        out_foldername = os.path.join('../data/lbp/', os.path.basename(os.path.dirname(left_path)))

        if not os.path.exists(out_foldername):
            os.mkdir(out_foldername)
        
        for radius in range(1,16):
            # Derive the number of points
            no_points = 8*radius
            
            # Compute the local binary patter for the left and right image
            lbp_left  = local_binary_pattern(left_image, no_points, radius, method='uniform')
            lbp_right = local_binary_pattern(right_image, no_points, radius, method='uniform')
            
            # Derive the out_filename where it will be stored
            out_lbp_left_filename  = os.path.join(out_foldername,'lbp-left-%d.npy'%(radius))
            out_lbp_right_filename = os.path.join(out_foldername,'lbp-right-%d.npy'%(radius))
            
            # Save the lbps
            np.save(out_lbp_left_filename,lbp_left)
            np.save(out_lbp_right_filename,lbp_right)

if __name__ == "__main__":
    main()
