import os
import argparse
import pandas as pd
import numpy as np
from skimage.io import imread
from deepcell.applications import Mesmer

# enrivonment:
# conda create -y -n napari-env -c conda-forge python=3.9 pip zarr
# conda activate napari-env
# python -m pip install "napari[all]"
# pip install deepcell

# Hackathon runs (using all membranes)
# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S06-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S06-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S06-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S06-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S06

# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S03-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/NCC/S03-NCC.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Calbindin/S03-Calbindin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP2/S03-AQP2.tiff,/projects/compsci/USERS/alizae/Hackathon2022/A-SMA/S03-A-SMA.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S03-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S03-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S03-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S03

# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S02-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/NCC/S02-NCC.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Calbindin/S02-Calbindin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP2/S02-AQP2.tiff,/projects/compsci/USERS/alizae/Hackathon2022/A-SMA/S02-A-SMA.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S02-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S02-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S02-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S02

# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S07-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S07-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S07-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S07-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S07

# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S01-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/NCC/S01-NCC.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Calbindin/S01-Calbindin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP2/S01-AQP2.tiff,/projects/compsci/USERS/alizae/Hackathon2022/A-SMA/S01-A-SMA.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S01-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S01-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S01-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S01

# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S05-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S05-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S05-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S05-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S05

# python -u /home/whitebr/run-mesmer.py --membrane-files /projects/compsci/USERS/alizae/Hackathon2022/Podocalyxin/S08-Podocalyxin.tiff,/projects/compsci/USERS/alizae/Hackathon2022/AQP1/S08-AQP1.tiff,/projects/compsci/USERS/alizae/Hackathon2022/Uromodulin/S08-Uromodulin.tiff --nuclear-file /projects/compsci/USERS/alizae/Hackathon2022/Nuclei/S08-Nuclei.tiff --mpp 0.65 --output-prefix /projects/compsci/whitebr/hackathon2022/S08

# working off of 
# https://github.com/vanvalenlab/deepcell-tf/blob/master/notebooks/applications/Mesmer-Application.ipynb
# and
# https://deepcell.readthedocs.io/en/master/API/deepcell.applications.html#mesmer

def create_mesmer_input(membrane_files, nucleus_file):
    """Create membrane and nuclear images for input to Mesmer.

    Parameter:
        membrane_files : str list
           paths to membrane marker (grayscale) images

        nucleus_file : str
           path to nuclear marker (grayscale) image

    Returns: 
        list(membrane_im, nucleus_im), where
           membrane_im : np.array
              an image representing the membrane, is the max projection
              of the normalized individual membrane marker images
           nuclear_im : np.array
              an image representing the (normalized) nuclear marker

    """
    ims = []
    for membrane_file in membrane_files:
        image = imread(membrane_file)
        image = image / np.iinfo(image.dtype).max
        image = image.astype("float32")
        image = image * np.finfo("float32").max
        ims.append(image)
        
    membrane_im = np.maximum.reduce(ims)
  
    nucleus_im = imread(nucleus_file)
    nucleus_im = nucleus_im / np.iinfo(nucleus_im.dtype).max
    nucleus_im = nucleus_im.astype("float32")
    nucleus_im = nucleus_im * np.finfo("float32").max

    return membrane_im, nucleus_im

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Mesmer on membrane and nuclear marker images')
    parser.add_argument('--membrane-files', dest='membrane_files', action='store',
                        required=True,
                        help='A comma-separated list of one or more membrane marker image files')
    parser.add_argument('--nuclear-file', dest='nuclear_file', action='store',
                        required=True,
                        help='Nuclear marker image file name')
    parser.add_argument('--mpp', dest='mpp', action='store',
                        type=float,
                        required=True,
                        default=0.65,
                        help='Microns per pixel of the membrane and nuclear marker images')
    parser.add_argument('--output-prefix', dest='output_prefix', action='store',
                        required=True,
                        help='Output prefix, with mesmer input stored in <output-prefix>input.npy and Mesmer output in <output-prefix>segmentation_predictions.npy')

    args = parser.parse_args()

    membrane_files = args.membrane_files.split(',')
    nuclear_file = args.nuclear_file
    output_prefix = args.output_prefix
    mpp = args.mpp

    print(membrane_files)
    print(nuclear_file)
    print(output_prefix)
    print(mpp)

    print('Reading and normalizing marker images')
    mem_img, nuc_img = create_mesmer_input(membrane_files, nuclear_file)

    print('Assemblig Mesmer input')
    im = np.stack((nuc_img, mem_img), axis=-1)
    im = np.expand_dims(im,0)

    # rgb_images = create_rgb_image(im, channel_colors=['green', 'blue'])
    ## Plot a subset of pixels near the center of the image, assuming that will
    ## be within the tissue
    idx = 0
    center_x = im[idx, ...].shape[0] / 2
    center_y = im[idx, ...].shape[1] / 2
    num_pixels = 200
    min_x = int(center_x - num_pixels / 2)
    max_x = int(center_x + num_pixels / 2)
    min_y = int(center_y - num_pixels / 2)
    max_y = int(center_y + num_pixels / 2)
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    # ax[0].imshow(im[idx, min_x:max_x, min_y:max_y, 0])
    # ax[1].imshow(im[idx, min_x:max_x, min_y:max_y, 1])
    # ax[2].imshow(rgb_images[idx, min_x:max_x, min_y:max_y])
    #
    # ax[0].set_title('Nuclear channel')
    # ax[1].set_title('Membrane channel')
    # ax[2].set_title('Overlay')
    # plt.show()

    app = Mesmer()

    # debug: run Mesmer on a small subimage
    print('Predicting with Mesmer')
    debug = False
    if debug:
        im = im[:,min_x:(min_x+100), min_y:(min_y+100),:]
    segmentation_predictions = app.predict(im, image_mpp = mpp)

    print('Saving output')
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    np.save(output_prefix + '/segmentation_predictions.npy', segmentation_predictions)
    np.save(output_prefix + '/input.npy', im)

    print('Done')


