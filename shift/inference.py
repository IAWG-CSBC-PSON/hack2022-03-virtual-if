from pathlib import Path
import torch
from models import create_model
from tifffile import imread
from tiler import Tiler, Merger
import numpy as np
from tifffile import imwrite


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    import sys

    checkpoint_dir = sys.argv[1]
    marker_folder_name = sys.argv[2]
    test_image = sys.argv[3]

#     checkpoint_dir = "/home/nhp/linux-share/hackathon"
#     marker_folder_name = "AQP1"
#     test_image = "/home/nhp/linux-share/hackathon/tissue-images/AF/S01-AF.tiff"
    
    sample_id = Path(test_image).name.split("-AF")[0]
    output_virtual_image_fp = (
        Path(checkpoint_dir)
        / marker_folder_name
        / f"{sample_id}-{marker_folder_name}-virtual.tiff"
    )

    opt = Namespace(
        aug=False,
        batch_size=8,
        beta1=0.5,
        checkpoints_dir=checkpoint_dir,
        continue_train=False,
        crop_size=256,
        dataroot="./",
        dataset_mode="aligned",
        direction="AtoB",
        display_env="main",
        display_freq=25,
        display_id=1,
        display_ncols=4,
        display_port=8097,
        display_server="http://localhost",
        display_winsize=256,
        epoch="latest",
        epoch_count=1,
        eps=0.01,
        gan_mode="lsgan",
        gpu_ids=[0],
        init_gain=0.02,
        init_type="normal",
        input_nc=3,
        isTrain=False,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_L1=100,
        lambda_identity=0.5,
        lambdadapt=False,
        load_iter=0,
        load_size=286,
        lr=0.0002,
        lr_decay_iters=50,
        lr_policy="linear",
        model="pix2pix",  # change this to pix2pix or cyclegan
        n_layers_D=3,
        name=marker_folder_name,
        ndf=64,
        netD="basic",
        netG="resnet_9blocks",
        ngf=64,
        niter=1,
        niter_decay=20,
        no_dropout=True,
        no_flip=False,
        no_html=False,
        norm="instance",
        num_threads=4,
        output_nc=1,
        phase="train",
        pool_size=50,
        preprocess="resize_and_crop",
        print_freq=100,
        save_by_iter=False,
        save_epoch_freq=5,
        save_latest_freq=7000,
        serial_batches=False,
        suffix="",
        thresh_path="./checkpoints",
        tmax=50,
        update_html_freq=1000,
        verbose=False,
    )

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)

    image = imread(test_image)
    image_dtype = image.dtype
    # create tiler for AF data
    tiler = Tiler(
        data_shape=image.shape,
        tile_shape=(3, 256, 256),
        channel_dimension=0,
        overlap=0.1,
    )

    # create a single channel merger tiler
    # otherwise we will have dimension mismatch issues
    merger_tiler = Tiler(data_shape=image.shape[1:], tile_shape=(256, 256), overlap=0.1)

    # create merger to store data
    merger = Merger(merger_tiler)
    opt.eval = True
    if opt.eval:
        model.eval()
    for tile_id, tile in tiler(image, progress_bar=True):
        # normalize tile to [0,1]
        tile = tile / np.iinfo(image_dtype).max
        # expand to be "batch-like"
        # code says inference doesn't work on batches so didn't attempt
        tile = np.expand_dims(tile, 0)
        tile = torch.from_numpy(tile).float()

        data = {"A": tile, "B": tile, "A_paths": [""], "B_paths": [""], "B_prev": [0]}

        model.set_input(data)  # unpack data from data loader

        model.test()  # run inference
        # process tile back to np.uint16
        out_tile = model.fake_B.detach().cpu().numpy()
        out_tile *= np.iinfo(image.dtype).max
        out_tile = out_tile.astype(image_dtype)

        merger.add(tile_id, np.squeeze(out_tile))

    del image
    # consumes lots of memory
    final_image = merger.merge(unpad=True)
    final_image = final_image.astype(image_dtype)
  
    # saved without tiling for quick vis in FIJI
    imwrite(
        output_virtual_image_fp,
        final_image,
        compress="deflate",
    )
