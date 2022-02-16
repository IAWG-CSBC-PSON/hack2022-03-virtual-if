## Evironment: python=3.9.7, torch=1.10.2, zarr=2.10.3, numpy=1.21.3, dask2022.2.0, tiler=0.5.6, tifffile=2022.2.9

import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import Union, List
import os
from pathlib import Path
import shutil
import numpy as np
import dask.array as da
from tifffile import imread
import zarr
from glob import glob
from skimage import img_as_ubyte
from tiler import Tiler
import time


def tifffile_to_dask(im_fp: Union[str, Path]) -> Union[da.array, List[da.Array]]:
    imdata = zarr.open(imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = [da.from_zarr(imdata[z]) for z in imdata.array_keys()]
    else:
        imdata = da.from_zarr(imdata)
    return imdata


def make_data_virtual_dir(
    data_path, 
    af_dir="AF",
    if_dirs=None,
    mask_dir="tissue-masks",
    samples=None
):
    af_samples = glob(os.path.join(data_path, af_dir, "*.tiff"))
    masks = glob(os.path.join(data_path, mask_dir, "*.tiff"))

    if if_dirs is None:
        if_dirs = sorted([
            d.split("/")[-1] for d in glob(os.path.join(data_path, "*")) 
            if af_dir not in d and mask_dir not in d and os.path.isdir(d)
        ])
    else:
        if_dirs.sort()
    
    if samples is None:
        samples = sorted(list(map(lambda x : x.split("/")[-1].split("-")[0], af_samples)))
    else:
        samples.sort()

    virtual_dir = {}
    for sample in samples:
        samp = {
            "AF": os.path.join(data_path, af_dir, f"{sample}-AF.tiff"),
            "tissue-mask": os.path.join(data_path, mask_dir, f"{sample}-tissue-mask.tiff")
        }
        for ifd in if_dirs:
            ifd_samp = os.path.join(data_path, ifd, f"{sample}-{ifd}.tiff")
            if os.path.isfile(ifd_samp):
                samp[ifd] = ifd_samp
            else:
                raise ValueError(f"Sample {sample} has no {ifd} IF data")

        virtual_dir[sample] = samp

    print("VIRTUAL DIRECTORY:")
    for sid, sample in virtual_dir.items():
        print(f"    {sid}:")
        for k, v in sample.items():
            print(f"\t{k}")
    print("\n")

    return virtual_dir


def physical_dirs_from_virtual(dir_path, virtual_dir):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    print("PHYSICAL DIRECTORIES:")

    for sid, sample in virtual_dir.items():
        if not os.path.isdir(os.path.join(dir_path, sid)):
            os.mkdir(os.path.join(dir_path, sid))
        print(f"    {os.path.join(dir_path, sid)}/")
        af_path = sample['AF']
        for n, path in sample.items():
            if n != "AF" and n != "tissue-mask":
                if not os.path.isdir(os.path.join(dir_path, sid, n)):
                    os.mkdir(os.path.join(dir_path, sid, n))
                print(f"\t{n}/")
                shutil.copyfile(path, os.path.join(dir_path, sid, n, f"{n}.tiff"))
                shutil.copyfile(af_path, os.path.join(dir_path, sid, n, f"AF.tiff"))



def write_tile_idxs(virtual_dir, write_dir, tile_shape=(512, 512), tile_overlap=0.2, mask_overlap=0.75):
    print("\nEXTRACTING TILES")

    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    for sid, sample in virtual_dir.items():
        print("    ", f"{sid} ", end="", flush=True)
        mask_im = tifffile_to_dask(sample['tissue-mask'])
        mask_tiler = Tiler(
            data_shape=mask_im.shape, 
            tile_shape=tile_shape, 
            overlap=tile_overlap,
            mode="constant",
            constant_value=0
        )
        assert (mask_tiler.data_shape == mask_im.shape).all()
        tile_overlaps = []
        for idx, tile in mask_tiler.iterate(mask_im):
            npx_in_mask = np.sum(tile) / 255
            overlap_with_mask = npx_in_mask / tile.size
            if overlap_with_mask >= mask_overlap:
                tile_overlaps.append(idx)
            del tile
        tile_data = {
            "idxs": tile_overlaps,
            "n_tiles": mask_tiler.n_tiles,
            "im_shape": mask_tiler.data_shape,
            "padded_im_shape": mask_tiler._new_shape,
            "tiler_pad_mode": mask_tiler.mode, ## Should be "constant"
            "tiler_pad_val": mask_tiler.constant_value, ## Should be 0
            "tile_shape": tuple(mask_tiler.tile_shape),
            "tile_overlap": mask_tiler.overlap
        }
        np.save(os.path.join(write_dir, f"{sid}.npy"), tile_data, allow_pickle=True)

    print("\nTILES EXTRACTED: ", write_dir, "\n")


class VIFDataset(Dataset):
    def __init__(
        self, 
        vif_virtual_dir,
        idx_file_dir,
        normalize_whole_slide=True,
        af_transform = None,
        if_transform = None,
    ):
        self.vif_dir = vif_virtual_dir
        self.idx_file_dir = idx_file_dir
        self.normalize_whole_slide = normalize_whole_slide
        self.af_transform = af_transform
        self.if_transform = if_transform
        self.dask_ims = {}
        
        self.get_data()

    def get_data(self):
        data = []
        for sid, sample in self.vif_dir.items():
            (idxs, n_tiles, im_shape, pad_im_shape, tile_shape, 
                tile_overlap, tiler_pad_mode, tiler_pad_val) = self.load_tile_data(sid)
            im_shape = tuple(im_shape)
            pad_im_shape = tuple(pad_im_shape)
            names = []
            ims = {}
            for n, s in sample.items():
                if n == 'tissue-mask':
                    continue
                names.append(n)

                im = tifffile_to_dask(s)
                if self.normalize_whole_slide:
                    im = im / np.iinfo(im.dtype).max

                if n == "AF":
                    assert im.shape == (3,) + im_shape, f"{s} has shape {im.shape} but should be {(3,) + im_shape}"
                    im_tile_shape = (3,) + tile_shape
                else:
                    assert im.shape == im_shape, f"{s} has shape {im.shape} but should be {im_shape}"
                    im_tile_shape = tile_shape
                im_tiler = Tiler(
                    data_shape=im.shape,
                    tile_shape=im_tile_shape,
                    overlap=tile_overlap,
                    mode=tiler_pad_mode,
                    constant_value=tiler_pad_val,
                )

                assert im_tiler.n_tiles == n_tiles, f"{s} has {im_tiler.n_tiles} tiles but should be {n_tiles}"

                if n == "AF":
                    assert tuple(im_tiler._new_shape) == (3,) + pad_im_shape, (f"{s} has post-padding shape {tuple(im_tiler._new_shape)} "
                        f"but should be {(3,) + pad_im_shape}")
                else:
                    assert tuple(im_tiler._new_shape) == pad_im_shape, (f"{s} has post-padding shape {tuple(im_tiler._new_shape)} " 
                        f"but should be {pad_im_shape}")

                ims[n] = (im, im_tiler)
            
            self.dask_ims[sid] = ims

            data += [(sid, names, idx) for idx in idxs]

        self.data = data

    def load_tile_data(self, sid):
        idx_file = os.path.join(self.idx_file_dir, f"{sid}.npy")
        if not os.path.isfile(idx_file):
            raise FileNotFoundError(f"{idx_file} does not exist")
        idx_data = np.load(idx_file, allow_pickle=True).item()
        return (idx_data["idxs"], idx_data["n_tiles"], idx_data["im_shape"], idx_data["padded_im_shape"],
            idx_data["tile_shape"], idx_data["tile_overlap"], idx_data["tiler_pad_mode"], idx_data["tiler_pad_val"])

    def __getitem__(self, index):
        sid, names, idx = self.data[index]
        af_im, af_tiler = self.dask_ims[sid]["AF"]
        ll_coord, ur_coord = af_tiler.get_tile_bbox(idx)
        ll_coord, ur_coord = ll_coord[1:], ur_coord[1:]
        af_tile = af_tiler.get_tile(af_im, idx).compute()
        af_tile = torch.from_numpy(af_tile).float()
        if not self.af_transform is None:
            af_tile = self.af_transform(af_tile)

        if_tiles = []
        channels = []
        for n in names:
            if n == "AF":
                continue
            channels.append(n)
            im, im_tiler = self.dask_ims[sid][n]
            im_ll_coord, im_ur_coord = im_tiler.get_tile_bbox(idx, with_channel_dim=False)
            assert (ll_coord == im_ll_coord).all() and (ur_coord == im_ur_coord).all(), (f"{sid} {n} {idx} has ll_coord {im_ll_coord} " 
                f"and ur_coord {im_ur_coord} but should be {ll_coord} and {ur_coord}")
            tile = im_tiler.get_tile(im, idx).compute()
            tile = torch.from_numpy(tile).float()
            if_tiles.append(tile)
        
        if_tiles = torch.stack(if_tiles, dim=0)
        if not self.if_transform is None:
            if_tiles = self.if_transform(if_tiles)
        return {
            "A": af_tile, 
            "B": if_tiles, 
            "sid": sid, 
            "tile_idx": idx,
            "ll_coord": ll_coord,
            "ur_coord": ur_coord, 
            "channels": channels
        }

    def __len__(self):
        return len(self.data)


def train_val_split(
    dataset, 
    batch_size,
    num_workers=os.cpu_count(),
    train_frac=0.9,
    seed=None, 
    shuffle=True
):
    indices = list(range(len(dataset)))

    if shuffle:
        print("Shuffling before train/val split. Random seed not provided. Defaulting to 0")
        np.random.seed(seed)
        np.random.shuffle(indices)

    n_train = int(len(dataset) * train_frac)
    train_indices, val_indices = indices[:n_train], indices[n_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    print(f"Train/Val split: {len(train_indices)}/{len(val_indices)}")

    trainloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler
    )

    validloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=valid_sampler
    )

    return trainloader, validloader


if __name__ == "__main__":

    ## Create a "virtual directory" - a nested dictionary of paired image paths
    ## virtual_dir[sample_id][image_name] = image_path
    ## print(virtual_dir["S01"]["AF"]) --> "/path/to/S01-AF.tiff"
    ## Will throw error if there is no data for a provided sample / marker combination.
    ## Assumes data directory is structured as:
    ## tissue_images/
    ##     AF/
    ##         S01-AF.tiff
    ##         S02-AF.tiff
    ##         ...
    ##     ...
    dp = "/home/users/strgar/strgar/vif-hack/data/images/tissue-images"
    samples = ["S04", "S05", "S06"]
    if_dirs = ["AQP1", "Podocalyxin", "Uromodulin"]
    virtual_dir = make_data_virtual_dir(dp, if_dirs=if_dirs, samples=samples)

    ## Create actual directory structure from the virtual directory
    ## Compatiable with the SHIFT aligned_dataset.py 
    ## -- https://gitlab.com/eburling/SHIFT/-/blob/master/data/aligned_dataset.py
    ## Each sample specific, non-AF image copied to shift_data_path/sample_id/image_marker
    ## AF image also copied, creating paired, sample-specific domain A --> domain B data folder
    ## Note: not yet tested with the SHIFT repo
    ## Note: whole slide images are copied, but can be easily modified to copy image tiles
    shift_data_path = "/home/users/strgar/strgar/vif-hack/data/shift"
    physical_dirs_from_virtual(shift_data_path, virtual_dir)

    ## Extract "good" tiles from tissue masks contained in the virtual directory
    ## Save the good tile indices along with some metadata to a file on disk
    ## This takes several minutes (or more with smaller tile) so nice to only do once
    ## Note for different tile sizes, overlap, etc. you will need to run again
    ## Writes .npy files to write_dir/sample_id.npy (e.g. write_dir/S01.npy)
    ## Files can be read with np.load(file_path)
    ## Data schema:
    ## {
    ##     "idxs": [tile_idx, tile_idx, ...],
    ##     "n_tiles": Number of tiles in the mask,
    ##     "im_shape": Mask shape (height, width),
    ##     "tile_shape": Tile shape (height, width),
    ##     "padded_im_shape": Mask shape after padding,
    ##     "tiler_pad_mode": mask_tiler padding mode, ## Should be "constant"
    ##     "tiler_pad_val": mask_tiler padding value, ## Should be 0
    ##     "tile_overlap": float (0.0 - 1.0)
    ## }
    tile_data_write_dir = "/home/users/strgar/strgar/vif-hack/data/tile_data"
    tile_shape = (512, 512)
    tile_overlap = 0.2
    mask_overlap = 0.75
    if not os.path.isdir(tile_data_write_dir):
        tile_shape=tile_shape
        tile_overlap=tile_overlap
        mask_overlap=mask_overlap
        write_tile_idxs(
            virtual_dir,
            tile_data_write_dir,
            tile_shape=tile_shape, 
            tile_overlap=tile_overlap, 
            mask_overlap=mask_overlap
        )


    ## Images were tiled according to "tile_shape" above
    ## Unless you have a lot of memory we likely we want to resize images to at most 256x256
    ## For some reason 
    af_transform = Compose([Resize((256, 256), InterpolationMode.BICUBIC),])
    if_transform = Compose([Resize((256, 256), InterpolationMode.BICUBIC),])

    ## Create the dataset
    ## Normalize whole slide mean scale image to [0, 1] using max dtype value (e.g. for uint16, 2**16-1)
    dataset = VIFDataset(
        virtual_dir,
        tile_data_write_dir,
        af_transform=af_transform,
        if_transform=if_transform,
        normalize_whole_slide=True,
    )

    print("Total number samples: ", len(dataset))

    ## Split the dataset into train and validation sets
    ## Not very familiar with Dask model, but it's best to use
    ## the main process for loading rather than workers...
    ## This might be useful -- https://pypi.org/project/dask-pytorch/
    trainloader, valloader = train_val_split(
        dataset,
        batch_size=3,
        num_workers=0,
        train_frac=0.9,
        seed=101,
        shuffle=True,
    )

    ## Sample a few training points from the trainloader
    ## Dataset __getitem__ returns a dictionary:
    # {
    #     "A": AF image (3, H, W), 
    #     "B": IF images (C, H, W), 
    #     "sid": Sample id, 
    #     "tile_idx": Tile index,
    #     "ll_coord": Lower left coordinate of tile,
    #     "ur_coord": Upper right coordinate of tile,, 
    #     "channels": List of marker names corresponding to IF images (e.g. ["AQP1", "Podocalyxin", "Uromodulin"])
    # }
    print("\nSAMPLE TRAINING DATA")
    for i, (meta, ims) in enumerate(trainloader):
        print(f"    Sample {i}/{len(trainloader)}")
        print("    METADATA")
        for k, v in meta.items():
            print(f"\t{k}: {v}")
        print("    IMAGE")
        print("\tshape:", ims.shape)
        print("\tdtype:", ims.dtype)
        print("\tmin:", np.min(ims.numpy(), (1, 2, 3)))
        print("\tmax:", np.max(ims.numpy(), (1, 2, 3)))
        print("\n")
        break
