from typing import Union, List
from pathlib import Path
from tifffile import imread
import zarr
import dask.array as da

# use zarr==2.10.3

def tifffile_to_dask(im_fp: Union[str,Path]) - > Union[da.array, List[da.Array]]:
    imdata = zarr.open(imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = [da.from_zarr(imdata[z]) for z in imdata.array_keys()]
    else:
        imdata = da.from_zarr(imdata)
    return imdata

  
af_im = tifffile_to_dask("./AF/S01-AF.tiff")

dask_im = tifffile_to_dask("./tissue-images/tissue-masks/S01-tissue-mask.tiff")
# read image into memory it's np.uint8 so smaller footprint
mask_im = dask_im.compute()

# initialize tiler
tiler = Tiler(data_shape=mask_im.shape,
              tile_shape=(512, 512),
              overlap=0.2)

# get index of all tiles that overlap with the mask greater than 75%
tile_overlaps = []
for idx, tile in tiler.iterate(mask_im):
    npx_in_mask = np.sum(tile) / 255
    overlap_with_mask = (np.sum(tile) / 255) / tile.size
    if overlap_with_mask > 0.75:
        tile_overlaps.append(idx)

# get tile on AF image... could then be written to disk... sent to model, etc.
good_tile = tiler.get_tile(af_im, tile_id=tile_overlaps[1])

