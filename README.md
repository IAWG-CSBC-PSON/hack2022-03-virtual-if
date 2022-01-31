# hack2022-03-virtual-if
Challenge 3: Virtual IF staining of label-free virtual IF staining

# Challenge Description
![image](https://user-images.githubusercontent.com/17855764/151869995-513d806f-1dce-4f34-a194-925edf310bca.png)

This challenge will involve using models to virtually stain label-free images.

## The data
The data is organized into sub-folders, each containing 4 or 8 images. All AF images are already registered and resampled to their corresponding IF images and in the same pixel space. Thus slicing is interchangable: `af_image[:,0:1000,0:1000]` would pull form the same area as `if_image[0:1000,:1000]` 


<pre>
<b>tissue-images</b>  
│
└───<b>AF</b>
│   │   S01-AF.tiff
│   │   S02-AF.tiff
│   │   ...
│   │   S08-AF.tiff
|
└───<b>AQP</b>
│   │   S01-AQP1.tiff
│   │   S02-AQP1.tiff
│   │   ...
│   │   S08-AQP1.tiff

...

└───<b>tissue-masks</b>
│   │   S01-tissue-mask.tiff
│   │   S02-tissue-mask.tiff
│   │   ...
│   │   S08-tissue-mask.tiff

</pre>

## File descriptions
`*-AF.tiff` : Label-free autofluorescence, this is the modality that will be used to develop the virtual stain, these are 16-bit three channel images

`*-AQP|A-SMA|AQP1|AQP2|Calbindin|ColIV-a12|NCC|Nuclei|Podocalyxin|Uromodulin.tiff` : individual immunofluorescence images from various markers. These are single channel 16-bit images.

`*-tissue-mask.tiff` : 8-bit binary masks of where values > 1 are on-tissue areas free from major artifacts

## Cosniderations for developing the virtual staining
4 of the AF images (S01-04) have corresponding IF for all labels, 4 only have AQP1,Nuclei,Podocalyxin, and Uromodulin. One way to organize the challenge is to try various approaches going from AF to a single label, i.e., AF -> AQP1 to discover what works, and then expand from there. It is NOT necessary to have a single model that predicts all species if better performance is achieved by splitting channels.

These images are on the order of 20000-30000 pixels in x and y dimension, so are quite large. Tiling the data will be critical and smarting tiling for train and validation sets is advised.

Evaluation metrics are still an area of exploration. Segmentation masks could be developed for some labels or for nuclei but this is an area this hackathon could explore.

## Necessary computing environment

Most virtual staining in recent years has been performed using deep learning. Efficient development will necessitate a GPU equiped machine and will probably most easily be done in python. The size of images, while large, can be managed by using out-of-memory operations in python through `tifffile` + `dask` + `zarr` . Other libraries that could be useful:  

```python
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

dask_im = tifffile_to_dask("./AF/S01-AF.tiff")

```

This produces a dask image that reads the chunks of the AF image, allowing read into memory of specific areas of the image without reading in the entire array into memory
```python
Out[1]: dask.array<from-zarr, shape=(3, 31279, 36802), dtype=uint16, chunksize=(1, 512, 512), chunktype=numpy.ndarray>
```

### tiling example
`tiler`[https://pypi.org/project/tiler/] is a python library for tiling arrays.
```python
from tiler import Tiler

tiler = Tiler(data_shape=dask_im.shape,
              tile_shape=(3, 256, 256),
              channel_dimension=0
              overlap=0.2)

tiler.get_tile(dask_im, tile_idx=600)

```
```python
Out[2]: dask.array<getitem, shape=(3, 512, 512), dtype=uint16, chunksize=(1, 509, 482), chunktype=numpy.ndarray>
```



## Some background on virtual staining

* https://gitlab.com/eburling/SHIFT
* transmitted image to fluorescence: https://doi.org/10.1016/j.cell.2018.03.040
* virtual histological staining from unlabelled tissue AF image: https://doi.org/10.1038/s41551-019-0362-y
* PhaseStain: https://www.nature.com/articles/s41377-019-0129-y
* Label-free prediction of three-dimensional fluorescence images: https://www.nature.com/articles/s41592-018-0111-2
* SHIFT: https://www.nature.com/articles/s41598-020-74500-3


