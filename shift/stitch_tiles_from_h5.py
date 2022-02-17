if __name__ == "__main__":
    from pathlib import Path
    from tiler import Tiler, Merger
    import numpy as np
    from tifffile import imwrite
    import h5py
    from tqdm import tqdm
    import sys
    
    h5_fp = sys.argv[1]
    # h5_fp = "/home/nhp/linux-share/hackathon/AQP1/S01-AQP1-virtual.h5"
    out_name = Path(h5_fp).name.split("-virtual.h5")[0]
    out_path = Path(h5_fp).parent / f"{out_name}-virtual.tiff"

    with h5py.File(h5_fp) as f:
        # get output shape
        out_shape = tuple(f["out_shape"][1:])
        # build merger
        merger_tiler = Tiler(data_shape=out_shape, tile_shape=(256, 256), overlap=0.1)
        merger = Merger(merger_tiler)
        
        # remove non-tile keys from list of tiles
        key_list = list(f.keys())
        key_list.pop(key_list.index("out_shape"))
        
        # get tile ids for Tiler
        tile_indices = np.asarray(key_list).astype(np.uint32)
        tile_indices = np.sort(tile_indices)
        
        # populate merger by reading in tiles
        for tile_id in tqdm(tile_indices):
            tile = np.asarray(f[str(tile_id)])
            merger.add(tile_id, np.squeeze(tile))
    
    # merge images and write tiff
    # N.B. this uses a lot of memory!
    final_image = merger.merge(unpad=True, dtype=np.uint16)
    imwrite(out_path, final_image, compression="deflate")
