from pathlib import Path
import h5py
from datasets import Dataset, DatasetDict

DATA_PATH = Path('/home/ids/smao-22/phd/neural-fcasa/recipes/ami/processed_data/')

splits = {
    "test": "hdf5/chunk.derev-hop160-tt.hdf5",
    "val": "hdf5/chunk.derev-hop160-cv.hdf5", 
    "train": "hdf5/chunk.derev-hop160-tr.hdf5",
}

def h5_generator(h5_path):
    with h5py.File(h5_path, "r") as f:
        grp_names = sorted(f.keys())
        for g in grp_names:
            grp = f[g]
            yield {
                "act": grp["act"][:],
                "wav": grp["wav"][:],
            }

dataset_dict = {}
for split_name, h5_path in splits.items():
    ds = Dataset.from_generator(
        h5_generator,
        gen_kwargs={"h5_path": DATA_PATH/h5_path},
        writer_batch_size=500,
        cache_dir=str(DATA_PATH)
    )
    dataset_dict[split_name] = ds

dataset = DatasetDict(dataset_dict)
dataset.save_to_disk(DATA_PATH/"hf_datasets")