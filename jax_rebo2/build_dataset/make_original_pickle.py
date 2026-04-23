import dpdata
import numpy as np
import pickle
from pathlib import Path
import h5py
from tqdm import tqdm


def expand_sys_str(root_dir):
    """Recursively iterate over directories taking those that contain `type.raw` file.

    If root_dir is a file but not a directory, it will be assumed as an HDF5 file.

    Parameters
    ----------
    root_dir : Union[str, Path]
        starting directory

    Returns
    -------
    List[str]
        list of string pointing to system directories

    Raises
    ------
    RuntimeError
        No system was found in the directory
    """
    root_dir = Path(root_dir)
    if root_dir.is_dir():
        matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
        if (root_dir / "type.raw").is_file():
            matches.append(str(root_dir))
    elif root_dir.is_file():
        # HDF5 file
        with h5py.File(root_dir, "r") as f:
            # list of keys in the h5 file
            f_keys = ["/"]
            f.visit(lambda x: f_keys.append("/" + x))
        matches = [
            f"{root_dir}#{d}" for d in f_keys if str(Path(d) / "type.raw") in f_keys
        ]
    else:
        raise OSError(f"{root_dir} does not exist.")
    if len(matches) == 0:
        raise RuntimeError(f"{root_dir} does not contain any systems!")
    return matches


def get_multi_system(path):
    if not isinstance(path, (list, tuple)):
        path = [path]
    system_paths = []
    for pp in path:
        system_paths.extend(expand_sys_str(pp))
    # systems = dpdata.MultiSystems(
    #     *[
    #         dpdata.LabeledSystem(s, fmt=("deepmd/npy" if "#" not in s else "deepmd/hdf5"))
    #         for s in system_paths
    #     ],
    # )
    systems = [
        dpdata.LabeledSystem(s, fmt=("deepmd/npy" if "#" not in s else "deepmd/hdf5"))
        for s in system_paths
    ]
    return systems


def make_dataset(datasets, self_energies=None):
    data = {}
    if self_energies is None:
        self_energies = {t: 0 for t in datasets[0]["atom_names"] if t != "Ar"}
    data["self_energies"] = self_energies  # dict atom name (e.g. 'C') to self-energy
    total_num_frames = sum([len(dataset) for dataset in datasets])
    rows = [{} for i in range(total_num_frames)]
    i = 0
    for dataset in tqdm(datasets, desc="Processing datasets"):
        for frame in dataset:
            species = np.array([frame["atom_names"][t] for t in frame["atom_types"]])
            rows[i] = {
                "species": species,
                "energy": frame["energies"][0] * 23.06035,
                "forces": frame["forces"][0] * 23.06035,
                "coordinates": frame["coords"][0],
                "orth_matrix": frame["cells"][0],
            }
            i += 1
    data["structures"] = rows
    return data


def has_argon(system):
    if "Ar" in system.data["atom_names"]:
        arindex = system.data["atom_names"].index("Ar")
        if system.data["atom_numbs"][arindex] > 0:
            return True
    return False


datasets = get_multi_system(
    [
        "/scratch/gpfs/AZP/jd6157/rebo2_jax/SiCl/DeepMDSiClData/TrainingData",
    ]
)

num_systems = len(datasets)
num_frames = sum([len(dataset) for dataset in datasets])
print(f"Found {num_systems} systems with {num_frames} frames.")
datasets = [dataset for dataset in datasets if not has_argon(dataset)]
data_dict = make_dataset(
    datasets,
    self_energies={
        "Si": 0.0,
        "Cl": 0.0,
    },
)
pickle.dump(data_dict, open("SiCl_dataset.pkl", "wb"))
