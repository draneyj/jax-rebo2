# jax_rebo2

A JAX implementation of the REBO2 interatomic potential.

## Installation

The repository vendors [`interpax`](https://github.com/draneyj/interpax) as a
git submodule. Initialize it before installing:

```bash
git clone <this repo>
cd jax-rebo2
git submodule update --init --recursive
pip install -e .
```

`pip install -e .` will install `jax_rebo2` in editable mode and will also
install the bundled `interpax` package from the local `./interpax` submodule.

### Optional extras

```bash
pip install -e ".[lammps]"    # LAMMPS Python bindings (testing/utils)
pip install -e ".[dataset]"   # dpdata (build_dataset/make_original_pickle.py)
pip install -e ".[all]"       # both of the above
```

### Tested environment

The pinned dependency floors in `setup.py` match a known-working
`conda` environment: Python 3.12, `jax` 0.8.2, `jax-md` 0.2.8,
`numpy` 2.1, `optax` 0.2.4, `flax` 0.10.3, `chex` 0.1.88,
`matplotlib` 3.10, `pandas` 2.2, `h5py` 3.12, `tqdm` 4.67.

## Layout

- `jax_rebo2/potential` – REBO2 potential implementations
- `jax_rebo2/train` – training / parameter-optimization scripts
- `jax_rebo2/build_dataset` – dataset preparation helpers
- `jax_rebo2/testing` – LAMMPS / molecular-energy comparison scripts
- `jax_rebo2/utils` – miscellaneous utilities
