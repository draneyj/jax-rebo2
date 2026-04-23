from initial_params_CHO import params
import optimize_potential
import jax
import pickle
import tqdm
import warnings
import sys
import numpy as np
import jax.numpy as jnp
import jax_md

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

print("Loading dataset...")
rows = pickle.load(open("CHO_dataset.pkl", "rb"))
rows = rows['structures']
stride = 1000
begin = int(sys.argv[1]) * stride
end = np.minimum((int(sys.argv[1]) + 1) * stride, len(rows))
rows = rows[begin:end]
interactions = pickle.load(open(f"interactions/{sys.argv[1]}_unpadded.pkl", "rb"))
cutoff = 3.0

for row in rows:
    row['species'] = optimize_potential.species_map(row['species'])
    row['energy'] /= 23.06035
    row['forces'] /= 23.06035

max_natoms = 256
max_nbonds = 5187
max_nneighbors = 40

for row, interaction in tqdm.tqdm(zip(rows, interactions)):
    interaction.pad(max_natoms, max_nbonds, max_nneighbors)
    box = jnp.array(row['orth_matrix'], dtype=jnp.float32)
    optimize_potential.pad_row(row, max_natoms)

pickle.dump(interactions,
    open(f"interactions/{sys.argv[1]}_padded.pkl", "wb"))
pickle.dump(rows,
    open(f"interactions/row_{sys.argv[1]}_padded.pkl", "wb"))