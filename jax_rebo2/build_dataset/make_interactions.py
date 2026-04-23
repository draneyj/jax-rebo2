from initial_params_CHO import params
import optimize_potential
import jax
import pickle
import tqdm
import warnings
import sys
import numpy as np
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
cutoff = 3.0

for row in rows:
    row['species'] = optimize_potential.species_map(row['species'])
    row['energy'] /= 23.06035
    row['forces'] /= 23.06035

interactions = []
boxes = []

print("Generating interactions and boxes...")
for row in tqdm.tqdm(rows):
    interaction, box = optimize_potential.get_interactions_box(row, cutoff)
    interactions.append(interaction)
    boxes.append(box)

pickle.dump(interactions,
    open(f"interactions/{sys.argv[1]}_unpadded.pkl", "wb"))

max_natoms = 0
max_nbonds = 0
max_nneighbors = 0
for interaction in interactions:
    max_natoms = max(max_natoms, interaction.max_natoms)
    max_nbonds = max(max_nbonds, interaction.max_nbonds)
    max_nneighbors = max(max_nneighbors, interaction.max_nneighbors)

print(f"Max natoms: {max_natoms}, max nbonds: {max_nbonds}, max nneighbors: {max_nneighbors}")
