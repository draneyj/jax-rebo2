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
rows = []
interactions = []
for i in range(17):
    print(f"loading {i}")
    rows += pickle.load(open(f"interactions/row_{i}_padded.pkl", "rb"))
    interactions += pickle.load(open(f"interactions/{i}_padded.pkl", "rb"))

pickle.dump(interactions,
    open(f"interactions/all_interactions_padded.pkl", "wb"))
pickle.dump(rows,
    open(f"interactions/all_rows_padded.pkl", "wb"))