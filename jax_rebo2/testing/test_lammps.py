import jax_md
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from functools import partial
import tqdm
import warnings
import multiprocessing
import os
import optimize_potential
import lammps


jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")


def make_lmp():
    units = "metal"
    atom_style = "atomic"
    pair_style = "rebo3"
    qeq = ""
    potential_file = "CH.rebo3_all_23"
    # potential_file = "CH.rebo3_2"

    atom_list = " ".join(["C", "H", "O"])
    pair_coeff_lines = f"pair_coeff  * * {potential_file} {atom_list}"

    lmp = lammps.lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"], name="develop")
    # lmp = lammps.lammps(cmdargs=["-echo", "screen", "-log", "none", "-nocite"], name="develop")
    lmp.commands_string(
        f"""
        units       {units}
        atom_style  atomic
        boundary    p p p

        region dummy_region block 0 1 0 1 0 1
        create_box {n_atom_types} dummy_region # random create_box to ensure 2 atom types

        pair_style  {pair_style}
        {pair_coeff_lines}

        compute pe all pe

        thermo_style    custom time temp
        thermo_modify   flush yes

        delete_atoms group all
        """
    )
    for i in range(n_atom_types):
        lmp.command(f"mass {i+1} 1.0")
    lmp.command("thermo_style custom step pe fmax")
    lmp.command("thermo 1")
    return lmp


def label_row_with_lammps(lmp, row):
    lmp.command("delete_atoms group all")
    R = jnp.array(row["coordinates"])
    species = jnp.array(row["species"])
    n_atoms = len(species)
    box = np.diag(row["orth_matrix"])

    lmp.command("delete_atoms group all")
    lmp.command(f"change_box all x final 0 {box[0]} y final 0 {box[1]} z final 0 {box[2]}")
    for i in range(1, n_atoms):
        if species[i] >= 0:
            lmp.command(f"create_atoms {species[i]+1} single {R[i, 0]} {R[i, 1]} {R[i, 2]}")

    lmp.command("run 0")

    energy = lmp.get_thermo("pe")
    f = lmp.numpy.extract_atom("f")
    r = lmp.numpy.extract_atom("x")
    return r, energy, f


def reorder_forces(force_positions, new_positions, forces):
    mindiffs = jnp.linalg.norm(force_positions[:, None] - new_positions[None, :], axis=-1)
    indices = jnp.argmin(mindiffs, axis=0)
    return forces[indices, :]


def compare_lammps_and_jax_md(lmp, row, interaction, params):
    positions_lammps, energy_lammps, forces_lammps = label_row_with_lammps(lmp, row)
    # optimize_potential.pad_row(row, 256)
    box = row['orth_matrix']
    energy_jax, nforce_jax = optimize_potential.energy_neg_force(
        params,
        box,
        *interaction,
    )
    force_jax = nforce_jax * -1.0
    # force_jax = force_jax[interaction[1] >= 0]
    forces_lammps = reorder_forces(forces_lammps, force_jax, forces_lammps)

    energy_diff = np.abs(energy_lammps - energy_jax)
    forces_diff = np.linalg.norm(forces_lammps - force_jax, axis=-1)

    return energy_diff, forces_diff


if __name__ == "__main__":
    n_atom_types = 3
    lmp = make_lmp()

    # with open("params/starting_params.pkl", "rb") as f:
    with open("params/all_23_params.pkl", "rb") as f:
        params = pickle.load(f)
    print("Loaded params.", flush=True)
    print("loading rows and interactions...", flush=True)
    rows = pickle.load(open("interactions/most_rows_padded.pkl", "rb"))
    interactions = pickle.load(
        open("interactions/most_interactions_padded_numpy.pkl", "rb")
    )
    print("Loaded rows and interactions.", flush=True)
    np.random.seed(72099)
    indicies = np.arange(len(rows))
    np.random.shuffle(indicies)
    # indicies = indicies[:5]
    rows = [rows[i] for i in indicies]
    interactions = [interactions[i] for i in indicies]
    interactions = [interaction.dump_arrays() for interaction in interactions]
    cutoff = 3.0

    energy_diffs = []
    forces_diffs = []

    rows_100 = rows[:100]
    interactions_100 = interactions[:100]
    pickle.dump(rows_100, open("interactions/rows_100.pkl", "wb"))
    pickle.dump(interactions_100, open("interactions/interactions_100.pkl", "wb"))

    for row, interaction in tqdm.tqdm(zip(rows[:10], interactions[:10]), desc="Processing rows"):
        # print(row, interaction)
        energy_diff, forces_diff = compare_lammps_and_jax_md(
            lmp, row, interaction, params
        )
        print("Energy diff:", energy_diff, flush=True)
        print("Forces diff:", forces_diff, flush=True)
        energy_diffs.append(energy_diff)
        forces_diffs.append(forces_diff)

    print("Energy differences:", np.mean(energy_diffs), np.std(energy_diffs))
    print("Forces differences:", np.mean(forces_diffs), np.std(forces_diffs))

    np.savez(
        "lammps_jax_md_comparison.npz",
        energy_diffs=energy_diffs,
        forces_diffs=forces_diffs,
    )