import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_compilation_cache_dir", "JAXCACHE")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_debug_nans", True)
import jax_md
import jax.numpy as jnp
import numpy as np
import optax
import newrebo2_interaction_list as rebo2
from initial_params_CHO import params
import pickle
from functools import partial
import tqdm
import warnings
import os
import argparse

# os.environ['XLA_FLAGS'] = (
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_triton_gemm_any=True "
#     "--xla_gpu_enable_command_buffer='' "
#     "--xla_disable_hlo_passes=collective-permute-motion "
#     "--xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE "
# )

# jax.config.update('jax_compiler_enable_remat_pass', False)
warnings.filterwarnings("ignore")

def species_map(species):
    species_mapped = np.zeros(len(species), dtype=int)
    for i, s in enumerate(species):
        if s == 'C':
            species_mapped[i] = 0
        elif s == 'H':
            species_mapped[i] = 1
        elif s == 'O':
            species_mapped[i] = 2
        else:
            raise ValueError(f"Unknown species: {s}")
    return species_mapped


def pad_row(row, max_natoms):
    """Pad the row to have a fixed number of atoms."""
    row['species'] = np.pad(row['species'], (0, max_natoms - len(row['species'])), constant_values=-1)
    row['forces'] = np.pad(row['forces'], ((0, max_natoms - len(row['forces'])), (0, 0)), constant_values=0.0)
    row['coordinates'] = np.pad(row['coordinates'], ((0, max_natoms - len(row['coordinates'])), (0, 0)), constant_values=0.0)


def self_energy(params, species, nspecies=3):
    self_energies = jnp.zeros(species.shape)
    for i in [0, 1, 2]:
        self_energies += jnp.where(species == i, params['self_energy'][i], 0.0)
    return jnp.sum(self_energies)


def get_interactions_box(row, cutoff, **kwargs):
    R = jnp.array(row['coordinates'])
    species = jnp.array(row['species'])
    box = jnp.array(row['orth_matrix'])
    displacement, shift = jax_md.space.periodic_general(box, fractional_coordinates=False)
    interactions = rebo2.allocate_interactions(
        R, species, displacement, cutoff, box, keep_nlists=False, **kwargs
    )
    return interactions, box


# @jax.jit
def row_energy_noself(params, interactions, box):
    displacement, shift = jax_md.space.periodic_general(box, fractional_coordinates=False)
    energy_hat = rebo2.rebo2_trainable(
                        params,
                        box,
                        *interactions,
                        nspecies = 3,
                        )
    return energy_hat


energy_neg_force = jax.jit(jax.value_and_grad(rebo2.rebo2_trainable, argnums=2))


def apply_rules(F_true):
    F_true = 0.5 * (F_true + F_true.swapaxes(0, 1))  # Ensure symmetry
    return F_true


def apply_derivative_rules(params, key1, key2):
    F_di = params[key1]
    F_dj = params[key2]
    F_true = 0.5 * (F_di + F_dj.swapaxes(2, 3))  # Ensure symmetry
    return F_true


@jax.jit
def fix_params(params):
    # ensure type_i vs type_j symmetry
    for key in params.keys():
        if len(params[key].shape) >= 2:
            if params[key].shape[0] == params[key].shape[1]:
                params[key] = 0.5 * (params[key] + params[key].swapaxes(0,1))
    # ensure N_i to N_j symmetry
    for key in ['F', 'F_dk', 'T', 'T_dk']:
        for i, arr1 in enumerate(params[key]):
            for j, subarray in enumerate(arr1):
                params[key] = params[key].at[i, j].set(apply_rules(subarray))
    # ensure d_i d_j symmetry

    for di_key, dj_key in [('F_di', 'F_dj'), ('T_di', 'T_dj')]:
        fixed_di = apply_derivative_rules(params, di_key, dj_key)
        params[di_key] = fixed_di
        params[dj_key] = fixed_di.swapaxes(2, 3)
    params['Dmax'] = jnp.minimum(params['Dmax'], 3.0)
    # keep things nonzero and stable
    for k in [
        "beta_1",
        "beta_2",
        "beta_3",
        "B1",
        "B2",
        "B3",
        "Dmin",
        "Dmax",
        "A",
        "Q",
        "alpha",
        "lambda_ijk",
        "rho",
        ]:
        params[k] = jnp.maximum(params[k], 0.0)
    return params


def interaction_loss(params, interactions, box, row, force_weight=1.0, energy_weight=1.0, regularization_weight=0.1):
    species = row['species']
    natoms = jnp.sum(species >= 0)  # Count only valid species

    energy_hat, nforce_hat = energy_neg_force(
        params,
        box,
        *interactions,
    )

    energy = row['energy']
    forces = row['forces']
    species = jnp.array(row['species'])
    energy_hat += self_energy(params, species)
    energy_error = jnp.square(energy_hat - energy) / natoms
    # nforce_hat = nforce_hat[:-1]
    force_error = jnp.sum(jnp.where((species >= 0)[:, None], jnp.square(-nforce_hat - forces), 0)) / natoms
    loss_value = energy_error * energy_weight + force_error * force_weight

    energy_error = jnp.abs(energy_hat - energy) / natoms
    force_error = jnp.sum(jnp.where((species >= 0), jnp.linalg.norm(-nforce_hat - forces, axis=-1), 0)) / natoms
    errors = dict(energy_error=energy_error, force_error=force_error)

    # add a penalty for short range attraction:
    # key, subkey = jax.random.split(key)
    # pair_interaction = jax.random.choice(subkey, pair_interactions)
    # pair_energy = rebo2.rebo2_trainable(params, fake_box, *pair_interaction.dump_arrays())
    # pair_loss = jax.nn.relu(-1 * pair_energy)
    # loss_value += pair_loss
    for key in ['F', 'T', 'P', 'F_di', 'F_dj', 'F_dk', 'T_di', 'T_dj', 'T_dk', 'P_di', 'P_dj', 'P_dk']:
        loss_value += regularization_weight * jnp.sum(jnp.abs(params[key]))

    return loss_value, errors

loss_nograd = jax.jit(interaction_loss)
loss_grad = jax.jit(jax.value_and_grad(loss_nograd, argnums=0, has_aux=True))

# def loss(params, rows, displacements, interactions, force_weight=1.0, energy_weight=1.0):
#     """Compute the loss function for optimization."""
#     # Compute the energy and forces
#     loss_values = 0
#     grad_loss = jax.tree_map(lambda x: jnp.zeros_like(x), params)
#     for interaction, displacement, row in tqdm.tqdm(zip(interactions, displacements, rows)):
#         this_loss_value, this_grad_loss = loss_grad(params, interaction, displacement, row, force_weight, energy_weight)
#         loss_values += this_loss_value
#         grad_loss += grad_loss
#     return loss_value, grad_loss


def loss_weights(epoch, num_epochs):
    force_weight =  1.0 * (1 - (num_epochs - epoch) / num_epochs)
    energy_weight = 1.0
    return force_weight, energy_weight


def optimize_potential(rows,
    interactions,
    boxes,
    initial_params,
    num_epochs=50,
    learning_rate=1e-3,
    validation_frac=0.0,
    output_prefix="REBO",
    regularization_weight=0.0,
    ):
    """Optimize the REBO2 potential parameters."""
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)

    # Define the loss function
    loss_fn = loss_grad

    best_validation_loss = np.inf
    # Optimization loop
    validation_indicies = np.random.choice(len(rows), int(validation_frac * len(rows)), replace=False) if validation_frac > 0 else np.array([])
    train_indicies = np.array([i for i in range(len(rows)) if i not in validation_indicies])

    validation_rows = [rows[i] for i in validation_indicies]
    validation_interactions = [interactions[i] for i in validation_indicies]
    validation_boxes = [boxes[i] for i in validation_indicies]

    rows = [rows[i] for i in train_indicies]
    interactions = [interactions[i] for i in train_indicies]
    boxes = [boxes[i] for i in train_indicies]
    key = jax.random.key(7201999)

    for step in range(num_epochs):
        print(f"Step {step + 1}/{num_epochs}")
        overall_loss = 0.0
       
        # train
        nrows = len(rows)
        errors_energy = np.zeros(nrows)  # To store energy and force errors
        errors_force = np.zeros(nrows)
        indicies = np.arange(nrows)
        np.random.shuffle(indicies)
        force_weight, energy_weight = loss_weights(step, num_epochs)
        for i in tqdm.tqdm(indicies, desc="Processing rows", total=nrows):
            interaction = interactions[i]
            box = boxes[i]
            row = rows[i]
            
            loss_value, grads = loss_fn(initial_params, interaction, box, row, force_weight, energy_weight, regularization_weight)
            grads['Dmin'] = 0.0
            grads['Dmax'] = 0.0
            loss_value, errors = loss_value  # Unpack the loss value and errors
            overall_loss += loss_value / len(rows)
            errors_energy[i] = errors['energy_error']
            errors_force[i] = errors['force_error']
            for key in grads.keys():
                grads[key] = jnp.nan_to_num(grads[key], nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and Infs
            # print(f"{grads['F'][0][0][8][0] = }")
            # print(f"{grads['F'][0][0][0][8] = }")
            updates, opt_state = optimizer.update(grads, opt_state)
            initial_params = optax.apply_updates(initial_params, updates)
            initial_params = fix_params(initial_params)  # Ensure parameters are valid
        # print(f"self energies preopt: {initial_params['self_energy']}")
        # set_self_energies(initial_params, rows + validation_rows, interactions + validation_interactions, boxes + validation_boxes)
        # print(f"self energies postopt: {initial_params['self_energy']}")
        # pickle.dump(initial_params, open(f"allparams/{output_prefix}_params_step_{step}.pkl", "wb"))
        print(f"Epoch Loss:\t{overall_loss}, energy errors: {errors_energy.mean()}, force errors: {errors_force.mean()}")

        # calculate validation loss
        val_errors_energy = np.zeros(len(validation_rows))  # To store energy and force errors
        val_errors_force = np.zeros(len(validation_rows))
        overall_val_loss = 0.0
        for i, (interaction, box, row) in tqdm.tqdm(enumerate(zip(validation_interactions, validation_boxes, validation_rows)), desc="Validating rows", total=len(validation_rows)):
            loss_value, errors = loss_nograd(initial_params, interaction, box, row)
            val_errors_energy[i] = errors['energy_error']
            val_errors_force[i] = errors['force_error']
            overall_val_loss += loss_value / len(validation_rows)
        
        print(f"Validation Loss:\t{overall_val_loss}, energy errors: {val_errors_energy.mean()}, force errors: {val_errors_force.mean()}")

        if overall_val_loss < best_validation_loss:
            best_validation_loss = overall_val_loss
            print(f"New best validation loss: {best_validation_loss}, saving parameters.")
            # Save the parameters if needed, e.g., using pickle or any other method
            pickle.dump(initial_params, open(f"params/{output_prefix}_{step}_params.pkl", "wb"))
        pickle.dump(initial_params, open(f"params/latest_params.pkl", "wb"))
    return initial_params


def set_self_energies(params, rows, interactions, boxes):
    """Calculate self-energies based on the current parameters and dataset."""
    current_energies = np.zeros(len(rows), dtype=np.float32)  # Initialize to zero
    current_energies = np.zeros(len(rows), dtype=np.float32)
    for i, (interaction, box) in tqdm.tqdm(enumerate(zip(interactions, boxes)), total=len(rows)):
        current_energies[i] = row_energy_noself(params, interaction, box)
    A = np.array(
        [[int(np.sum(interaction[1] == i)) for i in range(3)] for interaction in interactions]
    )
    true_energies = np.array([row['energy'] for row in rows])
    b = true_energies - current_energies
    self_energies = np.linalg.inv(A.T @ A) @ A.T @ b
    params['self_energy'] = jnp.array(self_energies, dtype=jnp.float32)
    # params['self_energy'] = jnp.array([-243.32243, -13.193096, -558.5242], dtype=jnp.float32)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Trains a REBO potential."
    )
    parser.add_argument(
        "-p",
        "--starting_params",
        type=str,
        help="Starting parameter file",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_prefix",
        type=str,
        help="prefix of the output param files",
        default="REBO",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-rw", "--regularization_weight", type=float, help="weight of spline regularization rate", default=0.0)
    args = parser.parse_args()
    print(args)
    print("Loading dataset...")
    if args.starting_params is not None:
        params = pickle.load(open(args.starting_params, "rb"))
    rows = pickle.load(open("interactions/most_rows_padded.pkl", "rb"))
    interactions = pickle.load(open("interactions/most_interactions_padded_numpy.pkl", "rb"))
    # params = pickle.load(open("params/fgk2_52_params.pkl", "rb"))
    np.random.seed(72099)
    indicies = np.arange(len(rows))
    np.random.shuffle(indicies)
    # indicies = indicies[:5]
    rows = [rows[i] for i in indicies]
    interactions = [interactions[i] for i in indicies]
    cutoff = 3.0

    # pad interactions:
    max_natoms = 256
    max_nbonds = 5187
    max_nneighbors = 40

    interactions = [interaction.dump_arrays() for interaction in interactions]
    boxes = [jnp.array(row['orth_matrix'], dtype=jnp.float32) for row in rows]

    # for i in tqdm.tqdm(range(len(rows)), desc="rePadding interactions"):
    #     natoms = np.sum(rows[i]['species'] >= 0)
    #     interactions[i] = list(interactions[i])
    #     interactions[i][2][interactions[i][2] == natoms] = -1
    #     interactions[i][3][interactions[i][3] == natoms] = -1
    #     interactions[i][7][interactions[i][7] == natoms] = -1
    #     interactions[i][10][interactions[i][10] == natoms] = -1
    #     interactions[i][13][interactions[i][13] == natoms] = -1
    #     interactions[i] = tuple(interactions[i])

    print("Dataset loaded.")
    print("Setting self-energies based on dataset...")
    set_self_energies(params, rows, interactions, boxes)
    print(f"Self-energies set based on dataset: {params['self_energy']}")
    print("Number of structures:", len(rows))
    print("Optimizing potential parameters...")
    new_params = optimize_potential(
        rows,
        interactions,
        boxes,
        params,
        validation_frac=0.2,
        num_epochs=200,
        learning_rate=args.learning_rate,
        output_prefix=args.output_prefix,
        regularization_weight=args.regularization_weight
    )
    print(f"Optimized Parameters: {new_params}")
