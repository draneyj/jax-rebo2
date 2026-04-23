import jax_md
from jax_md import energy, simulate, quantity
import jax
from jax import random, lax
import jax.numpy as jnp
import numpy as np
import optax
import newrebo2_interaction_list as rebo2
import pickle
from functools import partial

jax.config.update("jax_enable_x64", True)

dimension = 3
key = random.PRNGKey(0)
R_list = []
species_list = []
name_list = []
# ====== CH2 ====== -8.46925365496041 (JAX) vs -8.46925409822361 (LAMMPS)
R = jnp.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.09, 0.0],
    [0.0, -1.09, 0.0],
])
species = jnp.array([0, 1, 1])
R_list.append(R)
species_list.append(species)
name_list.append("CH2")

# ====== CH3 ====== -13.375005687483345 vs -13.374999948
R = jnp.array([
    [0.0,0.0,0.0],
    [0.0, 1.09, 0],
    [jnp.cos(jnp.pi/6) * 1.09, -jnp.sin(jnp.pi/6) * 1.09, 0],
    [-jnp.cos(jnp.pi/6) * 1.09, -jnp.sin(jnp.pi/6) * 1.09, 0.0],
])
species = jnp.array([0, 1, 1, 1])
R_list.append(R)
species_list.append(species)
name_list.append("CH3")

# ====== methane ====== -18.184994192975246 vs -18.185050242313
R = jnp.array([
    [0.0,0.0,0.0],
    [0.0, 1.01695, -0.115756],
    [-0.822766, -0.369042, -0.369042],
    [0.0, -0.115756, 1.01695],
    [0.822766, -0.369042, -0.369042],
])
species = jnp.array([0, 1, 1, 1, 1])
R_list.append(R)
species_list.append(species)
name_list.append("methane")

# ====== C2H ====== -11.57211419549892 (JAX) vs -11.5721588693423 (LAMMPS)
# print("C2H")
R = jnp.array([
    [-1.09, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.31, 0.0, 0.0],
])
species = jnp.array([1, 0, 0])
R_list.append(R)
species_list.append(species)
name_list.append("C2H")

# ====== C2H2 ====== -17.565107972184713 (JAX) vs -17.5651072645008 (LAMMPS)
# print("C2H2")
R = jnp.array([
    [-1.09, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.54, 0.0, 0.0],
    [2.63, 0.0, 0.0],
])
species = jnp.array([1, 0, 0, 1])
R_list.append(R)
species_list.append(species)
name_list.append("C2H2")

# ====== H3C2H2 ====== -26.560159071428153 (JAX) vs -26.5601367223639 (LAMMPS)
# print("H3C2H2")
R = jnp.array([
    [0, 0, 0],
    [-0.822766, -0.369042, -0.369042],
    [0.0, 1.01695, -0.115756],
    [0.0, -0.115756, 1.01695],
    [0.822766, -0.369042, -0.369042],
    [0.822766, 0.9, 0.1],
    [0.822766, -0.9, -0.1],
])
species = jnp.array([0, 1, 1, 1, 0, 1, 1])
R_list.append(R)
species_list.append(species)
name_list.append("H3C2H2")

# ====== Ethane (staggered) ====== -30.84584985523632 (JAX) vs 30.8457 (PAPER)
# print("Ethane (staggered)")
# R = jnp.array([
#     [-0.7560, 0.0000, 0.0000],
#     [0.7560, 0.0000, 0.0000],
#     [-1.1404, 0.6586, 0.7845],
#     [-1.1404, 0.3501, -0.9626],
#     [-1.1405, -1.0087, 0.1781],
#     [1.1404, -0.3501, 0.9626],
#     [1.1405, 1.0087, -0.1781],
#     [1.1404, -0.6586, -0.7845]
# ])
# species = jnp.array([0, 0, 1, 1, 1, 1, 1, 1])

# ====== Cyclopropene ======: -28.258583446229903 (JAX) vs -28.2588327762861 (LAMMPS)
# print("Cyclopropene")
R = jnp.array([
    [-0.8487, -0.0002, 0.0000],
    [0.4242, 0.6507, 0.0000],
    [0.4245, -0.6505, 0.0000],
    [-1.4015, -0.0004, .9258],
    [-1.4015, -0.0004, -0.9258],
    [0.9653, 1.5624, 0.2000],
    [0.9661, -1.5620, 0.0000]
])
species = jnp.array([0, 0, 0, 1, 1, 1, 1])
R_list.append(R)
species_list.append(species)
name_list.append("Cyclopropene")

# ====== Propyne ======: -30.307678058494155 (JAX) vs -30.3075603625181 (LAMMPS)
# print("Propyne")
# R = jnp.array([
#     [-1.3750, 0.0001, 0.0000],
#     [0.0873, -0.0002, 0.0001],
#     [1.2877, 0.0001, -0.0001],
#     [-1.7644, 0.0660, 1.0204],
#     [-1.7641, 0.8510, -0.5673],
#     [-1.7644, -0.9165, -0.4533],
#     [2.3527, 0.0003, -0.0002]
# ])
# species = jnp.array([0, 0, 0, 1, 1, 1, 1])
# R_list.append(R)
# species_list.append(species)
# name_list.append("Propyne")

# ====== Cyclopropane ======: -36.88740399299321 (JAX) vs 36.8887 (PAPER)
# print("Cyclopropane")
# R = jnp.array([
#     [0.3185, 0.8057, 0.0000],
#     [-0.8570, -0.1271, 0.0000],
#     [0.5386, -0.6786, 0.0000 ],
#     [0.5329, 1.3483, -0.9120],
#     [0.5329, 1.3483, 0.9121],
#     [-1.4342, -0.2127, 0.9120],
#     [-1.4342, -0.2126, -0.9121],
#     [0.9012, -1.1357, -0.9121],
#     [0.9012, -1.1356, 0.9121]
# ])
# species = jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
# R_list.append(R)
# species_list.append(species)
# name_list.append("Cyclopropane")

# ====== Propane ======: -43.5892485829059 (JAX) vs 43.5891 (PAPER)
# print("Propane")
# R = jnp.array([
#     [0.0000, -0.5689, 0.0000],
#     [-1.2571, 0.2844, 0.0000],
#     [1.2571, 0.2845, 0.0000],
#     [0.0000, -1.2183, 0.8824],
#     [0.0000, -1.2183, -0.8824],
#     [-1.2969, 0.9244, 0.8873],
#     [-1.2967, 0.9245, -0.8872],
#     [-2.1475, -0.3520, -0.0001],
#     [2.1475, -0.3520, 0.0000],
#     [1.2968, 0.9245, 0.8872],
#     [1.2968, 0.9245, -0.8872]
# ])
# species = jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# ====== Cis-2-butene ======: -50.20176942543069 (JAX) vs 50.2017 (PAPER)
# print("Cis-2-butene")
# R = jnp.array([
#    [-0.6703,    0.6200,    0.0000],
#    [ 0.6702,    0.6200,    0.0000],
#    [-1.5014,   -0.6201,    0.0000],
#    [ 1.5015,   -0.6200,    0.0000],
#    [-1.1945,    1.5716,    0.0113],
#    [ 1.1943,    1.5718,   -0.0113],
#    [-2.3834,   -0.4587,   -0.6295],
#    [-0.9908,   -1.4974,   -0.4042],
#    [-1.8500,   -0.8414,    1.0132],
#    [ 2.3833,   -0.4586,    0.6300],
#    [ 0.9908,   -1.4974,    0.4036],
#    [ 1.8506,   -0.8408,   -1.0131],])
# species = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# ====== isobutane ======: -56.33098523138038 (JAX) vs 56.3309 (PAPER)
# print("isobutane")
# R = jnp.array([
#       [ 0.000000000000000,      0.000000000000000,      0.376949000000000,],
#       [ 0.000000000000000,      0.000000000000000,      1.475269000000000,],
#       [ 0.000000000000000,      1.450290000000000,     -0.096234000000000,],
#       [ 0.000000000000000,      1.493997000000000,     -1.190847000000000,],
#       [-0.885482000000000,      1.984695000000000,      0.261297000000000,],
#       [ 0.885482000000000,      1.984695000000000,      0.261297000000000,],
#       [ 1.255988000000000,     -0.725145000000000,     -0.096234000000000,],
#       [ 1.293839000000000,     -0.746998000000000,     -1.190847000000000,],
#       [ 2.161537000000000,     -0.225498000000000,      0.261297000000000,],
#       [ 1.276055000000000,     -1.759198000000000,      0.261297000000000,],
#       [-1.255988000000000,     -0.725145000000000,     -0.096234000000000,],
#       [-1.293839000000000,     -0.746998000000000,     -1.190847000000000,],
#       [-1.276055000000000,     -1.759198000000000,      0.261297000000000,],
#       [-2.161537000000000,     -0.225498000000000,      0.261297000000000,],
# ])
# species = jnp.array([0,1,0,1,1,1,0,1,1,1,0,1,1,1,])

# ====== Napthalene ======: -93.87734723423716 (JAX) vs 93.8784 (PAPER)
# print("Napthalene")
R = jnp.array(
    [
        [2.4044, 0.7559, 0.0000],
        [2.4328, -0.6584, 0.0000],
        [1.2672, -1.3753, 0.0000],
        [0.0142, -0.7050, 0.0000],
        [-0.0142, 0.7048, 0.0000],
        [1.2108, 1.4252, 0.0000],
        [-1.2672, 1.3754, 0.0000],
        [-2.4328, 0.6585, 0.0000],
        [-2.4043, -0.7558, 0.0000],
        [-1.2108, -1.4254, 0.0000],
        [3.3509, 1.3062, 0.0000],
        [3.4006, -1.1703, 0.0000],
        [1.2810, -2.4710, 0.0000],
        [1.1803, 2.5206, 0.0000],
        [-1.2808, 2.4710, 0.0000],
        [-3.4008, 1.1701, 0.0000],
        [-3.3508, -1.3060, 0.0000],
        [-1.1805, -2.5207, 0.0000],
    ]
)
species = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,])
R_list.append(R)
species_list.append(species)
name_list.append("Napthalene")


params = pickle.load(open("params/all_23_params.pkl", "rb"))
# params = pickle.load(open("params/initial_params_CHO.pkl", "rb"))
# params = pickle.load(open("params/starting_params.pkl", "rb"))
for k in params.keys():
    params[k] = np.array(params[k])
displacement, shift = jax_md.space.free()
fake_box = jnp.diag(jnp.array([100.0, 100.0, 100.0]))

energy_list = []
# for R, species, name in zip(R_list, species_list, name_list):
#     interaction_fn, energy_fn = rebo2.rebo2(displacement, fake_box, params, species, nspecies=3)
#     # interaction_fn, energy_fn = rebo2.rebo2(displacement, fake_box, params, species, nspecies=3)
#     interactions = interaction_fn(R)
#     energy_fn = partial(energy_fn, interactions=interactions)
#     energy = energy_fn(R)
#     print(f"{name}: {energy} eV")
#     energy_list.append(energy)
for R, species, name in zip(R_list, species_list, name_list):
    interactions = rebo2.allocate_interactions(R, species, displacement, 3.0, fake_box,
        max_natoms=256,
        max_nbonds=5187,
        max_nneighbors=40)
    interactions = interactions.dump_arrays()
    energy = rebo2.rebo2_trainable(params, fake_box, *interactions)
    energy = float(energy)
    print(f"{name}: {energy} eV")
    energy_list.append(energy)

print("COMPLETE ===============================================================")
for energy, name in zip(energy_list, name_list):
    print(f"{name}: {energy}")