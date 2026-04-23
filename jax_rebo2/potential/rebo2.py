import numpy as np
import jax
import jax.numpy as jnp

from jax_md import space, smap, partition, quantity, util
from typing import Callable, Tuple, TextIO, Dict, Any, Optional
from functools import wraps, partial


from jax.scipy.special import erfc
from jax_md import interpolate
import interpax


# Types
f32 = util.f32
f64 = util.f32
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat
DisplacementOrMetricFn = space.DisplacementOrMetricFn

Box = space.Box


# Import Tersoff Functions for Testing


def _rebo2_cutoff(dr, Dmin, Dmax) -> Array:
    """The cut-off function of the Tersoff potential.
    Args:
      R: A Parameter that is the average of inner and outer cutoff radii
      D: A Parameter that is the half of the difference
         between inner and outer cutoff radii

    Returns:
      cut-off values
    """
    outer = jnp.where(dr < Dmax, 0.5 * (1 + jnp.cos(jnp.pi * (dr - Dmin) / (Dmax - Dmin))), 0)
    inner = jnp.where(dr < Dmin, 1, outer)
    return inner


# REBO II Potential Functions


def load_lammps_rebo2_parameters(file: TextIO, num_type: int) -> Array:
    # Establish variables to read in
    param_length = num_type * (num_type - 1) / 2

    # function to read in rebo2 parameters
    params = []
    params_per_line = 2

    # read parameters.
    # skip if the line has \# or empty
    # if the number of parameters in one line is less than params_per_line,
    # additional line is appended to match.
    skip = False
    for line in file.read().split("\n"):
        words = line.strip().split()
        nwords = len(words)
        if "#" in words or nwords == 0:
            continue

        if nwords < params_per_line and skip is False:
            line_keep = line
            skip = True
            continue

        line_keep += " " + line
        words = line_keep.strip().split()
        nwords = len(words)

        if nwords != params_per_line:
            raise ValueError(
                "Incorrect format: %d not in %d" % (nwords, params_per_line)
            )
        else:
            skip = False

    return params


# Given an array of distances between atoms and species, will sum tersoff weighted neighbors
def sum_species(Dmin, Dmax, s: int, dRij: Array, species: Array) -> Array:
    drij = space.distance(dRij)
    s_neighbor = jnp.array(jnp.empty_like(species))

    # TODO needs to sum over all species and return array not float
    for i in range(dRij.shape[0]):
        num_neighbor = 0
        for j in range(dRij.shape[1]):
            num_neighbor += jnp.where(
                (species[i] == s) & (i != j), _rebo2_cutoff(drij[i, j], Dmin, Dmax), 0
            )

    return s_neighbor


def _Ni_t(fc: Array) -> Array:
    Ni_total = jnp.sum(fc, axis=0)
    # copy the total to make an NxN array
    Ni_t = jnp.repeat(Ni_total[None, :], fc.shape[0], axis=0)
    Ni_t -= fc
    return Ni_t


def _Ni_species(fc: Array, species, target_species: int) -> Array:
    mask = species == target_species
    Ni_total = jnp.sum(mask * fc, axis=1)
    # copy the total to make an NxN array
    Ni_species = jnp.repeat(Ni_total[:, None], fc.shape[0], axis=1)
    Ni_species -= mask * fc
    return Ni_species


# Splining Function for conjugation F(x_ik)
def _Fx(x_ik):
    return jnp.where(
        x_ik < 2, 1, jnp.where(x_ik > 3, 0, (1 + jnp.cos(jnp.pi * (x_ik - 2))) / 2)
    )


# Conjugation Term for Carbon
def _N_conj(Ni_t, fc, species):
    species_i = jnp.repeat(species[None, :], species.shape[0], axis=0)
    species_k = jnp.repeat(species[:, None], species.shape[0], axis=1)
    # print(f"species_k shape: {species_k.shape}, species_k: {species_k}")
    mask = species_k == 0
    mask &= species_i == 0  # Carbon species

    # print(f"Ni_t shape: {Ni_t.shape}, Ni_t:\n{Ni_t}")
    # print(f"_Fx(Ni_t) shape: {_Fx(Ni_t).shape}, _Fx(Ni_t):\n{_Fx(Ni_t)}")
    
    term0 = jnp.where(mask, fc * _Fx(Ni_t), 0)  # Carbon species
    # print(f"term0 shape: {term0.shape}, term0:\n{term0}")
    term1 = jnp.sum(term0, axis=1)
    # print(f"term1 shape: {term1.shape}, term1: {term1}")
    term1 = jnp.repeat(term1[:, None], fc.shape[0], axis=1)
    # print(f"term1 after repeat shape: {term1.shape}, term1:\n{term1}")
    # remove j term from the sum
    term1 -= term0
    # print(f"term1 after removing j term shape: {term1.shape}, term1:\n{term1}")
    
    term2 = term1.T
    Nconj = 1 + term1**2 + term2**2
    return Nconj


def normalize(x: jnp.ndarray, axis: int, eps: float = 1e-12) -> jnp.ndarray:
    norm = jnp.sum(x * x, axis=axis, keepdims=True)
    # if the norm is close to zero
    # normalization should not be applied
    # the gradient should be the same as identify map
    norm = jnp.where(
        norm < eps,
        1.0,
        norm,
    )
    norm = jnp.sqrt(norm)
    return x / norm


def _b_ij_dh(dRij, drij, fc, Ni_t, Nj_t, Nij_conj, nspecies, species, T_lists) -> Array:
    # Compute cross products
    c1 = jnp.cross(dRij[:, :, None, :], dRij[None, :, :, :])
    # i != j, i != k, i != l, j != k, j != l, k != l
    mask_ijkl = (
        (1 - jnp.eye(dRij.shape[0], dtype=dRij.dtype))[:, :, None, None]  # i != j
        * (1 - jnp.eye(dRij.shape[0], dtype=dRij.dtype))[:, None, :, None]  # i != k
        * (1 - jnp.eye(dRij.shape[0], dtype=dRij.dtype))[:, None, None, :]  # i != l
        * (1 - jnp.eye(dRij.shape[0], dtype=dRij.dtype))[None, :, :, None]  # j != k
        * (1 - jnp.eye(dRij.shape[0], dtype=dRij.dtype))[None, :, None, :]  # j != l
        * (1 - jnp.eye(dRij.shape[0], dtype=dRij.dtype))[None, None, :, :]  # k != l
    )
    # make c1 and c2 unit vectors
    c1 = normalize(c1, axis=3)

    # Get static indices where mask is True

    # Vectorize over all valid index sets
    costheta = -jnp.einsum("ijkc,ijlc->ijkl", c1, c1)  # memory bottleneck... 171 GB
    # costheta = -jnp.sum(c1[..., :, None, :] * c1[..., None, :, :], axis=-1)
    # print(f"theta shape: {costheta.shape}, theta:\n{costheta[mask_ijkl > 0]}")
    fc_ik_fc_jl = fc[:, None, :, None] * fc[None, :, None, :]  # shape: (i, j, k, l)

    # Compute angular factor
    theta_factor = 1 - costheta ** 2
    # theta_factor = jnp.where(mask_ijkl, theta_factor, 0)  # Apply mask to ensure i != j, k != l

    # Broadcast and multiply: (i,j,k,l)
    weight = (
        theta_factor * fc_ik_fc_jl * mask_ijkl
    )
    # print(f"weight shape: {weight.shape}, weight:\n{weight[weight > 0]}")
    # Final reduction
    b = jnp.sum(weight, axis=(2, 3))  # shape: (i, j)

    T = np.zeros(Ni_t.shape, dtype=f64)
    species_i = jnp.repeat(species[:, None], species.shape[0], axis=1)
    species_j = jnp.repeat(species[None, :], species.shape[0], axis=0)
    for i in range(nspecies):
        for j in range(nspecies):
            mask = (species_i == i) & (species_j == j)
            T += jnp.where(mask, T_lists[i][j](Ni_t, Nj_t, Nij_conj), 0)
    # print(f"T shape: {T.shape}, T:\n{np.where(fc > 0, T, '_')}")
    # print(f"b shape: {b.shape}, b:\n{np.where(fc > 0, b, '_')}")
    return b * T


def _PI_rc(Ni_t, Nj_t, Nij_conj, F_lists, nspecies, species) -> Array:
    species_i = jnp.repeat(species[:, None], species.shape[0], axis=1)
    species_j = jnp.repeat(species[None, :], species.shape[0], axis=0)
    F = np.zeros(Ni_t.shape, dtype=f64)
    for i in range(nspecies):
        for j in range(nspecies):
            mask = (species_i == i) & (species_j == j)
            # print(f"i, j = {i}, {j}, mask: {mask}")
            F += jnp.where(mask, F_lists[i][j](Ni_t, Nj_t, Nij_conj), 0)
            # print(f"i, j = {i}, {j}, new F: {F}")
    return F


def _Q(Ni_t) -> Array:
    outer = jnp.where(Ni_t < 3.7, 0.5 * (1 + jnp.cos(2 * jnp.pi * (Ni_t - 3.2))), 0)
    inner = jnp.where(Ni_t < 3.2, 1, outer)
    return inner


def print3d_array(arr: Array):
    """Prints a 3D array in a readable format."""
    for i in range(arr.shape[0]):
        print(f"Slice {i}:")
        for j in range(arr.shape[1]):
            print(f"  Row {j}: {arr[i, j]}")
        print("\n")
    print("End of 3D array\n")


def _rebo2_bij_sigma(
    fc, Rij, rij, Ni_t, lambda_ijk, G_lists, gamma_lists, P_lists, rho, nspecies, species
) -> Array:
    """The bond-order term of the REBO II potential.
    Args:
      fc: A ndarray of shape [n, neighbors, dim] of pairwise distances between
        particles
      Rij: A ndarray of shape [n, neighbors, dim] of pairwise distances between
        particles
      lambda_ijk: A ndarray of shape [n, neighbors, dim] of pairwise distances between
        particles
      G_lists: A list of G functions for each species pair
      P_lists: A list of P functions for each species pair
    Returns:
      Bond-order values between i and j atoms
    """
    # g = jnp.zeros((Rij.shape[0], Rij.shape[1]), dtype=Rij.dtype)
    # for i in range(Rij.shape[0]):
    #     for j in range(Rij.shape[1]):
    #         for k in range(Rij.shape[1]):
    #             if i == j or i == k or j == k:
    #                 continue
    #             costheta = jnp.dot(Rij[i, j], Rij[i, k]) / (jnp.linalg.norm(Rij[i, j]) * jnp.linalg.norm(Rij[i, k]))
    #             G = G_lists[species[i]][species[j]](costheta)
    #             gamma = gamma_lists[species[i]][species[j]](costheta)
    #             g[i,j] += G + _Q(Ni_t[i, j]) * (gamma - G)

    # Compute all pairwise cos(theta) between (i, j, k)
    # Rij: [n, n, d]
    # Ni_t: [n, n]
    # species: [n]
    # G_lists, gamma_lists: [nspecies][nspecies] functions

    n = Rij.shape[0]

    # Normalize Rij vectors
    # Rij_norm = rij + 1e-12
    # Rij_norm = jnp.repeat(
    #     Rij_norm[:, :, None], Rij.shape[2], axis=2
    # )  # shape: (n, n, n)
    # Rij_unit = Rij / Rij_norm
    Rij_unit = normalize(Rij, axis=-1)

    # Compute costheta for all (i, j, k)
    # costheta[i, j, k] = dot(Rij[i, j], Rij[i, k]) / (||Rij[i, j]|| * ||Rij[i, k]||)
    costheta = jnp.einsum('ijd,ikd->ijk', Rij_unit, Rij_unit)
    n = Rij.shape[0]
    mask_ijk = (
        (1 - jnp.eye(n))[:, :, None]  # i != j
        * (1 - jnp.eye(n))[None, :, :]  # i != k
        * (1 - jnp.eye(n))[:, None, :]  # j != k
    )
    mask_ij = (1 - jnp.eye(n))
    mask_ijk_bool = mask_ijk.astype(bool)
    # Get species indices for i and j

    # Apply G and gamma functions for all (i, j, k)
    G = jnp.zeros_like(costheta)
    gamma = jnp.zeros_like(costheta)
    species_2d = jnp.repeat(species[:, None], n, axis=1)
    species_3d = jnp.repeat(species_2d[:, :, None], n, axis=2)
    for i in range(nspecies):
        mask = (species_3d == i)
        mask *= mask_ijk_bool  # Apply the mask to only valid (i, j, k) pairs
        G += jnp.where(
            mask, G_lists[i](jnp.clip(costheta, -1, 1)), 0
        )
        gamma += jnp.where(
            mask, gamma_lists[i](jnp.clip(costheta, -1, 1)), 0
        )

    Qval = _Q(Ni_t)
    Qval = jnp.repeat(Qval[:, None], n, axis=1)
    g = mask_ijk * (G + Qval * (gamma - G))
    
    # exp_term[i, j, k] = jnp.exp(lambda_ijk[i] * ((rho[ktype][itype]-rik)-(rho[jtype][itype]-rij)));
    lambda_ijk_2d = jnp.repeat(lambda_ijk[:, None], n, axis=1)
    lambda_ijk_3d = jnp.repeat(lambda_ijk_2d[:, :, None], n, axis=2)
    rij_3d = jnp.repeat(rij[:, :, None], n, axis=2)
    rik_3d = jnp.repeat(rij[:, None, :], n, axis=1)  # Assuming rik is the same as rij for now
    rho_ij = jnp.repeat(rho[:, :, None], n, axis=2)
    rho_ik = jnp.repeat(rho[:, None, :], n, axis=1)

    exp_term = jnp.exp(
        lambda_ijk_3d * ((rij_3d - rho_ij) - (rik_3d - rho_ik))
    )
    # print(f"rij:")
    # print3d_array(rij_3d - rho_ij)
    # print(f"rik:")
    # print3d_array(rik_3d - rho_ik)
    # print(f"lambda_ijk:")
    # print3d_array(lambda_ijk_3d)
    exp_term = jnp.where(mask_ijk, exp_term, 0)
    # print(f"expo sumpo: {jnp.sum(exp_term, axis=2)}")
    fc_ik = jnp.repeat(fc[:, None, :], n, axis=1)
    # sum over k
    # print(f"costheta:")
    # print3d_array((fc_ik * costheta))
    # print(f"g:")
    # print3d_array((fc_ik * g))
    # print(f"exp:")
    # print3d_array(exp_term)
    # print(f"product")
    # print3d_array(fc_ik * g * exp_term)
    gsum = jnp.sum(fc_ik * g * exp_term, axis=2)
    # print(f"fc_ik zero:\n{jnp.any(fc_ik < -1e-6)}")
    # print(f"g zero:\n{jnp.any(g < -1e-6)}")
    # print(f"exp_term zero:\n{jnp.any(exp_term < -1e-6)}")
    # print(f"fc_ik:")
    # print3d_array(fc_ik)
    # print(f"g*f: \n{fc_ik * g}")
    # print(f"gsum:\n{gsum}")

    NC = jnp.clip(_Ni_species(fc, species, 0), 0, 9)  # Carbon
    NH = jnp.clip(_Ni_species(fc, species, 1), 0, 9)  # Hydrogen
    NO = jnp.clip(_Ni_species(fc, species, 2), 0, 9)  # Oxygen
    # print(f"NC\n{NC}")
    # print(f"NH\n{NH}")
    # print(f"NO {NO}")
    P = jnp.zeros(rij.shape, dtype=Rij.dtype)
    species_i = jnp.repeat(species[:, None], species.shape[0], axis=1)
    species_j = jnp.repeat(species[None, :], species.shape[0], axis=0)
    for i in range(nspecies):
        for j in range(nspecies):
            mask = (species_i == i) & (species_j == j)
            mask = mask_ij * mask
            P += jnp.where(mask, P_lists[i][j](NC, NH, NO), 0)
    # print(f"P shape: {P.shape}, P:\n {P[fc>0]}")
    
    bij_sigma = jnp.where(mask_ij,
        1 / jnp.sqrt(1 + gsum + P),
        0)
    # print(f"gsum zero:\n{jnp.any(gsum < -0.01)}")
    # print(f"gsum:\n{jnp.any(jnp.isnan(gsum))}")
    # print(f"P zero:\n{jnp.any(P < -0.01)}")
    # print(f"P:\n{jnp.any(jnp.isnan(P))}")
    # print(f"bij_sigma:\n{bij_sigma[fc>0]}")
    bij_sigma = 0.5 * (bij_sigma + bij_sigma.T)  # average over ij and ji
    return bij_sigma


def _rebo2_bij(
    fc, dRij, F_lists, lambda_ijk, G_lists, gamma_lists, P_lists, rho, T_lists, nspecies, species
) -> Array:
    """The bond-order term of the REBO II potential.
    Args:
      # parameters for cut-off functions
      R: A Parameter that is the average of inner and outer cutoff radii
      D: A Parameter that is the half of the difference
         between inner and outer cutoff radii

      # parameters related to the angle Penalty function in the bond-order
      # function
      # h(\theta) = 1 + c^2/d^2 + c^2/(d^2 + (h - cos(\theta)^2))
      c: A Parameter that determines angle penalty
      d: A Parameter that determines angle penalty
      h: A cosine value that is a desirable angle between 3 atoms.

      # parameters related to the distance penalty function in the bond-order
      # function
      lam3: A Parameter that determines distance penalty value
      m: A Parameter that determines distance penalty value

      # parameters related to the bond-order function
      beta: A Parameter that determines bond-order value
      n: A Parameter that determines bond-order value

      dRij: A ndarray of shape [n, neighbors, dim] of pairwise distances between
       particles
      dRik: A ndarray of shape [n, neighbors, dim] of pairwise distances between
        particles TODO - Currently, it is the same as the dRij
      dRjl: A ndarray of shape [n, neighbors, dim] of pairwise distances between
        particles TODO - Currently, it is the same as the dRij


    Returns:
      Bond-order values between i and j atoms
    """

    drij = space.distance(dRij)
    # drik = space.distance(dRik)
    # drjl = space.distance(dRjl)

    # mask_ijk *= (1 - jnp.eye(mask_ijk.shape[-1], dtype=dRij.dtype))[None, :, :]

    # Sum the weighted number of C,H,O in the cutoff radius
    Ni_t = _Ni_t(fc)
    Nj_t = Ni_t.T
    # print(f"Ni_t shape: {Ni_t.shape}, Ni_t:\n{Ni_t[fc > 0]}")
    # print(f"Nj_t shape: {Nj_t.shape}, Nj_t:\n{Nj_t[fc > 0]}")
    
    # Calculate PI_rc via tricubic spline
    Nij_conj = _N_conj(Ni_t, fc, species)
    # print(f"Nij_conj:\n{np.where(fc>0,Nij_conj, '_')}")
    bij_pi = _PI_rc(Ni_t, Nj_t, Nij_conj, F_lists, nspecies, species)
    # print(f"bij_pi:\n{jnp.any(jnp.isnan(bij_pi))}")
    # print(f"bij_pi_RC\n{np.where(fc>0, bij_pi, '_')}")
    # print(f"bij_pi_RC\n{bij_pi}")
    bij_pi_dh = _b_ij_dh(dRij, drij, fc, Ni_t, Nj_t, Nij_conj, nspecies, species, T_lists)
    # print(f"bij_pi_dh:\n{jnp.any(jnp.isnan(bij_pi_dh))}")
    # print(f"bij_pi_dh\n{np.where(fc>0, bij_pi_dh, '_')}")
    bij_pi += bij_pi_dh
    # print(f"bij_pi after b_ij_dh {bij_pi}")
    bij_sigma = _rebo2_bij_sigma(fc, dRij, drij, Ni_t, lambda_ijk, G_lists, gamma_lists, P_lists, rho, nspecies, species)
    # print(f"bij_sigma:\n{jnp.any(jnp.isnan(bij_sigma))}")
    # print(f"bij_sigma {bij_sigma[fc > 0]}")
    bij = bij_pi + bij_sigma
    # print(f"bij shape: {bij.shape}, bij: {bij[fc > 0]}")
    return bij


def _rebo2_attractive(
    B1,
    B2,
    B3,
    beta_1,
    beta_2,
    beta_3,
    Dmin,
    Dmax,
    F_lists,
    lambda_ijk,
    G_lists,
    gamma_lists,
    P_lists,
    rho,
    T_lists,
    nspecies,
    species,
    dRij,
    mask_ijk,
) -> Array:
    """The attractive term of the REBO II potential.
    Args:
      dR12: A ndarray of shape [n, neighbors, dim] of pairwise distnaces between
        particles.
      dR13: A ndarray of shape [n, neighbors, dim] of pairwise distnaces between
        particles. TODO - Currently, it is the same as the dR12
      dR24: A ndarray of shape [n, neighbors, dim] of pairwise distnaces between
        particles. TODO - Currently, it is the same as the dR12
      R: A Parameter that is the average of inner and outer cutoff radii.
      D: A Parameter that is the half of the difference.
         between inner and outer cutoff radii.

      # parameters related to the angle Penalty function in the bond-order
      # function.
      # h(\theta) = 1 + c^2/d^2 + c^2/(d^2 + (h - cos(\theta)^2))
      c: A Parameter that determines angle penalty.
      d: A Parameter that determines angle penalty.
      h: A cosine value that is a desirable angle between 3 atoms.

      # parameters related to the distance penalty function in the bond-order
      # function.
      lam2 (beta): A Parameter that determines distance penalty value.
      lam3: A Parameter that determines distance penalty value.
      m: A Parameter that determines distance penalty value.

      # parameters related to the bond-order function
      beta: A Parameter that determines bond-order value.
      n: A Parameter that determines bond-order value.

    Returns:
      Attractive interaction energy for one pair of neighbors.
    """
    drij = space.distance(dRij)
    fc = _rebo2_cutoff(drij, Dmin, Dmax)

    fc -= jnp.diag(jnp.diag(fc))  # remove self-interaction
    fA = B1 * jnp.exp(-beta_1 * drij) + B2 * jnp.exp(-beta_2 * drij) + B3 * jnp.exp(-beta_3 * drij)
    # print(f"B1: {B1}\n\nB2: {B2}\n\nB3: {B3}\n\nbeta_1: {beta_1}\n\nbeta_2: {beta_2}\n\nbeta_3: {beta_3}")
    bij = _rebo2_bij(
        fc, dRij, F_lists, lambda_ijk, G_lists, gamma_lists, P_lists, rho, T_lists, nspecies, species
    )
    # print(f"fA: {fA}\n\nfc: {fc}\n\nfcfA: {fc*fA}\n\n bij: {bij}")
    # print(f"bij:\n{jnp.any(jnp.isnan(bij))}")
    return - fc * bij * fA


def _rebo2_repulsive(A: f64, Q: f64, alpha: f64, Dmin: f64, Dmax: f64, dr: Array) -> Array:
    """The repulsive term of the REBO II potential.
    Args:
      A: A scalar that determines repulsive energy (eV).
      Q: A scalar that represents Colombic repulsion
      lam1 (alpha): A scalar that determines the scale two-body distance (Angstrom).
      R: A scalar that is the average of inner and outer cutoff radii.
      D: A scalar that is the half of the difference
         between inner and outer cutoff radii.
    Returns:
      Repulsive interaction energy for one pair of neighbors.
    """
    
    fC = _rebo2_cutoff(dr, Dmin, Dmax)
    fC -= jnp.diag(jnp.diag(fC))  # remove self-interaction
    fR = A * (1 + Q / dr) * jnp.exp(-alpha * dr)
    return fC * fR


def _rebo2_make_3d_spline(knots_i, knots_j, knots_k, f, fx, fy, fz):
    f = interpax.Interpolator3D(knots_i, knots_j, knots_k, f, fx=fx, fy=fy, fz=fz, method='cubic', extrap=True)
    f = jax.vmap(f)
    return f


def _rebo2_make_1d_spline(knots_i, f, fx):
    f = interpax.Interpolator1D(knots_i, f, fx=fx, method='cubic', extrap=True)
    f = jax.vmap(jax.vmap(jax.vmap(f)))
    return f


def rebo2(
    displacement: DisplacementFn,
    params: Array,
    species: Array,
    nspecies: int = 3,
) -> Callable[[Array], Array]:
    """Computes REBO II potential.
    Args:
      displacement: The displacement function for the space.
      params: A dictionary of parameters for the potential.
      species: An array of species. .

    Returns:
      A function that computes the total energy.

    [2] D. Brenner et al "A second-generation reactive empirical bond order
    (REBO) potential energy expression for hydrocarbons" J. Phys.: Condens. Matter 14 (2002): 783.
    """
    # check number of parameters set.
    if species is None:
        params = params[0]

    # define a repulsive and an attractive function with given parameters.
    natoms = species.shape[0]
    species_i = jnp.repeat(species[:, None], species.shape[0], axis=1)
    species_j = jnp.repeat(species[None, :], species.shape[0], axis=0)
    Dmin = jnp.zeros((natoms, natoms), dtype=params["Dmin"].dtype)
    Dmax = jnp.zeros((natoms, natoms), dtype=params["Dmax"].dtype)
    A = jnp.zeros((natoms, natoms), dtype=params["A"].dtype)
    Q = jnp.zeros((natoms, natoms), dtype=params["Q"].dtype)
    alpha = jnp.zeros((natoms, natoms), dtype=params["alpha"].dtype)
    for i in jnp.arange(nspecies):
        for j in jnp.arange(nspecies):
            mask = (species_i == i) & (species_j == j)
            Dmin = Dmin + jnp.where(mask, params["Dmin"][i, j], 0)
            Dmax = Dmax + jnp.where(mask, params["Dmax"][i, j], 0)
            A = A + jnp.where(mask, params["A"][i, j], 0)
            Q = Q + jnp.where(mask, params["Q"][i, j], 0)
            alpha = alpha + jnp.where(mask, params["alpha"][i, j], 0)

    repulsive_fn = partial(
        _rebo2_repulsive,
        A,
        Q,
        alpha,
        Dmin,
        Dmax,
    )

    B1 = jnp.zeros((natoms, natoms), dtype=params["B1"].dtype)
    B2 = jnp.zeros((natoms, natoms), dtype=params["B2"].dtype)
    B3 = jnp.zeros((natoms, natoms), dtype=params["B3"].dtype)
    beta_1 = jnp.zeros((natoms, natoms), dtype=params["beta_1"].dtype)
    beta_2 = jnp.zeros((natoms, natoms), dtype=params["beta_2"].dtype)
    beta_3 = jnp.zeros((natoms, natoms), dtype=params["beta_3"].dtype)
    rho = jnp.zeros((natoms, natoms), dtype=params["rho"].dtype)
    lambda_ijk = jnp.zeros((natoms), dtype=params["lambda_ijk"].dtype)
    F_lists = []
    T_lists = []
    P_lists = []
    G_lists = []
    gamma_lists = []
    G_knots = jnp.array(params["G_knots"])
    for i in range(nspecies):
        # Initialize empty lists for each species
        F_lists.append([])
        T_lists.append([])
        P_lists.append([])

        G_lists.append(
            _rebo2_make_1d_spline(G_knots, jnp.array(params["G"][i]), jnp.array(params["dG"][i]))
        )
        gamma_lists.append(
            _rebo2_make_1d_spline(G_knots, jnp.array(params["gamma"][i]), jnp.array(params["dgamma"][i]))
        )
        mask = species == i
        lambda_ijk += jnp.where(mask, params["lambda_ijk"][i], 0)
        for j in range(nspecies):
            F_lists[i].append(
                _rebo2_make_3d_spline(
                    jnp.arange(10),  # knots_i
                    jnp.arange(10),  # knots_j
                    jnp.arange(10),  # knots_k
                    params["F"][i][j],  # f
                    params["F_di"][i][j],  # fx
                    params["F_dj"][i][j],  # fy
                    params["F_dk"][i][j],  # fz
                )
            )
            T_lists[i].append(
                _rebo2_make_3d_spline(
                    jnp.arange(10),  # knots_i
                    jnp.arange(10),  # knots_j
                    jnp.arange(10),  # knots_k
                    params["T"][i][j],  # f
                    params["T_di"][i][j],  # fx
                    params["T_dj"][i][j],  # fy
                    params["T_dk"][i][j],  # fz
                )
            )
            P_lists[i].append(
                _rebo2_make_3d_spline(
                    jnp.arange(10),  # knots_i
                    jnp.arange(10),  # knots_j
                    jnp.arange(10),  # knots_k
                    params["P"][i][j],  # f
                    params["P_di"][i][j],  # fx
                    params["P_dj"][i][j],  # fy
                    params["P_dk"][i][j],  # fz
                )
            )
            mask = (species_i == i) & (species_j == j)
            B1 += jnp.where(mask, params["B1"][i, j], 0)
            B2 += jnp.where(mask, params["B2"][i, j], 0)
            B3 += jnp.where(mask, params["B3"][i, j], 0)
            beta_1 += jnp.where(mask, params["beta_1"][i, j], 0)
            beta_2 += jnp.where(mask, params["beta_2"][i, j], 0)
            beta_3 += jnp.where(mask, params["beta_3"][i, j], 0)
            rho += jnp.where(mask, params["rho"][i, j], 0)

    attractive_fn = partial(
        _rebo2_attractive,
        B1,
        B2,
        B3,
        beta_1,
        beta_2,
        beta_3,
        Dmin,
        Dmax,
        F_lists,
        lambda_ijk,
        G_lists,
        gamma_lists,
        P_lists,
        rho,
        T_lists,
        nspecies,
        species,
    )

    # define compute functions.
    def compute_fn(R, **kwargs):
        d = partial(displacement, **kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)
        N = R.shape[0]
        mask = jnp.where(1 - jnp.eye(N), dr < Dmax, 0)
        mask = mask.astype(R.dtype)
        mask_ijk = mask[:, None, :] * mask[:, :, None]
        repulsive = util.safe_mask(mask, repulsive_fn, dr)
        # print(f"repulsive:\n{repulsive}")
        attractive = attractive_fn(dR, mask_ijk) * mask
        repulsive_term = util.high_precision_sum(repulsive)
        attractive_term = util.high_precision_sum(attractive)
        return 0.5 * (repulsive_term + attractive_term)

    return compute_fn


def rebo2_trainable(
    params: Dict,
    displacement: DisplacementFn,
    species: Array,
    R: Array,
    nspecies: int = 3,
) -> Array:
    """Computes REBO II potential.
    Args:
      displacement: The displacement function for the space.
      params: A dictionary of parameters for the potential.
      species: An array of species. .

    Returns:
      A function that computes the total energy.

    [2] D. Brenner et al "A second-generation reactive empirical bond order
    (REBO) potential energy expression for hydrocarbons" J. Phys.: Condens. Matter 14 (2002): 783.
    """
    # check number of parameters set.
    if species is None:
        params = params[0]

    # define a repulsive and an attractive function with given parameters.
    natoms = species.shape[0]
    species_i = jnp.repeat(species[:, None], species.shape[0], axis=1)
    species_j = jnp.repeat(species[None, :], species.shape[0], axis=0)
    Dmin = jnp.zeros((natoms, natoms), dtype=params["Dmin"].dtype)
    Dmax = jnp.zeros((natoms, natoms), dtype=params["Dmax"].dtype)
    A = jnp.zeros((natoms, natoms), dtype=params["A"].dtype)
    Q = jnp.zeros((natoms, natoms), dtype=params["Q"].dtype)
    alpha = jnp.zeros((natoms, natoms), dtype=params["alpha"].dtype)
    for i in jnp.arange(nspecies):
        for j in jnp.arange(nspecies):
            mask = (species_i == i) & (species_j == j)
            Dmin = Dmin + jnp.where(mask, params["Dmin"][i, j], 0)
            Dmax = Dmax + jnp.where(mask, params["Dmax"][i, j], 0)
            A = A + jnp.where(mask, params["A"][i, j], 0)
            Q = Q + jnp.where(mask, params["Q"][i, j], 0)
            alpha = alpha + jnp.where(mask, params["alpha"][i, j], 0)
    Dmin = Dmin + jnp.where((species_i == -1) | (species_j == -1), -2, 0)  # Handle -1 species
    Dmax = Dmax + jnp.where((species_i == -1) | (species_j == -1), -1, 0)  # Handle -1 species

    repulsive_fn = partial(
        _rebo2_repulsive,
        A,
        Q,
        alpha,
        Dmin,
        Dmax,
    )

    B1 = jnp.zeros((natoms, natoms), dtype=params["B1"].dtype)
    B2 = jnp.zeros((natoms, natoms), dtype=params["B2"].dtype)
    B3 = jnp.zeros((natoms, natoms), dtype=params["B3"].dtype)
    beta_1 = jnp.zeros((natoms, natoms), dtype=params["beta_1"].dtype)
    beta_2 = jnp.zeros((natoms, natoms), dtype=params["beta_2"].dtype)
    beta_3 = jnp.zeros((natoms, natoms), dtype=params["beta_3"].dtype)
    rho = jnp.zeros((natoms, natoms), dtype=params["rho"].dtype)
    lambda_ijk = jnp.zeros((natoms), dtype=params["lambda_ijk"].dtype)
    F_lists = []
    T_lists = []
    P_lists = []
    G_lists = []
    gamma_lists = []
    G_knots = jnp.array(params["G_knots"])
    for i in range(nspecies):
        # Initialize empty lists for each species
        F_lists.append([])
        T_lists.append([])
        P_lists.append([])

        G_lists.append(
            _rebo2_make_1d_spline(G_knots, jnp.array(params["G"][i]), jnp.array(params["dG"][i]))
        )
        gamma_lists.append(
            _rebo2_make_1d_spline(G_knots, jnp.array(params["gamma"][i]), jnp.array(params["dgamma"][i]))
        )
        mask = species == i
        lambda_ijk += jnp.where(mask, params["lambda_ijk"][i], 0)
        for j in range(nspecies):
            F_lists[i].append(
                _rebo2_make_3d_spline(
                    jnp.arange(10),  # knots_i
                    jnp.arange(10),  # knots_j
                    jnp.arange(10),  # knots_k
                    params["F"][i][j],  # f
                    params["F_di"][i][j],  # fx
                    params["F_dj"][i][j],  # fy
                    params["F_dk"][i][j],  # fz
                )
            )
            T_lists[i].append(
                _rebo2_make_3d_spline(
                    jnp.arange(10),  # knots_i
                    jnp.arange(10),  # knots_j
                    jnp.arange(10),  # knots_k
                    params["T"][i][j],  # f
                    params["T_di"][i][j],  # fx
                    params["T_dj"][i][j],  # fy
                    params["T_dk"][i][j],  # fz
                )
            )
            P_lists[i].append(
                _rebo2_make_3d_spline(
                    jnp.arange(10),  # knots_i
                    jnp.arange(10),  # knots_j
                    jnp.arange(10),  # knots_k
                    params["P"][i][j],  # f
                    params["P_di"][i][j],  # fx
                    params["P_dj"][i][j],  # fy
                    params["P_dk"][i][j],  # fz
                )
            )
            mask = (species_i == i) & (species_j == j)
            B1 += jnp.where(mask, params["B1"][i, j], 0)
            B2 += jnp.where(mask, params["B2"][i, j], 0)
            B3 += jnp.where(mask, params["B3"][i, j], 0)
            beta_1 += jnp.where(mask, params["beta_1"][i, j], 0)
            beta_2 += jnp.where(mask, params["beta_2"][i, j], 0)
            beta_3 += jnp.where(mask, params["beta_3"][i, j], 0)
            rho += jnp.where(mask, params["rho"][i, j], 0)

    attractive_fn = partial(
        _rebo2_attractive,
        B1,
        B2,
        B3,
        beta_1,
        beta_2,
        beta_3,
        Dmin,
        Dmax,
        F_lists,
        lambda_ijk,
        G_lists,
        gamma_lists,
        P_lists,
        rho,
        T_lists,
        nspecies,
        species,
    )

    # define compute functions.
    d = partial(displacement)
    dR = space.map_product(d)(R, R)
    dr = space.distance(dR)
    N = R.shape[0]
    mask = jnp.where(1 - jnp.eye(N), dr < Dmax, 0)
    mask = mask.astype(R.dtype)
    mask_ijk = mask[:, None, :] * mask[:, :, None]
    repulsive = util.safe_mask(mask, repulsive_fn, dr)
    # print(f"repulsive:\n{jnp.any(jnp.isnan(repulsive))}")
    attractive = attractive_fn(dR, mask_ijk) * mask
    # print(f"attractive:\n{jnp.any(jnp.isnan(attractive))}")
    repulsive_term = util.high_precision_sum(repulsive)
    attractive_term = util.high_precision_sum(attractive)
    return 0.5 * (repulsive_term + attractive_term)