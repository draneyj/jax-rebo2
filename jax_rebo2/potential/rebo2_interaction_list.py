import numpy as np
import jax
import jax.numpy as jnp

from jax_md import space, smap, partition, quantity, util
from typing import Callable, Tuple, TextIO, Dict, Any, Optional
from functools import wraps, partial

from jax.scipy.special import erfc
from jax_md import interpolate
import interpax
from dataclasses import dataclass, field

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


# other constants
thmin = -1.0
thmax = -0.995


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "R",
        "species",
        "ilist",
        "jlist",
        "species_i",
        "species_j",
        "mask_ij",
        "klist",
        "kmask",
        "kspecies",
        "llist",
        "lmask",
        "lspecies",
        "nilist",
        "nimask",
    ],
    meta_fields=[],
)
@dataclass
class Interactions_Arrays:
    R: Array
    species: Array
    ilist: Array
    jlist: Array
    species_i: Array
    species_j: Array
    mask_ij: Array
    klist: Array
    kmask: Array
    kspecies: Array
    llist: Array
    lmask: Array
    lspecies: Array
    nilist: Array
    nimask: Array

@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "R",
        "species",
        "ilist",
        "jlist",
        "species_i",
        "species_j",
        "mask_ij",
        "klist",
        "kmask",
        "kspecies",
        "llist",
        "lmask",
        "lspecies",
        "nilist",
        "nimask",
    ],
    meta_fields=[
        "nlist_dense",
        "nlist_sparse",
        "max_natoms",
        "max_nbonds",
        "max_nneighbors",
    ],
)
@dataclass
class Interactions:
    R: Array
    species: Array
    ilist: Array
    jlist: Array
    species_i: Array
    species_j: Array
    mask_ij: Array
    klist: Array
    kmask: Array
    kspecies: Array
    llist: Array
    lmask: Array
    lspecies: Array
    nilist: Array
    nimask: Array
    max_natoms: int
    max_nbonds: int
    max_nneighbors: int
    padded: bool = field(default=False, init=False)
    nlist_dense: Optional[NeighborList] = None
    nlist_sparse: Optional[NeighborList] = None
    def dump_arrays(self):
        """Dumps the interaction arrays to a dictionary."""
        return (self.R,
                self.species,
                self.ilist,
                self.jlist,
                self.species_i,
                self.species_j,
                self.mask_ij,
                self.klist,
                self.kmask,
                self.kspecies,
                self.llist,
                self.lmask,
                self.lspecies,
                self.nilist,
                self.nimask,
                )

    def pad(self, max_natoms: int, max_nbonds: int, max_nneighbors: int):
        """Pads the interaction arrays to the maximum sizes."""
        self.padded = True
        self.max_natoms = np.maximum(self.max_natoms, max_natoms)
        self.max_nbonds = np.maximum(self.max_nbonds, max_nbonds)
        self.max_nneighbors = np.maximum(self.max_nneighbors, max_nneighbors)
        extend_atoms = max_natoms - self.R.shape[0]
        self.R = jnp.pad(
            self.R,
            ((0, extend_atoms), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        
        extend_atoms = jnp.maximum(0, max_natoms - self.species.shape[0])
        self.species = jnp.pad(
            self.species, (0, extend_atoms), mode="constant", constant_values=-1
        )

        extend_bonds = jnp.maximum(0, max_nbonds - self.species_i.shape[0])
        bond_tuple = (0, extend_bonds)
        self.species_i = jnp.pad(
            self.species_i, bond_tuple, mode="constant", constant_values=-1
        )
        self.species_j = jnp.pad(
            self.species_j, bond_tuple, mode="constant", constant_values=-1
        )
        self.ilist = jnp.pad(
            self.ilist, bond_tuple, mode="constant", constant_values=-1
        )
        self.jlist = jnp.pad(
            self.jlist, bond_tuple, mode="constant", constant_values=-1
        )
        self.mask_ij = jnp.pad(
            self.mask_ij, bond_tuple, mode="constant", constant_values=False
        )

        extend_neighbors = jnp.maximum(0, max_nneighbors - self.klist.shape[1])
        neighbor_tuple = (
            (0, extend_bonds),
            (0, extend_neighbors),
        )
        self.klist = jnp.pad(
            self.klist,
            neighbor_tuple,
            mode="constant",
            constant_values=-1,
        )
        self.kmask = jnp.pad(
            self.kmask,
            neighbor_tuple,
            mode="constant",
            constant_values=False,
        )
        self.kspecies = jnp.pad(
            self.kspecies, neighbor_tuple, mode="constant", constant_values=-1
        )
        self.llist = jnp.pad(
            self.llist,
            neighbor_tuple,
            mode="constant",
            constant_values=-1,
        )
        self.lmask = jnp.pad(
            self.lmask,
            neighbor_tuple,
            mode="constant",
            constant_values=False,
        )
        self.lspecies = jnp.pad(
            self.lspecies, neighbor_tuple, mode="constant", constant_values=-1
        )

        extend_atoms = jnp.maximum(0, max_natoms - self.nilist.shape[0])
        self.nilist = jnp.pad(
            self.nilist,
            ((0, extend_atoms), (0, extend_neighbors)),
            mode="constant",
            constant_values=-1,
        )
        extend_atoms = jnp.maximum(0, max_natoms - self.nimask.shape[0])
        self.nimask = jnp.pad(
            self.nimask,
            ((0, extend_atoms), (0, extend_neighbors)),
            mode="constant",
            constant_values=False,
        )

        return self.R, self.species

    def reallocate_nlists(self, cutoff, box):
        displacement, shift = space.periodic_general(
            box, fractional_coordinates=False
        )
        R = self.R
        neighbor_function_sparse = partition.neighbor_list(
            displacement, box, cutoff, format=partition.Sparse
        )
        self.nlist_sparse = neighbor_function_sparse.allocate(R)
        neighbor_function_dense = partition.neighbor_list(
            displacement, box, cutoff, format=partition.Dense
        )
        self.nlist_dense = neighbor_function_dense.allocate(R)
        self.update(R)


    def update(self, R):
        if self.nlist_sparse is None or self.nlist_dense is None:
            interactions = Interactions(
                R=R,
                species=self.species,
                ilist=self.ilist,
                jlist=self.jlist,
                species_i=self.species_i,
                species_j=self.species_j,
                mask_ij=self.mask_ij,
                klist=self.klist,
                kmask=self.kmask,
                kspecies=self.kspecies,
                llist=self.llist,
                lmask=self.lmask,
                lspecies=self.lspecies,
                nilist=self.nilist,
                nimask=self.nimask,
                max_natoms=self.max_natoms,
                max_nbonds=self.max_nbonds,
                max_nneighbors=self.max_nneighbors,
            )
            return interactions
        self.nlist_sparse.update(R)
        self.nlist_dense.update(R)
        nlist_sparse = self.nlist_sparse
        nlist_dense = self.nlist_dense
        species = self.species
        mask = self.species >= 0

        natoms = R.shape[0]
        species_i = species[nlist_sparse.idx[0]]
        species_j = species[nlist_sparse.idx[1]]
        mask_ij = (
            (nlist_sparse.idx[0] < natoms)
            & (mask[nlist_sparse.idx[0]])
            & (mask[nlist_sparse.idx[1]])
        )

        klist = nlist_dense.idx[nlist_sparse.idx[0]]
        imask_2d = jnp.broadcast_to(
            (nlist_sparse.idx[0] < natoms)[:, None], klist.shape
        )
        klist += jnp.where(imask_2d, 0, natoms - klist)
        kmask = (klist < natoms) & mask[klist]
        kspecies = jnp.where(kmask, species[klist], -1)

        llist = nlist_dense.idx[nlist_sparse.idx[1]]
        jmask_2d = jnp.broadcast_to(
            (nlist_sparse.idx[1] < natoms)[:, None], llist.shape
        )
        llist += jnp.where(jmask_2d, 0, natoms - llist)
        lmask = (llist < natoms) & mask[klist]
        lspecies = jnp.where(lmask, species[llist], -1)
        interactions = Interactions(
                R=R,
                species=species,
                nlist_dense=nlist_dense,
                nlist_sparse=nlist_sparse,
                ilist=nlist_sparse.idx[0],
                jlist=nlist_sparse.idx[1],
                species_i=species_i,
                species_j=species_j,
                mask_ij=mask_ij,
                klist=klist,
                kmask=kmask,
                kspecies=kspecies,
                llist=llist,
                lmask=lmask,
                lspecies=lspecies,
                nilist=nlist_dense.idx,
                nimask=nlist_dense.idx < natoms,
                max_natoms=self.max_natoms,
                max_nbonds=self.max_nbonds,
                max_nneighbors=self.max_nneighbors,
            )
        # interactions.pad(self.max_natoms, self.max_nbonds, self.max_nneighbors)
        return interactions


def allocate_interactions(R, species, displacement, cutoff, box,
        max_natoms=None,
        max_nbonds=None,
        max_nneighbors=None,
        keep_nlists: bool = True,
    ) -> Interactions:
    neighbor_function_sparse = partition.neighbor_list(
        displacement, box, cutoff, format=partition.Sparse
    )
    nlist_sparse = neighbor_function_sparse.allocate(R)
    neighbor_function_dense = partition.neighbor_list(
        displacement, box, cutoff, format=partition.Dense
    )
    nlist_dense = neighbor_function_dense.allocate(R)

    mask = species >= 0
    natoms = R.shape[0]
    species_i = species[nlist_sparse.idx[0]]
    species_j = species[nlist_sparse.idx[1]]
    mask_ij = (
        (nlist_sparse.idx[0] < natoms)
        & (mask[nlist_sparse.idx[0]])
        & (mask[nlist_sparse.idx[1]])
    )

    klist = nlist_dense.idx[nlist_sparse.idx[0], :]
    imask_2d = jnp.broadcast_to((nlist_sparse.idx[0] < natoms)[:, None], klist.shape)
    klist += jnp.where(imask_2d, 0, natoms - klist)
    kmask = (klist < natoms) & mask[klist]
    kspecies = jnp.where(kmask, species[klist], -1)

    llist = nlist_dense.idx[nlist_sparse.idx[1], :]
    jmask_2d = jnp.broadcast_to((nlist_sparse.idx[1] < natoms)[:, None], llist.shape)
    llist += jnp.where(jmask_2d, 0, natoms - llist)
    lmask = (llist < natoms) & mask[klist]
    lspecies = jnp.where(lmask, species[llist], -1)

    if max_natoms is None:
        max_natoms = R.shape[0]
    if max_nbonds is None:
        max_nbonds = nlist_sparse.idx.shape[1]
    if max_nneighbors is None:
        max_nneighbors = nlist_dense.idx.shape[1]
    interactions = Interactions(
        R=R,
        species=species,
        nlist_dense=nlist_dense if keep_nlists else None,
        nlist_sparse=nlist_sparse if keep_nlists else None,
        ilist=nlist_sparse.idx[0],
        jlist=nlist_sparse.idx[1],
        species_i=species_i,
        species_j=species_j,
        mask_ij=mask_ij,
        klist=klist,
        kmask=kmask,
        kspecies=kspecies,
        llist=llist,
        lmask=lmask,
        lspecies=lspecies,
        nilist=nlist_dense.idx,
        nimask=nlist_dense.idx < natoms,
        max_natoms=max_natoms,
        max_nbonds=max_nbonds,
        max_nneighbors=max_nneighbors,
    )
    interactions.pad(max_natoms, max_nbonds, max_nneighbors)
    return interactions


def _slow_rebo2_cutoff(dr, Dmin, Dmax) -> Array:
    """The cut-off function of the Tersoff potential.
    Args:
      R: A Parameter that is the average of inner and outer cutoff radii
      D: A Parameter that is the half of the difference
         between inner and outer cutoff radii

    Returns:
      cut-off values
    """
    outer = jnp.where(
        dr < Dmax, 0.5 * (1 + jnp.cos(jnp.pi * (dr - Dmin) / (Dmax - Dmin))), 0
    )
    inner = jnp.where(dr < Dmin, 1, outer)
    return inner


def _rebo2_cutoff(dr, Dmin, Dmax) -> Array:
    """The cut-off function of the Tersoff potential.
    Args:
      R: A Parameter that is the average of inner and outer cutoff radii
      D: A Parameter that is the half of the difference
         between inner and outer cutoff radii

    Returns:
      cut-off values
    """
    t = (dr - Dmin) / (Dmax - Dmin)
    return jnp.where(t < 0, 1, jnp.where(t < 1, 1 - 10*t**3 + 15*t**4 - 6*t**5, 0))


def normalize(x: jnp.ndarray, axis: int, eps: float = 1e-12) -> jnp.ndarray:
    norm = jnp.sum(x * x, axis=axis, keepdims=True)
    # if the norm is close to zero
    # normalization should not be applied
    # the gradient should be the same as identify map
    norm = jnp.where(
        (norm < eps) | (norm == jnp.inf),
        1.0,
        norm,
    )
    norm = jnp.sqrt(norm)
    return x / norm


def _b_ij_dh(
    R_ij: Array,
    R_ik: Array,
    R_jl: Array,
    fc_ik: Array,
    fc_jl: Array,
    Ni_t: Array,
    Nj_t: Array,
    Nij_conj: Array,
    T_lists,
    nspecies: int,
    displacement: DisplacementOrMetricFn,
    interactions: Interactions,
) -> Array:

    # Compute cross products
    c1 = jax.vmap(jnp.cross)(-R_ij, R_ik)
    c2 = jax.vmap(jnp.cross)(R_ij, R_jl)


    c1 = normalize(c1, axis=-1)
    c2 = normalize(c2, axis=-1)

    costheta = np.sum(c1[:, None, :, :] * c2[:, :, None, :], axis=-1)

    costmp1 = jax.vmap(jax.vmap(jnp.dot))(-R_ij, R_ik) / (jnp.linalg.norm(R_ij, axis=-1) * jnp.linalg.norm(R_ik, axis=-1) + 1e-8)
    costmp2 = jax.vmap(jax.vmap(jnp.dot))(R_ij, R_jl) / (jnp.linalg.norm(R_ij, axis=-1) * jnp.linalg.norm(R_jl, axis=-1) + 1e-8)
    tsp1 = _rebo2_cutoff(-jnp.abs(costmp1), thmin, thmax)
    tsp2 = _rebo2_cutoff(-jnp.abs(costmp2), thmin, thmax)
    tsp = (1.0 - tsp1[:, None, :]) * (1.0 - tsp2[:, :, None])
    # print(f"costheta: {costheta}")
    # Vectorize over all valid index sets
    # costheta = jax.vmap(jax.vmap(jax.vmap(jnp.dot)))(c1, c2)
    mask = (
        interactions.mask_ij[:, None, None]
        & interactions.kmask[:, None, :]
        & interactions.lmask[:, :, None]
    )
    mask &= (interactions.jlist[:, None] != interactions.klist)[
        :, None, :
    ]  # Ensure j != k
    mask &= (interactions.ilist[:, None] != interactions.llist)[
        :, :, None
    ]  # Ensure i != l
    mask &= (
        interactions.klist[:, None, :] != interactions.llist[:, :, None]
    )  # Ensure k != l
    fc_ik_fc_jl = fc_ik[:, None, :] * fc_jl[:, :, None]
    theta_factor = 1 - costheta**2
    weight = jnp.where(mask, theta_factor * fc_ik_fc_jl * tsp, 0)
    # jax.debug.print(f"weight: {weight.shape}")
    b = jnp.sum(weight, axis=(-1, -2))  # shape: nbonds

    T = np.zeros(Ni_t.shape, dtype=f64)
    #jax.debug.print(
    #     "Tijargs:\nNi_T: {Ni_t}\nNj_T: {Nj_t}\nNij_conj: {Nij_conj}",
    #     Ni_t=Ni_t,
    #     Nj_t=Nj_t,
    #     Nij_conj=Nij_conj,
    # )

    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            mask = (interactions.species_i == i) & (interactions.species_j == j)
            T += jnp.where(mask, T_lists[i][j](jnp.clip(Ni_t, 0, 8), jnp.clip(Nj_t, 0, 8), jnp.clip(Nij_conj, 0, 8)), 0)
    #jax.debug.print("Tij: {T}", T=T)
    #jax.debug.print("TEtmp: {b}", b=b)
    #jax.debug.print("TE: {bt}", bt=b * T)
    return b * T


def _PI_rc(Ni_t, Nj_t, Nij_conj, F_lists, nspecies, interactions) -> Array:
    F = np.zeros(Ni_t.shape, dtype=f64)
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            mask = (interactions.species_i == i) & (interactions.species_j == j)
            F += jnp.where(mask, F_lists[i][j](jnp.clip(Ni_t, 0, 8), jnp.clip(Nj_t, 0, 8), jnp.clip(Nij_conj, 0, 8)), 0)
    return F


def _Q(Ni_t) -> Array:
    # outer = jnp.where(Ni_t < 3.7, 0.5 * (1 + jnp.cos(2 * jnp.pi * (Ni_t - 3.2))), 0)
    # inner = jnp.where(Ni_t < 3.2, 1, outer)
    # return inner
    return _rebo2_cutoff(Ni_t, 3.2, 3.7)


def print3d_array(arr: Array):
    """Prints a 3D array in a readable format."""
    for i in range(arr.shape[0]):
        print(f"Slice {i}:")
        for j in range(arr.shape[1]):
            print(f"  Row {j}: {arr[i, j]}")
        print("\n")
    print("End of 3D array\n")


costheta_ijk = lambda rijk: jnp.dot(rijk[0, :], rijk[1, :])
costheta_ijk_vmapped = jax.jit(jax.vmap(jax.vmap(costheta_ijk)))


def _rebo2_bij_sigma(
    R_ij: Array,
    r_ij: Array,
    fc_ij: Array,
    R_ik: Array,
    r_ik: Array,
    fc_ik: Array,
    Ni_t: Array,
    G_lists,
    gamma_lists,
    P_lists,
    nspecies,
    params,
    interactions,
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

    costheta = jax.vmap(jax.vmap(jnp.dot))(
        normalize(R_ij, axis=-1), normalize(R_ik, axis=-1)
    )
    #jax.debug.print("costheta: {costheta}", costheta=costheta)
    # Apply G and gamma functions for all (i, j, k)
    G = jnp.zeros_like(costheta)
    gamma = jnp.zeros_like(costheta)
    species_2d = jnp.broadcast_to(
        interactions.species_i[:, None], interactions.kspecies.shape
    )
    for i in [0, 1, 2]:
        mask = species_2d == i
        G += jnp.where(mask, G_lists[i](jnp.clip(costheta, -0.8, 0.8)), 0)
        gamma += jnp.where(mask, gamma_lists[i](jnp.clip(costheta, -1, 1)), 0)

    Qval = _Q(Ni_t)
    Qval = jnp.broadcast_to(Qval[:, None], G.shape)
    #jax.debug.print("Qval: {Qval}", Qval=Qval)
    #jax.debug.print("G: {G}", G=G)
    #jax.debug.print("gamma: {gamma}", gamma=gamma)
    g = G + Qval * (gamma - G)

    # exp_term[i, j, k] = jnp.exp(lambda_ijk[i] * ((rho[ktype][itype]-rik)-(rho[jtype][itype]-rij)));
    lambda_ijk = params["lambda_ijk"][interactions.species_i]
    lambda_ijk_2d = jnp.broadcast_to(lambda_ijk[:, None], costheta.shape)
    rho_ij = jnp.broadcast_to(
        params["rho"][interactions.species_i, interactions.species_j][:, None],
        interactions.kspecies.shape,
    )
    rho_ik = params["rho"][
        jnp.broadcast_to(interactions.species_i[:, None], interactions.kspecies.shape),
        interactions.kspecies,
    ]
    r_ij_2d = jnp.broadcast_to(r_ij[:, None], r_ik.shape)
    exp_term = jnp.exp(lambda_ijk_2d * ((r_ij_2d - rho_ij) - (r_ik - rho_ik)))

    mask_ijk = interactions.kmask[
        :,
        :,
    ]
    mask_ijk &= interactions.mask_ij[:, None]
    mask_ijk &= interactions.jlist[:, None] != interactions.klist[:, :]  # j != k
    mask_ijk = jnp.broadcast_to(mask_ijk, exp_term.shape)

    # sum over k
    gsum = jnp.sum(jnp.where(mask_ijk, fc_ik * g * exp_term, 0), axis=-1)
    #jax.debug.print("exp_term: {exp_term}", exp_term=exp_term)
    #jax.debug.print("g: {g}", g=g)
    NC = jnp.clip(
        _Ni_species(fc_ij, fc_ik, interactions.species_j, interactions.kspecies, 0),
        0,
        9,
    )  # Carbon
    NH = jnp.clip(
        _Ni_species(fc_ij, fc_ik, interactions.species_j, interactions.kspecies, 1),
        0,
        9,
    )  # Hydrogen
    NO = jnp.clip(
        _Ni_species(fc_ij, fc_ik, interactions.species_j, interactions.kspecies, 2),
        0,
        9,
    )  # Oxygen
    # print(jnp.max(NC), jnp.max(NH), jnp.max(NO))
    # #jax.debug.print("Max NC: {x}", x=jnp.max(NC))
    # #jax.debug.print("Max NH: {x}", x=jnp.max(NH))
    # #jax.debug.print("Max NO: {x}", x=jnp.max(NO))
    P = jnp.zeros(r_ij.shape, dtype=r_ij.dtype)
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            mask = (interactions.species_i == i) & (interactions.species_j == j)
            P += jnp.where(mask, P_lists[i][j](jnp.clip(NC, 0, 8), jnp.clip(NH, 0, 8), jnp.clip(NO, 0, 8)), 0)
    #jax.debug.print("gsum: {gsum}", gsum=gsum)
    #jax.debug.print("P: {P}", P=P)
    bij_sigma = jnp.where(interactions.mask_ij, 1 / jnp.sqrt(jnp.maximum(1 + gsum + P, 0.01)), 0)
    return bij_sigma


def _Ni_t(fc_ij, fc_ik) -> Array:
    Ni_t = jnp.sum(fc_ik, axis=1)
    Ni_t -= fc_ij
    return Ni_t


def _Ni_species(
    fc_ij: Array, fc_ik: Array, jspecies: Array, kspecies: Array, target_species: int
) -> Array:
    mask = kspecies == target_species
    Ni_species = jnp.sum(jnp.where(mask, fc_ik, 0), axis=1)
    Ni_species -= jnp.where(jspecies == target_species, fc_ij, 0)

    return Ni_species


def _NTT(
    displacement: DisplacementOrMetricFn,
    params: Dict[str, Array],
    interactions: Interactions,
) -> Array:
    """Calculate the NTT term for REBO II potential."""
    R = interactions.R
    R_n = interactions.R[interactions.nilist]
    R = jnp.broadcast_to(R[:, None, :], R_n.shape)
    dR_n = jax.vmap(space.map_bond(displacement))(R, R_n)
    r_n = space.distance(dR_n)
    species_n = interactions.species[interactions.nilist]  # Get species of neighbors
    species_i = jnp.broadcast_to(interactions.species[:, None], species_n.shape)
    fc_n = _rebo2_cutoff(
        r_n,
        params["Dmin"][species_i, species_n],
        params["Dmax"][species_i, species_n],
    )
    fc_n = jnp.where(interactions.nimask, fc_n, 0)
    NTT = jnp.sum(fc_n, axis=1)  # Sum over neighbors for each atom
    return NTT


# Splining Function for conjugation F(x_ik)
def _Fx(x_ik):
    # return jnp.where(
    #     x_ik < 2, 1, jnp.where(x_ik > 3, 0, (1 + jnp.cos(jnp.pi * (x_ik - 2))) / 2)
    # )
    return _rebo2_cutoff(x_ik, 2.0, 3.0)


# Conjugation Term for Carbon
def _N_conj(NTT, fc_ik, fc_jl, interactions):
    # Carbon species
    mask1 = interactions.kspecies == 0
    # print(f"OKOKOK1 {interactions.kspecies == 0}")
    # mask1 &= interactions.species_i[:, None] == 0
    # print(f"OKOKOK2 {interactions.species_i[:, None] == 0}")
    mask1 &= interactions.jlist[:, None] != interactions.klist
    # print(f"OKOKOK3 {interactions.jlist[:, None] != interactions.klist}")
    mask1 &= interactions.kmask
    # print(f"OKOKOK4 {interactions.kmask}")
    X = NTT[interactions.klist] - fc_ik
    #jax.debug.print("Xk: {X}", X=X)
    term0 = jnp.where(mask1, fc_ik * _Fx(X), 0)
    term1 = jnp.sum(term0, axis=1)

    mask2 = interactions.lspecies == 0
    # mask2 &= interactions.species_j[:, None] == 0
    mask2 &= interactions.ilist[:, None] != interactions.llist
    mask2 &= interactions.lmask
    X = NTT[interactions.llist] - fc_jl
    #jax.debug.print("Xl: {X}", X=X)
    term0 = jnp.where(mask2, fc_jl * _Fx(X), 0)
    term2 = jnp.sum(term0, axis=1)
    Nconj = 1 + term1**2 + term2**2
    return Nconj


def _rebo2_bij(
    displacement: DisplacementOrMetricFn,
    params: Dict[str, Array],
    interactions: Interactions,
    F_lists,
    T_lists,
    P_lists,
    G_lists,
    gamma_lists,
    nspecies: int,
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
    Rinf = jnp.array([1.0, 1.0, 1.0])
    fc_ij, r_ij = _fc_dr(displacement, params, interactions)
    fc_ij = jnp.where(interactions.mask_ij, fc_ij, 0)
    # calculate Ni_t, Nj_t
    R_i = interactions.R[interactions.ilist]
    species_i = interactions.species_i
    R_k = interactions.R[interactions.klist]
    species_k = interactions.kspecies
    R_i = jnp.broadcast_to(R_i[:, None, :], R_k.shape)
    species_i = jnp.broadcast_to(species_i[:, None], interactions.kspecies.shape)
    R_ik = jax.vmap(space.map_bond(displacement))(R_i, R_k)
    R_ik = jnp.where(interactions.kmask[:, :, None] & interactions.mask_ij[:, None, None], R_ik, Rinf)
    r_ik = space.distance(R_ik)
    r_ik = jnp.where(interactions.kmask & interactions.mask_ij[:, None], r_ik, 1e6)
    fc_ik = _rebo2_cutoff(
        r_ik,
        params["Dmin"][species_i, species_k],
        params["Dmax"][species_i, species_k],
    )
    fc_ik = jnp.where(interactions.kmask & interactions.mask_ij[:, None], fc_ik, 0)

    Ni_t = _Ni_t(fc_ij, fc_ik)

    R_j = interactions.R[interactions.jlist]
    R_j = jnp.where(interactions.mask_ij[:, None], R_j, Rinf)
    species_j = interactions.species_j
    R_l = interactions.R[interactions.llist]
    R_l = jnp.where(interactions.lmask[:, :, None] & interactions.mask_ij[:, None, None], R_l, Rinf)
    species_l = interactions.lspecies
    R_j = jnp.broadcast_to(R_j[:, None, :], R_l.shape)
    species_j = jnp.broadcast_to(species_j[:, None], interactions.lspecies.shape)
    R_jl = jax.vmap(space.map_bond(displacement))(R_j, R_l)
    R_jl = jnp.where(interactions.lmask[:, :, None] & interactions.mask_ij[:, None, None], R_jl, Rinf)
    r_jl = space.distance(R_jl)
    r_jl = jnp.where(interactions.lmask & interactions.mask_ij[:, None], r_jl, 1e6)
    fc_jl = _rebo2_cutoff(
        r_jl,
        params["Dmin"][species_j, species_l],
        params["Dmax"][species_j, species_l],
    )
    fc_jl = jnp.where(interactions.lmask & interactions.mask_ij[:, None], fc_jl, 0)

    Nj_t = _Ni_t(fc_ij, fc_jl)

    # calculate Nconj
    NTT = _NTT(displacement, params, interactions)
    Nij_conj = _N_conj(NTT, fc_ik, fc_jl, interactions)

    # Calculate PI_rc via tricubic spline
    bij_pi_rc = _PI_rc(Ni_t, Nj_t, Nij_conj, F_lists, nspecies, interactions)
    #jax.debug.print("Ni_t * Nj_t * Nij_conj: {val}", val=jnp.sum(Ni_t * Nj_t * Nij_conj * (fc_ij > 0)))
    # #jax.debug.print("Ni_t:\n{Ni_t}", Ni_t=Ni_t)
    #jax.debug.print("Ni_t_ntot: {val}", val=jnp.max(Ni_t*(fc_ij>0)))
    # #jax.debug.print("Nj_t:\n{Nj_t}", Nj_t=Nj_t)
    #jax.debug.print("Nj_t_ntot: {val}", val=jnp.max(Nj_t*(fc_ij>0)))
    # #jax.debug.print("Nij_conj:\n{Nij_conj}", Nij_conj=Nij_conj)
    #jax.debug.print("Nij_conj_ntot: {val}", val=jnp.max(Nij_conj*(fc_ij>0)))
    

    # Calculate dihedral terms
    R_ij = jax.vmap(space.map_bond(displacement))(R_i, R_j)
    bij_pi_dh = _b_ij_dh(
        R_ij,
        R_ik,
        R_jl,
        fc_ik,
        fc_jl,
        Ni_t,
        Nj_t,
        Nij_conj,
        T_lists,
        nspecies,
        displacement,
        interactions,
    )
    bij_sigma = _rebo2_bij_sigma(
        R_ij,
        r_ij,
        fc_ij,
        R_ik,
        r_ik,
        fc_ik,
        Ni_t,
        G_lists,
        gamma_lists,
        P_lists,
        nspecies,
        params,
        interactions,
    )
    # #jax.debug.print("bij_pi_rc: {bij_pi_rc}", bij_pi_rc=bij_pi_rc)
    # #jax.debug.print("bij_pi_rc&fc: {val}", val=bij_pi_rc * (fc_ij > 0))
    #jax.debug.print("bij_pi_rc_ntot: {val}", val=jnp.sum(bij_pi_rc * (fc_ij > 0)))
    # #jax.debug.print("bij_pi_dh: {bij_pi_dh}", bij_pi_dh=bij_pi_dh)
    # #jax.debug.print("bij_pi_dh&fc: {val}", val=bij_pi_dh * (fc_ij > 0))
    #jax.debug.print("bij_pi_dh_ntot: {val}", val=jnp.sum(bij_pi_dh * (fc_ij > 0)))
    # #jax.debug.print("bij_sigma: {bij_sigma}", bij_sigma=bij_sigma)
    # #jax.debug.print("bij_sigma&fc: {val}", val=bij_sigma * (fc_ij > 0))
    #jax.debug.print("bij_sigma_ntot: {val}", val=jnp.sum(bij_sigma * (fc_ij > 0)))
    bij = bij_pi_rc + bij_pi_dh + bij_sigma
    # #jax.debug.print("bij: {bij}", bij=bij)
    # #jax.debug.print("bij&fc: {val}", val=bij*(fc_ij>0))
    return bij


def _fc_dr(displacement, params, interactions) -> Array:
    dr = space.distance(
        space.map_bond(displacement)(
            interactions.R[interactions.ilist], interactions.R[interactions.jlist]
        )
    )
    dr = jnp.where(interactions.mask_ij, dr, 1e6)
    fc = _rebo2_cutoff(
        dr,
        params["Dmin"][interactions.species_i, interactions.species_j],
        params["Dmax"][interactions.species_i, interactions.species_j],
    )
    fc = jnp.where(interactions.mask_ij, fc, 0)
    return fc, dr


def _rebo2_attractive(
    displacement,
    params,
    F_lists,
    T_lists,
    P_lists,
    G_lists,
    gamma_lists,
    nspecies: int,
    interactions,
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

    fc, dr = _fc_dr(displacement, params, interactions)

    B1 = params["B1"][interactions.species_i, interactions.species_j]
    B2 = params["B2"][interactions.species_i, interactions.species_j]
    B3 = params["B3"][interactions.species_i, interactions.species_j]
    beta_1 = params["beta_1"][interactions.species_i, interactions.species_j]
    beta_2 = params["beta_2"][interactions.species_i, interactions.species_j]
    beta_3 = params["beta_3"][interactions.species_i, interactions.species_j]
    #jax.debug.print("dr: {dr}", dr=dr*(fc>0))
    #jax.debug.print("B1: {B1}", B1=B1*(fc>0))
    #jax.debug.print("B2: {B2}", B2=B2*(fc>0))
    #jax.debug.print("B3: {B3}", B3=B3*(fc>0))
    #jax.debug.print("beta_1: {beta_1}", beta_1=beta_1*(fc>0))
    #jax.debug.print("beta_2: {beta_2}", beta_2=beta_2*(fc>0))
    #jax.debug.print("beta_3: {beta_3}", beta_3=beta_3*(fc>0))
    fA = (
        B1 * jnp.exp(-beta_1 * dr)
        + B2 * jnp.exp(-beta_2 * dr)
        + B3 * jnp.exp(-beta_3 * dr)
    )
    #jax.debug.print("fA: {val}", val=fA * fc)
    bij = _rebo2_bij(
        displacement,
        params,
        interactions,
        F_lists,
        T_lists,
        P_lists,
        G_lists,
        gamma_lists,
        nspecies,
    )
    #jax.debug.print("bij: {val}", val=bij)
    #jax.debug.print("bij ntot: {val}", val=jnp.sum(bij*(fc>0)))
    #jax.debug.print("VAtot: {val}", val=jnp.sum(-fc * fA))
    # print(interactions.ilist)
    # print(interactions.jlist)
    return jnp.where(interactions.mask_ij, -fc * bij * fA, 0)


def _rebo2_repulsive(displacement, params, interactions) -> Array:
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
    fc, dr = _fc_dr(displacement, params, interactions)

    A = params["A"][interactions.species_i, interactions.species_j]
    Q = params["Q"][interactions.species_i, interactions.species_j]
    alpha = params["alpha"][interactions.species_i, interactions.species_j]
    fR = A * (1 + Q / dr) * jnp.exp(-alpha * dr)
    #jax.debug.print("dr: {dr}", dr=dr)
    #jax.debug.print("fR: {val}", val=fc * fR)
    return jnp.where(interactions.mask_ij, fc * fR, 0)


def _rebo2_make_3d_spline(knots_i, knots_j, knots_k, f, fx, fy, fz):
    zero_fx = fx * 0
    f = interpax.Interpolator3D(
        knots_i, knots_j, knots_k, f, fx=fx, fy=fy, fz=fz, fxy=zero_fx, fxz=zero_fx, fyz=zero_fx, fxyz=zero_fx, method="cubic", extrap=True
    )
    f = jax.vmap(f)
    return f


def _rebo2_make_1d_spline(knots_i, f, fx):
    f = interpax.Interpolator1D(knots_i, f, fx=fx, method="cubic", extrap=True)
    f = jax.vmap(jax.vmap(f))
    return f


def rebo2(
    displacement: DisplacementFn,
    box: Array,
    params: Array,
    species: Array,
    nspecies: int = 3,
    cutoff: Optional[float] = None,
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
    if cutoff is None:
        cutoff = jnp.max(params["Dmax"])

    interaction_fn = lambda R: allocate_interactions(
        R, species, displacement, cutoff, box
    )

    # set up splines
    F_lists = []
    T_lists = []
    P_lists = []
    G_lists = []
    gamma_lists = []
    G_knots = np.linspace(-1, 1, 25)
    for i in [0, 1, 2]:
        # Initialize empty lists for each species
        F_lists.append([])
        T_lists.append([])
        P_lists.append([])

        G_lists.append(
            _rebo2_make_1d_spline(
                G_knots, jnp.array(params["G"][i]), jnp.array(params["dG"][i])
            )
        )
        gamma_lists.append(
            _rebo2_make_1d_spline(
                G_knots,
                jnp.array(params["gamma"][i]),
                jnp.array(params["dgamma"][i]),
            )
        )
        for j in [0, 1, 2]:
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

    repulsive_fn = partial(
        _rebo2_repulsive,
        displacement,
        params,
    )
    attractive_fn = partial(
        _rebo2_attractive,
        displacement,
        params,
        F_lists,
        T_lists,
        P_lists,
        G_lists,
        gamma_lists,
        nspecies,
    )

    def energy_fn(R, interactions):
        interactions = interactions.update(R)

        # create interactions, pad to 25
        repulsive = repulsive_fn(interactions)
        attractive = attractive_fn(interactions)
        repulsive_term = util.high_precision_sum(repulsive)
        attractive_term = util.high_precision_sum(attractive)
        return 0.5 * (repulsive_term + attractive_term)

    return interaction_fn, energy_fn


@partial(jax.jit, static_argnames=['nspecies'])
def rebo2_trainable(
    params: Array,
    box: Array,
    R: Array,
    species: Array,
    ilist: Array,
    jlist: Array,
    species_i: Array,
    species_j: Array,
    mask_ij: Array,
    klist: Array,
    kmask: Array,
    kspecies: Array,
    llist: Array,
    lmask: Array,
    lspecies: Array,
    nilist: Array,
    nimask: Array,
    nspecies: int = 3,
) -> Callable[[Array], Array]:
    """Computes REBO II potential.
    Args:
      displacement: The displacement function for the space.
      params: A dictionary of parameters for the potential.
      species: An array of species.

    Returns:
      A function that computes the total energy.

    [2] D. Brenner et al "A second-generation reactive empirical bond order
    (REBO) potential energy expression for hydrocarbons" J. Phys.: Condens. Matter 14 (2002): 783.
    """
    interactions = Interactions_Arrays(
        R=R,
        species=species,
        ilist=ilist,
        jlist=jlist,
        species_i=species_i,
        species_j=species_j,
        mask_ij=mask_ij,
        klist=klist,
        kmask=kmask,
        kspecies=kspecies,
        llist=llist,
        lmask=lmask,
        lspecies=lspecies,
        nilist=nilist,
        nimask=nimask,
    )
    displacement, shift = space.periodic_general(box, fractional_coordinates=False)
    # set up splines
    F_lists = []
    T_lists = []
    P_lists = []
    G_lists = []
    gamma_lists = []
    G_knots = np.linspace(-1, 1, 25)
    for i in [0, 1, 2]:
        # Initialize empty lists for each species
        F_lists.append([])
        T_lists.append([])
        P_lists.append([])

        G_lists.append(
            _rebo2_make_1d_spline(
                G_knots, jnp.array(params["G"][i]), jnp.array(params["dG"][i])
            )
        )
        gamma_lists.append(
            _rebo2_make_1d_spline(
                G_knots,
                jnp.array(params["gamma"][i]),
                jnp.array(params["dgamma"][i]),
            )
        )
        for j in [0, 1, 2]:
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

    # interactions.update(R)

    # create interactions, pad to 25
    repulsive = _rebo2_repulsive(
        displacement,
        params,
        interactions,
    )
    attractive = _rebo2_attractive(
        displacement,
        params,
        F_lists,
        T_lists,
        P_lists,
        G_lists,
        gamma_lists,
        nspecies,
        interactions,
    )
    repulsive_term = util.high_precision_sum(repulsive)
    #jax.debug.print("repulsive_term: {repulsive_term}", repulsive_term=repulsive_term)
    attractive_term = util.high_precision_sum(attractive)
    #jax.debug.print("attractive_term: {attractive_term}", attractive_term=attractive_term)
    # jax.debug.print("total_energy: {total_energy}", total_energy=0.5 * (repulsive_term + attractive_term))
    return 0.5 * (repulsive_term + attractive_term)
