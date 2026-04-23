import lammps
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import pickle
import jax
import jax.numpy as jnp
import jax_md
import newrebo2_interaction_list as rebo2
from functools import partial
import warnings
import tqdm

plt.style.use(
    "~/.config/matplotlib/seaborn-v0_8-colorblindBIG.mplstyle"
)
# jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

def lmp_setup(potential="deepmd", graph=None, reax_pot=None, rebo3_pot=None):
    # lmp = lammps.lammps(cmdargs=["-log", "none"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"], name="develop")
    if potential == "nequip":
        lmp.command("newton off")
    if potential == "reaxff":
        lmp.command("units real")
        lmp.command("atom_style charge")
    else:
        lmp.command("units metal")
        lmp.command("atom_style atomic")
    lmp.command("dimension 3")
    lmp.command("boundary m m m")
    lmp.command("region box block -100 100 -100 100 -100 100")
    lmp.command("create_box 3 box")
    lmp.command("mass		* 1")
    lmp.command("thermo_style custom step pe")
    if potential == "REBO2":
        lmp.command("pair_style rebo")
        lmp.command("pair_coeff * * CH.rebo C H H")
    if potential == "REBO3":
        lmp.command("pair_style rebo3")
        lmp.command(f"pair_coeff * * {rebo3_pot} C H O")
    if potential == "zbl":
        lmp.command("pair_style zbl 5 6")
        lmp.command("pair_coeff 1 1 6 6")
        lmp.command("pair_coeff 1 2 6 1")
        lmp.command("pair_coeff 1 3 6 8")
        # lmp.command("pair_coeff 1 4 6 18")
        lmp.command("pair_coeff 2 2 1 1")
        lmp.command("pair_coeff 2 3 1 8")
        # lmp.command("pair_coeff 2 4 1 18")
        lmp.command("pair_coeff 3 3 8 8")
        # lmp.command("pair_coeff 3 4 8 18")
        # lmp.command("pair_coeff 4 4 18 18")
    if potential == "diff":
        lmp.command("pair_style hybrid/scaled -1.0 zbl 1.5 2.0 1.0 rebo")
        lmp.command("pair_coeff * * rebo CH.rebo C H")
        lmp.commands_string(
            """
        pair_coeff  1 1 zbl 6 6
        pair_coeff  1 2 zbl 6 1
        pair_coeff  2 2 zbl 1 1
        """
        )
    if potential == "deepmd":
        if graph is None:
            raise ValueError("Graph file must be provided for deepmd potential.")
        lmp.command("plugin load libdeepmd_lmp.so")
        lmp.command("plugin unload pair deepmd")
        lmp.command("plugin load libdeepmd_lmp.so")
        lmp.command(f"pair_style deepmd {graph}")
        lmp.command("pair_coeff * * C H O")
    if potential == "reaxff":
        if reax_pot is None:
            raise ValueError("ReaxFF file must be provided for reaxff potential.")
        lmp.command("pair_style reaxff NULL mincap 200 safezone 1.5 checkqeq no")
        lmp.command(f"pair_coeff * * {reax_pot} C H O")
        # lmp.command("fix reax_qeq all qeq/reaxff 1 0.0 10.0 1e-6 reaxff")
    return lmp


energy_neg_force = jax.jit(jax.value_and_grad(rebo2.rebo2_trainable, argnums=2))


def get_E_F_jax(ds, t1, t2, params):
    species = jnp.array([t1-1, t2-1])
    displacement, shift = jax_md.space.free()
    fake_box = jnp.diag(jnp.array([15.0, 15.0, 15.0]))

    energies = np.zeros(len(ds))
    forces = np.zeros(len(ds))
    R = jnp.array([[0.0, 0.0, 0.0], [ds[0], 0.0, 0.0],])
    base_interactions = rebo2.allocate_interactions(R, species, displacement, 3.0, fake_box)
    update = jax.jit(base_interactions.update)
    for i, d in tqdm.tqdm(enumerate(ds)):
        R = jnp.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0],])
        interactions = update(R)
        interactions_arrays = interactions.dump_arrays()
        
        energy, force = energy_neg_force(params, fake_box, *interactions_arrays)
        energies[i] = energy
        forces[i] = np.linalg.norm(force[0])  # JAX returns the negative force
    return energies, forces


def get_E_F(d, t1, t2, lmp):
    if lmp.get_natoms() > 0:
        lmp.command("delete_atoms group all")
    t1 = int(t1)
    t2 = int(t2)
    lmp.command(f"create_atoms {t1} single 0 0 0")
    lmp.command(f"create_atoms {t2} single {d} 0 0")
    # or read_data or move_atoms or whatever
    lmp.command("run 0")
    f0 = lmp.extract_atom("f")
    # q0 = lmp.extract_atom("q")
    # if q0 is not None:
    #     print(q0[0])
    pe = lmp.extract_compute("thermo_pe", 0, 0)
    return pe, np.array([f0[0][i] for i in range(3)])


def get_E_F_list(ds, t1, t2, lmp):
    if isinstance(lmp, dict):
        return get_E_F_jax(ds, t1, t2, lmp)
    Es = np.zeros(len(ds))
    Fs = np.zeros(len(ds))
    for i, d in enumerate(ds):
        E, F = get_E_F(d, t1, t2, lmp)
        Es[i] = E
        Fs[i] = np.linalg.norm(F)
    return Es, Fs


tmap = {1: "C", 2: "H", 3: "O", 4: "Ar"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots pairwise interactions for multiple potentials."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file for plot",
        default="pairwise.png",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        help="Models used for deepmd plot. First graph will be used to minimze.",
        default=[], # [f for f in os.listdir() if f.endswith(".pb")],
        nargs="+",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to pickle containing rebo2 parameters.",
        default=[], # [f for f in os.listdir() if f.endswith(".pb")],
        nargs="+",
    )
    parser.add_argument(
        "-x",
        "--reaxff",
        type=str,
        help="Models used for reaxff plot. First model will be used to minimze.",
        default=[f for f in os.listdir() if f.endswith(".pb")],
        nargs="+",
    )
    parser.add_argument(
        "-z",
        "--zbl",
        help="Whether to plot the ZBL potential.",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--rebo",
        help="Whether to plot the rebo2 potential.",
        action="store_true",
    )
    parser.add_argument(
        "-r3",
        "--rebo3",
        help="Path to rebo3 potential.",
        default=[], # [f for f in os.listdir() if f.endswith(".pb")],
        nargs="+",
    )
    parser.add_argument(
        "-rz",
        "--rebo-zbl",
        help="Whether to plot the ZBL potential.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--vline",
        help="location of vertical line",
        default=None,
    )
    parser.add_argument(
        "--model_type",
        help="type of deep potential (deepmd vs nequip)",
        default="deepmd",
    )

    args = parser.parse_args()
    lmp_list = []
    kwargs_list = []
    if args.zbl:
        lmp_zbl = lmp_setup(potential="zbl")
        zbl_kwargs = {"c": "k", "ls": "--", "label": "zbl"}
        lmp_list.append(lmp_zbl)
        kwargs_list.append(zbl_kwargs)
    if args.rebo:
        lmp_rebo = lmp_setup(potential="REBO2")
        rebo_kwargs = {"c": "r", "ls": "--", "label": "rebo"}
        lmp_list.append(lmp_rebo)
        kwargs_list.append(rebo_kwargs)
    if args.rebo_zbl:
        lmp_diff = lmp_setup(potential="diff")
        diff_kwargs = {"c": "b", "ls": "--", "label": "rebo-zbl"}
        lmp_list.append(lmp_diff)
        kwargs_list.append(diff_kwargs)

    models = args.models
    j = 0
    for i, m in enumerate(models):
        lmp_list.append(lmp_setup(potential=args.model_type, graph=m))
        kwargs_list.append({"c": "C" + str(i), "alpha": 0.5, "label": "deepmd", "ls": "--"})
        j += 1
    for i, m in enumerate(args.reaxff):
        lmp_list.append(lmp_setup(potential="reaxff", reax_pot=m))
        kwargs_list.append(
            {"c": "C" + str(j), "alpha": 0.5, "label": "reaxff " + m}
        )
        j += 1
    for i, m in enumerate(args.rebo3):
        lmp_rebo3 = lmp_setup(potential="REBO3", rebo3_pot=m)
        kwargs_list.append({"c": "C" + str(j), "alpha": 0.5, "label": "rebo3 " + m})
        lmp_list.append(lmp_rebo3)
        j += 1
    for i, m in enumerate(args.params):
        if not os.path.exists(m):
            print(f"File {m} does not exist, skipping.")
            continue
        params = pickle.load(open(m, "rb"))
        for k in params.keys():
            params[k] = np.array(params[k])
        lmp_list.append(params)
        kwargs_list.append(
            {"c": "C" + str(j), "alpha": 0.5, "label": "JAX rebo2 " + m}
        )
        j += 1
    # dft_data = np.loadtxt("pair_energies_forces.csv", delimiter=",", converters={0: str, 1: str})
    dft_data = pd.read_csv("pair_energies_forces.csv", header=None).values

    shiftdict = {}
    for s in dft_data:
        distance = s[2]
        s[0] = s[0].strip()
        s[1] = s[1].strip()
        if distance > 3.0:
            shiftdict[s[0] + s[1]] = s[3]

    fig, axs = plt.subplots(6, 3, figsize=(20, 20))
    print(axs.shape)
    ds = np.linspace(0.1, 3.05, 100)
    for i, t1 in enumerate([1, 2, 3]):
        for j, t2 in enumerate([1, 2, 3]):
            if t1 > t2:
                print(f"skipping... {t1}, {t2}")
                fig.delaxes(axs[2 * (t1 - 1), t2 - 1])
                fig.delaxes(axs[2 * (t1 - 1) + 1, t2 - 1])
                continue
            minimum_energy = 1e10
            for lmp, kwargs in zip(lmp_list, kwargs_list):
                print(f"processing... {t1}, {t2}")
                Es, Fs = get_E_F_list(ds, t1, t2, lmp)
                if "rebo" == kwargs["label"] and (t1 == 3 or t2 == 3):
                    continue
                factor = 1 / 23.06035 if "reax" in kwargs["label"] else 1
                # factor = 1
                bond_energies = (Es - Es[-1]) * factor
                axs[2 * (t1 - 1), t2 - 1].plot(ds, bond_energies, **kwargs)
                bond_energy = np.min(bond_energies)
                bond_length = ds[np.argmin(bond_energies)]
                print(
                    f"{kwargs['label']}: bond {tmap[t1]}, {tmap[t2]}: {bond_length:.2f} A {bond_energy:.2f} eV"
                )
                minimum_energy = np.minimum(minimum_energy, bond_energy)
                axs[2 * (t1 - 1) + 1, t2 - 1].plot(ds, Fs * factor, **kwargs)
                axs[2 * (t1 - 1), t2 - 1].set_title(f"{tmap[t1]}-{tmap[t2]}")
                axs[2 * (t1 - 1) + 1, t2 - 1].set_title(f"{tmap[t1]}-{tmap[t2]}")
            axs[2 * (t1 - 1), t2 - 1].set_ylim(bottom=minimum_energy - 10)

            for s in dft_data:
                # print(s[0], s[1], tmap[t1], tmap[t2])
                if (s[0] == tmap[t1] and s[1] == tmap[t2]) or (
                    s[0] == tmap[t2] and s[1] == tmap[t1]
                ):
                    factor = 1 / 23.06035
                    distance = s[2]
                    energy = s[3] - shiftdict[s[0] + s[1]]
                    force = s[4]
                    if distance > 3.0:
                        label = "DFT"
                    else:
                        label = None
                    axs[2 * (t1 - 1), t2 - 1].scatter(
                        distance, energy * factor, c="k", label=label, alpha=0.5
                    )
                    axs[2 * (t1 - 1) + 1, t2 - 1].scatter(
                        distance, force * factor, c="k", label=label, alpha=0.5
                    )
    for ax in axs[0::2, :].flatten():
        ax.set_ylim(bottom=-15, top=55)
        ax.set_xlabel("Distance [Å]")
        ax.set_ylabel("Energy [eV]")
        # ax.legend()
    for ax in axs[1::2, :].flatten():
        ax.set_ylim(-1, 25)
        ax.set_xlabel("Distance [Å]")
        ax.set_ylabel("||Force|| [eV / Å]")
        # ax.legend()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", fontsize=36)
    if args.vline is not None:
        for ax in axs.flatten():
            ax.axvline(float(args.vline), c="k", ls="--")
    fig.tight_layout()
    fig.savefig(args.output)

    # # print self energies:
    # for i, t1 in enumerate([1, 2, 3]):
    #     for lmp, kwargs in zip(lmp_list, kwargs_list):
    #         self_energy, _ = get_E_F(50, t1, t1, lmp)
    #         # self_energy = self_energy / 2
    #         tmap = {1: "C", 2: "H", 3: "O", 4: "Ar"}
    #         print(f"Potential {kwargs['label']} self energy for {tmap[t1]} is {self_energy}")
