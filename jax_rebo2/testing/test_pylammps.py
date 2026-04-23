

import numpy as np
import optax
import pickle
import tqdm
import lammps


def make_lmp():
    units = "metal"
    atom_style = "atomic"
    pair_style = "rebo3"
    qeq = ""
    potential_file = "CH.rebo3"
    # potential_file = "CH.rebo3_2"
    n_atom_types = 3  # C, H, O

    atom_list = " ".join(["C", "H", "O"])
    pair_coeff_lines = f"pair_coeff  * * {potential_file} {atom_list}"

    lmp = lammps.lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"], name="della")
    # lmp = lammps.lammps(cmdargs=["-echo", "screen", "-log", "none", "-nocite"], name="della")
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


def label_row_with_lammps(lmp):
    lmp.command("delete_atoms group all")
    n_atoms = 10
    R = np.random.rand(n_atoms, 3) * 10
    species = np.random.randint(0, 3, size=n_atoms)
    box = np.array([10.0, 10.0, 10.0])

    lmp.command("delete_atoms group all")
    lmp.command(f"change_box all x final 0 {box[0]} y final 0 {box[1]} z final 0 {box[2]}")
    for i in range(1, n_atoms):
        if species[i] >= 0:
            lmp.command(f"create_atoms {species[i]+1} single {R[i, 0]} {R[i, 1]} {R[i, 2]}")

    lmp.command("run 0")

    energy = lmp.get_thermo("pe")
    f = lmp.extract_atom("f")
    forces = np.array([[f[i][0], f[i][1], f[i][2]] for i in range(n_atoms)])
    return energy, forces

label_row_with_lammps(make_lmp())