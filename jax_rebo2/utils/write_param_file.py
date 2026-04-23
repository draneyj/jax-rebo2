import numpy as np
import pickle
import interpax
import jax
import matplotlib.pyplot as plt
import argparse
import newrebo2_interaction_list as rebo2


def write_param(file, params, key, name):
    for p in params[key].flatten():
        file.write(f"{p} ")
    file.write(f"{name}\n")

def check_and_write_param(file, params, key, name, nrepeats, default_value):
    if key in params.keys():
        write_param(file, params, key, name)
    else:
        file.write(nrepeats * f"{default_value} " + f"{name}\n")

def get_g_coeffs(knots, g_values, dg_values, npoints=6):
    spline = interpax.Interpolator1D(knots, g_values, fx=dg_values, method="cubic", extrap=True)
    xs = []
    for i in range(len(knots) - 1):
        xs.append(np.linspace(knots[i], knots[i + 1], npoints))
    coefs = []
    for i,x in enumerate(xs):
        y = jax.vmap(spline)(x)
        poly = np.polynomial.polynomial.Polynomial.fit(x, y, deg=npoints - 1)
        coefs.append(poly.convert().coef)
    return np.concatenate(coefs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="creates rebo2 parameter file from learned parameters."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to pickle containing rebo2 parameters.",
        default="params",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file.",
        default="CH.rebo3_default",
    )
    args = parser.parse_args()
    outfile = open(args.output, "w")
    outfile.write(f"# Script by Jack Draney 2025.\n")
    outfile.write(f"# REBO2 parameters created with script from {args.params}.\n")

    params = pickle.load(open(args.params, "rb"))
    nspecies = int(np.sqrt(np.prod(params["A"].shape)))
    outfile.write(f"{nspecies}\tnspecies\n")
    write_param(outfile, params, "Dmin", "rcmin")
    write_param(outfile, params, "Dmax", "rcmax")
    if "rcmaxp" in params.keys():
        write_param(outfile, params, "rcmaxp", "rcmaxp")
    else:
        write_param(outfile, params, "Dmax", "rcmaxp")
    check_and_write_param(outfile, params, "smin", "smin", 1, "0.1")
    check_and_write_param(outfile, params, "Nmin", "Nmin", 1, "2.0")
    check_and_write_param(outfile, params, "Nmax", "Nmax", 1, "3.0")
    check_and_write_param(outfile, params, "NCmin", "NCmin", 1, "3.2")
    check_and_write_param(outfile, params, "NCmax", "NCmax", 1, "3.7")
    write_param(outfile, params, "Q", "Q")
    write_param(outfile, params, "alpha", "alpha")
    write_param(outfile, params, "A", "A")
    for i in range(nspecies):
        for j in range(nspecies):
            for k in range(3):
                outfile.write(f"{params['B'+str(k+1)][i, j]} ")
    outfile.write("B\n")
    for i in range(nspecies):
        for j in range(nspecies):
            for k in range(3):
                outfile.write(f"{params['beta_'+str(k+1)][i, j]} ")
    outfile.write("Beta\n")
    write_param(outfile, params, "rho", "rho")
    write_param(outfile, params, "lambda_ijk", "lambdaijk")
    check_and_write_param(outfile, params, "rcLJmin", "rcLJmin", nspecies**2, "0.0")
    check_and_write_param(outfile, params, "rcLJmax", "rcLJmax", nspecies**2, "0.0")
    check_and_write_param(outfile, params, "bLJmin", "bLJmin", nspecies**2, "0.0")
    check_and_write_param(outfile, params, "bLJmax", "bLJmax", nspecies**2, "0.0")
    check_and_write_param(outfile, params, "epsilon", "epsilon", nspecies**2, "0.0")
    check_and_write_param(outfile, params, "sigma", "sigma", nspecies**2, "0.0")
    check_and_write_param(outfile, params, "epsilonT", "epsilonT", nspecies**2, "0.0")

    # write g
    for i in range(nspecies):
        outfile.write(f"\n# g{i}1 and g{i}2\n\n")
        outfile.write(f"{len(params['G_knots'])}\n")
        for knot in params['G_knots']:
            outfile.write(f"{knot}\n")
        outfile.write("\n")

        g_coeffs = get_g_coeffs(params['G_knots'], params['G'][i], params['dG'][i], npoints=6)
        gamma_coeffs = get_g_coeffs(params['G_knots'], params['gamma'][i], params['dgamma'][i], npoints=6)
        for coeff in gamma_coeffs:
            outfile.write(f"{coeff:20.10f}\n")
        outfile.write("\n")
        for coeff in g_coeffs:
            outfile.write(f"{coeff:20.10f}\n")
    
    # write p
    for i in range(nspecies):
        for j in range(nspecies):
            knots = np.arange(10)
            outfile.write(f"\n# p{i}{j}\n\n")
            outfile.write(f"{2 * nspecies}\n")
            for k in range(nspecies):
                outfile.write(f"0.0\n10.0\n")
            f = params['P'][i, j]
            fx = params['P_di'][i, j]
            fy = params['P_dj'][i, j]
            fz = params['P_dk'][i, j]
            zero_fx = fx*0
            spline = interpax.Interpolator3D(
                knots, knots, knots, f, fx=fx, fy=fy, fz=fz, fxy=zero_fx, fxz=zero_fx, fyz=zero_fx, fxyz=zero_fx, method="cubic", extrap=True
            )

            for k1 in knots:
                for k2 in knots:
                    for k3 in knots:
                        cs = spline.coef(k1,k2,k3).reshape(-1)
                        for c in cs:
                            outfile.write(f"{c:20.10f}\n")
            outfile.write("\n")
    
    # write pi (F)
    for i in range(nspecies):
        for j in range(nspecies):
            knots = np.arange(10)
            outfile.write(f"\n# pi{i}{j}\n\n")
            outfile.write(f"{2 * nspecies}\n")
            for k in range(nspecies):
                outfile.write(f"0.0\n10.0\n")
            f = params['F'][i, j]
            fx = params['F_di'][i, j]
            fy = params['F_dj'][i, j]
            fz = params['F_dk'][i, j]
            zero_fx = fx*0
            spline = interpax.Interpolator3D(
                knots, knots, knots, f, fx=fx, fy=fy, fz=fz, fxy=zero_fx, fxz=zero_fx, fyz=zero_fx, fxyz=zero_fx, method="cubic", extrap=True
            )

            for k1 in knots:
                for k2 in knots:
                    for k3 in knots:
                        cs = spline.coef(k1,k2,k3).reshape(-1)
                        for c in cs:
                            outfile.write(f"{c:20.10f}\n")
            outfile.write("\n")
    
    # write T
    for i in range(nspecies):
        for j in range(nspecies):
            knots = np.arange(10)
            outfile.write(f"\n# T{i}{j}\n\n")
            outfile.write(f"{2 * nspecies}\n")
            for k in range(nspecies):
                outfile.write(f"0.0\n10.0\n")
            f = params['T'][i, j]
            fx = params['T_di'][i, j]
            fy = params['T_dj'][i, j]
            fz = params['T_dk'][i, j]
            zero_fx = fx*0
            spline = interpax.Interpolator3D(
                knots, knots, knots, f, fx=fx, fy=fy, fz=fz, fxy=zero_fx, fxz=zero_fx, fyz=zero_fx, fxyz=zero_fx, method="cubic", extrap=True
            )
            for k1 in knots:
                for k2 in knots:
                    for k3 in knots:
                        cs = spline.coef(k1,k2,k3).reshape(-1)
                        for c in cs:
                            outfile.write(f"{c:20.10f}\n")
            outfile.write("\n")

    # write H
    if 'H' in params.keys():
        for i in range(nspecies):
            for j in range(nspecies):
                knots = np.arange(10)
                outfile.write(f"\n# H{i}{j}\n\n")
                outfile.write(f"{2 * nspecies}\n")
                for k in range(nspecies):
                    outfile.write(f"0.0\n10.0\n")
                f = params['H'][i, j]
                fx = params['H_di'][i, j]
                fy = params['H_dj'][i, j]
                fz = params['H_dk'][i, j]
                zero_fx = fx*0
                spline = interpax.Interpolator3D(
                    knots, knots, knots, f, fx=fx, fy=fy, fz=fz, fxy=zero_fx, fxz=zero_fx, fyz=zero_fx, fxyz=zero_fx, method="cubic", extrap=True
                )

                for k1 in knots:
                    for k2 in knots:
                        for k3 in knots:
                            cs = spline.coef(k1,k2,k3).reshape(-1)
                            for c in cs:
                                outfile.write(f"{c:20.10f}\n")
                outfile.write("\n")
