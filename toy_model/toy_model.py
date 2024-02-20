#!/usr/bin/env python3

"""
This is a simple model to test the emulator and mcmc

It read in the sampled parameter file and generate outputs

"""

import numpy as np
import pathlib
from shutil import copy

def toy_model(para_dict, x):
    """This is a simple toy model"""
    rel_error = 0.01
    y1 = para_dict["A"]*np.exp(-para_dict["B"]*x**2.)
    y1 = y1*(1. + np.random.normal(0., rel_error, len(x)))
    y2 = para_dict["C"]*np.cosh(para_dict["D"]*x)
    y2 = y2*(1. + np.random.normal(0., rel_error, len(x)))
    Y = np.concatenate((y1, y2))
    return Y


def pseudo_expdata(x):
    para_dict_true = {
        "A": 0.2,
        "B": 0.5,
        "C": 0.7,
        "D": 0.3,
    }
    Y_exp = toy_model(para_dict_true, x)
    Y_stat_err = 0.1*Y_exp
    Y_sys_err = 0.1*Y_exp
    output = np.array([range(len(Y_exp)), Y_exp, Y_stat_err, Y_sys_err])
    return output


def main():
    """This is the main function"""
    import argparse

    parser = argparse.ArgumentParser(description='A simple test model')
    parser.add_argument(
        '-i', '--input_file', metavar='', type=str,
        default='parameter_0', help='parameter input file')
    parser.add_argument(
        '--exp', action='store_true',
        help='output pseudo exp data to fit')
    args = parser.parse_args()

    x = np.linspace(-5, 5., 21)
    if args.exp:
        exp_data = pseudo_expdata(x)
        np.savetxt("pseudo_expdata.txt", exp_data.transpose(), delimiter=" ",
                   fmt="%.4e", header="idx  Y  Y_stat_err  Y_sys_err")
    else:
        para_dict = {}
        para_file = open(args.input_file, "r")
        for line in para_file:
            par = line.split()
            para_dict.update({str(par[0]): float(par[1])})
        para_file.close()

        Y = toy_model(para_dict, x)

        outputs = np.array([range(len(Y)), Y, 0.01*Y])

        sample_id = args.input_file.split("_")[1]
        folder_name = 'run_{}'.format(sample_id)
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
        copy(args.input_file, "{}/parameter_{}".format(folder_name, sample_id))
        np.savetxt("{}/Bayesian_output.txt".format(folder_name),
                   outputs.transpose(), delimiter=" ", fmt="%.4e")

if __name__ == '__main__':
    main()
