#!/usr/bin/env python3

"""
This is a simple model to test the emulator and mcmc

It read in the sampled parameter file and generate outputs

"""

import numpy as np

def model(para_dict, x):
    """This is a simple model"""
    rel_error = 0.1
    y1 = (para_dict["A"]*(np.exp(-para_dict["B"]*x**2.)
                          + rel_error*np.random.random(len(x))))
    y2 = (para_dict["C"]*(np.cosh(para_dict["D"]*x)
                          + rel_error*np.random.random(len(x))))
    Y = np.concatenate(y1, y2)
    return Y

def main():
    """This is the main function"""
    import argparse

    parser = argparse.ArgumentParser(description='A simple test model')
    parser.add_argument(
        '-i', '--input_file', metavar='', type=str,
        default='sample_0', help='parameter input file')
    args = parser.parse_args()

    para_dict = {}
    para_file = open(args.input_file, "r")
    for line in para_file:
        par = line.split()
        para_dict.update({str(par[0]): float(par[1])})
    para_file.close()

    x = np.linspace(-5, 5., 21)
    Y = model(para_dict, x)

    sample_id = args.input_file.split("_")[1]
    np.savetxt("output_{}.txt".format(sample_id), Y, delimiter=" ", fmt="%.4e")

if __name__ == '__main__':
    main()
