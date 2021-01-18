#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from diffusion import Domain
from diffusion.analytical import Solver, solution_1D, animate_1D


def make_animation(file=None, draw=False):
    """Make an animation of the analytical solution"""

    domain = Domain([20, 10], [1, 3], [0, 0.4, 0])
    solver = Solver(domain)
    eV = solver.find_eigV(max_lambda=100, zero=1e-4)
    xq = np.linspace(0, domain.total_length, 1001)
    eM = solver.find_eigM(xq)

    t = np.linspace(0, 10, 101)
    G = [solution_1D(tn, xq, len(xq)//2, 'arbitrary', lambdas=eV, nus=eM, peak=1) for tn in t]
    anim = animate_1D(G, t, xq, domain.barriers)

    if file:
        Writer = animation.writers['ffmpeg']  # requires ffmpeg installed
        writer = Writer(fps=30)
        anim.save(file, writer=writer)

    if draw:
        plt.show()  # this plays the animation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='animation')
    parser.add_argument('--file', dest='file',
                        type=str,
                        help='File path for saving the animation')
    parser.add_argument('--draw', dest='draw',
                        action='store_true',
                        help='Draw the animation on screen')
    args = parser.parse_args()
    make_animation(file=args.file, draw=args.draw)
