from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plot_config as cfg
from scipy import fft, signal

def plotSet(title, xLabel, yLabel, xStart, xEnd, yStart, yEnd):
    # font size and resolution
    plt.rcParams.update({'font.size': cfg.FONT_SIZE})
    plt.rcParams['figure.dpi'] = cfg.DPI

    # figure and axis sizes, labels
    fig = plt.figure(figsize=(cfg.WIDTH/cfg.DPI, cfg.HEIGHT/cfg.DPI))
    ax = fig.add_axes([0.1, 0.12, 0.85, 0.8])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)
    ax.set_ylim(yStart, yEnd)

    return fig, ax

def plot_surface(xs, ys, xLabel=None, yLabel=None, zLabel=None, show=False):
    # font size and resolution
    plt.rcParams.update({'font.size': cfg.FONT_SIZE})
    plt.rcParams['figure.dpi'] = cfg.DPI

    # figure and axis sizes, labels
    fig = plt.figure(figsize=(cfg.WIDTH/cfg.DPI, cfg.HEIGHT/cfg.DPI))
    ax = fig.add_subplot(111, projection='3d')

    Z = np.tile(ys, (len(ys), 1))
    ys = np.linspace(xs[0], xs[-1], len(xs))
    X, Y = np
    line = ax.plot_surface(X, Y, Z)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)

    if show is True:
        plt.show()
    return fig, ax, line

