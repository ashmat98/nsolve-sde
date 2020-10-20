
from warper import simulate_2d_only_memory, simulate_2d_only_memory_anharmonic_2, simulate_2d_only_memory_anharmonic_1, simulate_3d_only_memory_anharmonic_1

from matplotlib import pyplot as plt
import numpy as np

from multiprocessing import Pool

import os, pickle
from tqdm.notebook import tqdm

import pandas as pd


from nsdesolve import Uxy_2, Uxyz_1

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams["figure.facecolor"] = "white"


def get_cov(R,S,H):
    Z = np.zeros(len(R))
    return np.array([[R,Z,Z,H],
                     [Z,R,-H,Z],
                     [Z,-H,S,Z],
                     [H,Z,Z,S]])

def omega_kumulant(A,B,g0,ka):
    bigO=A + 2*B/ka**2/g0
    
    op2 = 1/2 * (bigO+abs(bigO)*np.sqrt(1+8*B/g0/bigO**2))
    om2 = 1/2 * (bigO-abs(bigO)*np.sqrt(1+8*B/g0/bigO**2))
    return op2, om2
 
def get_RSH_harmonic_onlyka(o0,g0,b,ka):
    mu = ka
    S = 1/g0/2/ka**2*(ka**2+o0**2+g0*ka+b**2)
    R = 1/g0/2/o0**2/ka**2 * (o0**2+ka**2)
    H = - 1/2/g0/ka**2*b
    
    return R,S,H, 2*(R*S+H**2)

def mean_with_err(arr, axis):
    return arr.mean(axis=axis), arr.std(axis=axis)/np.sqrt(arr.shape[axis])