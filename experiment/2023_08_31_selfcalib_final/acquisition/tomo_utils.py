# -*- coding: utf-8 -*-
"""
Tomographic utilities for data acquisition.
Device-independent tomography project
=================
This module defines few standard table for preparation/projection of Pauli eigenstates
using waveplates. It is used in user_tomo_generators.

Author:
-------
Robert Stárek

Last Updated:
-------------
Date: 31/8/2023
Version: 1.2
"""

import itertools
import numpy as np
import WaveplateBox as wp
import KetSugar as ks

BASE_STATES = {
    "H": ks.LO,
    "V": ks.HI,
    "D": ks.HLO,
    "A": ks.HHI,
    "R": ks.CLO,
    "L": ks.CHI,
}
BASE_STATE_LIST = [ks.LO, ks.HI, ks.HLO, ks.HHI, ks.CLO, ks.CHI]

STD_PROJ_TABLE = np.array([
    [0.0, 0.0], 
    [45.0, 0.0], 
    [22.5, 0.0], 
    [-22.5, 0.0], 
    [22.5, 45.0], 
    [-22.5, -45.0]
    ])

STD_PREP_TABLE = np.array([
    [0.0, 0.0], 
    [45.0, 0.0], 
    [22.5, 0.0], 
    [-22.5, 0.0], 
    [-22.5, -45.0], 
    [22.5, 45.0]
    ])

def remap_angle(x):
    return ((x - np.pi / 2) % np.pi) - np.pi / 2

def deg_round_remap(x):
    return np.round(np.rad2deg(remap_angle(x)), 3)

def generate_projection_angles_deg(proj_kets, projector=ks.LO, dhwp=0, dqwp=0):
    table = []
    for ket in proj_kets:
        opt_res = wp.SearchForProj(ket, projector, dhwp, dqwp, tol=1e-12)
        try:
            assert opt_res["fun"] < -0.999
        except:
            opt_res["fun"]
            raise
        hwp, qwp = opt_res["x"]
        hwp_deg = deg_round_remap(hwp)
        qwp_deg = deg_round_remap(qwp)
        table.append((hwp_deg, qwp_deg))
    return np.array(table)

def generate_preparation_angles_deg(proj_kets, input_ket=ks.LO, dhwp=0, dqwp=0):
    table = []
    for ket in proj_kets:
        opt_res = wp.SearchForKet(ket, input_ket, dhwp, dqwp, tol=1e-12)
        assert opt_res["fun"] < -0.999
        hwp, qwp = opt_res["x"]
        hwp_deg = deg_round_remap(hwp)
        qwp_deg = deg_round_remap(qwp)
        table.append((hwp_deg, qwp_deg))
    return np.array(table)

def tomography_generator(angle_tables, wp_labels):
    for angles in itertools.product(*angle_tables):
        yield {key: angle for key, angle in zip(wp_labels, np.concatenate(angles).ravel())}

