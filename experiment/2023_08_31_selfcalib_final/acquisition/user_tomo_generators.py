# -*- coding: utf-8 -*-
"""
User tomographic generator to be used in acquire.py.
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

import numpy as np
from tomo_utils import tomography_generator, STD_PREP_TABLE, STD_PROJ_TABLE, deg_round_remap, BASE_STATE_LIST, generate_preparation_angles_deg, generate_projection_angles_deg
import KetSugar as ks
import WaveplateBox as wp
DEG = np.pi/180.

#Retardance deviation

# #acquired from calibration data wp 3,4,13,14, gets good data
# DHWP1 = 4.5*DEG
# DQWP2 = -3.6*DEG
# DHWP3 = 4.5*DEG
# DQWP4 = -1.3*DEG

 #acquired from calibration data wp 1,2,5,6, 
DHWP1 = 2.1*DEG
DQWP2 = -5.3*DEG
DHWP3 = 5.8*DEG
DQWP4 = -4.5*DEG

# #acquired from calibration data wp 1,2,5,6, 90deg flip
#DHWP1 = -7.*DEG
#DQWP2 = -4.2*DEG
#DHWP3 = 2.8*DEG
#DQWP4 = -3.8*DEG

def process_1_qubit(wp_labels):
    """
    Iterator that yield 36 measurement waveplate settings needed to perform
    Pauli tomography of a single-qubit process.
    Args:
        wp_labels ... list of strings, keys to the waveplates dictionary
                      it should be ordered as hwp1, qwp2, hwp3, qwp4,
                      where first two waveplates 1 and 2 are used to prepare state
                      and 3, 4 serve for projections.
    Yields:
        move ... dictionary with waveplates names as keys and 
                 waveplates angles in degrees as values
        info ... tuple with additonal info, here just index of settings
    """
    gen = tomography_generator([STD_PREP_TABLE, STD_PROJ_TABLE], wp_labels)
    i = 0
    for move in gen:
        i = i + 1
        info = (i, 0, 0, 0)
        yield move, info

def calibrated_process_1_qubit(wp_labels):
    cal_prep_table = generate_preparation_angles_deg(BASE_STATE_LIST, ks.LO, DHWP1, DQWP2)
    cal_proj_table = generate_projection_angles_deg(BASE_STATE_LIST, ks.LO, DHWP3, DQWP4)
    gen = tomography_generator([cal_prep_table, cal_proj_table], wp_labels)
    i = 0
    for move in gen:
        i = i + 1
        info = (i, 0, 0, 0)
        yield move, info

# few save samling schemes in Bloch coordianates (theta...collatitude, phi...longitude)
samplings = np.load('probes_samplings.npz')

# waveplate settings in radians for pauli tomography
std_projection_table = (
    (0, 0),
    (np.pi/4, 0),
    (np.pi/8, 0),
    (-np.pi/8, 0),
    (np.pi/8, np.pi/4),
    (-np.pi/8, -np.pi/4)
)

# same as std_projection_table
# but used in reversed measurement
rev_meas_table = (
    (0, 0),
    (np.pi/4, 0),
    (np.pi/8, 0),
    (-np.pi/8, 0),
    (-np.pi/8, -np.pi/4),
    (np.pi/8, np.pi/4)
)


def bloch_coord_to_wp_angles(theta, phi):
    """
    Analytical formula to prepare state located at 
    collatitude theta and longitude phi on the Bloch sphere.
    Returns angles of waveplates in radians.
    """
    c = np.pi/4 if (theta > np.pi/2) else 0
    qwp = -.5*np.arcsin(np.sin(theta)*np.sin(phi))
    hwp = .25*np.arctan(np.tan(theta)*np.cos(phi)) + 0.5*qwp + c
    return hwp, qwp


def generate_tomogram_fwd_stp1(wp_labels):
    """
    Calibration iterator with 8 probe states. Forward mode.
    Args:
        wp_labels ... list of strings, keys to the waveplates dictionary
                      it should be ordered as hwp1, qwp2, hwp3, qwp4,
                      where first two waveplates 1 and 2 are used to prepare state
                      and 3, 4 serve for projections.
    Yields:
        move ... dictionary with waveplates names as keys and 
                 waveplates angles in degrees as values
        info ... tuple with additonal info, here just index of settings
    """
    sampling = samplings['n8']
    i = 0
    for theta, phi in sampling:
        hwp1, qwp1 = bloch_coord_to_wp_angles(theta, phi)
        for hwp3, qwp4 in std_projection_table:
            angles = [hwp1, qwp1, hwp3, qwp4]
            move = {key: deg_round_remap(angle)
                    for key, angle in zip(wp_labels, angles)}
            info = (i, 0, 0, 0)
            i = i + 1
            yield move, info


def generate_tomogram_rev_stp1(wp_labels):
    """
    Calibration iterator with 8 probe states. Measurement in reversed mode.
    Now Pauli projections are done effectively on preparation side, while probe preparation is done on projection side.
    Args:
        wp_labels ... list of strings, keys to the waveplates dictionary
                      it should be ordered as hwp1, qwp2, hwp3, qwp4,
                      where first two waveplates 1 and 2 are used to prepare state
                      and 3, 4 serve for projections.
    Yields:
        move ... dictionary with waveplates names as keys and 
                 waveplates angles in degrees as values
        info ... tuple with additonal info, here just index of settings
    """
    sampling = samplings['n8']
    i = 0
    for hwp_prep, qwp_prep in rev_meas_table:
        for (theta, phi) in sampling:
            hwp_proj, qwp_proj = bloch_coord_to_wp_angles(theta, -phi)
            angles = (hwp_prep, qwp_prep, hwp_proj, qwp_proj)
            move = {key: deg_round_remap(angle)
                    for key, angle in zip(wp_labels, angles)}
            info = (i, 0, 0, 0)
            i = i + 1
            yield move, info


def get_check_tomo_8(dhwp1=0, dqwp2=0, dhwp3=0, dqwp4=0):
    """
    Get testing iterator, assuming deviations in waveplate retardance.
    Test is performed on the same 8 states used for calibration.
    Args:
        dhwp1, dqwp2 ... retardance deviation in radians for preparation waveplates
        dhwp3, dqwp4 ... retardance deviation in radians for projection waveplates

    The returned function is an iterator with following args/returns:
    Args:
        wp_labels ... list of strings, keys to the waveplates dictionary
                      it should be ordered as hwp1, qwp2, hwp3, qwp4,
                      where first two waveplates 1 and 2 are used to prepare state
                      and 3, 4 serve for projections.
    Yields:
        move ... dictionary with waveplates names as keys and 
                 waveplates angles in degrees as values
        info ... tuple with additonal info, here just index of settings     
    """
    def iterator_function(wp_labels):
        sampling = samplings['n8']
        fixed_preparation_angles = []
        for theta, phi in sampling:
            optimum_f = wp.SearchForKet(ks.BlochKet(
                theta, phi), ks.LO, dhwp1, dqwp2, 1e-11)
            assert optimum_f['fun'] < -0.99
            fixed_preparation_angles.append(optimum_f['x'])

        fixed_projection_angles = []
        for ket in BASE_STATE_LIST:
            optimum_f = wp.SearchForProj(ket, ks.LO, dhwp3, dqwp4, 1e-11)
            assert optimum_f['fun'] < -0.99
            fixed_projection_angles.append(optimum_f['x'])

        i = 0
        for hwp1, qwp2 in fixed_preparation_angles:
            for hwp3, qwp4 in fixed_projection_angles:
                angles = (hwp1, qwp2, hwp3, qwp4)
                move = {key: deg_round_remap(angle)
                        for key, angle in zip(wp_labels, angles)}
                info = (i, 0, 0, 0)
                i = i + 1
                yield move, info
    return iterator_function

check_tomo_8_n = get_check_tomo_8()
check_tomo_8_c = get_check_tomo_8(
    dhwp1 = DHWP1,
    dqwp2 = DQWP2,
    dhwp3 = DHWP3,
    dqwp4 = DQWP4
)

def get_check_tomo_dense(dhwp1=0, dqwp2=0, dhwp3=0, dqwp4=0):
    """
    Same as get_check_tomo_8, but uses 58 probe states.
    """
    Nx = 9
    Ny = 9

    def foo(wp_labels):
        thetas = np.linspace(0, 180, Ny)*DEG
        phis = np.linspace(-180, 180, Nx)[1:]*DEG
        sampling = []
        for theta in thetas:
            for phi in ([0] if (theta == 0 or theta == np.pi) else phis):
                sampling.append((theta, phi))

        fixed_preparation_angles = []
        for theta, phi in sampling:
            optimum_f = wp.SearchForKet(ks.BlochKet(
                theta, phi), ks.LO, dhwp1, dqwp2, 1e-9)
            assert optimum_f['fun'] < -0.99
            fixed_preparation_angles.append(optimum_f['x'])

        if (dhwp3 == 0) and (dqwp4 == 0):
            fixed_projection_angles = std_projection_table
        else:
            fixed_projection_angles = []
            for ket in BASE_STATE_LIST:
                optimum_f = wp.SearchForProj(ket, ks.LO, dhwp3, dqwp4, 1e-9)
                assert optimum_f['fun'] < -0.99
                fixed_projection_angles.append(optimum_f['x'])

        i = 0
        for hwp1, qwp2 in fixed_preparation_angles:
            for hwp3, qwp4 in fixed_projection_angles:
                angles = (hwp1, qwp2, hwp3, qwp4)
                move = {key: deg_round_remap(angle)
                        for key, angle in zip(wp_labels, angles)}
                info = (i, 0, 0, 0)
                i = i + 1
                yield move, info
    return foo


check_tomo_58_n = get_check_tomo_dense()
check_tomo_58_c = get_check_tomo_dense(
    dhwp1 = DHWP1,
    dqwp2 = DQWP2,
    dhwp3 = DHWP3,
    dqwp4 = DQWP4
)
