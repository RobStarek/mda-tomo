# -*- coding: utf-8 -*-
"""
Data acquisitin script.
Device-independent tomography project
=================
This script controlls the motorized rotation mounts (Newport smc100c+pr50cc) with waveplates
to perform quantum state tomography on various probe states.
A programmable multimeter (TTi1906) is used to measure photocurrent from the detector photodiode.

Usage:
------
measurement_settings.toml configures waveplate calibrations, output file,
defines serial interface to the devices and is used to select user
measurement iterator. The output is saved into .h5 file.

Author:
-------
Robert Stárek

Last Updated:
-------------
Date: 31/8/2023
Version: 1.2
"""

from time import strftime, sleep
import toml
import numpy as np
import h5py
import device_utils as du
from smcStack import SMCStack as Actuator
DEG = np.pi/180.

if __name__ == '__main__':
    # Load measurement settings and module with user iterators.
    SETTINGS = toml.load('measurement_settings.toml')
    USER_MODULE = __import__(SETTINGS['tomo_generators']['path'])
    USER_ITERS = [getattr(USER_MODULE, it)
                  for it in SETTINGS['tomo_generators']['iterators']]

    print('user_iters:')
    for i, element in enumerate(USER_ITERS):
        print(i, element)

    # open comports to the devices
    HW_ADDR = du.get_hw_addresses(SETTINGS['device_usb_info'])
    acquire, close_detector, n_read = du.get_detection_func_tti(HW_ADDR, None)
    waveplates = {key: (item['id'], key, item['x0'])
                  for key, item in SETTINGS['wpdef'].items()}
    actuator = Actuator(HW_ADDR["waveplates"], waveplates, 1)
    waveplate_selection = SETTINGS['wpselection']['wp_set'] + \
        SETTINGS['wpselection']['wp_prep'] + SETTINGS['wpselection']['wp_proj']
    n_wp = len(waveplate_selection)

    # Create output files
    output_file = h5py.File(f"{SETTINGS['general']['filename']}.h5", "w")
    output_file.create_group('metadata')
    for category, item in SETTINGS.items():
        for key, subitem in item.items():
            output_file['metadata'].attrs[f'{category}/{key}'] = str(subitem)
    output_file['metadata'].attrs['date'] = strftime('%d/%m/%Y %H:%M:%S')

    for i_gen, generator in enumerate(USER_ITERS):
        print(f"I: {i_gen}")
        subdata = []
        sub_dset_move = output_file.create_dataset(
            f"iter_{i_gen}_move", (1296, n_wp), dtype='f', maxshape=(None, n_wp))
        sub_dset_read = output_file.create_dataset(
            f"iter_{i_gen}_int", (1296, n_read), dtype='f', maxshape=(None, n_read))
        sub_dset_info = output_file.create_dataset(
            f"iter_{i_gen}_info", (1296, 5), dtype='f', maxshape=(None, 5))
        last_info = None
        logs_count = 0
        for j_move, (move, info) in enumerate(generator(waveplate_selection)):
            # Log progress
            if j_move == 0:
                sleep(5)
            print(f"J: {j_move}")
            move_log = [move[key] for key in waveplate_selection]
            print(move)

            # Manipulate devices
            actuator(move)
            actuator.WaitForMovement(waveplate_selection)
            reading = acquire(SETTINGS['general']['averaging'])

            # Update H5 file
            if j_move > (sub_dset_read.shape[0]-1):
                sub_dset_read.resize((j_move + 128, n_read))
                sub_dset_move.resize((j_move + 128, n_wp))

            # Save reading and waveplate positions into output file.
            sub_dset_move[j_move] = move_log
            sub_dset_read[j_move] = reading

            # Logging of info from user iterators.
            if last_info is None:
                last_info = (j_move, *info)
                print(f'Info: {last_info}')
                sub_dset_info.resize((logs_count + 128, 5))
                sub_dset_info[logs_count] = np.array(last_info)
                logs_count = logs_count + 1
                sub_dset_info.flush()
            elif last_info[1:] != info:
                last_info = (j_move, *info)
                print(f'Info: {last_info}')
                sub_dset_info.resize((logs_count + 128, 5))
                sub_dset_info[logs_count] = np.array(last_info)
                logs_count = logs_count + 1
                sub_dset_info.flush()
            sub_dset_move.flush()
            sub_dset_read.flush()

        # finally resize the file to the actual number of entries
        sub_dset_read.resize((j_move+1, n_read))
        sub_dset_move.resize((j_move+1, n_wp))
        sub_dset_info.resize((logs_count, 5))
        sub_dset_move.flush()
        sub_dset_read.flush()

    # close comports and files
    close_detector()
    actuator.Close()
    output_file.close()

    print("Done")
