# -*- coding: utf-8 -*-
"""
Measurement device utilities.

Device-independent tomography project
=================
Helper module used to simplify finding motorized mounts and multimeters.
It provides function get_detection_func_tti() that 
generates acquire function. 
If I ever need to replace the multimeter with a different detector, I just 
add/modify get_detection_func_xxx() and the main code, acquire.py, should not be modified.

Author:
-------
Robert Stárek

Last Updated:
-------------
Date: 31/8/2023
Version: 1.2
"""

from time import sleep, localtime, strftime
from serial.tools import list_ports
from TTi1906 import TTiMinimal

def autodetect_port_addr(vendorid, productid, serial_number=None):
    """
    Attempt to find comport by vendor, product id and serial number.
    If not found, returns None.
    """
    visible_ports = list_ports.comports()
    for port in visible_ports:
        if (port.vid == vendorid) and (port.pid == productid):
            if serial_number is None:
                return port.device
            elif port.serial_number in serial_number:
                return port.device

def get_hw_addresses(device_dict):
    return {key : autodetect_port_addr(
        entry['vid'], 
        entry['pid'],
        entry['sn']) 
        for key, entry in device_dict.items()}

def get_detection_func_tti(detector_addresses, logger = None):
    detector_h = TTiMinimal(detector_addresses.get("ttiH")) #H proj
    detector_v = TTiMinimal(detector_addresses.get("ttiV")) #V proj
    n_chan = 2
    def sync_reading(avg=1):
        s_h = 0
        s_v = 0
        for i in range(avg):
            detector_v.issue_read()
            detector_h.issue_read()
            sleep(detector_h.dt)
            s_v += detector_v.pick_reading()
            s_h +=  detector_h.pick_reading() 
        current_time = strftime("%H:%M:%S", localtime())
        msg = f'{current_time}: TTi: Reading: H: {s_h/avg}, V: {s_v/avg}'
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        return (s_h/avg, s_v/avg)

    def exit_function():
        detector_h.close()
        detector_v.close()

    return sync_reading, exit_function, n_chan
