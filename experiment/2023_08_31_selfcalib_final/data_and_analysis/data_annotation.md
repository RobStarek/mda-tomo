## Main data

* `wp_3_4_13_14_cal8qwp.h5` - 
    * calibration of waveplates 3, 4, 13, 14 using 8 probe states. 
    * Dataset `iter_0_int` contains calibration data for projection waveplates (forward mode).
    * Dataset `iter_1_int` contains calibration data for preparation waveplates (reversed mode). 
    * The calibration was done with QWP rotated to cca -12 deg in between of preparation and projection.
    * Measurement tables (sequence of waveplate angles) is saved in dataset `iter_{i}_move` (in degrees).
    * this file is analyzed with `analyze_calibration_data.ipynb` and retardance deviation is extracted and then tested using 58 probe states, saved in file `wp_3_4_13_14_chech_58.h5`.
    * Each dataset contains 8x6 intensity readings, first entry in each line is reading from transmissive port and the second entry is reading from reflective port of the analyzer PBS.

* `wp_3_4_13_14_check_58.h5` -
    * test of discovered retardances of waveplates 3, 4, 13, and 14. 
    * Dataset `iter_0_int` is reference - perfect waveplates are asssumed.
    * Dataset `iter_1_int` contains test data - retardance was taken into account.    
    * **These data to be presented in paper.**
    * `iter_2_int` and `iter_3_int` dataset contain uncalibrated and calibrated process tomography of free space, respectively. This is a benchmark of calibration.


## Supplementary data

* `wp_1_2_5_6_cal8_90.h5` -
    * calibration data for waveplates 1, 2, 5, 6
    * it is organized the same way as `wp_3_4_13_14_cal8qwp.h5`
    * Here we added 90 degrees f0r angular calibration of quarter-wave plates.
    * It turns out that when we get the fast/slow axis in the disagreement with the model, the calibration fail.
    * This is demonstrated in `wp_1_2_5_6_check_8_90.h5`, where calibrated results are worse than the reference.

* `wp_1_2_5_6_cal8_0.h5` -
    * calibration data for waveplates 1, 2, 5, 6
    * it is organized the same way as `wp_3_4_13_14_cal8qwp.h5`
    * We kept the original angular zero positons of waveplates. 
    * It turns out that when we get the fast/slow axis in the agreement with the model, the calibration works.
    * This is demonstrated in `wp_1_2_5_6_check_8_0.h5`, where calibrated results are better than the reference.

* `depol_wp_1_2_5_6_cal_8.h5` - 
    * input beam was depolarized by partial insertion of the HWP rotated at 45
    * the resulting beam input into the two preparation waveplates was therefore a mixture of H and V
    * the result is similar up to 

* `wp_3_4_13_14_c_proc_qwp.h5` and `wp_3_4_13_14_n_proc_qwp.h5`
    * Process tomography of quarter wave plate that was used in between preparation and projection.
    * It shows that with calibration, the results are better.



