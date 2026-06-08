#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import math

import numpy.ma as ma

import spectrum_functions as sf
import bin_plots as bp
import binning_functions as bf


try:
    from pyplatefit import fit_spec,Platefit, Linefit, plot_fit
    HAS_PYPLATEFIT = True
    print("[INFO] pyplatefit detected. Running with full spectral fitting mode.")
except ImportError:
    HAS_PYPLATEFIT = False
    print("[WARNING] pyplatefit is not installed.")
    print("To proceed without it, you must provide:")
    print("  1. An emission line list (e.g., ['OII3727b', 'HGAMMA', 'HBETA', 'OIII5007'])")
    print("  2. Observed wavelengths for each line")
    print("  3. A wavelength window size (Recommendation: 2x the observed FWHM in the central spaxel)\n")
    print("Open the .py file and input these!")


def bin_spaxels(cube,dec_center,ra_center, input_redshift,sn_threshold, rd, pa,inc,n_annuli, binmap_savefig_location,
                observed_emission_lines=None, observed_wavelengths=None, observed_FWHM=None):

    #Central Spectrum for getting wavelength windows
    y_center,x_center=cube.wcs.sky2pix((dec_center,ra_center),nearest=True)[0]
    sp_best = cube[:,y_center,x_center]

    # Check if pyplatefit is installed
    if HAS_PYPLATEFIT:

        print("[INFO] Running bin_spaxels in Mode A (pyplatefit spectral fitting).")
        #Get global spectrum; used to check lines with highest SN
        em_lines,fit_spec_global=sf.global_spec_em(cube,input_redshift,dec_center,ra_center)
        #Decide line_set
        if len(em_lines)==4:
            line_1,line_2,line_3,line_4=em_lines
            line_set=0
        if len(em_lines)==5:
            line_1,line_2,line_3,line_4,line_5=em_lines
            line_set=1

        # If pyplatefit is installed
        res_best_1 = fit_spec(sp_best,z = input_redshift, ziter=True, fit_all=False)
        line_tab = res_best_1['lines']
        sp_best=res_best_1['line_spec']

        #Display the fits
        res_best = res_best_1

        #Get wavelength and FWHM of lines
        LBDA_OBS_line_1,FWHM_OBS_line_1,LBDA_LEFT_line_1,LBDA_RIGHT_line_1 = sf.get_line_obs_wl(line_tab,line_1)
        LBDA_OBS_line_2,FWHM_OBS_line_2,LBDA_LEFT_line_2,LBDA_RIGHT_line_2 = sf.get_line_obs_wl(line_tab,line_2)
        LBDA_OBS_line_3,FWHM_OBS_line_3,LBDA_LEFT_line_3,LBDA_RIGHT_line_3 = sf.get_line_obs_wl(line_tab,line_3)
        LBDA_OBS_line_4,FWHM_OBS_line_4,LBDA_LEFT_line_4,LBDA_RIGHT_line_4= sf.get_line_obs_wl(line_tab,line_4)

        if line_set==1:
            LBDA_OBS_line_5,FWHM_OBS_line_5,LBDA_LEFT_line_5,LBDA_RIGHT_line_5 = sf.get_line_obs_wl(line_tab,line_5)



        LBDA_LEFT_line_1 = sp_best.wave.pixel(LBDA_LEFT_line_1, nearest=True, unit=None)
        LBDA_LEFT_line_2 =sp_best.wave.pixel(LBDA_LEFT_line_2, nearest=True, unit=None)
        LBDA_LEFT_line_3=sp_best.wave.pixel(LBDA_LEFT_line_3, nearest=True, unit=None)
        LBDA_LEFT_line_4 =sp_best.wave.pixel(LBDA_LEFT_line_4, nearest=True, unit=None)



        LBDA_RIGHT_line_1 = sp_best.wave.pixel(LBDA_RIGHT_line_1, nearest=True, unit=None)
        LBDA_RIGHT_line_2 = sp_best.wave.pixel(LBDA_RIGHT_line_2, nearest=True, unit=None)
        LBDA_RIGHT_line_3 = sp_best.wave.pixel(LBDA_RIGHT_line_3, nearest=True, unit=None)
        LBDA_RIGHT_line_4 = sp_best.wave.pixel(LBDA_RIGHT_line_4, nearest=True, unit=None)



        if line_set==1:
            line_center_line_5 =sp_best.wave.pixel(LBDA_OBS_line_5, nearest=True, unit=None)
            LBDA_LEFT_line_5 = sp_best.wave.pixel(LBDA_LEFT_line_5, nearest=True, unit=None)
            LBDA_RIGHT_line_5 = sp_best.wave.pixel(LBDA_RIGHT_line_5, nearest=True, unit=None)


    else:
        print("[WARNING] pyplatefit is not installed. Switching to Mode B (Window Fallback).")
        print("To proceed without it, you must provide:")
        print("  1. An emission line list (e.g., ['OII3727b', 'HGAMMA', 'HBETA', 'OIII5007'])")
        print("  2. Observed wavelengths for each line")
        print("  3. A wavelength window size (Recommendation: 2x the observed FWHM in the central spaxel)\n")

        # Enforce that the user MUST provide the fallback arguments
        if (observed_emission_lines is None or
            observed_wavelengths is None or
            observed_FWHM is None):
            raise ValueError(
                "Missing required parameters for Fallback Mode!\n"
                "Since pyplatefit is not installed, you must provide:\n"
                "  - observed_emission_lines\n"
                "  - observed_wavelengths\n"
                "  - observed_FWHM"
            )

        print("[INFO] Validated fallback inputs. Computing window sizes from FWHM...")
        em_lines=observed_emission_lines
        LBDA_OBS_line_1,LBDA_OBS_line_2,LBDA_OBS_line_3,LBDA_OBS_line_4 = observed_wavelengths
        FWHM_OBS_line_1,FWHM_OBS_line_2,FWHM_OBS_line_3,FWHM_OBS_line_4 = observed_FWHM

        #Decide line_set
        if len(em_lines)==4:
            line_1,line_2,line_3,line_4=em_lines
            line_set=0
        if len(em_lines)==5:
            line_1,line_2,line_3,line_4,line_5=em_lines
            line_set=1









####################
    line_center_line_1 = sp_best.wave.pixel(LBDA_OBS_line_1, nearest=True, unit=None)
    line_center_line_2 =sp_best.wave.pixel(LBDA_OBS_line_2, nearest=True, unit=None)
    line_center_line_3=sp_best.wave.pixel(LBDA_OBS_line_3, nearest=True, unit=None)
    line_center_line_4 =sp_best.wave.pixel(LBDA_OBS_line_4, nearest=True, unit=None)




    #Get window size
    window_size_line_1 =  int(np.round(FWHM_OBS_line_1/1.25))
    window_size_line_2 =  int(np.round(FWHM_OBS_line_2/1.25))
    window_size_line_3 = int(np.round(FWHM_OBS_line_3/1.25))
    window_size_line_4 =  int(np.round(FWHM_OBS_line_4/1.25))

    if line_set==1:
        window_size_line_5 =  int(np.round(FWHM_OBS_line_5/1.25))
########
    if line_set==0:
        line_center_line_5,window_size_line_5,LBDA_OBS_line_5 = 0,0,0


    if line_set==1:
        test_data = bf.sub_cube_sn_sii(cube,ra_center, dec_center, pa, inc,line_center_line_1,line_center_line_2,line_center_line_3,line_center_line_4,line_center_line_5,window_size_line_1,window_size_line_2,window_size_line_3,window_size_line_4,window_size_line_5,LBDA_OBS_line_4,LBDA_OBS_line_5,FWHM_OBS_line_4,FWHM_OBS_line_5,line_set)

    if line_set==0:
        test_data = bf.sub_cube_sn(cube,ra_center, dec_center, pa, inc,line_center_line_1,line_center_line_2,line_center_line_3,line_center_line_4,window_size_line_1,window_size_line_2,window_size_line_3,window_size_line_4,line_set)

#############
    r_max = rd
    n_annuli=n_annuli
    x, y, radius, theta, r_in, r_out, d_theta,I = bf.calc_bin_positions(r_max, n_annuli)


    cutoff_sn = 0.
    test_data_sn_line_1 = test_data[:, -7]
    test_data_sn_line_2 = test_data[:, -6]
    test_data_sn_line_3 = test_data[:, -5]
    test_data_sn_line_4 = test_data[:, -4]



    #---------Initialization--------------
    # Set max delta r and delta theta; set delta r and delta theta
    delta_r_max, delta_theta_max = rd, math.radians(390)

    delta_r_arr = radius

    delta_theta_arr = theta

    spx_to_bin = test_data

    #Initialize binmap
    final_binmap = []

    #Mark SN above threshold
    spx_sn_line_1 = spx_to_bin[:, -7]
    spx_sn_line_2 = spx_to_bin[:, -6]
    spx_sn_line_3 = spx_to_bin[:, -5]
    spx_sn_line_4 = spx_to_bin[:, -4]


    spx_all_lines_above_thresh = (spx_sn_line_1>sn_threshold) & (spx_sn_line_2>sn_threshold)&(spx_sn_line_3>sn_threshold) & (spx_sn_line_4>sn_threshold)

    """(i) Loop over all spaxels individually. Perform spectral fitting on each. If the spaxel’s S/N
    is above the set threshold, assign it a unqiue bin ID number remove spaxel from future
    binning."""

    spx_thresh = np.where(spx_all_lines_above_thresh)


    spx_to_bin[:,-2][spx_thresh] = 0 #binhow

    spx_to_bin[:,-1][spx_thresh] = 1#set as already binned

    #Get x,y coordinates of each bin
    x_hsn,y_hsn = spx_to_bin[spx_thresh][:,0],spx_to_bin[spx_thresh][:,1]
    #Get r, theta coordinates of each bin
    r_hsn,theta_hsn  = spx_to_bin[spx_thresh][:,2],spx_to_bin[spx_thresh][:,3]
    #Get SN of each line
    sn_hsn_line_1 = spx_to_bin[spx_thresh][:,-7]
    sn_hsn_line_2 = spx_to_bin[spx_thresh][:,-6]
    sn_hsn_line_3 = spx_to_bin[spx_thresh][:,-5]
    sn_hsn_line_4 = spx_to_bin[spx_thresh][:,-4]


    #Get flux and variance for each spaxel for the 4 lines
    fl_hsn_line_1 = spx_to_bin[spx_thresh][:,4]
    var_hsn_line_1 = spx_to_bin[spx_thresh][:,5]

    fl_hsn_line_2 = spx_to_bin[spx_thresh][:,6]
    var_hsn_line_2 = spx_to_bin[spx_thresh][:,7]

    fl_hsn_line_3 = spx_to_bin[spx_thresh][:,8]
    var_hsn_line_3 = spx_to_bin[spx_thresh][:,9]

    fl_hsn_line_4 = spx_to_bin[spx_thresh][:,10]
    var_hsn_line_4 = spx_to_bin[spx_thresh][:,11]


    bin_no =0
    # Loop through the indices of the arrays
    for i in range(len(x_hsn)):
        # Append the values as a list to final_bin_map
        bin_no = bin_no+1
        final_binmap.append([bin_no,[[x_hsn[i], y_hsn[i]]],[[r_hsn[i],theta_hsn[i]]] ,sn_hsn_line_1[i],sn_hsn_line_2[i],sn_hsn_line_3[i],sn_hsn_line_4[i], fl_hsn_line_1[i], var_hsn_line_1[i],fl_hsn_line_2[i], var_hsn_line_2[i],fl_hsn_line_3[i], var_hsn_line_3[i],fl_hsn_line_4[i], var_hsn_line_4[i],0])

    #This section is the binning loop
    """(ii) For each remaining unbinned spaxel, coadd the spaxel with other spaxels within ∆r
    and ∆θ of the spaxel’s coordinates. (∆r and ∆θ define some initial bin size in radial
    coordinates.) Perform spectral fitting on the coadded spectrum and record the S/N of the
    weakest emission line in this bin.

    (iii) Find the bin with the lowest S/N, but still above the S/N threshold. Assign these spaxels
    with a bin ID number, and remove them from future binning.

    (iv) Repeat above two steps (ii) & (iii) until there are no bins above threshold.

    (v) Increase ∆r and/or ∆θ (i.e. increase bin size) and goto step (ii). These increases follow
    some predefined sequence. Once ∆r and/or ∆θ reach a maximum size limit, continue to
    next step."""

    a=0

    delta_r,delta_theta = delta_r_arr[a],delta_theta_arr[a]
    b=0
    while (np.shape(np.where(spx_to_bin[:, -1] == -1))[1] != 0) and ((delta_r < delta_r_max) and (delta_theta < delta_theta_max)):
    # For a delta r and delta theta, find unbinned neighbors


        lowest_sn_idx_temp_bin=0

        while lowest_sn_idx_temp_bin!=-1:
            # Create bins for a delta r and delta theta
            temp_bin=bf.create_bins(spx_to_bin, delta_r, delta_theta, sn_threshold)#Excludes already binned spaxels when finding neighbors

            # Find bin with lowest SN and flag them in spx_to_bin
            lowest_sn_idx_temp_bin=bf.lowest_sn_bin(temp_bin,sn_threshold)#lowest_sn_idx in the temp_bin

            if lowest_sn_idx_temp_bin!=-1:
                #Go to -2th element of temp_bin to find the lowest_sn_idx in spx_to_bin
                lowest_sn_idx_spx_to_bin=temp_bin[lowest_sn_idx_temp_bin][-2]# k value


                #index of central spaxel in spx_to_bin
                central_spaxel_idx = lowest_sn_idx_spx_to_bin #it is the k value... the current index in the loop

                #indices of bin neighbors in spx_to_bin: 2nd element of the temporary
                bin_neighbors = temp_bin[lowest_sn_idx_temp_bin][2]


                #Mark the spaxels as binned
                # Mark the central spaxel and its neighbors as binned
                spx_to_bin[central_spaxel_idx, -1] = 1  # Central spaxel
                spx_to_bin[bin_neighbors, -1] = 1  # Neighbors
                #Bin how IDs
                spx_to_bin[central_spaxel_idx, -2] = 1  # Central spaxel
                spx_to_bin[bin_neighbors, -2] = 1  # Neighbors

                #Add bins to final_binmap
                bin_no = bin_no+1
                coords_bin_r_theta = temp_bin[lowest_sn_idx_temp_bin][0]
                coords_bin = temp_bin[lowest_sn_idx_temp_bin][1]

                fl_bin_line_1 =  temp_bin[lowest_sn_idx_temp_bin][11]
                var_bin_line_1 = temp_bin[lowest_sn_idx_temp_bin][12]

                fl_bin_line_2 =  temp_bin[lowest_sn_idx_temp_bin][13]
                var_bin_line_2 = temp_bin[lowest_sn_idx_temp_bin][14]

                fl_bin_line_3 =  temp_bin[lowest_sn_idx_temp_bin][15]
                var_bin_line_3 = temp_bin[lowest_sn_idx_temp_bin][16]

                fl_bin_line_4 =  temp_bin[lowest_sn_idx_temp_bin][17]
                var_bin_line_4 = temp_bin[lowest_sn_idx_temp_bin][18]


                SN_bin_line_1 =  temp_bin[lowest_sn_idx_temp_bin][-6]
                SN_bin_line_2 =  temp_bin[lowest_sn_idx_temp_bin][-5]
                SN_bin_line_3 =  temp_bin[lowest_sn_idx_temp_bin][-4]
                SN_bin_line_4 =  temp_bin[lowest_sn_idx_temp_bin][-3]
                #print(bin_no)
                #bin_status
                final_binmap.append([bin_no,coords_bin,coords_bin_r_theta,SN_bin_line_1,SN_bin_line_2,SN_bin_line_3,SN_bin_line_4, fl_bin_line_1, var_bin_line_1,fl_bin_line_2, var_bin_line_2,fl_bin_line_3, var_bin_line_3,fl_bin_line_4, var_bin_line_4,1])

        a = a+1
        #print(a)
        if (a==len(delta_r_arr)):
            print('End of arrray')
            break
        else:
            delta_r,delta_theta = delta_r_arr[a],delta_theta_arr[a]


    # Get still unbinned spaxels
    # They will go for the last step of accretion
    unbinned_idx=np.where(spx_to_bin[:,-2]==-1)[0]
    x_unbin,y_unbin = spx_to_bin[unbinned_idx][:,0],spx_to_bin[unbinned_idx][:,1]

    r_unbinned,theta_unbinned = spx_to_bin[unbinned_idx][:,2],spx_to_bin[unbinned_idx][:,3]

    fl_unbin_line_1 = spx_to_bin[unbinned_idx][:,4]
    var_unbin_line_1 = spx_to_bin[unbinned_idx][:,5]

    fl_unbin_line_2 = spx_to_bin[unbinned_idx][:,6]
    var_unbin_line_2 = spx_to_bin[unbinned_idx][:,7]

    fl_unbin_line_3 = spx_to_bin[unbinned_idx][:,8]
    var_unbin_line_3 = spx_to_bin[unbinned_idx][:,9]

    fl_unbin_line_4 = spx_to_bin[unbinned_idx][:,10]
    var_unbin_line_4 = spx_to_bin[unbinned_idx][:,11]



    sn_unbin_line_1 = spx_to_bin[unbinned_idx][:,-7]
    sn_unbin_line_2 = spx_to_bin[unbinned_idx][:,-6]
    sn_unbin_line_3 = spx_to_bin[unbinned_idx][:,-5]
    sn_unbin_line_4 = spx_to_bin[unbinned_idx][:,-4]

    bin_no = -1


    # Loop through the indices of the arrays
    for i in range(len(x_unbin)):
        # Append the values as a list to final_bin_map
        final_binmap.append([bin_no,[[x_unbin[i], y_unbin[i]]], [[r_unbinned[i],theta_unbinned[i]]],sn_unbin_line_1[i],sn_unbin_line_2[i],sn_unbin_line_3[i],sn_unbin_line_4[i], fl_unbin_line_1[i], var_unbin_line_1[i],fl_unbin_line_2[i], var_unbin_line_2[i],fl_unbin_line_3[i], var_unbin_line_3[i],fl_unbin_line_4[i], var_unbin_line_4[i],-1])


    data_accr = np.array(final_binmap,dtype='object')


    #Calculate mean r

    # Loop through each bin's information
    for bin_data in data_accr:

        # Extract the r, theta coordinates
        r_theta_coords = bin_data[2]

        # Calculate the mean of r (first element of each pair in the list)
        mean_r = np.mean([coord[0] for coord in r_theta_coords])

        # Replace the second element with the mean of r, wrapped in a list
        bin_data[2] = mean_r  # Wrap the mean of r in a list


    """(vi) For each remaining unbinned spaxel. Accrete the spaxel to the nearest bin at a greater
    radius than it. If the S/N of the new bin is greater than previous then record the new bin.
    Otherwise discard the spaxel and leave the bin unchanged."""


    # The part finds neighbor spaxels within delta r and delta theta of each unbinned spaxel
    bin_no = data_accr[:, 0]
    xy_bin_coords = data_accr[:, 1]
    r_mean_bin_coords = data_accr[:, 2]



    sn_bin_line_1 = data_accr[:,3]
    sn_bin_line_2 = data_accr[:,4]
    sn_bin_line_3 = data_accr[:,5]
    sn_bin_line_4 = data_accr[:,6]


    fl_bin_line_1= data_accr[:,7]
    var_bin_line_1= data_accr[:,8]

    fl_bin_line_2= data_accr[:,9]
    var_bin_line_2= data_accr[:,10]

    fl_bin_line_3= data_accr[:,11]
    var_bin_line_3= data_accr[:,12]

    fl_bin_line_4= data_accr[:,13]
    var_bin_line_4= data_accr[:,14]






    flag_bin= data_accr[:,-1]
    rows_to_remove =[]

    for k in range(len(data_accr)):
        if (flag_bin[k] == -1):
            r_unbin = r_mean_bin_coords[k]
            #find bins at greater radius
            idx_bins_greater_radius = np.where((bin_no !=-1)&(r_mean_bin_coords>r_unbin))
            if len(data_accr[idx_bins_greater_radius])!=0: #if such bin exists:
                #Find all bins greater than r unbin
                bins_greater_radius = data_accr[idx_bins_greater_radius]
                #Out of all bins greater than r unbin,find the nearest one
                nearest_bin_idx=np.argmin(bins_greater_radius[:,2])

                nearest_bin = bins_greater_radius[nearest_bin_idx]

                xy_spaxel_unbin = xy_bin_coords[k]
                #fluxes and variances
                fl_spaxel_unbin_line_1 = fl_bin_line_1[k]
                var_spaxel_unbin_line_1 = var_bin_line_1[k]

                fl_spaxel_unbin_line_2 = fl_bin_line_2[k]
                var_spaxel_unbin_line_2 = var_bin_line_2[k]

                fl_spaxel_unbin_line_3 = fl_bin_line_3[k]
                var_spaxel_unbin_line_3 = var_bin_line_3[k]

                fl_spaxel_unbin_line_4 = fl_bin_line_4[k]
                var_spaxel_unbin_line_4 = var_bin_line_4[k]

                #SN of lines
                sn_spaxel_unbin_line_1 = sn_bin_line_1[k]
                sn_spaxel_unbin_line_2 = sn_bin_line_2[k]
                sn_spaxel_unbin_line_3 = sn_bin_line_3[k]
                sn_spaxel_unbin_line_4 = sn_bin_line_4[k]

                #bin data
                bin_no_nearest_bin = nearest_bin[0]

                fl_nearest_bin_line_1 = nearest_bin[7]
                var_nearest_bin_line_1 = nearest_bin[8]

                fl_nearest_bin_line_2 = nearest_bin[9]
                var_nearest_bin_line_2 = nearest_bin[10]

                fl_nearest_bin_line_3 = nearest_bin[11]
                var_nearest_bin_line_3 = nearest_bin[12]

                fl_nearest_bin_line_4 = nearest_bin[13]
                var_nearest_bin_line_4 = nearest_bin[14]

                sn_nearest_bin_line_1 = nearest_bin[3]
                sn_nearest_bin_line_2 = nearest_bin[4]
                sn_nearest_bin_line_3 = nearest_bin[5]
                sn_nearest_bin_line_4 = nearest_bin[6]

                xy_bin_coords_nearest_bin = nearest_bin[1]

                #Coadded flux and variance
                fl_new_line_1 = fl_spaxel_unbin_line_1+fl_nearest_bin_line_1
                var_new_line_1 = var_spaxel_unbin_line_1+var_nearest_bin_line_1

                fl_new_line_2 = fl_spaxel_unbin_line_2+fl_nearest_bin_line_2
                var_new_line_2 = var_spaxel_unbin_line_2+var_nearest_bin_line_2

                fl_new_line_3 = fl_spaxel_unbin_line_3+fl_nearest_bin_line_3
                var_new_line_3 = var_spaxel_unbin_line_3+var_nearest_bin_line_3

                fl_new_line_4 = fl_spaxel_unbin_line_4+fl_nearest_bin_line_4
                var_new_line_4 = var_spaxel_unbin_line_4+var_nearest_bin_line_4

                #SN of bins
                sn_new_line_1 = fl_new_line_1/np.sqrt(var_new_line_1)
                sn_new_line_2 = fl_new_line_2/np.sqrt(var_new_line_2)
                sn_new_line_3 = fl_new_line_3/np.sqrt(var_new_line_3)
                sn_new_line_4 = fl_new_line_4/np.sqrt(var_new_line_4)

                if (sn_new_line_1 > sn_nearest_bin_line_1) and (sn_new_line_2 > sn_nearest_bin_line_2) and (sn_new_line_3 > sn_nearest_bin_line_3) and (sn_new_line_4 > sn_nearest_bin_line_4):
                    print('Accreting spaxels...')
                    #At which index of data_accr should I add the spaxels
                    add_spx_at = np.where(data_accr[:,0]==bin_no_nearest_bin)[0][0]
                    #print(' ')
                    print('Adding spaxel',xy_bin_coords[k][0],'at index',k,'to bin',bin_no_nearest_bin,'at index',add_spx_at,'of data_accr')
                    data_accr[add_spx_at][1].append(xy_bin_coords[k][0])

                    #fluxes and variance
                    data_accr[add_spx_at][7] = fl_new_line_1
                    data_accr[add_spx_at][8] = var_new_line_1
                    data_accr[add_spx_at][9] = fl_new_line_2
                    data_accr[add_spx_at][10] = var_new_line_2
                    data_accr[add_spx_at][11] = fl_new_line_3
                    data_accr[add_spx_at][12]= var_new_line_2
                    data_accr[add_spx_at][13]= fl_new_line_4
                    data_accr[add_spx_at][14]= var_new_line_4


                    #SN
                    data_accr[add_spx_at][3] = sn_new_line_1
                    data_accr[add_spx_at][4] = sn_new_line_2
                    data_accr[add_spx_at][5] = sn_new_line_3
                    data_accr[add_spx_at][6] = sn_new_line_4
                    data_accr[add_spx_at][-1]=2
                    print(' ')
                    rows_to_remove.append(k)



    data_accr=np.delete(data_accr, rows_to_remove, axis=0)
    binmap_before_acc=np.array(final_binmap,dtype='object')
    binmap_after_acc=np.array(data_accr,dtype='object')
    print('final_binmap contains', len(data_accr), 'bins')
    # Call the function
    bf.check_unique_pixel_coordinates(binmap_before_acc)
    bf.check_unique_pixel_coordinates(binmap_after_acc)
    bin_sn_check=binmap_after_acc[binmap_after_acc[:,0]!=-1]
    bp.plot_combined_binmap_from_my_list(binmap_after_acc,binmap_savefig_location)

    print("Done!")

    return binmap_after_acc
