import numpy as np
from spectrum_functions import extract_spectrum,extract_spectrum_doublet
import astropy.units as u

def calc_bin_positions(r_max, n_annuli):
    r_edge = np.linspace(0., r_max, n_annuli+1, dtype=float)
    r = (r_edge[:-1] + r_edge[1:]) / 2.
    r_in = r_edge[:-1]
    r_out = r_edge[1:]

    n = 6 * (np.arange(n_annuli, dtype=int) + 1)
    radius = np.repeat(r, n)
    N = np.repeat(n, n)
    r_in = np.repeat(r_in, n)
    r_out = np.repeat(r_out, n)

    I = np.concatenate([np.arange(i, dtype=float) for i in n])
    d_theta = 2. * np.pi / N
    theta = I * d_theta

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y, radius, theta, r_in, r_out, d_theta,I

def compute_galactocentric_coords(cube,x, y, ra_center_deg, dec_center_deg, pa_deg, inc_deg):
    coord_sky = cube.wcs.pix2sky([y, x], unit=u.deg)#gives dec,ra not ra,dec
    dec_deg, ra_deg = coord_sky[0][0], coord_sky[0][1]#ra,dec are in degrees

    ra_arcsec = ra_deg *3600.
    dec_arcsec = dec_deg*3600.
    ra_center_arcsec = ra_center_deg*3600.
    dec_center_arcsec = dec_center_deg*3600.
    pa_arcsec = pa_deg*3600.
    inc_arcsec = inc_deg*3600.

    dec_center_rad=dec_center_deg*(np.pi/180)
    pa_rad = pa_deg*(np.pi/180.)
    inc_rad = inc_deg*(np.pi/180.)

    # Convert to radians
    #ra, dec, ra_center, dec_center, pa, inc = map(np.radians, [ra, dec, ra_center, dec_center, pa, inc])

    # Compute angular offsets
    delta_alpha = (ra_arcsec- ra_center_arcsec) * np.cos(dec_center_rad)
    delta_delta = (dec_arcsec - dec_center_arcsec)


    # Cartesian coordinates in the sky plane
    x_sky = delta_alpha# in arcsec
    y_sky = delta_delta # in arcsec


    # Rotate by position angle
    x_prime = x_sky * np.cos(pa_rad) + y_sky * np.sin(pa_rad) #arcsec
    y_prime = -x_sky * np.sin(pa_rad) + y_sky * np.cos(pa_rad)#arcsec

    # Correct for inclination
    x_gal = x_prime # in arcsec
    y_gal = y_prime / np.cos(inc_rad)# in arcsec

    # Compute galactocentric radius and azimuthal angle
    r = np.sqrt(x_gal**2 + y_gal**2) #in arcsec
    # Handle floating-point precision for theta
    epsilon = 1e-10  # Threshold for "zero" (adjust based on your data scale)
    if np.abs(x_gal) < epsilon and np.abs(y_gal) < epsilon:
        theta = 0.0  # Explicitly set theta=0 at the center
    else:
        theta = np.arctan2((y_gal/3600)*np.pi/180, (x_gal/3600)*np.pi/180) #theta is in radians


    return (r, theta), (x_gal,y_gal)



def calc_mean_flux_sn(spec):
    # Get flux and variance from the spectrum data
    flux = spec.data
    flux_var = spec.var

    mask1 = (flux_var>0)# & (flux>0)

    flux = flux[mask1]
    flux_var =flux_var[mask1]
    # Create a mask where both flux and flux_var are not NaN
    mask = ~np.isnan(flux) & ~np.isnan(flux_var)

    # Apply the mask to flux and flux_var
    valid_flux = flux[mask]
    valid_flux_var = flux_var[mask]
    #if len(mask) == len(flux):
        #print('bad line')
    # Calculate average flux and variance only for valid pixels
    average_flux = np.mean(valid_flux)#np.mean(valid_flux)
    average_variance = np.mean(valid_flux_var)#np.mean(valid_flux_var)

    # Calculate signal-to-noise ratio
    signal = average_flux
    noise = np.sqrt(average_variance)
    sn = signal / noise

    return average_flux, average_variance, sn



#Take cube as input and give spaxel coordinates, spectrum object and S/N

def sub_cube_sn(cube,ra_center, dec_center, pa, inc,line_center_line_1,line_center_line_2,line_center_line_3,line_center_line_4,window_size_line_1,window_size_line_2,window_size_line_3,window_size_line_4,line_set):
    # Generate spaxel coordinates
    xx, yy = np.meshgrid(np.arange(cube.shape[2]),
                     np.arange(cube.shape[1]))
    #SN -ve flag
    sn_flag = 0
    sp_coords = np.column_stack((xx.ravel(), yy.ravel()))
    # Initialize a list to store combined data
    combined_data_spec = []
    # Loop over all spaxels
    for coord in sp_coords:
        x_in, y_in = coord
        if (line_center_line_1 == None) and (line_center_line_2 == None) and (line_center_line_3 == None) and (line_center_line_4 == None):
            print('Extracting the full spectrum for the spaxel')
            print('All line center values should be given!')
            spec = cube[:, y_in, x_in]  # Extract signal array for spaxel
        else:
            if line_set==1:
                spec_line_1 = extract_spectrum(cube,x_in,y_in,line_center_line_1,window_size_line_1)#
                lcll =line_center_line_4
                lclr =line_center_line_5
                wsll = window_size_line_4
                wslr = window_size_line_5
                wl_ll = LBDA_OBS_line_4
                wl_lr = LBDA_OBS_line_5
                fwhm_ll = FWHM_OBS_line_4
                fwhm_lr = FWHM_OBS_line_5
                spec_line_4 = extract_spectrum_doublet(cube,x_in,y_in,lcll,lclr,wsll,wslr,wl_ll,wl_lr,fwhm_ll,fwhm_lr)
            else:
                spec_line_1 = extract_spectrum(cube,x_in,y_in,line_center_line_1,window_size_line_1+2)#[OII] 3727b
                spec_line_4 = extract_spectrum(cube,x_in,y_in,line_center_line_4,window_size_line_4)
            spec_line_2 = extract_spectrum(cube,x_in,y_in,line_center_line_2,window_size_line_2)
            spec_line_3 = extract_spectrum(cube,x_in,y_in,line_center_line_3,window_size_line_3)





        flux_line_1, variance_line_1, sn_line_1 = calc_mean_flux_sn(spec_line_1)
        flux_line_2, variance_line_2, sn_line_2 = calc_mean_flux_sn(spec_line_2)
        flux_line_3, variance_line_3, sn_line_3 = calc_mean_flux_sn(spec_line_3)
        flux_line_4, variance_line_4, sn_line_4 = calc_mean_flux_sn(spec_line_4)


        if (sn_line_1<0) and (sn_line_2<0) and (sn_line_3<0) and (sn_line_4<0):
            sn_flag = -1
        else:
            sn_flag = 0
        r = compute_galactocentric_coords(cube,x_in, y_in, ra_center, dec_center, pa, inc)[0][0]
        theta = compute_galactocentric_coords(cube,x_in, y_in, ra_center, dec_center, pa, inc)[0][1]
        #print(x_in,y_in)
        # Append the data
        combined_data_spec.append([x_in, y_in, r, theta, flux_line_1, variance_line_1, flux_line_2, variance_line_2, flux_line_3, variance_line_3, flux_line_4, variance_line_4, sn_line_1,sn_line_2,sn_line_3,sn_line_4,sn_flag,-1,-1])
    return np.array(combined_data_spec,dtype='object')

#Important... the function should:
#1) Exclude already binned spaxels when finding neighbors. Otherwise, a spaxel can will belong to multiple bins!
#This ensured by (bin_flag != 1)

#2) Exclude the central spaxel from its neighbors. This problem ocurs because r_coords includes all spaxels.
# The command np.arange(len(spx_to_bin)) != k) takes care of it


#I made these changes to Binning_algorithm_in_steps-part6_accreting on my laptop on 1/23/25
def sub_cube_sn_sii(cube,ra_center, dec_center, pa, inc,line_center_line_1,line_center_line_2,line_center_line_3,line_center_line_4,line_center_line_5,window_size_line_1,window_size_line_2,window_size_line_3,window_size_line_4,window_size_line_5,LBDA_OBS_line_4,LBDA_OBS_line_5,FWHM_OBS_line_4,FWHM_OBS_line_5,line_set):
    # Generate spaxel coordinates
    xx, yy = np.meshgrid(np.arange(cube.shape[2]),
                     np.arange(cube.shape[1]))
    #SN -ve flag
    sn_flag = 0
    sp_coords = np.column_stack((xx.ravel(), yy.ravel()))
    # Initialize a list to store combined data
    combined_data_spec = []
    # Loop over all spaxels
    for coord in sp_coords:
        x_in, y_in = coord
        if (line_center_line_1 == None) and (line_center_line_2 == None) and (line_center_line_3 == None) and (line_center_line_4 == None):
            print('Extracting the full spectrum for the spaxel')
            print('All line center values should be given!')
            spec = cube[:, y_in, x_in]  # Extract signal array for spaxel
        else:
            if line_set==1:
                spec_line_1 = extract_spectrum(cube,x_in,y_in,line_center_line_1,window_size_line_1)#
                lcll =line_center_line_4
                lclr =line_center_line_5
                wsll = window_size_line_4
                wslr = window_size_line_5
                wl_ll = LBDA_OBS_line_4
                wl_lr = LBDA_OBS_line_5
                fwhm_ll = FWHM_OBS_line_4
                fwhm_lr = FWHM_OBS_line_5
                spec_line_4 = extract_spectrum_doublet(cube,x_in,y_in,lcll,lclr,wsll,wslr,wl_ll,wl_lr,fwhm_ll,fwhm_lr)
                spec_line_2 = extract_spectrum(cube,x_in,y_in,line_center_line_2,window_size_line_2)
                spec_line_3 = extract_spectrum(cube,x_in,y_in,line_center_line_3,window_size_line_3)
            else:
                print('This function only works for line_set = 1')
                #spec_line_1 = extract_spectrum(cube,x_in,y_in,line_center_line_1,window_size_line_1+2)#[OII] 3727b
                #spec_line_4 = extract_spectrum(cube,x_in,y_in,line_center_line_4,window_size_line_4)
        flux_line_1, variance_line_1, sn_line_1 = calc_mean_flux_sn(spec_line_1)
        flux_line_2, variance_line_2, sn_line_2 = calc_mean_flux_sn(spec_line_2)
        flux_line_3, variance_line_3, sn_line_3 = calc_mean_flux_sn(spec_line_3)
        flux_line_4, variance_line_4, sn_line_4 = calc_mean_flux_sn(spec_line_4)


        if (sn_line_1<0) and (sn_line_2<0) and (sn_line_3<0) and (sn_line_4<0):
            sn_flag = -1
        else:
            sn_flag = 0
        r = compute_galactocentric_coords(cube,x_in, y_in, ra_center, dec_center, pa, inc)[0][0]
        theta = compute_galactocentric_coords(cube,x_in, y_in, ra_center, dec_center, pa, inc)[0][1]
        #print(x_in,y_in)
        # Append the data
        combined_data_spec.append([x_in, y_in, r, theta, flux_line_1, variance_line_1, flux_line_2, variance_line_2, flux_line_3, variance_line_3, flux_line_4, variance_line_4, sn_line_1,sn_line_2,sn_line_3,sn_line_4,sn_flag,-1,-1])
    return np.array(combined_data_spec,dtype='object')









# The function finds neighbor spaxels within delta r and delta theta of each unbinned spaxel
def create_bins(spx_to_bin, delta_r, delta_theta, sn_threshold):
    x_coords = spx_to_bin[:, 0]
    y_coords = spx_to_bin[:, 1]
    r_coords = spx_to_bin[:, 2]
    theta_coords = spx_to_bin[:, 3]
    spaxel_flag = spx_to_bin[:, -2]

    fl_avg_line_1 = spx_to_bin[:, 4]
    var_avg_line_1 = spx_to_bin[:, 5]

    fl_avg_line_2 = spx_to_bin[:, 6]
    var_avg_line_2 = spx_to_bin[:, 7]

    fl_avg_line_3 = spx_to_bin[:, 8]
    var_avg_line_3 = spx_to_bin[:, 9]

    fl_avg_line_4 = spx_to_bin[:, 10]
    var_avg_line_4 = spx_to_bin[:, 11]

    sn_spxs_line_1 = spx_to_bin[:, -7]
    sn_spxs_line_2 = spx_to_bin[:, -6]
    sn_spxs_line_3 = spx_to_bin[:, -5]
    sn_spxs_line_4 = spx_to_bin[:,-4]



    bin_flag = spx_to_bin[:, -1]
    bin_how = spx_to_bin[:, -2]
    bin_data = []

    for k in range(len(spx_to_bin)):
        if (bin_flag[k] == 1) and (bin_how[k] == 1):  # Skip already binned spaxels
            continue

        x, y, r, theta = x_coords[k], y_coords[k], r_coords[k], theta_coords[k]
        fl_line_1, var_line_1, sn_spx_line_1 = fl_avg_line_1[k], var_avg_line_1[k], sn_spxs_line_1[k]
        fl_line_2, var_line_2, sn_spx_line_2 = fl_avg_line_2[k], var_avg_line_2[k], sn_spxs_line_2[k]
        fl_line_3, var_line_3, sn_spx_line_3 = fl_avg_line_3[k], var_avg_line_3[k], sn_spxs_line_3[k]
        fl_line_4, var_line_4, sn_spx_line_4 = fl_avg_line_4[k], var_avg_line_4[k], sn_spxs_line_4[k]

        if bin_how[k] == 0:  # High S/N lines should be excluded
            #bin_data.append([r, theta, [[x, y]], [[x, y]], fl, var, fl, var, sn_spx, k, 0])
            continue
        # Find neighbors within delta_r and delta_theta
        #Exclude the central spaxel itself
        # Exclude already binned spaxels
        neigh_array = np.where((abs(r_coords - r) <= delta_r) &
                               (abs(theta_coords - theta) <= delta_theta) &
                               (np.arange(len(spx_to_bin)) != k)& (bin_flag != 1))

        if len(neigh_array[0]) == 0:#No neighbors
            #print('No neighbor for index',k,'in spx_to_bin')
            bin_data.append([[[r, theta]], [[x, y]], [], fl_line_1, var_line_1,fl_line_2, var_line_2,fl_line_3, var_line_3,fl_line_4, var_line_4, fl_line_1, var_line_1,fl_line_2, var_line_2,fl_line_3, var_line_3,fl_line_4, var_line_4, sn_spx_line_1,sn_spx_line_2,sn_spx_line_3,sn_spx_line_4, k, -1])
        else:
            flux_neigh_line_1 = np.sum(spx_to_bin[neigh_array[0], 4])
            var_neigh_line_1 = np.sum(spx_to_bin[neigh_array[0], 5])

            flux_neigh_line_2 = np.sum(spx_to_bin[neigh_array[0], 6])
            var_neigh_line_2 = np.sum(spx_to_bin[neigh_array[0], 7])

            flux_neigh_line_3 = np.sum(spx_to_bin[neigh_array[0], 8])
            var_neigh_line_3 = np.sum(spx_to_bin[neigh_array[0], 9])

            flux_neigh_line_4 = np.sum(spx_to_bin[neigh_array[0], 10])
            var_neigh_line_4 = np.sum(spx_to_bin[neigh_array[0], 11])

            coadd_flux_line_1 = fl_line_1 + flux_neigh_line_1
            coadd_var_line_1 = var_line_1 + var_neigh_line_1
            sn_coadd_line_1 = coadd_flux_line_1 / np.sqrt(coadd_var_line_1)

            coadd_flux_line_2 = fl_line_2 + flux_neigh_line_2
            coadd_var_line_2 = var_line_2 + var_neigh_line_2
            sn_coadd_line_2 = coadd_flux_line_2 / np.sqrt(coadd_var_line_2)

            coadd_flux_line_3 = fl_line_3 + flux_neigh_line_3
            coadd_var_line_3 = var_line_3 + var_neigh_line_3
            sn_coadd_line_3 = coadd_flux_line_3 / np.sqrt(coadd_var_line_3)

            coadd_flux_line_4 = fl_line_4 + flux_neigh_line_4
            coadd_var_line_4 = var_line_4 + var_neigh_line_4
            sn_coadd_line_4 = coadd_flux_line_4 / np.sqrt(coadd_var_line_4)


            neighbor_coords = spx_to_bin[neigh_array[0], :2].tolist()
            neighbor_coords_r_theta = spx_to_bin[neigh_array[0], 2:4].tolist()
            bin_data.append([[[r, theta]]+neighbor_coords_r_theta, [[x, y]] + neighbor_coords, neigh_array[0], fl_line_1, var_line_1, fl_line_2, var_line_2, fl_line_3, var_line_3, fl_line_4, var_line_4,
                             coadd_flux_line_1, coadd_var_line_1,coadd_flux_line_2, coadd_var_line_2,coadd_flux_line_3, coadd_var_line_3,coadd_flux_line_4, coadd_var_line_4, sn_coadd_line_1,sn_coadd_line_2,sn_coadd_line_3,sn_coadd_line_4, k, -1])

    return np.array(bin_data, dtype='object')


def lowest_sn_bin(bin_data, sn_threshold):
    sn_values = []
    for row in bin_data:
        #Extract SN values for each spaxel
        SN_line_1= row[19]
        SN_line_2= row[20]
        SN_line_3= row[21]
        SN_line_4= row[22]

        #Find the weakest emission line in a bin
        sns_values = np.array([SN_line_1,SN_line_2,SN_line_3,SN_line_4],dtype=float)
        lowest_sn_value = np.min(sns_values)
        sn_values.append(lowest_sn_value)#An array of SN of the weakest line of each bin

    sn_values = np.array(sn_values)
    valid_indices = np.where(sn_values > sn_threshold)[0]
    return valid_indices[np.argmin(sn_values[valid_indices])] if valid_indices.size > 0 else -1


# Function to check if pixel coordinates are unique across bins
def check_unique_pixel_coordinates(final_binmap):
    seen_coordinates = set()  # Set to track the coordinates we've seen

    for bin in final_binmap:
        pixel_coords = bin[1]  # Extract the list of pixel coordinates

        # Convert list of coordinates to a tuple of tuples to make it hashable
        try:
            pixel_coords_tuple = tuple(map(tuple, pixel_coords))  # Converting list of lists to tuple of tuples
        except TypeError:
            print(f"Error processing coordinates for bin {bin[0]}: {pixel_coords}")
            continue  # Skip this bin if it can't be processed

        # Check if the tuple already exists in the set
        if pixel_coords_tuple in seen_coordinates:
            print(f"Duplicate found! Bin {bin[0]} has duplicate coordinates.")
            return False
        seen_coordinates.add(pixel_coords_tuple)

    print("All bins have unique pixel coordinates.")
    return True
