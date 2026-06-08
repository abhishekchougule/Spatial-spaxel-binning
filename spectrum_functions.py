from mpdaf.obj import Cube,Image,WaveCoord,Spectrum,deg2sexa,sexa2deg
from pyplatefit import fit_spec,Platefit, Linefit, plot_fit
from mpdaf.sdetect.source import Source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import math


def global_spec_em(cube, input_redshift, dec_center, ra_center):
    global_spectrum = cube.sum(axis=(1,2))
    #logger = logging.getLogger('pyplatefit')
    #logger.setLevel('INFO')

    try:
        res = fit_spec(global_spectrum, z=input_redshift, ziter=True, fit_all=False)
        line_table = res['lines'].copy()
    except ValueError as e:
        if "NaN values detected" in str(e):
            print('Fitting Global Spectrum failed! Trying central pixel instead...')
            y_center, x_center = cube.wcs.sky2pix((dec_center, ra_center), nearest=True)[0]
            sp_best = cube[:, y_center, x_center]

            res = fit_spec(sp_best, z=input_redshift, ziter=True, fit_all=False)
            line_table = res['lines'].copy()
        else:
            raise  # Re-raise other ValueErrors

    # Convert to Pandas
    line_table = line_table.to_pandas().reset_index()

    # Split into df1 (Balmer lines) and df2 (Forbidden lines)
    df1 = line_table[line_table['LINE'].isin(['HGAMMA', 'HBETA', 'HALPHA'])].copy()
    df2 = line_table[line_table['LINE'].isin(['OII3727b', 'OIII5007', 'SII6717', 'SII6731'])].copy()

    # ----- df1: Select top 2 Balmer lines by SNR -----
    df1_top2 = df1.sort_values(by='SNR', ascending=False).drop_duplicates(subset='LINE').head(2)

    # ----- df2: Handle SII doublet and select top 2 forbidden line groups -----
    sii_lines = df2[df2['LINE'].isin(['SII6717', 'SII6731'])]

    if len(sii_lines) == 2:
        # Properly combine S/N assuming continuum-dominated noise for fair ranking
        snr_6717 = sii_lines.loc[sii_lines['LINE'] == 'SII6717', 'SNR'].values[0]
        snr_6731 = sii_lines.loc[sii_lines['LINE'] == 'SII6731', 'SNR'].values[0]
        sii_snr = (snr_6717 + snr_6731) / np.sqrt(2)
        sii_lbda = sii_lines['LBDA_OBS'].mean()
    elif len(sii_lines) == 1:
        sii_snr = sii_lines['SNR'].values[0]
        sii_lbda = sii_lines['LBDA_OBS'].values[0]
    else:
        sii_snr = None

    # Isolate non-SII lines to build the pooled ranking frame
    non_sii = df2[~df2['LINE'].isin(['SII6717', 'SII6731'])].drop_duplicates(subset='LINE')

    if sii_snr is not None:
        sii_entry = pd.DataFrame([{
            'LINE': 'SII_doublet',
            'SNR': sii_snr,
            'LBDA_OBS': sii_lbda
        }])
        combined_df2 = pd.concat([non_sii, sii_entry], ignore_index=True)
    else:
        combined_df2 = non_sii

    # Select top 2 forbidden components based on the fair SNR ranking
    df2_top_lines = combined_df2.sort_values(by='SNR', ascending=False).head(2)

    # ----- Map back to original line naming configurations -----
    # If the doublet won a slot, fetch BOTH original lines from df2
    if 'SII_doublet' in df2_top_lines['LINE'].values:
        sii_both = df2[df2['LINE'].isin(['SII6717', 'SII6731'])]
        other_forbidden = df2[df2['LINE'].isin(df2_top_lines['LINE']) & ~df2['LINE'].isin(['SII_doublet'])]
        df2_final = pd.concat([other_forbidden, sii_both], ignore_index=True)
    else:
        # If SII didn't make the cut, keep the chosen lines as they are
        df2_final = df2[df2['LINE'].isin(df2_top_lines['LINE'])]

    # ----- Final Combination and Sorting -----
    df_combined = pd.concat([df1_top2, df2_final], ignore_index=True)

    # Sort strictly by wavelength so 'SII6717' and 'SII6731' fall cleanly into place
    df_combined_sorted = df_combined.sort_values(by='LBDA_OBS').reset_index(drop=True)

    lines_arr = list(df_combined_sorted['LINE'])

    return lines_arr, res




def get_visible_lines(z_list, wave_min=4700, wave_max=9400):
    """
    For each redshift in z_list, returns the visible Balmer and forbidden lines
    (grouping doublets like [O II], [S II]) based on the instrument wavelength range.

    Parameters:
    - z_list (list of float): List of redshift values
    - wave_min (float): Minimum observed wavelength (default: 4700 Å for MUSE)
    - wave_max (float): Maximum observed wavelength (default: 9400 Å for MUSE)

    Returns:
    - results (list of dict): Each dict contains 'z', 'Balmer_lines', 'Forbidden_lines'
    """

    # Balmer lines (single lines)
    balmer_lines = {
        'Hγ': 4341.68,
        'Hβ': 4862.68,
        'Hα': 6564.61,
    }

    # Forbidden lines (group doublets)
    forbidden_lines = {
        '[O II]': (3726.03, 3728.82),  # doublet
        '[O III]': (5008.24,),         # single line
        '[S II]': (6718.29, 6732.67),  # doublet
    }

    results = []

    for z in z_list:
        balmer_seen = [line for line, λ0 in balmer_lines.items()
                       if wave_min <= λ0 * (1 + z) <= wave_max]

        forbidden_seen = []
        for line, λ0s in forbidden_lines.items():
            for λ0 in λ0s:
                λ_obs = λ0 * (1 + z)
                if wave_min <= λ_obs <= wave_max:
                    forbidden_seen.append(line)
                    break  # include only once even if both components visible

        # Pick 2 lines from each category
        balmer_pick = balmer_seen[:2]
        forbidden_pick = forbidden_seen[:2]

        results.append({
            'z': round(z, 5),
            'Balmer_lines': ', '.join(balmer_pick),
            'Forbidden_lines': ', '.join(forbidden_pick),
        })

    return results

def get_line_obs_wl(line_tab,line_name):
    LBDA_OBS = line_tab[line_tab['LINE']==line_name]['LBDA_OBS'][0]
    FWHM_OBS = line_tab[line_tab['LINE']==line_name]['FWHM_OBS'][0]
    LBDA_LEFT = line_tab[line_tab['LINE']==line_name]['LBDA_LEFT'][0]
    LBDA_RIGHT = line_tab[line_tab['LINE']==line_name]['LBDA_RIGHT'][0]
    return LBDA_OBS,FWHM_OBS,LBDA_LEFT,LBDA_RIGHT


def extract_spectrum(cube,x_in,y_in,line_center,window_size):
    spec_extr = cube[line_center-window_size:line_center+window_size,y_in,x_in]
    return spec_extr

def extract_spectrum_doublet(cube,x_in,y_in,lcll,lclr,wsll,wslr,wl_ll,wl_lr,fwhm_ll,fwhm_lr):
    #lcll:line_center_line_left
    #lclr:line_center_line_right
    #wsll:window_size_line_left
    #wslr:window_size_line_right
    #wl_ll:LBDA_OBS_line_left
    #wl_lr:LBDA_OBS_line_right
    #fwhm_ll:FWHM_OBS_line_left
    #fwhm_lr:FWHM_OBS_line_right
    spec_extr = cube[lcll-wsll-2:lclr+wslr+2,y_in,x_in]
    spec_extr.mask_region(lmin=wl_ll+fwhm_ll,lmax=wl_lr-fwhm_lr-1.25, unit=u.angstrom)
    return spec_extr

def get_line_data(line_tab,line_name):
    FLUX = line_tab[line_tab['LINE']==line_name]['FLUX'][0]
    FLUX_ERR = line_tab[line_tab['LINE']==line_name]['FLUX_ERR'][0]
    SNR = line_tab[line_tab['LINE']==line_name]['SNR'][0]
    EQW = line_tab[line_tab['LINE']==line_name]['EQW'][0]
    VDISP = line_tab[line_tab['LINE']==line_name]['VDISP'][0]
    FWHM_OBS = line_tab[line_tab['LINE']==line_name]['FWHM_OBS'][0]
    return FLUX,FLUX_ERR,SNR,EQW,VDISP,FWHM_OBS


