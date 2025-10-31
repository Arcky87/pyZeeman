import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from astropy.io import fits
from loaders import *
import matplotlib.pyplot as plt

# Импорты из существующих модулей
from extract_order_spectrum import extract_order_summed, load_trace_data
from thar_calibration import (
    find_and_plot_lines,
    interactive_wavelength_calibration,
    find_nearest_line,
    find_peaks_for_order,
    plot_extracted_spectrum,
    fit_dispersion_poly
)
from thar_auto_calibration import load_thar_atlas, load_calibration_solution

data_dir = Path('/data/Observations/test_pyzeeman_final/')
temp_dir = data_dir / 'temp'
traced_dir = data_dir / 'TRACED_ORDERS'
calib_dir = data_dir / 'CALIBRATIONS'
calib_dir.mkdir(exist_ok=True)
atlas_file = Path('thar.dat')

reference1 = calib_dir / 'reference_solution.json'
reference2 = calib_dir / 'o011ref_solution.json'
atlas_lines = load_thar_atlas(atlas_file)

ref18 = load_calibration_solution(reference1)
ref11 = load_calibration_solution(reference2)

ref_model18 = np.poly1d(ref18['model'])
ref_model11 = np.poly1d(ref11['model'])

calib_points18 = [(float(p),float(w)) for p,w in ref18['calib_points'].items()]
calib_points11 = [(float(p),float(w)) for p,w in ref11['calib_points'].items()]

calib_points18.pop(-3)

#ref_wls_full = sorted(set(ref_wls_interactive) | set(atlas_lines.tolist()))

fig,(ax1,ax2) = plt.subplots(1,2)

ax1.plot([item[0] for item in calib_points18], 
         [a-b for a,b in zip(sorted([item[0] for item in calib_points18]),
                             sorted([item[0] for item in calib_points11]))], 'rx')

dif_model = ref_model18(np.arange(4600)) - ref_model11(np.arange(4600))

ax2.plot(sorted([item[1] for item in calib_points18]),
         [sorted([item[1] for item in calib_points18])-ref_model18(sorted([item[0] for item in calib_points18]))][0],'rx')
ax2.plot(sorted([item[1] for item in calib_points18]),
         [sorted([item[1] for item in calib_points11])-ref_model11(sorted([item[0] for item in calib_points11]))][0],'gx')


plt.show()