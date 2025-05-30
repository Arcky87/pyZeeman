import astropy.io.fits as pyfits
import shutil
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

##################################################################

def plot_diff_across(dir_name, obj_name, **kwargs):
    clobj_name = obj_name.split('.')[0] + '_CRR.fits'
    print(obj_name, clobj_name)
    mask_dir = dir_name / "cosmic_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    hdul = pyfits.open(dir_name.joinpath(obj_name))
    hdur = pyfits.open(dir_name.joinpath(clobj_name))
    aver_orig = hdul[0].data.squeeze().mean(axis=1)
    aver_cleared = hdur[0].data.mean(axis=1)
    
    orig_peaks, orig_vals = find_peaks(aver_orig, height=2,distance=25)
    clear_peaks, clear_vals = find_peaks(aver_cleared, height=2,distance=25)

    orig_max_ind = np.argsort(orig_vals['peak_heights'])[:-15:-1]
    clear_max_ind = np.argsort(clear_vals['peak_heights'])[:-15:-1]

    fig,(ax1,ax2) = plt.subplots(1,2, **kwargs)
    ax1.plot(aver_orig)
    ax2.plot(aver_cleared)
#    ax1.plot(orig_peaks, aver_orig[orig_peaks], "x")
    for x,y in zip(orig_peaks[orig_max_ind],orig_vals['peak_heights'][orig_max_ind]):
        label = "{:.1f}".format(y)
        ax1.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
    for x,y in zip(clear_peaks[clear_max_ind],clear_vals['peak_heights'][clear_max_ind]):
        label = "{:.1f}".format(y)
        ax2.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
#       ax1.annotate(f'{aver_orig[ind]}',xy=(orig_peaks[i],aver_orig[ind]))
    ax1.set_title('Original')
    ax2.set_title('Astroscrappy cleared')
    mask_path = mask_dir/f"sig_reduce_{obj_name.split('.')[0]}.pdf"
    plt.savefig(mask_path, format="pdf", bbox_inches="tight")
    #plt.show()
    plt.close()

if __name__ == '__main__':
    from pathlib import Path
    
    # Directory containing the lists and data (created by lister.py)
    temp_directory = Path('./TEMP')
    
    # Input list (object frames from lister.py)
    obj_list = 'obj_list.txt'
    
    # Check if input list exists
    if (temp_directory / obj_list).exists():
        print('Plotting differences between original and cosmic-ray reduced spectra...')
              
        # Read object list and process each file
        with open(temp_directory/ obj_list, 'r') as f:
            for line in f:
                name = line.strip()
                print(f'Processing {name}...')
                
                # Plot with larger figure size
                plot_diff_across(
                    dir_name=temp_directory.parent,
                    obj_name=os.path.basename(name),
                    figsize=(15, 6)  # Wider figure for better visibility
                )
                
        print('All difference plots have been saved in cosmic_masks/')
    else:
        print(f'Input list {obj_list} not found in {temp_directory}')