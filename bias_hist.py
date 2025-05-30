import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from astropy.visualization import hist
from astropy.stats import mad_std
import astropy.io.fits as pyfits
from pathlib import Path

##################################################################

def plot_bias_hist(dir_name, list_name):
    """
    Plot histograms of bias frames with statistics
    """
    with open(dir_name.joinpath(list_name), 'r') as f:
        for line in f:
            name = line.strip()
            path = dir_name.parent
            namepath = path / name
            print(f"Calculating histogram for {namepath}")
            try:
                with pyfits.open(namepath) as hdul:
                    data = hdul[0].data
                    
                    # Calculate statistics
                    mean = np.mean(data)
                    median = np.median(data)
                    std = np.std(data)
                    mad = mad_std(data)
                    
                    # Create figure with two subplots
                    plt.figure(figsize=(10,10))
                    
                    # Regular histogram
                    hist(data[:,100:].flatten(), bins=800);  
                    plt.xlabel('Counts')
                    plt.ylabel('NoP')
                    plt.semilogy()
                    
                    # Save plot
                    # plt.savefig(f"bias_hist_{os.path.splitext(os.path.basename(name))[0]}.pdf", 
                    #           format="pdf", bbox_inches="tight")
                    plt.show()
#                    plt.close()
                    
                    print(f"Processed: {name}")
                    print(f"Mean: {mean:.2f}, Median: {median:.2f}")
                    print(f"STD: {std:.2f}, MAD: {mad:.2f}")
                    print("-" * 50)
                    
            except IOError:
                print(f"Can't open file: {name}")
                continue
    
    return "Completed"

if __name__ == '__main__':
    # Directory containing the lists (created by lister.py)
    temp_directory = Path('./TEMP')
    
    # Input list (bias frames from lister.py)
    input_list = 'bias_list.txt'
    
    # Check if input list exists
    if (temp_directory / input_list).exists():
        print('Processing bias frames for histogram analysis...')
        result = plot_bias_hist(temp_directory, input_list)
        if result == "Completed":
            print('Successfully created histograms for all bias frames')
    else:
        print(f'Input list {input_list} not found in {temp_directory}')
