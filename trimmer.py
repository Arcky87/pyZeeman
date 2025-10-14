import astropy.io.fits as pyfits
import os
import numpy
import copy
import shutil

##################################################################
def trimmer(dir_name, list_name, area, flip):
    with open(dir_name.joinpath(list_name), 'r') as f:
        for line in f:
            name = line.strip()
            try:
                hdulist = pyfits.open(name, mode = 'update', do_not_scale_image_data=True)
                prihdr = hdulist[0].header
                data = hdulist[0].data.squeeze().copy()
                print(f'FITs has {data.shape} dims')
                hdulist.close()
                print(data.shape[0], (int(area[1])-int(area[0])),data.shape[1], (int(area[3])-int(area[2])))
                if data.shape[0]>=(int(area[1])-int(area[0])) and data.shape[1]>=(int(area[3])-int(area[2])):
                    trimmed_data = copy.copy(data[int(area[0]):int(area[1]),int(area[2]):int(area[3])])
                    if flip == 'X':
                        flipped_data = numpy.flip(trimmed_data, 1)
                    elif flip == 'Y':
                        flipped_data = numpy.flip(trimmed_data, 0)
                    elif flip == 'XY':
                        flipped_data = numpy.flip(trimmed_data)
                    else:
                        flipped_data = trimmed_data.copy()
                    hdulist[0].data = flipped_data
                    prihdr['NAXIS1'] = flipped_data.shape[1]
                    prihdr['NAXIS2'] = flipped_data.shape[0]
                    prihdr['HISTORY'] = 'overscan trimmed'
                    prihdr['HISTORY'] = 'Data flipped along '+str(flip)
                    try:
                        hdulist.writeto(name, overwrite=True)
                    except IOError:
                        print(f"ERROR: Can't write file {name}")
                else:
                    print (f"Frame {name} has wrong size")

            except IOError:
                print (f"Can't open file: {name}")
    f.close()
    return("Trimmed")

if __name__ == '__main__':
    from pathlib import Path
    
    # Directory containing the lists (created by lister.py)
    temp_directory = Path('./TEMP')
    
    # Trimming area parameters [y_start, y_end, x_start, x_end]
    area = [0, 625, 0, 4600]  # Example values, adjust according to your needs
    
    # Process each type of image
    for list_name in ['bias_list.txt', 'dark_list.txt', 'flat_list.txt', 'thar_list.txt', 'obj_list.txt']:
        if (temp_directory / list_name).exists():
            print(f'Processing {list_name}...')
            result = trimmer(temp_directory, list_name, area, flip='None')
            if result == "Trimmed":
                print(f'Successfully trimmed files from {list_name}')
