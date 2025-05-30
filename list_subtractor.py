import astropy.io.fits as pyfits
import os
import numpy

import warnings
warnings.simplefilter("ignore")
##################################################################
def list_subtractor(list_name, subtrahend_name, stype):
    hdulist = pyfits.open(subtrahend_name)
    delta = hdulist[0].data.squeeze().copy()
    prihdr = hdulist[0].header
    hdulist.close()

    with open(list_name, 'r') as f:
        for line in f:
            print(f"Subtracting from {line}")
            name = line.strip()
            hdulist = pyfits.open(name, mode = 'update')
            data = hdulist[0].data.squeeze().copy()
            prihdr = hdulist[0].header
            hdulist.close()

            if data.shape[0]==delta.shape[0] and data.shape[1]==delta.shape[1]:
                data=numpy.float32(data)-delta
                prihdr['HISTORY'] = stype+' subtracted'
                hdu = pyfits.PrimaryHDU(data, prihdr)
                hdulist = pyfits.HDUList([hdu])
                hdulist.writeto(name, overwrite=True)
            else:
                print ("Frame", name, "has wrong size")
    f.close()

    return ("Cleaned")

if __name__ == '__main__':
    from pathlib import Path
    
    temp_directory = Path('./TEMP')
    data_directory = Path('.')
    input_list = 'obj_crr_list.txt'

    status = list_subtractor(temp_directory / input_list, data_directory.joinpath('s_bias.fits'), 'Bias')
    print (f" {status}")

    
