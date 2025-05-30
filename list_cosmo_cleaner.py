from LA_Cosmic import detCos
from astroscrappy import detect_cosmics
import shutil
import os
import logging
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

##################################################################

def list_cosmo_cleaner(dir_name, list_name, out_list_name, rdn=5.0, gf=2.0, **kwargs):
    try:
        with open(dir_name.joinpath(list_name), 'r') as infile:
            input_files = [line.strip() for line in infile.readlines()]
        with open(dir_name.joinpath(out_list_name), 'r') as outfile:
            output_files = [line.strip() for line in outfile.readlines()]

            if len(input_files) != len(output_files):
                print("Error: The number of input and output files must match.")
                return

        for input_file, output_file in zip(input_files, output_files):
            try:
                # Check if input file exists
                if not os.path.isfile(input_file):
                    print(f"Input file not found: {input_file}")
                    continue

                with pyfits.open(input_file) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    if data is None:
                        print(f"No data found in file: {input_file}")
                        continue
                    mask, cleaned_data = detect_cosmics(data, readnoise=rdn, gain=gf,**kwargs)
                    pyfits.writeto(output_file, cleaned_data, header, overwrite=True)
                    print(f"Processed and saved: {output_file}")
            except Exception as e:
                print(f"Error processing file {input_file}: {e}")
    except Exception as e:
        print(f"Error reading file lists: {e}")



#    with open(dir_name.joinpath(list_name), 'r') as f:
#        for line in f:
#            name = line.strip()
#            out_name = os.path.splitext(name)[0] + '_CRR.fits'
#            hdu = pyfits.open(name, mode = 'update', do_not_scale_image_data=True)
#            prihdr = hdu[0].header
#            data = hdu[0].data.squeeze().copy()
#            print(f'FITs has {data.shape} dims')
#            hdu.close()
#            detCos(image=name,  out_clean=out_name)
#            cosmics, cleared = detect_cosmics(data,sigclip=5, gain=2.78, readnoise=5.6) 
#           print(out_name, file=f_out)
#           print(out_name)
#           logging.info(f"{out_name}")
#           print()
#    f.close()

#    f_out=open(dir_name.joinpath(out_list_name), 'a')
#    f_out.close()
#    print('File names saved in ', dir_name.joinpath(out_list_name))
#    return("Cleaned")
