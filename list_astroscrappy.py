from astroscrappy import detect_cosmics
import astropy.io.fits as pyfits
import shutil
import os
import logging
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

##################################################################

def list_cosmo_cleaner(dir_name, list_name, out_list_name, mask_list_name, plot=False, rdn=5.0, gf=2.0, **kwargs):

    f_out=open(dir_name.joinpath(out_list_name), 'a')
    m_out=open(dir_name.joinpath(mask_list_name), 'a')

    with open(dir_name.joinpath(list_name), 'r') as f:
        for line in f:
            name = line.strip()
            out_name = os.path.splitext(name)[0] + '_CRR.fits'
            mask_name = os.path.splitext(name)[0] + '_cosmics.png'
            mask_dir = Path('./cosmic_masks') 
            mask_dir.mkdir(parents=True, exist_ok=True)
            with pyfits.open(name) as hdul:
                print(f'Scrapping {name}')
                data = hdul[0].data.squeeze()
                print(f'Squeezed data has {data.shape} dimensions')
                header = hdul[0].header
                mask, cleaned_data = detect_cosmics(data, readnoise=rdn, gain=gf,**kwargs)
                pyfits.writeto(out_name, cleaned_data, header, overwrite=True)
                plt.imsave(mask_dir / mask_name, mask, cmap='copper')
            print(out_name, file=f_out)
            print(mask_name, file=m_out)
            print(f"Processed and saved: {out_name}")
            logging.info(f"{out_name}")
            if plot==True:
                z_scale = ZScaleInterval()
                z1,z2 = z_scale.get_limits(cleaned_data)
                fig, (ax1,ax2) = plt.subplots(2,1,figsize=(19,9))
                ax1.imshow(cleaned_data,vmin=z1,vmax=z2,cmap='coolwarm')
                ax2.imshow(mask,cmap='coolwarm')
                plt.show()
    f.close()
    f_out.close()
    m_out.close()

    print('File names saved in ', dir_name.joinpath(out_list_name))
    return("Cleaned")

if __name__ == '__main__':
    from pathlib import Path
    
    # Directory containing the lists (created by lister.py)
    temp_directory = Path('./TEMP')
    
    # Input list (object frames from lister.py)
    input_list = 'obj_list.txt'
    
    # Output lists
    output_list = 'obj_crr_list.txt'  # List of cosmic-ray cleaned files
    mask_list = 'obj_mask_list.txt'   # List of cosmic ray mask images
    
    # Check if input list exists
    if (temp_directory / input_list).exists():
        print(f'Processing object frames for cosmic ray removal...')
        # Process with default readnoise=5.0 and gain=2.0
        result = list_cosmo_cleaner(
            dir_name=temp_directory,
            list_name=input_list,
            out_list_name=output_list,
            mask_list_name=mask_list,
            plot=False,  # Set to True to show cleaning results
            rdn=5.6,     # Adjust readnoise if needed
            gf=2.78       # Adjust gain if needed
        )
        if result == "Cleaned":
            print('Successfully cleaned cosmic rays from object frames')
    else:
        print(f'Input list {input_list} not found in {temp_directory}')
