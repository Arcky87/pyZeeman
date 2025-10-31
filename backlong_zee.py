import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import os
from astropy.visualization import ZScaleInterval
from scipy.ndimage import median_filter

def subtract_scattered_light(dir_name, list_name, out_list_name, border_width=90, plot=False):
    """
    Вычитает рассеянный свет из изображения, аппроксимируя фон линейным трендом в каждом столбце
    
    Parameters:
        image: 2D numpy array [spatial, spectral] - входное изображение
        border_width: int - количество пикселей по краям для аппроксимации фона
            
    Returns:
        tuple: (corrected_image, background_image)
            corrected_image - изображение после вычитания фона
            background_image - матрица рассеянного света
    """
    
    f_out=open(dir_name.joinpath(out_list_name), 'a')
    with open(dir_name.joinpath(list_name), 'r') as f:
        for line in f:
            name = line.strip()
            out_name = os.path.splitext(name)[0] + '_bt.fits'
            out_bg_name = os.path.splitext(name)[0] + '_background.fits'
            with pyfits.open(name) as hdul:
                print(f'Subtracting {name}')
                data = hdul[0].data.squeeze()
                header = hdul[0].header

                # Инициализация матрицы фона
                background = np.zeros_like(data)
                corrected = np.zeros_like(data)
    
                # Координаты по пространственной оси
                y = np.arange(data.shape[0])
    
                # Обрабатываем каждый столбец отдельно
                for col in range(data.shape[1]):
                      # 1. Берем данные с краев столбца
                    top_part = data[:border_width, col]
                    bottom_part = data[-border_width:, col]
        
                        # 2. Медианные значения для устойчивости к выбросам
                    y_top = y[:border_width]
                    median_top = np.median(top_part)
        
                    y_bottom = y[-border_width:]
                    median_bottom = np.median(bottom_part)
        
                    # 3. Линейная аппроксимация (2 точки)
                    slope = (median_bottom - median_top) / (y_bottom.mean() - y_top.mean())
                    intercept = median_top - slope * y_top.mean()
        
                    # 4. Строим фон для всего столбца
                    background[:, col] = slope * y + intercept
        
                    # 5. Вычитаем фон (с защитой от отрицательных значений)
 #                   corrected[:, col] = data[:, col] - background[:, col]
 #                   corrected[:, col] = np.clip(corrected[:, col], 0, None)

                # 6. Медианная фильтрация фона background
                rady = 0 
                radx = 100
                thresh = 0.0
                kernel = np.ones((2*rady+1, 2*radx+1), bool)
                bg_filtered = median_filter(background,footprint=kernel)
                diff = np.abs(bg_filtered - background)
                mask = diff > thresh
                bg_fil = np.where(mask, bg_filtered, background)
                corrected = data - bg_fil
                corrected = np.clip(corrected, 0, None) # убираем отрицательные значения
    
                pyfits.writeto(out_name, corrected, header, overwrite=True)
                pyfits.writeto(out_bg_name, bg_fil, header, overwrite=True)
            print(out_name, file=f_out)
            print(f"Processed and saved: {out_name}")
            if plot==True:
                z_scale = ZScaleInterval()
                
                plt.figure(figsize=(15, 5))

                plt.subplot(141)
                z1, z2 = z_scale.get_limits(data)
                plt.imshow(data, cmap='coolwarm', aspect='auto', vmin=z1, vmax=z2)
                plt.title("Original Image")
                plt.colorbar()

                plt.subplot(142)
                plt.imshow(bg_fil, cmap='coolwarm', aspect='auto')
                plt.title("Scattered Light Model")
                plt.colorbar()

                z1, z2 = z_scale.get_limits(corrected)
                plt.subplot(143)
                plt.imshow(corrected, cmap='coolwarm', aspect='auto', vmin=z1, vmax=z2)
                plt.title("Background Subtracted")
                plt.colorbar()

                plt.subplot(144)
                plt.imshow(mask, cmap='coolwarm', aspect='auto')
                plt.title("Bac mask")
                plt.colorbar()

                plt.tight_layout()
                plt.show()
                #plt.pause(3)
                #plt.close()
    f.close()
    f_out.close()

    print('File names saved in ', dir_name.joinpath(out_list_name))
    return("Cleaned")


if __name__ == "__main__":
    from pathlib import Path
    # Пример использования
    print("Testing scattered light removal...")
    
    temp_directory = Path('./TEMP')
    data_directory = Path('.')
    input_list = 'obj_crr_list.txt'
    output_list = 'obj_crr_bt_list.txt'

    if (temp_directory / input_list).exists():
        print(f'Removing scattered light from object spectra...')
    
    # Обработка
    result = subtract_scattered_light(dir_name=temp_directory,
                                                     list_name=input_list,
                                                     out_list_name=output_list,
                                                     plot=False,
                                                     border_width=90)
    if result == "Cleaned":
            print('Successfully subtracted background from object frames')
    else:
        print(f'Input list {input_list} not found in {temp_directory}')
