import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import os
from astropy.visualization import ZScaleInterval

def subtract_scattered_light(dir_name, list_name, out_list_name, border_width=60, plot=False):
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
                    corrected[:, col] = data[:, col] - background[:, col]
                    corrected[:, col] = np.clip(corrected[:, col], 0, None)
    
                pyfits.writeto(out_name, corrected, header, overwrite=True)
                pyfits.writeto(out_bg_name, background, header, overwrite=True)
            print(out_name, file=f_out)
            print(f"Processed and saved: {out_name}")
            if plot==True:
                z_scale = ZScaleInterval()
                
                plt.figure(figsize=(15, 5))

                plt.subplot(131)
                z1, z2 = z_scale.get_limits(data)
                plt.imshow(data, cmap='coolwarm', aspect='auto', vmin=z1, vmax=z2)
                plt.title("Original Image")
                plt.colorbar()

                plt.subplot(132)
                plt.imshow(background, cmap='coolwarm', aspect='auto')
                plt.title("Scattered Light Model")
                plt.colorbar()

                z1, z2 = z_scale.get_limits(corrected)
                plt.subplot(133)
                plt.imshow(corrected, cmap='coolwarm', aspect='auto', vmin=z1, vmax=z2)
                plt.title("Background Subtracted")
                plt.colorbar()

                plt.tight_layout()
                #plt.show()
                plt.pause(3)
                plt.close()
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
                                                     plot=True,
                                                     border_width=60)
    if result == "Cleaned":
            print('Successfully subtracted background from object frames')
    else:
        print(f'Input list {input_list} not found in {temp_directory}')
