import numpy as np
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

def extract_spectrum_1d(image_data, trace_data, order_number, 
                       extraction_method='sum', aperture_width=None,
                       background_regions=None, cosmic_ray_mask=None,
                       interpolate_trace=True, plot=False, 
                       output_path=None, save_format='fits'):
    """
    Экстракция одномерного спектра из двумерного изображения по трассировке порядка
    
    Parameters:
    -----------
    image_data : np.array or str/Path
        2D массив изображения или путь к FITS файлу
    trace_data : dict
        Данные трассировки от load_traced_orders()
    order_number : int
        Номер порядка для экстракции
    extraction_method : str
        Метод экстракции: 'sum', 'mean', 'optimal', 'gaussian_weighted'
    aperture_width : float or None
        Ширина апертуры в пикселях (если None, используется полная ширина порядка)
    background_regions : tuple or None
        Области для вычитания фона: ((y1_start, y1_end), (y2_start, y2_end))
    cosmic_ray_mask : np.array or None
        Маска космических лучей (True = плохой пиксель)
    interpolate_trace : bool
        Интерполировать трассировку для субпиксельной точности
    plot : bool
        Показать диагностические графики
    output_path : str/Path or None
        Путь для сохранения результата
    save_format : str
        Формат сохранения: 'fits', 'txt', 'json'
    
    Returns:
    --------
    dict : Результат экстракции:
        {
            'wavelength_axis': np.array,  # ось по дисперсии (пиксели)
            'spectrum': np.array,         # экстрагированный спектр
            'error': np.array,           # ошибки (если доступны)
            'background': np.array,      # фон (если вычитался)
            'snr': np.array,            # отношение сигнал/шум
            'extraction_info': dict      # информация об экстракции
        }
    """
    
    # Загрузка изображения
    if isinstance(image_data, (str, Path)):
        with fits.open(image_data) as hdul:
            image = hdul[0].data.astype(float)
            header = hdul[0].header
    else:
        image = image_data.astype(float)
        header = None
    
    # Поиск данных порядка
    order_data = None
    for order in trace_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        raise ValueError(f"Порядок {order_number} не найден в trace_data")
    
    # Извлечение трассировки
    x_trace = order_data['trace_full']['x']
    y_center = order_data['trace_full']['y_center']
    y_upper = order_data['trace_full']['y_upper']
    y_lower = order_data['trace_full']['y_lower']
    
    # Определение рабочего диапазона по x
    x_min = max(0, int(np.min(x_trace)))
    x_max = min(image.shape[1], int(np.max(x_trace)) + 1)
    x_range = np.arange(x_min, x_max)
    
    # Интерполяция трассировки для субпиксельной точности
    if interpolate_trace and len(x_trace) > 1:
        f_center = interpolate.interp1d(x_trace, y_center, kind='cubic', 
                                      bounds_error=False, fill_value='extrapolate')
        f_upper = interpolate.interp1d(x_trace, y_upper, kind='cubic',
                                     bounds_error=False, fill_value='extrapolate')
        f_lower = interpolate.interp1d(x_trace, y_lower, kind='cubic',
                                     bounds_error=False, fill_value='extrapolate')
        
        y_center_interp = f_center(x_range)
        y_upper_interp = f_upper(x_range)
        y_lower_interp = f_lower(x_range)
    else:
        # Использование ближайших значений
        indices = np.searchsorted(x_trace, x_range)
        indices = np.clip(indices, 0, len(x_trace) - 1)
        y_center_interp = y_center[indices]
        y_upper_interp = y_upper[indices]
        y_lower_interp = y_lower[indices]
    
    # Применение ширины апертуры
    if aperture_width is not None:
        half_width = aperture_width / 2
        y_upper_interp = y_center_interp + half_width
        y_lower_interp = y_center_interp - half_width
    
    # Инициализация массивов результатов
    spectrum = np.zeros(len(x_range))
    error = np.zeros(len(x_range))
    background = np.zeros(len(x_range))
    n_pixels = np.zeros(len(x_range))
    
    # Создание маски космических лучей
    if cosmic_ray_mask is None:
        cosmic_ray_mask = np.zeros_like(image, dtype=bool)
    
    # Экстракция для каждого столбца
    for i, x in enumerate(x_range):
        if x < 0 or x >= image.shape[1]:
            continue
            
        y_low = int(np.floor(y_lower_interp[i]))
        y_high = int(np.ceil(y_upper_interp[i]))
        
        # Ограничение границами изображения
        y_low = max(0, y_low)
        y_high = min(image.shape[0], y_high)
        
        if y_high <= y_low:
            continue
        
        # Извлечение столбца
        column = image[y_low:y_high, x]
        mask_column = cosmic_ray_mask[y_low:y_high, x]
        
        # Маскирование плохих пикселей
        good_pixels = ~mask_column & np.isfinite(column)
        
        if np.sum(good_pixels) == 0:
            continue
        
        # Вычисление фона
        bg_value = 0
        if background_regions is not None:
            bg_values = []
            for bg_start, bg_end in background_regions:
                bg_start = max(0, int(bg_start))
                bg_end = min(image.shape[0], int(bg_end))
                if bg_end > bg_start:
                    bg_region = image[bg_start:bg_end, x]
                    bg_mask = cosmic_ray_mask[bg_start:bg_end, x]
                    bg_good = ~bg_mask & np.isfinite(bg_region)
                    if np.sum(bg_good) > 0:
                        bg_values.extend(bg_region[bg_good])
            
            if len(bg_values) > 0:
                bg_value = np.median(bg_values)
        
        background[i] = bg_value
        
        # Применение метода экстракции
        if extraction_method == 'sum':
            spectrum[i] = np.sum(column[good_pixels] - bg_value)
            error[i] = np.sqrt(np.sum(np.abs(column[good_pixels]) + bg_value))
            
        elif extraction_method == 'mean':
            spectrum[i] = np.mean(column[good_pixels] - bg_value)
            error[i] = np.sqrt(np.mean(np.abs(column[good_pixels]) + bg_value))
            
        elif extraction_method == 'gaussian_weighted':
            # Гауссовы веса относительно центра порядка
            y_coords = np.arange(y_low, y_high)
            weights = np.exp(-0.5 * ((y_coords - y_center_interp[i]) / 
                                   (order_data.get('median_width', 5) / 2.35))**2)
            weights = weights[good_pixels]
            weights /= np.sum(weights) if np.sum(weights) > 0 else 1
            
            spectrum[i] = np.sum((column[good_pixels] - bg_value) * weights)
            error[i] = np.sqrt(np.sum(np.abs(column[good_pixels]) * weights**2))
            
        elif extraction_method == 'optimal':
            # Упрощенная оптимальная экстракция
            # Требует профиль порядка и модель шума
            y_coords = np.arange(y_low, y_high)
            profile = np.exp(-0.5 * ((y_coords - y_center_interp[i]) / 
                                   (order_data.get('median_width', 5) / 2.35))**2)
            profile = profile[good_pixels]
            profile /= np.sum(profile**2) if np.sum(profile**2) > 0 else 1
            
            variance = np.abs(column[good_pixels]) + bg_value + 1  # readout noise
            weights = profile / variance
            weights /= np.sum(weights) if np.sum(weights) > 0 else 1
            
            spectrum[i] = np.sum((column[good_pixels] - bg_value) * weights)
            error[i] = 1.0 / np.sqrt(np.sum(weights)) if np.sum(weights) > 0 else np.inf
        
        n_pixels[i] = np.sum(good_pixels)
    
    # Вычисление SNR
    snr = spectrum / error
    snr[~np.isfinite(snr)] = 0
    
    # Формирование результата
    result = {
        'wavelength_axis': x_range.astype(float),
        'spectrum': spectrum,
        'error': error,
        'background': background,
        'snr': snr,
        'extraction_info': {
            'order_number': order_number,
            'extraction_method': extraction_method,
            'aperture_width': aperture_width,
            'x_range': (x_min, x_max),
            'y_range': (float(np.min(y_lower_interp)), float(np.max(y_upper_interp))),
            'n_pixels_used': n_pixels,
            'median_snr': float(np.median(snr[snr > 0])) if np.any(snr > 0) else 0,
            'source_file': trace_data['metadata'].get('source_file', 'unknown')
        }
    }
    
    # Визуализация
    if plot:
        _plot_extraction_diagnostics(image, x_range, y_center_interp, 
                                    y_upper_interp, y_lower_interp, 
                                    spectrum, order_number)
    
    # Сохранение результата
    if output_path is not None:
        _save_extracted_spectrum(result, output_path, save_format)
    
    print(f"Экстракция порядка {order_number} завершена:")
    print(f"  Диапазон X: {x_min}-{x_max} пикселей")
    print(f"  Метод: {extraction_method}")
    print(f"  Медианное SNR: {result['extraction_info']['median_snr']:.2f}")
    
    return result


def _plot_extraction_diagnostics(image, x_range, y_center, y_upper, y_lower, 
                                spectrum, order_number):
    """Диагностические графики экстракции"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Изображение с трассировкой
    ax = axes[0, 0]
    x_min, x_max = int(np.min(x_range)), int(np.max(x_range))
    y_min = int(np.min(y_lower)) - 10
    y_max = int(np.max(y_upper)) + 10
    
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)
    
    im_section = image[y_min:y_max, x_min:x_max]
    ax.imshow(im_section, aspect='auto', origin='lower', 
              extent=[x_min, x_max, y_min, y_max])
    
    ax.plot(x_range, y_center, 'r-', linewidth=2, label='Центр')
    ax.plot(x_range, y_upper, 'g--', linewidth=1, label='Верхняя граница')
    ax.plot(x_range, y_lower, 'g--', linewidth=1, label='Нижняя граница')
    ax.set_title(f'Трассировка порядка {order_number}')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')
    ax.legend()
    
    # 2. Экстрагированный спектр
    ax = axes[0, 1]
    ax.plot(x_range, spectrum, 'b-', linewidth=1)
    ax.set_title(f'Экстрагированный спектр порядка {order_number}')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Интенсивность')
    ax.grid(True, alpha=0.3)
    
    # 3. Профиль порядка (средний)
    ax = axes[1, 0]
    if len(x_range) > 0:
        mid_x = x_range[len(x_range)//2]
        if mid_x < image.shape[1]:
            y_low = int(y_lower[len(x_range)//2]) - 5
            y_high = int(y_upper[len(x_range)//2]) + 5
            y_low = max(0, y_low)
            y_high = min(image.shape[0], y_high)
            
            profile = image[y_low:y_high, mid_x]
            y_coords = np.arange(y_low, y_high)
            ax.plot(y_coords, profile, 'k-', linewidth=1)
            ax.axvline(y_center[len(x_range)//2], color='r', linestyle='-', label='Центр')
            ax.axvline(y_upper[len(x_range)//2], color='g', linestyle='--', label='Границы')
            ax.axvline(y_lower[len(x_range)//2], color='g', linestyle='--')
            ax.set_title(f'Профиль порядка в X={mid_x}')
            ax.set_xlabel('Y (пиксели)')
            ax.set_ylabel('Интенсивность')
            ax.legend()
    
    # 4. Статистика экстракции
    ax = axes[1, 1]
    snr = spectrum / np.sqrt(np.abs(spectrum) + 1)
    ax.plot(x_range, snr, 'purple', linewidth=1)
    ax.set_title('Отношение сигнал/шум')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('SNR')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _save_extracted_spectrum(result, output_path, save_format):
    """Сохранение экстрагированного спектра"""
    
    output_path = Path(output_path)
    order_num = result['extraction_info']['order_number']
    
    if save_format == 'fits':
        # Создание FITS файла
        from astropy.table import Table
        
        table = Table({
            'wavelength': result['wavelength_axis'],
            'flux': result['spectrum'],
            'error': result['error'],
            'background': result['background'],
            'snr': result['snr']
        })
        
        fits_file = output_path / f"spectrum_order_{order_num:02d}.fits"
        table.write(fits_file, format='fits', overwrite=True)
        
    elif save_format == 'txt':
        txt_file = output_path / f"spectrum_order_{order_num:02d}.txt"
        data = np.column_stack([
            result['wavelength_axis'],
            result['spectrum'],
            result['error'],
            result['snr']
        ])
        np.savetxt(txt_file, data, 
                  header='# X_pixel  Flux  Error  SNR',
                  fmt='%.6f')
        
    elif save_format == 'json':
        import json
        json_file = output_path / f"spectrum_order_{order_num:02d}.json"
        
        # Конвертация numpy arrays в списки для JSON
        json_data = {
            'wavelength_axis': result['wavelength_axis'].tolist(),
            'spectrum': result['spectrum'].tolist(),
            'error': result['error'].tolist(),
            'background': result['background'].tolist(),
            'snr': result['snr'].tolist(),
            'extraction_info': result['extraction_info']
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)


def extract_multiple_orders(image_data, trace_data, order_numbers=None, **kwargs):
    """
    Экстракция нескольких порядков одновременно
    
    Parameters:
    -----------
    image_data : np.array or str/Path
        2D массив изображения или путь к FITS файлу
    trace_data : dict
        Данные трассировки от load_traced_orders()
    order_numbers : list or None
        Список номеров порядков (None для всех)
    **kwargs : dict
        Параметры для extract_spectrum_1d()
    
    Returns:
    --------
    dict : Словарь с экстрагированными спектрами для каждого порядка
    """
    
    if order_numbers is None:
        order_numbers = trace_data['order_numbers']
    
    results = {}
    
    for order_num in order_numbers:
        print(f"\nЭкстракция порядка {order_num}...")
        try:
            result = extract_spectrum_1d(image_data, trace_data, order_num, **kwargs)
            results[order_num] = result
        except Exception as e:
            print(f"Ошибка при экстракции порядка {order_num}: {e}")
            continue
    
    print(f"\nЭкстракция завершена для {len(results)} порядков")
    return results

