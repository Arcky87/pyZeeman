import numpy as np
from pathlib import Path
import astropy.io.fits as pyfits
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.ndimage import gaussian_filter1d
import warnings
from astropy.visualization import ZScaleInterval
from scipy.stats import linregress
from loaders import *
import os
import json

import argparse
import glob
import time

#==== My functions ====
def gaussian(x, amplitude, center, sigma, offset):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

def super_gaussian(x, amplitude, center, sigma, power, offset):
    return amplitude * np.exp(-(np.abs(x - center)/(2*sigma))**(2*power)) + offset

#=== Supa-Dupa Gaussian Piskunov style with truncation and trending! Buy it!
def gaussian_pisk(x, amplitude, center, sigma, offset, slope):
    z = (x - center) / sigma
    ez = np.exp(-z**2 / 2.0) * (np.abs(z) <= 7.0)
    return amplitude * ez + offset + slope * x

def moffat_profile(radius, intensity, width, slope, background, center):
     return intensity * (1 + ((radius - center) / width)**2)**(-slope) + background


# ========== REDUCE GETXWD to detect slice bounds ==========

def fit_gaussian(x, y, initial_guess=None):
    """
    Подгонка гауссианой по Пискунову (REDUCE)
    
    Parameters:
    -----------
    x : array_like
        Координаты
    y : array_like
        Значения профиля
    initial_guess : list, optional
        Начальное приближение [amplitude, center, sigma, offset]
    
    Returns:
    --------
    popt : array
        Оптимальные параметры [amplitude, center, sigma, offset]
    pcov : array
        Матрица ковариации
    fitted_curve : array
        Подогнанная кривая
    """
    if initial_guess is None:
        # Автоматическое определение начального приближения
        amplitude = np.max(y) - np.min(y)
        center = x[np.argmax(y)]
        sigma = (x[-1] - x[0]) / 6  # Примерная оценка
        offset = np.min(y)
        slope = (y[-1] - y[0]) / (x[-1] - x[0])  # Грубая оценка наклона
        initial_guess = [amplitude, center, sigma, offset, slope]
    
    try:
        popt, pcov = curve_fit(gaussian_pisk, x, y, p0=initial_guess)
        fitted_curve = gaussian_pisk(x, *popt)
        return popt, pcov, fitted_curve
    except Exception as e:
        warnings.warn(f"Gaussian fitting failed: {e}")
        return initial_guess, None, gaussian_pisk(x, *initial_guess)

def estimate_width_getxwd(profile, y_positions, gauss=True, pixels=True):
    """
    """
    
    if gauss:
        # Gaussian fitting
       # y_local = np.arange(len(profile))
        popt, _, fitted = fit_gaussian(y_positions, profile)
        amplitude, center_local, sigma, offset, slope = popt
        
        # Iterative robust fitting with outliers rejection
       #Original:  kgood=where(abs(pg-prof) lt 0.33*ag(0), ngood)
       # with f=a(0)*exp(-z^2/2)+a(3)+a(4)*x
       # where: z=(x-a(1))/a(2)
       # 
       #  7 because the Gaussian model determines 6 parameters

        residuals = np.abs(fitted - profile)
        good_mask = residuals < 0.33 * amplitude
        
        if np.sum(good_mask) > 7 and np.sum(good_mask) <= len(profile):
            popt, _, _ = fit_gaussian(y_positions[good_mask], profile[good_mask])
            amplitude, center, sigma, offset, slope = popt

        
        # Profile y limits (truncate on 7 sigmas as Nikolay did)
        z_max = 2.0 # ПАРАМЕТР ДЛЯ ПОДГОНКИ 

        y_min_limit = center - z_max * sigma - 2
        y_max_limit = center + z_max * sigma + 2

        yym1 = max(y_positions[0], y_min_limit)
        yym2 = min(y_positions[-1], y_max_limit)

     #   yym1 = max(0, int(np.floor(center_local - z_max * sigma - 2)))
      #  yym2 = min(len(profile) - 1, int(np.ceil(center_local + z_max * sigma + 2)))
        pkfrac = 0.85
        if pixels:
            width = pkfrac*(yym2 - yym1) 
        else:
            width = pkfrac * (yym2 - yym1 + 1) / len(profile)
        return width, center, popt

    else:
        # Пороговая модель getxwd
        pmin = np.min(profile) #background trough counts
        pmax = np.max(profile) #order peak counts
        threshold = np.sqrt(max(pmin, 1) * pmax) * 1.0 # Геометрическое среднее
        
        keep_mask = profile > threshold

        nkeep = np.sum(keep_mask)
        pkfrac = 0.9
        
        if pixels:
            # IDL: xwd[0,0]=0.5+0.5*nkeep+1, xwd[1,0]=0.5+0.5*nkeep+1
            width = pkfrac * (1.5 + 1.0*nkeep)  # fraction of order to extract
        else:
            width = pkfrac * (0.5 + 0.5 * nkeep + 1) / len(profile) # fraction of order to extract
        
        # Оценка центра как среднего значения пикселей выше порога
        if nkeep > 0:
            center_indices = np.where(keep_mask)[0]
            center_local = np.mean(center_indices)
            center = y_positions[int(center_local)] if int(center_local) < len(y_positions) else y_positions[len(y_positions)//2]
        else:
            center = y_positions[len(y_positions)//2]
    return width, center

def plot_vertical_profile(profile, peaks=None, output_dir=None):
    """Plot vertical profile with detected peaks
    """
    
    plt.figure(figsize=(12, 6))
    plt.plot(profile, 'k-', label='Profile')
    
    if peaks is not None:
        plt.plot(peaks, profile[peaks], 'rx', label='Detected orders')
        
        # Add peak numbers
        for i, peak in enumerate(peaks):
            plt.text(peak, profile[peak], f' {i+1}', 
                    color='red', fontsize=8)
    
    plt.title('Vertical Profile with Detected Orders')
    plt.xlabel('Y pixel')
    plt.ylabel('Median intensity')
    plt.grid(True, alpha=0.3)
    plt.legend()
     #plt.show()
    
    if output_dir:
        plt.savefig(output_dir / 'vertical_profile.pdf')
    
    return profile

def plot_order_fit(image, x, profile, y_coordinates, order_num, width_getxwd, fit_params=None):
    """Plot order profile fit at a specific position"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Image section
    plt.subplot(131)
    y_center = y_coordinates[len(profile)//2]  # Center of the profile
    y_start = max(0, int(y_center - 50))
    y_end = min(image.shape[0], int(y_center + 50))
    x_start = max(0, x - 100)
    x_end = min(image.shape[1], x + 100)

    plt.imshow(image[y_start:y_end, x_start:x_end], aspect='auto', cmap='gray',
               extent=[x_start, x_end, y_end, y_start])
    plt.axvline(x, color='r', linestyle='--', alpha=0.5)

    # Показываем ширину от getxwd 
    if fit_params is not None:
        width = width_getxwd  
        center_img = fit_params[1]
        plt.axhline(center_img - width/2, color='purple',ls=':', alpha=0.7, label='Gaussian_width')
        plt.axhline(center_img + width/2, color='purple',ls=':', alpha=0.7)
    else:
        plt.axhline(y_center - width_getxwd/2, color='g',ls='--', label='getxwd width')
        plt.axhline(y_center + width_getxwd/2, color='g',ls='--')
    
    plt.title(f'Order {order_num} at x={x}')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.legend()
    
    # Plot 2: Profile and fit
    plt.subplot(132)
    plt.plot(y_coordinates, profile, 'ko-', label='Data', markersize=3)
    
    if fit_params is not None:
        y_fit = np.linspace(y_coordinates[0], y_coordinates[-1], 100)
        fit_curve = gaussian_pisk(y_fit, *fit_params)
        plt.plot(y_fit, fit_curve, 'r-', label='Gaussian fit')
        
        center = fit_params[1]
        width = width_getxwd  # FWHM
        plt.axvline(center, color='g', linestyle='--', label='Center')
        plt.axvline(center - width/2, color='b', linestyle=':', alpha=0.5)
        plt.axvline(center + width/2, color='b', linestyle=':', alpha=0.5)

    else:
        center_idx = y_coordinates[len(profile) // 2]
        plt.axvline(center_idx - width_getxwd/2, color='orange', linestyle='-', alpha=0.8, label='getxwd bounds')
        plt.axvline(center_idx + width_getxwd/2, color='orange', linestyle='-', alpha=0.8)

    plt.title('Profile and Fit')
    plt.xlabel('Y pixel')
    plt.ylabel('Intensity')
    plt.legend()
    
    # Plot 3: Residuals if fit exists
    plt.subplot(133)
    if fit_params is not None:
        fit_at_data = gaussian_pisk(y_coordinates, *fit_params)
        residuals = profile - fit_at_data
        plt.plot(y_coordinates, residuals, 'ko-', markersize=3)
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.title('Fit Residuals')
        plt.ylabel('Residual')
        plt.xlabel('Y pixel')
    else:
        plt.text(0.5, 0.5, 'Using getxwd\nwidth estimation', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
        plt.title('Width from getxwd')
    
    plt.tight_layout()
    #plt.show()
    plt.pause(0.01)
    plt.close()

def find_order_boundaries(image, peaks,border_width=50):
    # 1. Аппроксимируем фон по краям изображения
    spatial_axis = np.arange(image.shape[0])

    # Берем данные из краевых областей
    left_bg = image[:border_width, :].mean(axis=1)
    right_bg = image[-border_width:, :].mean(axis=1)
    bg_data = np.concatenate([left_bg, right_bg])
    bg_positions = np.concatenate([spatial_axis[:border_width], 
                                 spatial_axis[-border_width:]])
    
        # Линейная аппроксимация фона
    slope, intercept, _, _, _ = linregress(bg_positions, bg_data)

    # 2. Функция для поиска границ в заданной области
    def find_boundary(search_start, search_end):
        profile = image[search_start:search_end, :].mean(axis=1)
        x = np.arange(search_start, search_end)
        bg = slope * x + intercept
        diff = np.abs(profile - bg)
        return search_start + np.argmin(diff)
    
    # 3. Внутренние границы между порядками
    boundaries = []
    if len(peaks) >= 2:
        for i in range(len(peaks)-1):
            search_radius = min(20, (peaks[i+1] - peaks[i])/4)
            search_center = (peaks[i] + peaks[i+1]) / 2
            boundary = find_boundary(
                int(max(0, search_center - search_radius)),
                int(min(image.shape[0], search_center + search_radius))
            )
            boundaries.append(boundary)
  
    # Левая граница (левее первого пика)
    if len(peaks) > 0:
        left_boundary = find_boundary(peaks[0]-20,peaks[0])
        boundaries.insert(0, left_boundary)
        
        # Правая граница (правее последнего пика)
        right_boundary = find_boundary( peaks[-1],peaks[-1]+20)
        boundaries.append(right_boundary)

    return {
        'boundaries': np.array(boundaries),
        'background_model': (slope, intercept)
    }

def find_order_boundaries_from_flat(flat_file, 
                                   n_orders=14,
                                   smooth=True, 
                                   smooth_sigma=1.0,
                                   plot=True,
                                   save_plots=False,
                                   plots_path=None):
    """
    Определение границ спектральных порядков по флэт-кадру
    
    Parameters:
    -----------
    flat_file : str
        Путь к среднему флэт-кадру для определения границ порядков
    n_orders : int
        Количество порядков для поиска
    smooth : bool
        Применить сглаживание к вертикальному профилю
    smooth_sigma : float
        Параметр сглаживания
    plot : bool
        Показывать графики
    save_plots : bool
        Сохранять графики на диск
    plots_path : Path, optional
        Путь для сохранения графиков
    
    Returns:
    --------
    dict : Результаты анализа границ
        {
            'peaks': np.array,
            'bounds': dict,
            'flat_data': np.array,
            'flat_header': Header,
            'vertical_profile': np.array,
            'success': bool,
            'error': str or None
        }
    """
    
    try:
        print(f"Анализ флэт-кадра {flat_file}...")
        
        # Загрузка флэт-кадра
        flat_data, flat_header, _ = fits_loader(flat_file)
        print(f"Загружен флэт-кадр размером {flat_data.shape}")
        
        # Анализируем вертикальный профиль
        print("Анализ вертикального профиля...")
        vertical_profile = np.median(flat_data, axis=1)
        
        if smooth:
            vertical_profile = gaussian_filter1d(vertical_profile, sigma=smooth_sigma)
            print(f"Применено сглаживание с sigma={smooth_sigma}")
        
        # Поиск пиков (спектральных порядков)
        prominence = np.percentile(vertical_profile, 25)
        peaks, properties = find_peaks(vertical_profile, prominence=prominence, width=4)
        
        # Отбираем нужное количество порядков
        if n_orders is not None and len(peaks) > n_orders:
            peak_heights = vertical_profile[peaks]
            strongest_indices = np.argsort(peak_heights)[-n_orders:]
            peaks = np.sort(peaks[strongest_indices])
        
        print(f"Найдено {len(peaks)} спектральных порядков: {peaks}")
        
        # Визуализация найденных порядков
        if plot:
            plot_vertical_profile(vertical_profile, peaks, 
                                output_dir=plots_path if save_plots else None)
        
        # Определение границ порядков
        print("Определение границ спектральных порядков...")
        bounds = find_order_boundaries(flat_data, peaks, border_width=50)
        
        print(f"Границы порядков: {bounds['boundaries']}")
        
        # Визуализация границ
        if plot:
            plt.figure(figsize=(12, 6))
            profile = flat_data.mean(axis=1)
            plt.plot(profile, label='Средний профиль')
            
            bg_line = bounds['background_model'][0] * np.arange(len(profile)) + bounds['background_model'][1]
            plt.plot(bg_line, '--', label='Модель фона')
            
            for i, bound in enumerate(bounds['boundaries']):
                plt.axvline(x=bound, color='g', linestyle=':', alpha=0.7, 
                           label='Границы порядков' if i == 0 else "")
            
            for i, peak in enumerate(peaks):
                plt.axvline(x=peak, color='r', linestyle=':', alpha=0.8,
                           label='Центры порядков' if i == 0 else "")
            
            plt.title("Определение границ спектральных порядков")
            plt.xlabel("Пространственная ось (пиксели)")
            plt.ylabel("Интенсивность")
            plt.legend()
            
            if save_plots and plots_path:
                plt.savefig(plots_path / "order_boundaries.pdf")
            plt.show()
        
        return {
            'peaks': peaks,
            'bounds': bounds,
            'flat_data': flat_data,
            'flat_header': flat_header,
            'vertical_profile': vertical_profile,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Ошибка при анализе флэт-кадра {flat_file}: {e}"
        print(f"ОШИБКА: {error_msg}")
        return {
            'peaks': None,
            'bounds': None,
            'flat_data': None,
            'flat_header': None,
            'vertical_profile': None,
            'success': False,
            'error': error_msg
        }

def save_order_boundaries(peaks, bounds, flat_file, output_path):
    """
    Сохранение границ порядков в JSON файл
    
    Parameters:
    -----------
    peaks : np.array
        Позиции пиков порядков
    bounds : dict
        Данные границ от find_order_boundaries()
    flat_file : str
        Путь к исходному флэт-файлу
    output_path : Path
        Путь к директории для сохранения
    
    Returns:
    --------
    Path : Путь к сохраненному файлу
    """
    
    boundaries_data = {
        'metadata': {
            'flat_file': flat_file,
            'n_orders': len(peaks),
            'peaks': [float(p) for p in peaks],
            'boundaries': [float(b) for b in bounds['boundaries']],
            'background_model': bounds['background_model']
        }
    }
    
    boundaries_file = output_path / "order_boundaries.json"
    with open(boundaries_file, 'w') as f:
        json.dump(boundaries_data, f, indent=4)
    
    print(f"Границы порядков сохранены в {boundaries_file}")
    return boundaries_file

def trace_single_spectrum(spec_file, peaks, bounds, 
                         n_points_for_fit=10,
                         getxwd_gauss=True,
                         plot=True, 
                         save_plots=False,
                         plots_path=None):
    """
    Трассировка спектральных порядков для одного изображения
    
    Parameters:
    -----------
    spec_file : str
        Путь к FITS файлу спектра
    peaks : array-like
        Позиции пиков порядков
    bounds : dict
        Данные границ порядков от find_order_boundaries()
    n_points_for_fit : int
        Количество узлов аппроксимации порядка
    getxwd_gauss : bool
        Использовать гауссову аппроксимацию в getxwd
    plot : bool
        Показывать промежуточные графики
    save_plots : bool
        Сохранять графики на диск
    plots_path : Path, optional
        Путь для сохранения графиков
    
    Returns:
    --------
    dict : Результаты трассировки
        {
            'metadata': dict,
            'orders': list,
            'success': bool,
            'error': str or None
        }
    """
    
    start_time = time.time()
    base_name = Path(spec_file).stem
    
    try:
        # Загрузка спектра
        spec_data, spec_header, _ = fits_loader(spec_file)
        print(f"Загружен спектр размером {spec_data.shape}")
        
        # Трассировка каждого порядка
        traced_orders = []
        
        for order_num, peak in enumerate(peaks, 1):
            print(f"\nТрассировка порядка {order_num}/{len(peaks)}...")

            centers = []
            order_widths = []
            x_positions = []
            
            # Определяем границы для этого порядка
            y_start = bounds['boundaries'][order_num - 1]
            y_end = bounds['boundaries'][order_num]
            
            # Трассировка вдоль дисперсионного направления
            step_size = max(1, spec_data.shape[1] // n_points_for_fit)
           
            for x in range(0, spec_data.shape[1], step_size):
                try:
                    # Окно для усреднения
                    x_start = max(0, x - step_size // 2)
                    x_end = min(spec_data.shape[1], x + step_size // 2)
                   
                    # Извлекаем локальный профиль
                    local_profile = np.median(spec_data[y_start:y_end, x_start:x_end], axis=1)
                    y_positions = np.arange(y_start, y_end)
            
                    # Применяем getxwd для определения центра и ширины
                    if getxwd_gauss:
                        width_getxwd, center_getxwd, local_popt = estimate_width_getxwd(
                            local_profile, y_positions, gauss=getxwd_gauss)
                        
                        if plot and x % (step_size * 5) == 0:  # Показываем каждый 5-й результат
                            plot_order_fit(spec_data, x, local_profile, y_positions,
                                         order_num, width_getxwd, fit_params=local_popt)
                    else:
                        width_getxwd, center_getxwd = estimate_width_getxwd(
                            local_profile, y_positions, gauss=getxwd_gauss)
                        
                        if plot and x % (step_size * 5) == 0:  # Показываем каждый 5-й результат
                            plot_order_fit(spec_data, x, local_profile, y_positions,
                                         order_num, width_getxwd, fit_params=None)
                    
                    centers.append(center_getxwd)
                    order_widths.append(width_getxwd)
                    x_positions.append(x)
                    
                except Exception as e:
                    print(f"  ОШИБКА в позиции x={x}: {e}")
                    continue
                    
            # Аппроксимация полиномом Чебышева
            if len(centers) > 3:
                print(f"Аппроксимация порядка {order_num} полиномом Чебышева...")
                
                # Полиномиальная аппроксимация центров
                trace_coeffs = chebfit(x_positions, centers, deg=2)
                width = np.median(order_widths)
                
                # Генерируем полную трассу
                x_full = np.arange(spec_data.shape[1])
                y_center_full = chebval(x_full, trace_coeffs)
                y_upper_full = y_center_full - width / 2
                y_lower_full = y_center_full + width / 2
                
                order_data = {
                    'order_number': int(order_num),
                    'peak_position': int(peak),
                    'median_width': float(width),
                    'chebyshev_coeffs': trace_coeffs.tolist(),
                    'trace_points': {
                        'x_measured': [float(x) for x in x_positions],
                        'y_measured': [float(y) for y in centers],
                        'widths_measured': [float(w) for w in order_widths]
                    },
                    'trace_full': {
                        'x': [int(x) for x in x_full],
                        'y_center': [float(y) for y in y_center_full],
                        'y_upper': [float(y) for y in y_upper_full],
                        'y_lower': [float(y) for y in y_lower_full]
                    }
                }
                
                traced_orders.append(order_data)
                print(f"  Порядок {order_num}: медианная ширина = {width:.2f} пикс")
                print(f"  Коэффициенты Чебышева: {trace_coeffs}")

            else:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: Недостаточно точек для порядка {order_num}")
                
        # Визуализация результата трассировки
        if plot and traced_orders:
            z_scale = ZScaleInterval()
            z1, z2 = z_scale.get_limits(spec_data)
            
            plt.figure(figsize=(15, 8))
            plt.imshow(spec_data, aspect='auto', cmap='coolwarm', vmin=z1, vmax=z2)
            
            for i, order in enumerate(traced_orders):
                # Показываем измеренные точки
                plt.scatter(order['trace_points']['x_measured'], 
                           order['trace_points']['y_measured'], 
                           c='red', s=50, 
                           label='Измеренные центры' if i == 0 else "", 
                           alpha=0.8, zorder=5)
        
                # Показываем аппроксимацию
                plt.plot(order['trace_full']['x'], order['trace_full']['y_center'], 
                        'yellow', linewidth=2, 
                        label='Аппроксимация Чебышева' if i == 0 else "")
                plt.plot(order['trace_full']['x'], order['trace_full']['y_upper'], 
                        'cyan', linewidth=1, alpha=0.8,
                        label='Границы порядков' if i == 0 else "")
                plt.plot(order['trace_full']['x'], order['trace_full']['y_lower'], 
                        'cyan', linewidth=1, alpha=0.8)
                    
            plt.title(f'{base_name} - трассированные порядки')
            plt.xlabel('X пиксель')
            plt.ylabel('Y пиксель')
            plt.legend()
            
            if save_plots and plots_path:
                plt.savefig(plots_path / f"{base_name}_traces.pdf")
            plt.pause(0.01)
            plt.close()
        
        # Формируем результаты
        result_data = {
            'metadata': {
                'source_file': spec_file,
                'processing_time': float(time.time() - start_time),
                'n_orders_traced': int(len(traced_orders)),
                'parameters': {
                    'getxwd_gauss': getxwd_gauss,
                    'n_points_for_fit': n_points_for_fit
                }
            },
            'orders': traced_orders,
            'success': True,
            'error': None
        }
        
        return result_data
        
    except Exception as e:
        error_msg = f"Ошибка при обработке {spec_file}: {e}"
        print(f"ОШИБКА: {error_msg}")
        return {
            'metadata': {
                'source_file': spec_file,
                'processing_time': float(time.time() - start_time),
                'n_orders_traced': 0
            },
            'orders': [],
            'success': False,
            'error': error_msg
        }

def trace_orders(flat_file="s_flat.fits", 
                spec_list="TEMP/obj_crr_bt_list.txt",
                output_dir="traced_orders",
                n_orders=14,
                n_points_for_fit=10,
                smooth=True, 
                smooth_sigma=1.0, 
                getxwd_gauss=True,
                plot=True, 
                save_plots=False, 
                overwrite=False,
                save_format='json'):
    """
    Пакетная трассировка спектральных порядков
    
    Parameters:
    -----------
    flat_file : str
        Путь к среднему флэт-кадру для определения границ порядков
    spec_list : str
        Файл со списком спектров для обработки
    output_dir : str
        Директория для сохранения результатов
    n_orders : int
        Количество порядков
    n_points_for_fit : int
        Количество узлов аппроксимации порядка
    smooth : bool
        Применить сглаживание к вертикальному профилю
    smooth_sigma : float
        Параметр сглаживания
    getxwd_gauss : bool
        Использовать гауссову аппроксимацию в getxwd
    plot : bool
        Показывать промежуточные графики
    save_plots : bool
        Сохранять графики на диск
    overwrite : bool
        Перезаписывать существующие файлы
    save_format : str
        Формат сохранения результатов ('json', 'fits', 'both')
    """
    
    # Создаем директорию для результатов
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if save_plots:
        plots_path = output_path / "plots"
        plots_path.mkdir(exist_ok=True)
    else:
        plots_path = None
    
    print("="*80)
    print("ПАКЕТНАЯ ТРАССИРОВКА СПЕКТРАЛЬНЫХ ПОРЯДКОВ")
    print("="*80)
    
    # ===============================================================
    # ШАГ 1: Определение границ порядков по флэт-кадру
    # ===============================================================
    print(f"\nШаг 1: Анализ флэт-кадра {flat_file}...")
    
    boundary_analysis = find_order_boundaries_from_flat(
        flat_file=flat_file,
        n_orders=n_orders,
        smooth=smooth,
        smooth_sigma=smooth_sigma,
        plot=plot,
        save_plots=save_plots,
        plots_path=plots_path
    )
    
    if not boundary_analysis['success']:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {boundary_analysis['error']}")
        return

    peaks = boundary_analysis['peaks']
    bounds = boundary_analysis['bounds']
    
    # ===============================================================
    # ШАГ 2: Сохранение границ порядков
    # ===============================================================
    print(f"\nШаг 2: Сохранение границ порядков...")
    
    boundaries_file = save_order_boundaries(peaks, bounds, flat_file, output_path)
     
    # ===============================================================
    # ШАГ 3: Загрузка списка спектров
    # ===============================================================
    print(f"\nШаг 3: Загрузка списка спектров из {spec_list}...")
    
    try:
        with open(spec_list, 'r') as f:
            spectrum_files = [line.strip() for line in f if line.strip()]
        print(f"Найдено {len(spectrum_files)} спектров для обработки")
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить список спектров {spec_list}: {e}")
        return
    
    # ===============================================================
    # ШАГ 4: Пакетная обработка спектров
    # ===============================================================
    print(f"\nШаг 4: Обработка спектров...")
    
    processed_count = 0
    failed_count = 0
    
    for spec_idx, spec_file in enumerate(spectrum_files, 1):
        print(f"\n{'='*60}")
        print(f"Обработка спектра {spec_idx}/{len(spectrum_files)}: {spec_file}")
        print(f"{'='*60}")
        
        # Проверяем, нужно ли перезаписывать
        base_name = Path(spec_file).stem
        output_json = output_path / f"{base_name}_traced.json"
        output_fits = output_path / f"{base_name}_traced.fits"
        
        if not overwrite and (output_json.exists() or output_fits.exists()):
            print(f"ПРОПУСК: Файл уже обработан (используйте --overwrite для перезаписи)")
            continue

        # Трассировка одного спектра
        result_data = trace_single_spectrum(
            spec_file=spec_file,
            peaks=peaks,
            bounds=bounds,
            n_points_for_fit=n_points_for_fit,
            getxwd_gauss=getxwd_gauss,
            plot=plot,
            save_plots=save_plots,
            plots_path=plots_path
        )
        
        if result_data['success']:
            # ===============================================================
            # ШАГ 5: Сохранение результатов
            # ===============================================================
            print(f"\nСохранение результатов для {spec_file}...")
            
            if save_format in ['json', 'both']:
                with open(output_json, 'w') as f:
                    json.dump(result_data, f, indent=4)
                print(f"Сохранено в JSON: {output_json}")
            
            if save_format in ['fits', 'both']:
                # Создаем маску порядков
                spec_data, spec_header, _ = fits_loader(spec_file)
                orders_mask = np.zeros_like(spec_data, dtype=np.int16)
                
                for order_data in result_data['orders']:
                    order_num = order_data['order_number']
                    x_coords = np.array(order_data['trace_full']['x'])
                    y_upper = np.array(order_data['trace_full']['y_upper'])
                    y_lower = np.array(order_data['trace_full']['y_lower'])
                    
                    for x in x_coords:
                        if 0 <= x < orders_mask.shape[1]:
                            y_start = max(0, int(np.floor(y_upper[x])))
                            y_end = min(orders_mask.shape[0], int(np.ceil(y_lower[x])))
                            orders_mask[y_start:y_end, x] = order_num
                
                # Сохраняем в FITS с дополнительным слоем
                primary_hdu = pyfits.PrimaryHDU(spec_data, header=spec_header)
                mask_hdu = pyfits.ImageHDU(orders_mask, name='ORDERS_MASK')
                
                # Добавляем метаданные в заголовок
                mask_hdu.header['EXTNAME'] = 'ORDERS_MASK'
                mask_hdu.header['NORDERS'] = len(result_data['orders'])
                mask_hdu.header['TRACEALG'] = 'GETXWD+CHEBYSHEV'
                
                hdul = pyfits.HDUList([primary_hdu, mask_hdu])
                hdul.writeto(output_fits, overwrite=overwrite)
                print(f"Сохранено в FITS: {output_fits}")
            
            processed_count += 1
            
        else:
            failed_count += 1
            print(f"ОШИБКА: {result_data['error']}")
            response = input('Продолжить обработку? (y/n): ')
            if response.lower() not in ['y', 'yes', '']:
                break
    
    # ===============================================================
    # Финальная статистика
    # ===============================================================
    print(f"\n{'='*80}")
    print("ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"{'='*80}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {failed_count}")
    print(f"Результаты сохранены в: {output_path}")
    print(f"Границы порядков: {boundaries_file}")
    
    if save_plots:
        print(f"Графики сохранены в: {plots_path}")

if __name__ == '__main__':
  
    # ===============================================================
    # Основной запуск
    # ===============================================================
    parser = argparse.ArgumentParser(description='Пакетная трассировка спектральных порядков')
    parser.add_argument('--flat', default='s_flat.fits', help='Флэт-кадр для определения границ')
    parser.add_argument('--list', default='TEMP/obj_crr_bt_list.txt', help='Список спектров')
    parser.add_argument('--output', default='traced_orders', help='Директория для результатов')
    parser.add_argument('--n_orders', type=int, help='Количество порядков')
    parser.add_argument('--no-gauss', action='store_false', dest='getxwd_gauss', 
                       help='Не использовать гауссову аппроксимацию')
    parser.add_argument('--no-plot', action='store_false', dest='plot',
                       help='Отключить визуализацию')
    parser.add_argument('--save-plots', action='store_true',
                       help='Сохранять графики на диск')
    parser.add_argument('--overwrite', action='store_true',
                       help='Перезаписывать существующие файлы')
    parser.add_argument('--format', choices=['json', 'fits', 'both'], default='json',
                       help='Формат сохранения результатов')
    
    args = parser.parse_args()
    
    trace_orders(flat_file="s_flat.fits", 
                spec_list="TEMP/obj_crr_bt_list.txt",
                output_dir="traced_orders",
                n_orders=14,
                n_points_for_fit=10,
                smooth=True, 
                smooth_sigma=1.0, 
                getxwd_gauss=False,
                plot=True, 
                save_plots=True, 
                overwrite=True,
                save_format='json')

    print('\nDone!')