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

#==== My functions ====
def gaussian(x, amplitude, center, sigma, offset):
    """Gaussian function for fitting"""
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

def super_gaussian(x, amplitude, center, sigma, power, offset):
    return amplitude * np.exp(-(np.abs(x - center)/(2*sigma))**(2*power)) + offset

#=== Supa-Dupa Gaussian Piskunov style with truncation and trending! Buy it!
def gaussian_pisk(x, amplitude, center, sigma, offset, slope):
    z = (x - center) / sigma
    ez = np.exp(-z**2 / 2.0) * (np.abs(z) <= 7.0)
    return amplitude * ez + offset + slope * x


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
        
        if np.sum(good_mask) > 7 and np.sum(good_mask) < len(profile):
            popt, _, _ = fit_gaussian(y_positions[good_mask], profile[good_mask])
            amplitude, center, sigma, offset, slope = popt

        print(amplitude, center, sigma, offset, slope)
        
        # Profile y limits (truncate on 7 sigmas as Nikolay did)
        z_max = 7.0

        y_min_limit = center - z_max * sigma - 2
        y_max_limit = center + z_max * sigma + 2

        yym1 = max(y_positions[0], y_min_limit)
        yym2 = min(y_positions[-1], y_max_limit)

     #   yym1 = max(0, int(np.floor(center_local - z_max * sigma - 2)))
      #  yym2 = min(len(profile) - 1, int(np.ceil(center_local + z_max * sigma + 2)))
        
        if pixels:
            width = yym2 - yym1 
        else:
            width = (yym2 - yym1 + 1) / len(profile)
        
        #center = y_positions[int(center_local)] if int(center_local) < len(y_positions) else y_positions[len(y_positions)//2]
        print(f"estimate_width_getxwd returned {width}, {center}, {popt} with gaussian fit")
        return width, center, popt

    else:
        # Пороговая модель getxwd
        pmin = np.min(profile) #background trough counts
        pmax = np.max(profile) #order peak counts
        threshold = np.sqrt(max(pmin, 1) * pmax) *0.9 # Геометрическое среднее
        print(f"Threshold of the slice {threshold}")
        
        keep_mask = profile > threshold

        nkeep = np.sum(keep_mask)
        
        if pixels:
            # IDL: xwd[0,0]=0.5+0.5*nkeep+1, xwd[1,0]=0.5+0.5*nkeep+1
            width = (0.5+ nkeep +2)  # fraction of order to extract
        else:
            width = (0.5 + 0.5 * nkeep + 1) / len(profile) # fraction of order to extract
        
        # Оценка центра как среднего значения пикселей выше порога
        if nkeep > 0:
            center_indices = np.where(keep_mask)[0]
            center_local = np.mean(center_indices)
            center = y_positions[int(center_local)] if int(center_local) < len(y_positions) else y_positions[len(y_positions)//2]
        else:
            center = y_positions[len(y_positions)//2]
    print(f"estimate_width_getxwd returned {width}, {center} without gaussian fit")
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
    
    if output_dir:
        plt.savefig(output_dir / 'vertical_profile.pdf')
    plt.show()
    
    return profile

def plot_order_fit(image, x, profile, y_coordinates, order_num, width_getxwd, fit_params=None):
    """Plot order profile fit at a specific position"""
    plt.figure(figsize=(15, 5))
    
    print(f"Image shape for coodinate setup y_end {image.shape[0]}, x_end {image.shape[1]}")
    # Plot 1: Image section
    plt.subplot(131)
    y_center = y_coordinates[len(profile)//2]  # Center of the profile
    y_start = max(0, int(y_center - 50))
    y_end = min(image.shape[0], int(y_center + 50))
    x_start = max(0, x - 100)
    x_end = min(image.shape[1], x + 100)

    print(f"""Coordinates for imshow at plot_order_fit are y_center={y_center}, 
            y_start={y_start}, y_end={y_end}, x_start={x_start}, x_end={x_end}""")

    plt.imshow(image[y_start:y_end, x_start:x_end], aspect='auto', cmap='gray',
               extent=[x_start, x_end, y_end, y_start])
    plt.axvline(x, color='r', linestyle='--', alpha=0.5)

    # Показываем ширину от getxwd 
    if fit_params is not None:
        width = width_getxwd  
        center_img = fit_params[1]
        print(f"WIDTH FOR FIRST SUBPLOT. Params for plotting the fit (from fit_params). fit_params[2]: {fit_params[2]}, width (fit_params[2] * 2.355): {width}, width_getxwd: {width_getxwd}")
        plt.axhline(center_img - width/2, color='purple',ls=':', alpha=0.7, label='Gaussian_width')
        plt.axhline(center_img + width/2, color='purple',ls=':', alpha=0.7)
    else:
        print(f"WIDTH FOR FIRST SUBPLOT. Params for plotting the fit (without gaussian). width_getxwd:{width_getxwd}, y_center:{y_center}")
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
    plt.show()
    # plt.pause(1)
    # plt.close()

def find_order_boundaries(image, peaks,border_width=50):
    """
    """
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

def trace_orders(image, n_orders=None, getxwd_gauss=True, smooth=False, smooth_along=False, smooth_sigma=1.0, plot=False):

    print("Step 1: Analyzing vertical profile...")
    vertical_profile = np.median(image, axis=1)
    
    if smooth:
        vertical_profile = gaussian_filter1d(vertical_profile, sigma=smooth_sigma)
        print(f"Applied Gaussian smoothing with sigma={smooth_sigma}")

    # Find peaks
    prominence = np.percentile(vertical_profile, 25) 
    print(f"prominence: {prominence}")
    peaks, properties = find_peaks(vertical_profile, prominence=prominence, width=4)
    
    # Отбираем нужное количество порядков
    if n_orders is not None and len(peaks) > n_orders:
        # Take n_orders strongest peaks
        peak_heights = vertical_profile[peaks]
        strongest_indices = np.argsort(peak_heights)[-n_orders:]
        peaks = np.sort(peaks[strongest_indices])
    
    plot_vertical_profile(vertical_profile, peaks)
    
    # Обработка каждого отдельного порядка
    traces = []
    widths = []

    # Оцениваем размеры окон
    bounds = find_order_boundaries(image, peaks, border_width=50)
    plt.figure(figsize=(12, 6))

    # Пространственный профиль (усредненный по длинам волн)
    profile = image.mean(axis=1)
    plt.plot(profile, label='Средний профиль')

    # Фоновая модель
    bg_line = bounds['background_model'][0] * np.arange(len(profile)) + bounds['background_model'][1]
    plt.plot(bg_line, '--', label='Модель фона')
  
    # Внутренние границы
    for bound in bounds['boundaries']:
        plt.axvline(x=bound, color='g', linestyle=':', alpha=0.7, 
                    label='Границы слайсов' if bound == bounds['boundaries'][0] else "")

    plt.title("Определение границ по фону")
    plt.xlabel("Пространственная ось (пиксели)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.show()
    
    for order_num, peak in enumerate(peaks, 1):
        print(f"\nProcessing Order {order_num}...")
        centers = []
        order_widths = []
        x_positions = []
        
        # Sample points along dispersion
        for x in range(0, image.shape[1], 460): #TODO сделать произвольный шаг трассировки (4600/шаг)
            print(f"x = {x}")
            # Используем оцененные размеры окон
            x_start = max(0, x - image.shape[1] // 460)
            x_end = min(image.shape[1], x + image.shape[1] // 460)
            y_start = bounds['boundaries'][order_num -1] 
            y_end = bounds['boundaries'][order_num] 

            # y_start = max(0, peak - 15)
            # y_end = min(image.shape[0], peak + 15)
            
            local_profile = np.median(image[y_start-1:y_end+1, x_start:x_end], axis=1)
            if smooth_along:
                local_profile = gaussian_filter1d(local_profile, sigma=1) ## EXPERIMENTAL
            y_positions = np.arange(y_start-1, y_end+1)
           
            # Используем алгоритм getxwd
            try:
                if getxwd_gauss:
                    width_getxwd, center_getxwd, local_popt = estimate_width_getxwd(
                        local_profile, y_positions, 
                        gauss=getxwd_gauss)
                    print(f"  getxwd at x={x}: width={width_getxwd:.2f}, center={center_getxwd:.2f}")
                    
                    if plot:
                        # Plot результат
                        plot_order_fit(image, x, local_profile, y_positions,
                                 order_num, width_getxwd,fit_params=local_popt)  
                else:
                    width_getxwd, center_getxwd = estimate_width_getxwd(
                        local_profile, y_positions, 
                        gauss=getxwd_gauss)
                    print(f"  getxwd at x={x}: width={width_getxwd:.2f}, center={center_getxwd:.2f}")
                    
                    if plot:
                        # Plot результат
                        plot_order_fit(image, x, local_profile, y_positions,
                                     order_num, width_getxwd, fit_params=None)
                
                # Store results
                centers.append(center_getxwd)
                order_widths.append(width_getxwd)
                x_positions.append(x)
                
            except Exception as e:
                print(f" getxwd failed at x={x}: {e}")
                continue
                       
        if len(centers) > 3:
            fit_result = {
                    'metadata': {},
                    'orders': []
            }
            # Fit polynomial to centers
            trace_coeffs = chebfit(x_positions, centers, deg=2)
            width = np.median(order_widths)
            
            traces.append(trace_coeffs)
            widths.append(width)
            
            print(f"Order {order_num}: median width = {width:.2f}")
            
 
            x_fit = np.arange(image.shape[1])
            y_fit = chebval(x_fit, trace_coeffs)

            trace_upper = y_fit - width / 2
            trace_lower = y_fit + width / 2

            order_data = {
                'slice_number': order_num,
                'width': float(width),
                'boundaries': {
                    'x': x_fit.tolist(),
                    'y_center': y_fit.tolist(),
                    'y_upper': trace_upper.tolist(),
                    'y_lower': trace_lower.tolist()
                }
            }
            fit_result['orders'].append(order_data)


             # Plot trace
            if plot:          
                z_scale = ZScaleInterval()
                z1,z2 = z_scale.get_limits(image)
                plt.figure(figsize=(12, 6))
                plt.imshow(image, aspect='auto', cmap='coolwarm',vmin=z1, vmax=z2)
                plt.plot(x_positions, centers, 'r.', label='Measured centers', markersize=8)
                plt.plot(x_fit, y_fit, 'b-', label='Polynomial fit', linewidth=1.5)
                plt.plot(x_fit, trace_upper, 'y--',linewidth=1,alpha=0.8)
                plt.plot(x_fit, trace_lower, 'y--',linewidth=1,alpha=0.8)
                plt.title(f'Order {order_num} Trace (Width: {width:.2f} pixels)')
                plt.xlabel('X pixel')
                plt.ylabel('Y pixel')
                plt.legend()
                plt.show()
    
    return fit_result

if __name__ == '__main__':
    # Load flat field
    file_path = ('./s_flat.fits')
    flat_data, flat_header,_ = fits_loader(file_path)
    result = trace_orders(flat_data, n_orders=14, smooth=True,
                                  getxwd_gauss=False, smooth_sigma=3)
    
    print(f'\nFound {len(result['orders'])} orders')
    base_name = os.path.splitext(file_path)[0]
    output_file=file_path.strip()
    output_path = base_name + '_orders.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print('Done!')