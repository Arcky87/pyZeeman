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
    Случай для одного порядка (getxwd lines 62--100)

    Profile width estimation (getxwd from REDUCE)
    Calculate the location of order peaks in center of image (local coordinates)
    
    Parameters:
    -----------
    profile : array
        Profile intensity accross the slice
    y_positions : array
        y-coordinates of the profile
    gauss : bool
        Fit with gaussian (yes/no)
    pixels : bool
        Return result in pixels (either in fraction)
    pkfrac : float
        Allowable fraction of peak
    
    Returns:
    --------
    width : float
        full width of the profile
    center : float
        profile center
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
        
        return width, center, popt

    else:
        # Пороговая модель getxwd
        pmin = np.min(profile) #background trough counts
        pmax = np.max(profile) #order peak counts
        threshold = np.sqrt(max(pmin, 1) * pmax) * 0.5
        
        keep_mask = profile > threshold
        nkeep = np.sum(keep_mask)
        
        if pixels:
            # IDL: xwd[0,0]=0.5+0.5*nkeep+1, xwd[1,0]=0.5+0.5*nkeep+1
            width = (0.5 + 0.5 * nkeep + 1)  # fraction of order to extract
        else:
            width = (0.5 + 0.5 * nkeep + 1) / len(profile) # fraction of order to extract
        
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
    
    if output_dir:
        plt.savefig(output_dir / 'vertical_profile.pdf')
    plt.show()
    
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

    print(f"Image section: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]") 
    plt.imshow(image[y_start:y_end, x_start:x_end], aspect='auto', cmap='gray',
               extent=[x_start, x_end, y_end, y_start])
    plt.axvline(x, color='r', linestyle='--', alpha=0.5)

    # Показываем ширину от getxwd 
    if fit_params is not None:
        width = fit_params[2] * 2.355  # FWHM
        center_img = fit_params[1]
   #     print(f"width: {width}, {center_img},  {center_img - width/2,center_img + width/2}")
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
        width = fit_params[2] * 2.355  # FWHM
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

def estimate_window_sizes(image, peaks):
    """
    Оценивает оптимальные размеры окон для анализа профилей
    
    Parameters:
        image: 2D array - изображение
        peaks: array - позиции пиков
        
    Returns:
        dict: размеры окон по X и Y
    """
    # 1. Оценка Y-окна из расстояния между порядками
    peak_separations = np.diff(peaks)
    typical_separation = np.median(peak_separations)
    y_window = int(typical_separation * 0.35)
 
    # 2. Оценка X-окна через автокорреляцию
    def estimate_x_scale(image):
        # Берем центральную часть изображения
        center_y = image.shape[0] // 2
        profile = image[center_y, :]
        
        # Вычисляем автокорреляцию
        autocorr = np.correlate(profile, profile, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Находим первый минимум после центрального пика
        peaks_auto, _ = find_peaks(-autocorr)
        if len(peaks_auto) > 0:
            return peaks_auto[0]
        else:
            return image.shape[1] // 20  # fallback: 5% от ширины
    
    x_scale = estimate_x_scale(image)
    x_window = max(x_scale * 2, image.shape[1] // 20)
    
    # 3. Проверка отношения сигнал/шум
    def estimate_snr(image, x_window, y_window):
        center_x = image.shape[1] // 2 # width / 2
        center_y = image.shape[0] // 2 # heigth / 2
        
        # Берем центральную область
        region = image[max(0, center_y - y_window):min(image.shape[0], center_y + y_window),
                      max(0, center_x - x_window):min(image.shape[1], center_x + x_window)]
        
        signal = np.median(region)
        noise = np.std(region - signal)
        return signal / noise if noise > 0 else 0
    
    # Корректируем размеры окон на основе SNR
    snr = estimate_snr(image, x_window, y_window)
    
    return {
        'x_window': x_window,
        'y_window': y_window,
        'typical_separation': typical_separation,
        'snr': snr
    }

def trace_orders(image, n_orders=None, getxwd_gauss=True, smooth=False, smooth_sigma=1.0):
    """
    """
    print("Step 1: Analyzing vertical profile...")
    vertical_profile = np.median(image, axis=1)
    
    if smooth:
        vertical_profile = gaussian_filter1d(vertical_profile, sigma=smooth_sigma)
        print(f"Applied Gaussian smoothing with sigma={smooth_sigma}")

    # Find peaks
    prominence = np.percentile(vertical_profile, 25) 
    print(f"prominence: {prominence}")
    peaks, properties = find_peaks(vertical_profile, prominence=prominence, width=4)
    
    if n_orders is not None and len(peaks) > n_orders:
        # Take n_orders strongest peaks
        peak_heights = vertical_profile[peaks]
        strongest_indices = np.argsort(peak_heights)[-n_orders:]
        peaks = np.sort(peaks[strongest_indices])
    
    plot_vertical_profile(vertical_profile, peaks)
    
    # Process each order
    traces = []
    widths = []

    # Оцениваем размеры окон
    windows = estimate_window_sizes(image, peaks)
    print(f"Estimated windows: X={windows['x_window']}, Y={windows['y_window']}")
    print(f"Order separation: {windows['typical_separation']:.1f}")
    print(f"SNR: {windows['snr']:.1f}")
    
    for order_num, peak in enumerate(peaks, 1):
        print(f"\nProcessing Order {order_num}...")
        centers = []
        order_widths = []
        x_positions = []
        
        # Sample points along dispersion
        for x in range(0, image.shape[1], 460): #TODO сделать произвольный шаг трассировки (4600/шаг)
            # Используем оцененные размеры окон
            x_start = max(0, x - windows['x_window']//2)
            x_end = min(image.shape[1], x + windows['x_window']//2)
            y_start = max(0, peak - windows['y_window'])
            y_end = min(image.shape[0], peak + windows['y_window'])

            # y_start = max(0, peak - 15)
            # y_end = min(image.shape[0], peak + 15)
            
            local_profile = np.median(image[y_start:y_end, x_start:x_end], axis=1)
            y_positions = np.arange(y_start, y_end)
            
            # Используем алгоритм getxwd
            try:
                if getxwd_gauss:
                    width_getxwd, center_getxwd, local_popt = estimate_width_getxwd(
                        local_profile, y_positions, 
                        gauss=getxwd_gauss)
                    print(f"  getxwd at x={x}: width={width_getxwd:.2f}, center={center_getxwd:.2f}")
                
                    # Plot результат
                    plot_order_fit(image, x, local_profile, y_positions,
                                 order_num, width_getxwd,fit_params=local_popt)  
                else:
                    width_getxwd, center_getxwd = estimate_width_getxwd(
                        local_profile, y_positions, 
                        gauss=getxwd_gauss)
                    print(f"  getxwd at x={x}: width={width_getxwd:.2f}, center={center_getxwd:.2f}")

                    # Plot результат
                    plot_order_fit(image, x, local_profile, y_positions,
                                 order_num, width_getxwd, fit_params=None)
                
                # Store results
                centers.append(center_getxwd)
                order_widths.append(width_getxwd)
                x_positions.append(x)
                
            except Exception as e:
                print(f"  getxwd failed at x={x}: {e}")
                continue
                       
        if len(centers) > 3:
            # Fit polynomial to centers
            trace_coeffs = chebfit(x_positions, centers, deg=2)
            width = np.median(order_widths)
            
            traces.append(trace_coeffs)
            widths.append(width)
            
            print(f"Order {order_num}: median width = {width:.2f}")
            
            # Plot trace
            x_fit = np.arange(image.shape[1])
            y_fit = chebval(x_fit, trace_coeffs)

            trace_upper = y_fit - width / 2
            trace_lower = y_fit + width / 2
            
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
    
    return traces, widths

if __name__ == '__main__':
    # Load flat field
    temp_directory = Path('./TEMP')
    flat_file = 's_flat.fits'
    
    print(f'Reading {flat_file}...')
    with pyfits.open(flat_file) as hdul:
        flat_data = hdul[0].data
    
    # Trace orders with getxwd
    print('Tracing orders with getxwd algorithm...')
    traces, widths = trace_orders(flat_data, n_orders=14, smooth=True,
                                  getxwd_gauss=False)
    
    print(f'\nFound {len(traces)} orders')
    
    # Save traces to file
    temp_directory.mkdir(exist_ok=True)
    print('Saving traces...')
    with open(temp_directory / 'simple_traces_getxwd.txt', 'w') as f:
        f.write('# Traces generated using getxwd algorithm\n')
        f.write('# Format: Order number, Trace coefficients, Width\n')
        f.write('-' * 50 + '\n')
        for i, (trace, width) in enumerate(zip(traces, widths)):
            f.write(f'Order {i+1}\n')
            f.write(f'Trace coefficients: {" ".join(map(str, trace))}\n')
            f.write(f'Width: {width}\n')
            f.write('-' * 40 + '\n')
    
    print('Done!')


# # С getxwd (гауссовская модель)
# traces, widths = trace_orders(image, use_getxwd=True, getxwd_gauss=True)

# # С getxwd (пороговая модель)  
# traces, widths = trace_orders(image, use_getxwd=True, getxwd_gauss=False)

# # Старый метод
# traces, widths = trace_orders(image, use_getxwd=False)