import numpy as np
from pathlib import Path
import astropy.io.fits as pyfits
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval

def gaussian(x, amplitude, center, sigma, offset):
    """Gaussian function for fitting"""
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

def super_gaussian(x, amplitude, center, sigma, power, offset):
    return amplitude * np.exp(-(np.abs(x - center)/(2*sigma))**(2*power)) + offset

def plot_vertical_profile(image, peaks=None, output_dir=None):
    """Plot vertical profile with detected peaks"""
    vertical_profile = np.median(image, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(vertical_profile, 'k-', label='Profile')
    
    if peaks is not None:
        plt.plot(peaks, vertical_profile[peaks], 'rx', label='Detected orders')
        
        # Add peak numbers
        for i, peak in enumerate(peaks):
            plt.text(peak, vertical_profile[peak], f' {i+1}', 
                    color='red', fontsize=8)
    
    plt.title('Vertical Profile with Detected Orders')
    plt.xlabel('Y pixel')
    plt.ylabel('Median intensity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_dir:
        plt.savefig(output_dir / 'vertical_profile.pdf')
    plt.show()
    
    return vertical_profile

def plot_order_fit(image, x, y_center, profile, fit_params, order_num):
    """Plot order profile fit at a specific position"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Image section
    plt.subplot(131)
    y_start = max(0, int(y_center - 50))
    y_end = min(image.shape[0], int(y_center + 50))
    x_start = max(0, x - 100)
    x_end = min(image.shape[1], x + 100)
  
    plt.imshow(image[y_start:y_end, x_start:x_end], aspect='auto', cmap='gray')
    plt.axvline(x - x_start, color='r', linestyle='--', alpha=0.5)
    
    if fit_params is not None:
        width = fit_params[2] * 2.355  # FWHM
        plt.axhline((y_center - y_start) - width/2, color='g',ls='--')
        plt.axhline((y_center - y_start) + width/2, color='g',ls='--')
    plt.title(f'Order {order_num} at x={x}')
    
    # Plot 2: Profile and fit
    plt.subplot(132)
    y = np.arange(len(profile))
    plt.plot(y, profile,  'ko-', label='Data', markersize=3)
    
    if fit_params is not None:
        y_fit = np.linspace(y[0], y[-1], 100)
        fit = super_gaussian(y_fit, *fit_params)
        plt.plot(y_fit, fit, 'r-', label='Gaussian fit')
        
        center = fit_params[1]
        width = fit_params[2] * 2.355  # FWHM
        plt.axvline(center, color='g', linestyle='--', label='Center')
        plt.axvline(center - width/2, color='b', linestyle=':', alpha=0.5)
        plt.axvline(center + width/2, color='b', linestyle=':', alpha=0.5)
    
    plt.title('Profile and Fit')
    plt.xlabel('Intensity')
    plt.ylabel('Y pixel')
    plt.legend()
    
    # Plot 3: Residuals if fit exists
    plt.subplot(133)
    if fit_params is not None:
        fit_at_data =super_gaussian(y, *fit_params)
        residuals = profile - fit_at_data
        plt.plot(y, residuals, 'ko-', markersize=3)
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.title('Fit Residuals')
        plt.ylabel('Residual')
        plt.xlabel('Y pixel')
    
    plt.tight_layout()
 #   plt.show()
    plt.pause(1)
    plt.close()

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
    y_window = int(typical_separation * 0.5)  # 50% от расстояния между порядками
    
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
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        
        # Берем центральную область
        region = image[center_y-y_window:center_y+y_window,
                      center_x-x_window:center_x+x_window]
        
        signal = np.median(region)
        noise = np.std(region - signal)
        return signal / noise if noise > 0 else 0
    
    # Корректируем размеры окон на основе SNR
    snr = estimate_snr(image, x_window, y_window)
    # if snr < 10:  # если SNR низкий
    #     x_window = int(x_window * 1.5)  # увеличиваем окно
    #     y_window = int(y_window * 1.2)
    
    return {
        'x_window': x_window,
        'y_window': y_window,
        'typical_separation': typical_separation,
        'snr': snr
    }

def trace_orders(image, n_orders=None):
    """Trace spectral orders with visual debugging"""
    print("Step 1: Analyzing vertical profile...")
    vertical_profile = np.median(image, axis=1)
    
    # Find peaks
    prominence = np.percentile(vertical_profile, 25) 
    print(f"prominence: {prominence}")
    peaks, properties = find_peaks(vertical_profile, prominence=prominence, width=4)
    
    if n_orders is not None and len(peaks) > n_orders:
        # Take n_orders strongest peaks
        peak_heights = vertical_profile[peaks]
        strongest_indices = np.argsort(peak_heights)[-n_orders:]
        peaks = np.sort(peaks[strongest_indices])
    
    # Plot vertical profile
    plot_vertical_profile(image, peaks)
    
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
        for x in range(0, image.shape[1], 460):
            # # Get local profile
            # x_start = max(0, x - 50)
            # x_end = min(image.shape[1], x + 50)
            # y_start = max(0, peak - 30)
            # y_end = min(image.shape[0], peak + 30)

            # Используем оцененные размеры окон
            x_start = max(0, x - windows['x_window']//2)
            x_end = min(image.shape[1], x + windows['x_window']//2)
            y_start = max(0, peak - windows['y_window'])
            y_end = min(image.shape[0], peak + windows['y_window'])
            
            local_profile = np.median(image[y_start:y_end, x_start:x_end], axis=1)
            power_gauss=0.0
            try:
                # Fit Gaussian
                p0 = [np.max(local_profile) - np.min(local_profile),  # amplitude
                      len(local_profile) // 2,  # center
                      len(local_profile) / 10,  # sigma
                      power_gauss,                        # power
                      np.min(local_profile)]    # offset
                              
                popt, _ = curve_fit(super_gaussian, np.arange(len(local_profile)), 
                                  local_profile, p0=p0,
                                  bounds = ([0, 0, 0, 0.0, 0],             # нижние границы
                                            [np.inf, len(local_profile), np.inf, 4.0, np.inf]))  # верхние границы
       
                print(f"""
                      Estimated curve parameters:
                      Amplitude {popt[0]},
                      Center {popt[1]},
                      Sigma width {popt[2]},
                      Power of sigma {popt[3]},
                      Offset {popt[4]}""")                
                
                # Plot fit
                plot_order_fit(image, x, y_start + popt[1], local_profile, popt, order_num)
                
                # Store results if fit looks good
                if 0 < popt[1] < len(local_profile):
                    centers.append(y_start + popt[1])
                    order_widths.append(popt[2] * 2.355)  # FWHM
                    x_positions.append(x)
                
            except RuntimeError:
                print(f"  Failed to fit at x={x}")
                continue
        
        if len(centers) > 3:
            # Fit polynomial to centers
            trace_coeffs = chebfit(x_positions, centers, deg=2)
            width = np.median(order_widths)
            
            traces.append(trace_coeffs)
            widths.append(width)
            
            # Plot trace
            x_fit = np.arange(image.shape[1])
            y_fit = chebval(x_fit, trace_coeffs)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(image, aspect='auto', cmap='gray')
            plt.plot(x_positions, centers, 'r.', label='Measured centers')
            plt.plot(x_fit, y_fit, 'b-', label='Polynomial fit')
            plt.title(f'Order {order_num} Trace')
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
    
    # Trace orders
    print('Tracing orders...')
    traces, widths = trace_orders(flat_data, n_orders=14)
    
    print(f'\nFound {len(traces)} orders')
    
    # Save traces to file
    print('Saving traces...')
    with open(temp_directory / 'simple_traces.txt', 'w') as f:
        for i, (trace, width) in enumerate(zip(traces, widths)):
            f.write(f'Order {i+1}\n')
            f.write(f'Trace coefficients: {" ".join(map(str, trace))}\n')
            f.write(f'Width: {width}\n')
            f.write('-' * 40 + '\n')
    
    print('Done!')
