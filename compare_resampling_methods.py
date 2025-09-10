import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json

# --- Критические импорты, которые нужны вашим функциям ---
from astropy.io import fits
from astropy.stats import mad_std
from scipy.signal import find_peaks, correlate
from astropy.modeling import models, fitting
# from scipy.optimize import curve_fit # Закомментировано в вашем коде

# --- Импорты из specutils и astropy для переинтерполяции ---
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.manipulation import resample
from astropy import units as u


# ==============================================================================
# ЧАСТЬ 1: ВАШ КОД, ПРЕДОСТАВЛЕННЫЙ В ПРЕДЫДУЩЕМ СООБЩЕНИИ
# Нижеследующие функции скопированы как есть, без изменений.
# ==============================================================================

def fits_loader(file_path, hdu_index=0, dtype=np.float32):
    file_path = Path(file_path)
    with fits.open(file_path) as hdul:
        data = hdul[hdu_index].data.astype(dtype)
        header = hdul[hdu_index].header
    return data, header, str(file_path)

def load_atlas_lines(file_path):
    try:
        lines = np.loadtxt(file_path, comments='#', usecols=(1,))
        print(f"Успешно загружено {len(lines)} линий из атласа '{file_path}'.")
        return np.sort(lines)
    except FileNotFoundError:
        print(f"Ошибка: Файл атласа не найден по пути: {file_path}")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла атласа '{file_path}': {e}")
        return None

def extract_all_orders(image_data, traced_data):
    all_fluxes = []
    orders_to_extract = traced_data.get('orders', [])
    x = None
    for order_info in orders_to_extract:
        order_num = order_info['order_number']
        x, flux_array = extract_order(image_data, traced_data, order_num)
        all_fluxes.append(flux_array)
    print("\nЭкстракция всех порядков завершена.")
    if x is None:
        # Если ни один порядок не был извлечен, создаем пустую ось X
        x = np.arange(image_data.shape[1])
    return x, all_fluxes

def find_peaks_for_order(x_coords, flux, peak_params):
    noise_estimate = mad_std(flux)
    prominence_threshold = peak_params['prominence_sigma'] * noise_estimate
    peaks, properties = find_peaks(
        flux, prominence=prominence_threshold, width=peak_params['width_range'],
        distance=peak_params['distance_pixels'], rel_height=0.5)
    if len(peaks) == 0:
        return np.array([])
    fitter = fitting.LevMarLSQFitter()
    final_peak_centers = []
    for i, peak_idx in enumerate(peaks):
        try:
            fwhm_guess = properties['widths'][i]
            window_radius = int(np.ceil(fwhm_guess * 3))
            start_idx = max(0, peak_idx - window_radius)
            end_idx = min(len(flux) - 1, peak_idx + window_radius + 1)
            if (end_idx - start_idx) < 5: continue
            x_window = x_coords[start_idx:end_idx]
            y_window = flux[start_idx:end_idx]
            gauss_part = models.Gaussian1D(
                amplitude=flux[peak_idx] - np.median([y_window[0], y_window[-1]]),
                mean=x_coords[peak_idx],
                stddev=fwhm_guess / 2.3548)
            bg_part = models.Const1D(amplitude=np.median([y_window[0], y_window[-1]]))
            compound_model = gauss_part + bg_part
            fitted_model = fitter(compound_model, x_window, y_window, maxiter=2000)
            fit_center = fitted_model.mean_0.value
            fit_amplitude = fitted_model.amplitude_0.value
            fit_sigma = fitted_model.stddev_0.value
            if not (x_window[0] < fit_center < x_window[-1]) or fit_sigma < 0 or fit_amplitude < 0:
                 continue
            final_peak_centers.append(fit_center)
        except (RuntimeError, ValueError):
            continue
    return np.array(final_peak_centers)

def fit_dispersion_poly(pixel_coords, lambda_coords, poly_deg):
    if len(pixel_coords) <= poly_deg:
        raise ValueError(f"Для подгонки полинома {poly_deg} нужно {poly_deg + 1} точек. Дано: {len(pixel_coords)}")
    coeffs = np.polyfit(pixel_coords, lambda_coords, poly_deg)
    poly_model = np.poly1d(coeffs)
    return poly_model

def interactive_wavelength_calibration(found_peaks, atlas_lines, ax_spectrum, x_coords_orig, flux_orig, order_num, gain, ron_e):
    print("\n" + "="*80 + "\n        Интерактивный режим калибровки по длинам волн\n" + "="*80)
    calib_points = {}
    pixel_centers = np.array(found_peaks)
    disp_model = None
    while True:
        print("\n--- Статус калибровки ---")
        print(f"Найдено сопоставлений: {len(calib_points)}")
        if calib_points:
            sorted_pixels = sorted(calib_points.keys())
            for px in sorted_pixels:
                print(f"  - Пиксель {px:.3f} -> {calib_points[px]:.4f} Å")
        cmd = input("\nДействия: [a]dd, [d]elete, [p]redict, [f]it, [q]uit: ").lower()

        if cmd == 'a':
            try:
                px_input = float(input("  Введите координату пика (пиксель): "))
                closest_peak_idx = np.argmin(np.abs(pixel_centers - px_input))
                px_val = pixel_centers[closest_peak_idx]
                lambda_input = float(input(f"  Введите длину волны для пика {px_val:.3f} (Å): "))
                calib_points[px_val] = lambda_input
                print(f"-> Добавлено: ({px_val:.3f}, {lambda_input:.4f})")
                if len(calib_points) >= 2:
                    current_deg = min(len(calib_points) - 1, 5)
                    try:
                        px, lmb = zip(*calib_points.items())
                        disp_model = fit_dispersion_poly(px, lmb, current_deg)
                        print(f"-> Модель обновлена (полином {current_deg}-й степени).")
                    except ValueError as e:
                        print(f"! {e}"); disp_model = None
            except ValueError: print("! Некорректный ввод. Вводите числа.")
        elif cmd == 'd':
            if not calib_points: print("! Нет точек для удаления."); continue
            try:
                px_to_del = float(input("  Введите пиксель точки для удаления: "))
                key_to_del = min(calib_points.keys(), key=lambda k: abs(k-px_to_del))
                if abs(key_to_del - px_to_del) < 1:
                    del calib_points[key_to_del]
                    print(f"-> Точка {key_to_del:.3f} удалена."); disp_model = None
                else: print(f"! Точка ~{px_to_del} не найдена.")
            except ValueError: print("! Некорректный ввод.")
        elif cmd == 'p':
            if disp_model is None: print("! Сначала добавьте >= 2 точек."); continue
            unidentified_peaks = [p for p in pixel_centers if p not in calib_points]
            if not unidentified_peaks: print("-> Все пики сопоставлены."); continue
            print("\n" + "-"*80 + f"\n{'Пик (пикс)':>12} | {'Предсказано (Å)':>18} | {'Ближайший в атласе (Å)':>22} | {'Разница (Å)':>12}\n" + "-"*80)
            for peak_px in unidentified_peaks:
                predicted_lambda = disp_model(peak_px)
                nearest_atlas_lambda = find_nearest_line(predicted_lambda, atlas_lines)
                diff = predicted_lambda - nearest_atlas_lambda
                print(f"{peak_px:12.3f} | {predicted_lambda:18.4f} | {nearest_atlas_lambda:22.4f} | {diff:12.4f}")
            print("-" * 80)
        elif cmd == 'f':
            if len(calib_points) < 4: print("! Недостаточно точек для надежного фита (нужно >= 4)."); continue
            try:
                final_deg = int(input(f"  Введите степень полинома для финализации (рекомендуется 3-5): "))
                px, lmb = zip(*calib_points.items())
                final_model = fit_dispersion_poly(px, lmb, final_deg)
                is_finalized = finalize_and_resample(final_model, calib_points, found_peaks, atlas_lines,
                                        x_coords_orig, flux_orig, order_num, gain, ron_e)
                if is_finalized:
                    print("\n" + "="*80 + "\nФинальное решение принято и будет использовано.\n" + "="*80)
                    return {'model_coeffs': final_model.coef.tolist(), 'model_degree': final_model.order,
                            'order_num': order_num, 'calib_points': calib_points}
            except ValueError as e: print(f"Ошибка: {e}. Попробуйте снова.")
        elif cmd == 'q':
            if input("Выйти из калибровки без сохранения? [y/n]: ").lower() == 'y': break
    print("--- Выход из режима калибровки без сохранения решения ---"); return None

def fit_polynomial_robustly(x_data, y_data, degree, sigma_threshold=3.0, max_iterations=3):
    if len(x_data) < degree + 1: return None, None
    model_poly = models.Polynomial1D(degree=degree)
    fitter = fitting.LinearLSQFitter()
    current_x, current_y = np.copy(x_data), np.copy(y_data)
    fitted_model = None
    for i in range(max_iterations):
        num_points_before = len(current_x)
        if len(current_x) < degree + 1: print("  Предупреждение: недостаточно точек."); return None, None
        try: fitted_model = fitter(model_poly, current_x, current_y)
        except Exception as e: print(f"  Ошибка аппроксимации: {e}"); return None, None
        residuals = current_y - fitted_model(current_x)
        residual_sigma = mad_std(residuals, ignore_nan=True)
        inlier_mask = np.abs(residuals) < sigma_threshold * residual_sigma if residual_sigma > 0 else np.ones_like(residuals, dtype=bool)
        current_x, current_y = current_x[inlier_mask], current_y[inlier_mask]
        num_points_after = len(current_x)
        if num_points_after == num_points_before: break
    if fitted_model is None: return None, None
    final_residuals = y_data - fitted_model(x_data)
    final_stat_sigma = mad_std(final_residuals, ignore_nan=True)
    final_mask = np.abs(final_residuals) < sigma_threshold * final_stat_sigma if final_stat_sigma > 0 else np.ones_like(final_residuals, dtype=bool)
    return fitted_model, final_mask

def match_peaks_robustly(ref_spectrum, target_spectrum, ref_positions):
    ref_segment = ref_spectrum[500:-500] - np.median(ref_spectrum[500:-500])
    target_segment = target_spectrum[500:-500] - np.median(target_spectrum[500:-500])
    correlation = correlate(ref_segment, target_segment, mode='same')
    lag_array = np.arange(-len(correlation)//2, len(correlation)//2)
    global_shift_estimate = lag_array[np.argmax(correlation)]
    
    refined_search_window = 18
    target_peak_params = {'prominence_sigma': 4, 'width_range': (2.1, 12), 'distance_pixels': 8}
    target_positions = find_peaks_for_order(np.arange(len(target_spectrum)), target_spectrum, target_peak_params)
    
    matched_ref_pos, shifts = [], []
    if len(target_positions) == 0: return np.array([]), np.array([])
    for ref_pos in ref_positions:
        distances = np.abs(target_positions - (ref_pos + global_shift_estimate))
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < refined_search_window:
            target_pos = target_positions[closest_idx]
            matched_ref_pos.append(ref_pos)
            shifts.append(target_pos - ref_pos)
    return np.array(matched_ref_pos), np.array(shifts)

# ==============================================================================
# ЧАСТЬ 2: ФУНКЦИИ-ЗАГЛУШКИ ДЛЯ НЕДОСТАЮЩИХ ЗАВИСИМОСТЕЙ
# ВАЖНО: Замените содержимое этих функций на ваш реальный код!
# ==============================================================================

def load_traced_orders(json_file):
    """
    Загружает результаты трассировки спектральных порядков из JSON файла
    
    Parameters:
    -----------
    json_file : str or Path
        Путь к JSON файлу с результатами трассировки
    
    Returns:
    --------
    dict : Словарь с метаданными и данными трассировки
        {
            'metadata': dict,
            'orders': list of dict,
            'n_orders': int
        }
    """
    json_path = Path(json_file)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Файл трассировки не найден: {json_file}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON файла {json_file}: {e}")
    
    # Проверяем структуру данных
    if 'metadata' not in data or 'orders' not in data:
        raise ValueError(f"Неправильная структура файла трассировки: {json_file}")
    
    # Добавляем удобные поля
    data['n_orders'] = len(data['orders'])
    
    return data

def get_order_trace(traced_data, order_number, x_range=None):
    """
    Извлекает трассировку для конкретного порядка
    
    Parameters:
    -----------
    traced_data : dict
        Данные трассировки от load_traced_orders()
    order_number : int
        Номер порядка для извлечения
    x_range : tuple, optional
        (x_start, x_end) - диапазон X для ограничения трассы
    
    Returns:
    --------
    dict : Данные трассировки порядка
        {
            'order_number': int,
            'x': np.array,
            'y_center': np.array,
            'y_upper': np.array,
            'y_lower': np.array,
            'width': float,
            'coeffs': np.array
        }
    """
    # Ищем нужный порядок
    order_data = None
    for order in traced_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        available_orders = [o['order_number'] for o in traced_data['orders']]
        raise ValueError(f"Порядок {order_number} не найден. "
                        f"Доступные порядки: {available_orders}")
    
    # Извлекаем координаты
    x = np.array(order_data['trace_full']['x'])
    y_center = np.array(order_data['trace_full']['y_center'])
    y_upper = np.array(order_data['trace_full']['y_upper'])
    y_lower = np.array(order_data['trace_full']['y_lower'])
    
    # Ограничиваем диапазон если нужно
    if x_range is not None:
        x_start, x_end = x_range
        mask = (x >= x_start) & (x <= x_end)
        x = x[mask]
        y_center = y_center[mask]
        y_upper = y_upper[mask]
        y_lower = y_lower[mask]
    
    return {
        'order_number': order_data['order_number'],
        'x': x,
        'y_center': y_center,
        'y_upper': y_upper,
        'y_lower': y_lower,
        'width': order_data['median_width'],
        'coeffs': np.array(order_data['chebyshev_coeffs'])
    }


def extract_order(image_data, traced_data, order_number):
    img_height, img_width = image_data.shape
    trace = get_order_trace(traced_data, order_number)

    x_coords = trace['x']
    y_upper_trace = trace['y_upper']
    y_lower_trace = trace['y_lower']

    extracted_flux = np.zeros(img_width)

    for i in range(len(x_coords)):
        x_col = int(x_coords[i])

        if not (0 <= x_col < img_width):
            continue

        y_start = int(np.ceil(y_upper_trace[i]))
        y_end = int(np.floor(y_lower_trace[i]))

        if y_start <= y_end:
            column_flux = np.sum(image_data[y_start:y_end, x_col])
            extracted_flux[x_col] = column_flux

    return np.arange(img_width), extracted_flux

def find_nearest_line(reference_lambda, atlas_lines):
    """Находит ближайшую линию в атласе к заданной длине волны."""
    # np.searchsorted требует отсортированного массива, что мы обеспечиваем в load_atlas_lines
    idx = np.searchsorted(atlas_lines, reference_lambda, side="left")
    
    if idx == 0:
        return atlas_lines[0]
    if idx == len(atlas_lines):
        return atlas_lines[-1]
    
    # Сравниваем с левым и правым соседом
    left_neighbor = atlas_lines[idx - 1]
    right_neighbor = atlas_lines[idx]
    if abs(reference_lambda - left_neighbor) < abs(reference_lambda - right_neighbor):
        return left_neighbor
    else:
        return right_neighbor

def finalize_and_resample(final_model, calib_points_dict, all_found_peaks, atlas_lines, x_coords_orig, flux_orig, order_num,gain,ron_e):
    """
    Выполняет финальные шаги: создает ЕДИНЫЙ СВОДНЫЙ ГРАФИК с остатками для ВСЕХ линий, 
    пересчитывает спектр и сохраняет результаты.
    """
    print("\n--- Финализация решения ---")
    
    # --- 1. Подготовка данных ---
    px_ident = np.array(list(calib_points_dict.keys()))
    lmb_ident = np.array(list(calib_points_dict.values()))
    unidentified_peaks_px = np.setdiff1d(all_found_peaks, px_ident)
    residuals_ident = lmb_ident - final_model(px_ident)
    rms = np.sqrt(np.mean(residuals_ident**2))

    if len(unidentified_peaks_px) > 0:
        unid_model_lmb = final_model(unidentified_peaks_px)
        unid_atlas_lmb = np.array([find_nearest_line(l, atlas_lines) for l in unid_model_lmb])
        residuals_unid = unid_atlas_lmb - unid_model_lmb
    else:
        unid_atlas_lmb = []
        residuals_unid = []

    x_min, x_max = x_coords_orig.min(), x_coords_orig.max()
    lambda_min, lambda_max = final_model(x_min), final_model(x_max)
    if lambda_min > lambda_max:
        lambda_min, lambda_max = lambda_max, lambda_min

    # --- 2. АНАЛИЗ РЕШЕНИЯ И СОЗДАНИЕ ЕДИНОГО ГРАФИКА ---
    print("1. Анализ остатков и создание сводного графика решения...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Дисперсионное решение для порядка №{order_num}", fontsize=16)

    # --- Верхний график: Решение на фоне данных ---
    
    x_smooth = np.linspace(x_min, x_max, 500)
    ax1.plot(x_smooth, final_model(x_smooth), 'r-', lw=2, label=f"Полином {final_model.order}-й степени")
    ax1.scatter(px_ident, lmb_ident, facecolors='none', edgecolors='blue', s=150, lw=1.5,
                label=f"Опорные точки, RMS={rms:.5f}")
    ax1.scatter(unidentified_peaks_px, unid_atlas_lmb, c='gray', marker='|', s=100, 
                lw=1.5, label=f"Найденные (не опознаны)")

    ax1.set_ylabel("Длина волны (Å)"); ax1.set_xlim(x_min, x_max); ax1.set_ylim(lambda_min, lambda_max)
    ax1.grid(True, linestyle=':', which='both')
    
    info_text = (f"RMS ошибки (по опорным точкам): {rms:.5f} Å\n"
                 f"Точек в решении: {len(px_ident)}\n"
                 f"Степень полинома: {final_model.order}")
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.legend(loc='lower right')

    # --- Нижний график: Остатки для ВСЕХ найденных линий ---
    ax2.axhline(0, color='red', linestyle='--', lw=1)
    # Остатки для ОПОРНЫХ точек (Истинная λ - Модельная λ)
    ax2.scatter(px_ident, residuals_ident, marker='x', color='blue', s=70, lw=1.5,
                label='Опорные точки')
    
    # Остатки для НЕОПОЗНАННЫХ точек (Ближайшая λ в атласе - Модельная λ)
    if len(unidentified_peaks_px) > 0:
        ax2.scatter(unidentified_peaks_px, residuals_unid, marker='+', color='gray', s=50,
                    label='Неопознанные точки')

    ax2.set_xlabel("Положение на детекторе (пиксели)"); ax2.set_ylabel("Остатки (Атлас - Модель), Å")
    ax2.grid(True, linestyle=':')
    ax2.legend(loc='upper right')

    pdf_filename = f"order_{order_num}_dispersion_solution.pdf"
    fig.savefig(pdf_filename)
    print(f"   -> Сводный график решения сохранен в файл: {pdf_filename}")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Оставляем место для suptitle
    plt.show()

    user_choice = input("\nГрафик остатков вас устраивает? Продолжить с переинтерполяцией и сохранением? [y/n]: ").lower()
    if user_choice == 'y':
        print("\n--- Финализация и сохранение результатов ---")
        print("\n1. Переинтерполяция спектра на линейную сетку длин волн...")

        CRPIX1 = 1.0      
        CRVAL1 = final_model(x_coords_orig[0])
        NAXIS1 = len(x_coords_orig)
        lambda_start = final_model(x_coords_orig[0])
        lambda_end = final_model(x_coords_orig[-1])
        CDELT1 = (lambda_end - lambda_start) / (NAXIS1 - 1)
       # dispersion_func = final_model.deriv()
       # CDELT1 = np.min(np.abs(dispersion_func(x_coords_orig)))
        target_dispersion = (CRVAL1 + (np.arange(NAXIS1) - (CRPIX1 - 1)) * CDELT1) * u.AA
        
        # ПЗС-ошибки по потокам считаются по заданным GAIN, RON
        error_adu = calculate_ccd_error(flux_orig, gain, ron_e)
        uncertainty = StdDevUncertainty(error_adu * u.adu)
# Готовим монотонную сетку для specutils
        original_wavelengths = final_model(x_coords_orig) * u.AA
        sort_indices = np.argsort(original_wavelengths)

        # Создаем объект Spectrum1D. Важно: сетка должна быть строго монотонной.
        input_spectrum = Spectrum1D(
            flux=flux_orig[sort_indices] * u.adu,
            spectral_axis=original_wavelengths[sort_indices],
            uncertainty=uncertainty[sort_indices]
        )

        flux_resampler = resample.FluxConservingResampler(extrapolation_treatment='zero_fill')
        resampled_spectrum = flux_resampler(input_spectrum, target_dispersion)

        print("\n2. Сохранение результатов линеаризации опорного спектра...")
        wcs_filename = f"order_{order_num}_wcs_standard.txt"
        with open(wcs_filename, 'w') as f:
            f.write("# FITS WCS standard keywords defined from reference spectrum\n")
            f.write(f"NAXIS1 = {NAXIS1}\n")
            f.write(f"CRPIX1 = {CRPIX1}\n")
            f.write(f"CRVAL1 = {CRVAL1:.8f}\n")
            f.write(f"CDELT1 = {CDELT1:.8f}\n")
        print(f"  -> Стандартные параметры WCS сохранены в: {wcs_filename}")
        
        
        return True  # <--- Возвращаем True, если пользователь согласен
    else:
        print("-> Отмена. Возврат в меню калибровки...")
        return False # <--- Возвращаем False, если пользователь хочет попробовать снова

def find_and_plot_lines(x_coords, flux, ax, prominence_sigma, width_range, distance_pixels):

    noise_estimate = mad_std(flux)
    prominence_threshold = prominence_sigma * noise_estimate
    print(f"  Оценка уровня шума: {noise_estimate:.2f} ADU")
    print(f"  Порог значимости (Prominence > {prominence_sigma}*шум): {prominence_threshold:.2f} ADU")
    print(f"  Диапазон ширины: {width_range} пикс.")
    print(f"  - Мин. дистанция: {distance_pixels} пикс.")

    peaks, properties = find_peaks(
        flux, 
        prominence=prominence_threshold, 
        width=width_range,
        distance=distance_pixels
    )

    # Удаление старых маркеров и фитов перед отрисовкой новых
    artists_to_remove = []
    artists_to_check = ax.get_lines() + ax.texts
    for artist in artists_to_check:
        if artist.get_gid() in ["line_marker", "line_fit"]:
            artists_to_remove.append(artist)
    for artist in artists_to_remove:
        artist.remove()

    if len(peaks) == 0:
        print("--> Спектральные линии не найдены с текущими параметрами.")
        ax.figure.canvas.draw_idle()
        return
    
    print(f"\nНайдено {len(peaks)} кандидатов. Выполняется фитинг гауссианой...")
    print("-" * 80)
    print(f"{'#':>3} {'Фит. Центр (X)':>18} {'Амплитуда':>15} {'FWHM (пикс)':>15} {'Лок. Фон':>12} {'СКО фита':>10}")
    print("-" * 80)
    
    fit_count = 0
    fitter = fitting.LevMarLSQFitter()
    final_peak_centers = []
 
    for i, peak_idx in enumerate(peaks):

        fwhm_guess = properties['widths'][i]
        window_radius = int(np.ceil(fwhm_guess * 3))
        start_idx = max(0, peak_idx - window_radius)
        end_idx = min(len(flux) - 1, peak_idx + window_radius + 1)

        if (end_idx - start_idx) < 5:
            continue

        x_window = x_coords[start_idx:end_idx]
        y_window = flux[start_idx:end_idx]

# Initial gaussian estimate

        gauss_part = models.Gaussian1D(
                amplitude=flux[peak_idx] - np.median([y_window[0], y_window[-1]]),
                mean=x_coords[peak_idx],
                stddev=fwhm_guess / 2.3548 # FWHM to sigma
                )
        bg_part = models.Const1D(amplitude=np.median([y_window[0], y_window[-1]]))
        compound_model = gauss_part + bg_part
        #center_guess = x_coords[peak_idx]
        #background_guess = np.median([y_window[0], y_window[-1]])
        #amplitude_guess = flux[peak_idx] - background_guess
        #sigma_guess = fwhm_guess / 2.3548      

        #p0 = [amplitude_guess, center_guess, sigma_guess, background_guess]   
        #popt, pcov = curve_fit(gaussian_with_bg, x_window, y_window, p0=p0)
        #fit_amplitude, fit_center, fit_sigma, fit_bg = popt
        fitted_model = fitter(compound_model, x_window, y_window, maxiter=5)
        fit_center = fitted_model.mean_0.value
        fit_amplitude = fitted_model.amplitude_0.value
        fit_sigma = fitted_model.stddev_0.value
        fit_bg = fitted_model.amplitude_1.value

        if not (x_window[0] < fit_center < x_window[-1]) or fit_sigma < 0 or fit_amplitude < 0:
                 print(f" !  Пропускаем пик #{i+1} в {fit_center:.2f}: фит сошелся к нефизичным параметрам.")
                 continue
        fit_count += 1
        fit_fwhm = fit_sigma * 2.3548  
        
        residuals = y_window - fitted_model(x_window)
        rms_fit = np.sqrt(np.mean(residuals**2))
        print(f"{fit_count:3d} {fit_center:18.3f} {fit_amplitude:15.2f} {fit_fwhm:15.3f} {fit_bg:12.2f} {rms_fit:10.2f}")
        
        final_peak_centers.append(fit_center)

       # ax.axvline(x=fit_center, color='limegreen', linestyle='--', alpha=0.8, gid="line_marker")
        ax.text(fit_center, fit_amplitude + fit_bg, f' {fit_center:.2f}', 
                    color='limegreen', rotation=90, va='bottom', ha='center', fontsize=9, gid="line_marker")
        
        x_fit_smooth = np.linspace(x_window[0], x_window[-1], 200)
        y_fit_smooth = fitted_model(x_fit_smooth)
        ax.plot(x_fit_smooth, y_fit_smooth, color='orange', alpha=0.9, lw=2, gid="line_fit")

    if fit_count == 0:
        print("--> Не удалось успешно подогнать ни одну из найденных линий.")
    else:
        print(f"Успешно подогнано {fit_count} из {len(peaks)} линий.")

    current_handles, current_labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(current_labels, current_handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    ax.figure.canvas.draw_idle()

    return final_peak_centers


# ==============================================================================
# ЧАСТЬ 3: ОСНОВНОЙ СКРИПТ (ОРКЕСТРАТОР И ПЕРЕИНТЕРПОЛЯЦИЯ)
# ==============================================================================

def calculate_ccd_error(flux_adu, gain, ron_e):
    """Вычисляет ошибку в ADU по формуле ПЗС."""
    flux_safe = np.maximum(flux_adu, 0)
    shot_noise_e = np.sqrt(flux_safe * gain)
    total_noise_e = np.sqrt(shot_noise_e**2 + ron_e**2)
    return total_noise_e / gain

def resample_single_spectrum(flux_orig, x_coords_orig, effective_model_func, target_grid, gain, ron_e):
    """Выполняет переинтерполяцию одного спектра на целевую сетку."""
    # specutils может выдать ошибку, если ось не монотонна
    try:
        original_wavelengths = effective_model_func(x_coords_orig)
    except Exception as e:
        print(f"  Ошибка при вычислении effective_model_func для пикселей. Проверьте модель сдвига. {e}")
        return None
        
    sort_indices = np.argsort(original_wavelengths)
    
    input_spectrum = Spectrum1D(
        flux=flux_orig[sort_indices] * u.adu,
        spectral_axis=original_wavelengths[sort_indices] * u.AA,
        uncertainty=StdDevUncertainty(calculate_ccd_error(flux_orig, gain, ron_e)[sort_indices] * u.adu)
    )

    flux_resampler = resample.FluxConservingResampler(extrapolation_treatment='zero_fill')
    return flux_resampler(input_spectrum, target_grid)

def main():
    parser = argparse.ArgumentParser(description="Сравнение методов коррекции искажений.")
    parser.add_argument("fits_file", type=str, help="Путь к FITS-файлу.")
    parser.add_argument("trace_file", type=str, help="Путь к JSON-файлу с трассировкой.")
    parser.add_argument("atlas_file", type=str, help="Путь к файлу атласа линий.")
    args = parser.parse_args()

    GAIN, RON_E = 0.68, 3.8
    PEAK_PARAMS = {'prominence_sigma': 7, 'width_range': (2.1, 12), 'distance_pixels': 8}
    SHIFT_POLY_DEGREE = 3

    print("--- 1. Загрузка данных ---")
    image_data, _, _ = fits_loader(args.fits_file)
    traced_data = load_traced_orders(args.trace_file)
    atlas_lines = load_atlas_lines(args.atlas_file)
    x_coords_orig, all_spectra_1d = extract_all_orders(image_data, traced_data)

    print("\n--- 2. Интерактивная калибровка опорного порядка ---")
    ref_order_num = 7
    ref_order_index = ref_order_num - 1
    ref_spectrum = all_spectra_1d[ref_order_index]
    ref_peaks = find_peaks_for_order(x_coords_orig, ref_spectrum, PEAK_PARAMS)
    print(f"   -> Найдено {len(ref_peaks)} пиков в опорном спектре №{ref_order_num}.")

    # Создаем график для интерактивной функции
    fig_calib, ax_calib = plt.subplots(figsize=(17, 7))
    ax_calib.plot(x_coords_orig, ref_spectrum)
    ax_calib.set_title(f"Интерактивная калибровка для порядка {ref_order_num}")
    ax_calib.set_xlabel("Пиксели"); ax_calib.set_ylabel("Интенсивность")
    current_peaks = find_and_plot_lines(x_coords_orig, ref_spectrum, ax_calib, **PEAK_PARAMS)

    plt.show(block=False) # Показываем окно, но не блокируем выполнение скрипта

    # Вызываем ВАШУ интерактивную функцию с ПРАВИЛЬНЫМИ аргументами
    calibration_result = interactive_wavelength_calibration(
        found_peaks=current_peaks, atlas_lines=atlas_lines, ax_spectrum=ax_calib,
        x_coords_orig=x_coords_orig, flux_orig=ref_spectrum,
        order_num=ref_order_num, gain=GAIN, ron_e=RON_E
    )
    plt.close(fig_calib) # Закрываем интерактивное окно

    if calibration_result is None:
        print("\nКалибровка отменена. Выход."); return

    ref_disp_model = np.poly1d(calibration_result['model_coeffs'])
    print("\n   -> Опорное дисперсионное решение D(p_ref) успешно получено.")

    print("\n--- 3. Поиск моделей искажений S_i(p_ref) ---")
    distortion_models = {}
    for i, target_spectrum in enumerate(all_spectra_1d):
        target_order_num = traced_data['orders'][i]['order_number']
        if target_order_num == ref_order_num:
            distortion_models[target_order_num] = np.poly1d([0.0])
            continue
        matched_ref_pos, shifts = match_peaks_robustly(ref_spectrum, target_spectrum, ref_peaks)
        shift_model, _ = fit_polynomial_robustly(matched_ref_pos, shifts, SHIFT_POLY_DEGREE)
        distortion_models[target_order_num] = shift_model if shift_model is not None else np.poly1d([0.0])
        print(f"   -> Модель искажений для порядка {target_order_num} найдена.")

    print("\n--- 4. Определение единой целевой сетки ---")
    NAXIS1 = len(x_coords_orig)
    CRVAL1 = ref_disp_model(x_coords_orig[0])
    CDELT1 = (ref_disp_model(x_coords_orig[-1]) - CRVAL1) / (NAXIS1 - 1)
    universal_target_grid = (CRVAL1 + np.arange(NAXIS1) * CDELT1) * u.AA
    print(f"   -> Целевая сетка создана (от {universal_target_grid[0]:.2f} до {universal_target_grid[-1]:.2f} с шагом {CDELT1:.4f} Å/пикс).")

    print("\n--- 5. Переинтерполяция двумя методами ---")
    resampled_A, resampled_B = [], []
    for i, flux_orig in enumerate(all_spectra_1d):
        order_num = traced_data['orders'][i]['order_number']
        print(f"   -> Обработка порядка {order_num}...")
        shift_model = distortion_models[order_num]
        def model_A(p): return ref_disp_model(p - shift_model(p))
        def model_B(p):
            p_ref = p; [p_ref := p - shift_model(p_ref) for _ in range(4)]; return ref_disp_model(p_ref)
        spec_A = resample_single_spectrum(flux_orig, x_coords_orig, model_A, universal_target_grid, GAIN, RON_E)
        spec_B = resample_single_spectrum(flux_orig, x_coords_orig, model_B, universal_target_grid, GAIN, RON_E)
        if spec_A: resampled_A.append(spec_A)
        if spec_B: resampled_B.append(spec_B)

    print("\n--- 6. Создание итоговых графиков ---")
    if not resampled_A or not resampled_B: print("Нечего отображать, пересэмплирование не удалось."); return
    
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 14), sharex=True, sharey=True)
    offset = np.max(resampled_A[ref_order_index].flux.value) * 0.7
    for i, spec in enumerate(resampled_A):
        order_num = traced_data['orders'][i]['order_number']
        ax_a.plot(spec.spectral_axis, spec.flux.value, lw=1, label=f"П.{order_num}")
    ax_a.set_title("Метод А (Аппроксимация)", fontsize=16)
    ax_a.grid(True, linestyle=':'); ax_a.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.15))

    for i, spec in enumerate(resampled_B):
        order_num = traced_data['orders'][i]['order_number']
        ax_b.plot(spec.spectral_axis, spec.flux.value, lw=1, label=f"П.{order_num}")
    ax_b.set_title("Метод Б (Итерации)", fontsize=16)
    ax_b.set_xlabel("Длина волны (Å)"); ax_b.grid(True, linestyle=':')
    
    fig.supylabel("Поток (смещен для наглядности)")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


if __name__ == '__main__':
    main()