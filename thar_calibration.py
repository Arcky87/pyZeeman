import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from loaders import load_atlas_lines, fits_loader, load_traced_orders
from visualize_trace import *
from astropy.stats import mad_std
from astropy.modeling import models,fitting 
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.manipulation import resample
from astropy import units as u
import json

def calculate_ccd_error(flux_adu, gain, ron_e):
    """
    Вычисляет ошибку для каждого пикселя по уравнению шума ПЗС.
    
    Args:
        flux_adu (np.ndarray): Массив потока в ADU.
        gain (float): Коэффициент усиления в e-/ADU.
        ron_e (float): Шум считывания в электронах (e-).
        
    Returns:
        np.ndarray: Массив ошибок в тех же единицах, что и поток (ADU).
    """
    # Убедимся, что поток не отрицательный (после вычитания фона) для расчета фотонного шума
    signal_e = np.maximum(0, flux_adu) * gain
    
    # Полный шум в электронах
    noise_e = np.sqrt(signal_e + ron_e**2)
    
    # Возвращаем шум в единицах ADU
    noise_adu = noise_e / gain
    return noise_adu

def find_peaks_for_order(x_coords, flux, peak_params):
    """
    Неинтерактивная версия для поиска и фитинга пиков.
    Возвращает отсортированный массив точных пиксельных центров.
    """
    noise_estimate = mad_std(flux)
    prominence_threshold = peak_params['prominence_sigma'] * noise_estimate
    
    peaks, properties = find_peaks(
        flux, 
        prominence=prominence_threshold, 
        width=peak_params['width_range'],
        distance=peak_params['distance_pixels'],
        rel_height=0.5
    )
    
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
                stddev=fwhm_guess / 2.3548 # FWHM to sigma
                )
            bg_part = models.Const1D(amplitude=np.median([y_window[0], y_window[-1]]))
            compound_model = gauss_part + bg_part
                        
            # center_guess = x_coords[peak_idx]
            # background_guess = np.median([y_window[0], y_window[-1]])
            # amplitude_guess = flux[peak_idx] - background_guess
            # sigma_guess = fwhm_guess / 2.3548

        #    p0 = [amplitude_guess, center_guess, sigma_guess, background_guess]
        #    popt, _ = curve_fit(gaussian_with_bg, x_window, y_window, p0=p0, maxfev=5000)

            fitted_model = fitter(compound_model, x_window, y_window, maxiter=2000)
            fit_center = fitted_model.mean_0.value
            fit_amplitude = fitted_model.amplitude_0.value
            fit_sigma = fitted_model.stddev_0.value
            
            #fit_amplitude, fit_center, fit_sigma, _ = popt
            if not (x_window[0] < fit_center < x_window[-1]) or fit_sigma < 0 or fit_amplitude < 0:
                 continue
            final_peak_centers.append(fit_center)
        except RuntimeError:
            continue

    return np.array(sorted(final_peak_centers))

def gaussian_with_bg(x, amplitude, center, sigma, background):
    """
    Модель гауссианы с постоянным фоном.
    FWHM = 2.355 * sigma
    """
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2)) + background

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
    
def fit_dispersion_poly(pixel_coords, lambda_coords, poly_deg):
    """Подгоняет полином к точкам (пиксель, длина волны)."""
    if len(pixel_coords) <= poly_deg:
        raise ValueError(f"Для подгонки полинома степени {poly_deg} нужно как минимум {poly_deg + 1} точек. Предоставлено: {len(pixel_coords)}")
    
    # np.polyfit(x, y, deg) ожидает именно такой порядок
    coeffs = np.polyfit(pixel_coords, lambda_coords, poly_deg)
    # Создаем объект полинома для удобного вычисления значений
    poly_model = np.poly1d(coeffs)
    return poly_model

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

        flux_resampler = resample.FluxConservingResampler()
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

def interactive_wavelength_calibration(found_peaks, atlas_lines, ax_spectrum, x_coords_orig, flux_orig, order_num,gain,ron_e,initial_calib_points=None):
    """
    Основная интерактивная функция для калибровки по длинам волн.
    `found_peaks` - это DataFrame или dict с точными центрами линий.
    """
    print("\n" + "="*80)
    print("        Интерактивный режим калибровки по длинам волн")
    print("="*80)
    
    # Контейнер для наших надежных сопоставлений: {пиксель: длина_волны}
    calib_points = {}
    # Получаем список пиксельных координат из результатов фитинга
    pixel_centers = np.array(found_peaks)  
    # Инициализируем модель пустой
    disp_model = None
    if initial_calib_points:
        calib_points = initial_calib_points.copy()
        print(f"-> Загружено {len(calib_points)} точек из предыдущей сессии.")
        if len(calib_points) >= 2:
            try:
                deg = min(len(calib_points) - 1, 5)
                px, lmb = zip(*calib_points.items())
                disp_model = fit_dispersion_poly(px, lmb, deg)
                print(f"-> Начальная модель (полином {deg}-й степени) построена.")
            except ValueError as e: print(f"! Ошибка построения начальной модели: {e}")
        else: calib_points = {}
    while True:
        print("\n--- Статус калибровки ---")
        print(f"Найдено сопоставлений: {len(calib_points)}")
        if calib_points:
            # Сортируем для красивого вывода
            sorted_pixels = sorted(calib_points.keys())
            for px in sorted_pixels:
                print(f"  - Пиксель {px:.3f} -> {calib_points[px]:.4f} Å")
        
        # --- Главное меню ---
        cmd = input(
            "\nДействия: [a]dd - добавить сопоставление, [d]elete - удалить, \n"
            "          [p]redict - предсказать линии, [f]it - финализировать, [q]uit - выйти: "
        ).lower()
        # --- ДОБАВИТЬ ТОЧКУ ---
        if cmd == 'a':
            try:
                px_input = float(input("  Введите координату пика (пиксель): "))
                # Ищем ближайший найденный пик к введенному значению
                closest_peak_idx = np.argmin(np.abs(pixel_centers - px_input))
                px_val = pixel_centers[closest_peak_idx]
                
                lambda_input = float(input(f"  Введите длину волны для пика {px_val:.3f} (Å): "))
                
                calib_points[px_val] = lambda_input
                print(f"-> Добавлено: ({px_val:.3f}, {lambda_input:.4f})")

                # Если точек достаточно, сразу обновляем модель
                if len(calib_points) >= 2:
                    current_deg = min(len(calib_points) - 1, 5) # Начинаем с линейной/квадратичной
                    try:
                        px, lmb = zip(*calib_points.items())
                        disp_model = fit_dispersion_poly(px, lmb, current_deg)
                        print(f"-> Модель обновлена (полином {current_deg}-й степени).")
                    except ValueError as e:
                        print(f"! {e}")
                        disp_model = None
            except ValueError:
                print("! Некорректный ввод. Вводите числа.")

        # --- УДАЛИТЬ ТОЧКУ ---
        elif cmd == 'd':
            if not calib_points:
                print("! Нет точек для удаления.")
                continue
            try:
                px_to_del = float(input("  Введите пиксель точки, которую нужно удалить: "))
                # Ищем ключ, ближайший к введенному значению
                key_to_del = min(calib_points.keys(), key=lambda k: abs(k-px_to_del))
                if abs(key_to_del - px_to_del) < 1: # Проверка, что мы не удаляем что-то случайное
                    del calib_points[key_to_del]
                    print(f"-> Точка с пикселем {key_to_del:.3f} удалена.")
                    disp_model = None # Сбрасываем модель, ее нужно пересчитать
                else:
                    print(f"! Точка с пикселем ~{px_to_del} не найдена в списке.")

            except ValueError:
                print("! Некорректный ввод.")
        
        # --- ПРЕДСКАЗАТЬ ЛИНИИ ---
        elif cmd == 'p':
            if disp_model is None:
                print("! Невозможно предсказать. Сначала добавьте хотя бы 2 точки для построения модели.")
                continue
            
            unidentified_peaks = [p for p in pixel_centers if p not in calib_points]
            if not unidentified_peaks:
                print("-> Все найденные пики уже сопоставлены.")
                continue

            print("\n" + "-"*80)
            print(f"{'Пик (пикс)':>12} | {'Предсказано (Å)':>18} | {'Ближайший в атласе (Å)':>22} | {'Разница (Å)':>12}")
            print("-" * 80)

            for peak_px in unidentified_peaks:
                predicted_lambda = disp_model(peak_px)
                nearest_atlas_lambda = find_nearest_line(predicted_lambda, atlas_lines)
                diff = predicted_lambda - nearest_atlas_lambda
                print(f"{peak_px:12.3f} | {predicted_lambda:18.4f} | {nearest_atlas_lambda:22.4f} | {diff:12.4f}")
            print("-" * 80)
        
        # --- ФИНАЛИЗАЦИЯ ---
        elif cmd == 'f':
            if len(calib_points) < 4:
                print("! Недостаточно точек для надежного фита. Рекомендуется хотя бы 5-6.")
                continue
            
            try:
                final_deg = int(input(f"  Введите степень полинома для финальной подгонки (рекомендуется 3 или 4): "))
                px, lmb = zip(*calib_points.items())
                final_model = fit_dispersion_poly(px, lmb, final_deg)
                
                is_finalized = finalize_and_resample(final_model, 
                                        calib_points,       # Словарь {пиксель: λ} опознанных точек
                                        found_peaks,        # Полный список всех найденных пикселей
                                        atlas_lines,        # Полный список всех линий атласа
                                        x_coords_orig, 
                                        flux_orig, 
                                        order_num,
                                        gain,ron_e
                                    )
                if is_finalized:
                        print("\n" + "="*80 + "\nФинальное решение принято и сохранено.\n" + "="*80)
                        if input("\nПрименить эту калибровку к графику? [y/n]: ").lower() == 'y':
                            ax_spectrum.set_xlabel("Длина волны (Å)")
                            new_ticks = final_model(ax_spectrum.get_xticks())
                            ax_spectrum.set_xticks(ax_spectrum.get_xticks())
                            ax_spectrum.set_xticklabels([f"{t:.1f}" for t in new_ticks])
                            ax_spectrum.figure.canvas.draw_idle()
                            print("-> Ось X на графике обновлена.")
                            print(type(final_model),type(order_num),type(calib_points),type(found_peaks))
                        return {
                            'model': final_model.coef.tolist(),
                            'model_degree': final_model.o,
                            'order_num': order_num,
                            'calib_points': calib_points,
                            'all_peaks_px': found_peaks,
                        }            
            except ValueError as e:
                print(f"Ошибка: {e}. Попробуйте снова.") 
        elif cmd == 'q':
            if input("Вы уверены, что хотите выйти из калибровки? [y/n]: ").lower() == 'y':
                break
                
    print("--- Выход из режима калибровки без сохранения решения ---")
    return None

if __name__ == '__main__':
    import pickle

    FITS_FILE_PATH = Path('/data/Observations/test_pyzeeman/o018.fts')
    TRACED_ORDERS_JSON_PATH = Path('/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json')
    ATLAS_FILE_PATH = Path('/data/Observations/test_pyzeeman/thar_lines_blue.txt') # Путь к вашему атласу
    SOLUTION_FILE_PATH = FITS_FILE_PATH.with_suffix('.distortion_model.json')

    atlas_lines = load_atlas_lines(ATLAS_FILE_PATH)
    image_data, header, _ = fits_loader(FITS_FILE_PATH)
    traced_data = load_traced_orders(TRACED_ORDERS_JSON_PATH)

    plt.ion()
    visualize_spectrum_with_orders(image_data, traced_data, header)
   # current_peaks = []

    #ref_order_info = {'num': None, 'model': None, 'peaks': []}
    fig1d = None
    ax1d = None
    while True:
        prompt = "\nВведите номер порядка для извлечения (или нажмите Enter для выхода): "
        user_input = input(prompt)
        if not user_input: break

        peak_params = {
                'prominence_sigma': 15,
                'width_range': [1.8, 3.5],
                'distance_pixels': 10
            }
            
        order_to_extract = int(user_input)
        x, flux = extract_order(image_data, traced_data, order_to_extract)
        fig1d, ax1d = plot_extracted_spectrum(x, flux, order_to_extract, ax=ax1d)

    #    find_peaks_for_order(x, flux,peak_params)

        current_peaks = find_and_plot_lines(x, flux, ax1d, **peak_params)
        ref_order_info = interactive_wavelength_calibration(
            current_peaks, atlas_lines, ax1d, x, flux, order_to_extract,2.78,5.6)
        
        if ref_order_info:
            print(f"\nСохранение информации об эталонном порядке в файл:\n{SOLUTION_FILE_PATH}")
            with open(SOLUTION_FILE_PATH, 'w') as f:
                json.dump(ref_order_info, f, indent=4)
            print("-> Сохранение успешно завершено.")
            
            # После успешной калибровки одного порядка можно выйти из цикла
            break 
        else:
            print("\nКалибровка отменена. Вы можете выбрать другой порядок.")
    plt.ioff()
    plt.show()