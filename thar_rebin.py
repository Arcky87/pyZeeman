import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loaders import load_atlas_lines, fits_loader, load_traced_orders, get_trace_summary
from visualize_trace import *
from thar_calibration import find_peaks_for_order, fit_dispersion_poly,find_and_plot_lines

import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import mad_std


import json
import argparse 
from scipy.signal import correlate,find_peaks
from scipy.optimize import curve_fit
from astropy.stats import mad_std

def fit_polynomial_robustly(x_data, y_data, degree, sigma_threshold=3.0, abs_residual_limit=None, max_iterations=3):
    """
    Выполняет ИТЕРАТИВНУЮ робастную аппроксимацию полиномом с использованием astropy.modeling.
    """
    if len(x_data) < degree + 1:
        return None, None

    # --- Определяем модель и метод фиттинга ---
    model_poly = models.Polynomial1D(degree=degree)
    fitter = fitting.LinearLSQFitter()
    
    current_x = np.copy(x_data)
    current_y = np.copy(y_data)
    fitted_model = None

    for i in range(max_iterations):
        num_points_before = len(current_x)
        
        # Шаг 1: Аппроксимация по текущему набору "хороших" точек
        if len(current_x) < degree + 1:
            print("  Предупреждение: на итерации осталось недостаточно точек.")
            return None, None
        
        # --- ИЗМЕНЕНИЕ ASTROPY: Выполняем подгонку ---
        try:
            fitted_model = fitter(model_poly, current_x, current_y)
        except Exception as e:
            print(f"  Ошибка: не удалось выполнить аппроксимацию на итерации: {e}")
            return None, None
        
        residuals = current_y - fitted_model(current_x)

        # Шаг 3: Применение фильтров
        residual_sigma = mad_std(residuals, ignore_nan=True)
        statistical_mask = np.abs(residuals) < sigma_threshold * residual_sigma if residual_sigma > 0 else np.ones_like(residuals, dtype=bool)
        inlier_mask = statistical_mask
        
   #     if abs_residual_limit is not None:
   #         absolute_mask = np.abs(residuals) < abs_residual_limit
   #         inlier_mask = np.logical_and(statistical_mask, absolute_mask)

        # Шаг 4: Обновление набора "хороших" точек
        current_x = current_x[inlier_mask]
        current_y = current_y[inlier_mask]
        num_points_after = len(current_x)
        
        print(f"  Итерация {i+1}: отброшено {num_points_before - num_points_after} точек. Осталось: {num_points_after}.")

        if num_points_after == num_points_before:
            print("  Набор точек стабилизировался. Завершаем итерации.")
            break

    # --- Финальный этап ---
    # Создаем маску для ИСХОДНЫХ данных для корректного построения графика
    if fitted_model is None:
        return None, None

    final_residuals = y_data - fitted_model(x_data)
    final_stat_sigma = mad_std(final_residuals, ignore_nan=True)
    final_statistical_mask = np.abs(final_residuals) < sigma_threshold * final_stat_sigma if final_stat_sigma > 0 else np.ones_like(final_residuals, dtype=bool)
    final_mask = final_statistical_mask
    
  #  if abs_residual_limit is not None:
  #      final_absolute_mask = np.abs(final_residuals) < abs_residual_limit
  #      final_mask = np.logical_and(final_statistical_mask, final_absolute_mask)
        
    num_inliers = np.sum(final_mask)
    print(f"  Финальная модель: {num_inliers} из {len(x_data)} точек подходят под модель.")

    return fitted_model, final_mask

def match_peaks_robustly(ref_spectrum, target_spectrum, ref_positions):
    """
    Находит соответствующие пики, используя двухпроходный метод:
    1. Кросс-корреляция для нахождения глобального сдвига.
    2. Уточненный поиск пиков в окнах вокруг сдвинутых позиций.

    Returns:
        tuple: (массив сопоставленных опорных позиций, массив вычисленных сдвигов)
    """
    # --- Проход 1: Кросс-корреляция ---
    # Мы используем только центральную часть спектра, чтобы избежать краевых эффектов
    # и уменьшить объем вычислений. Этого достаточно для нахождения глобального сдвига.

    ref_segment = ref_spectrum[500:-500]
    target_segment = target_spectrum[500:-500]

    # Вычитаем медиану, чтобы убрать фон и сфокусироваться на структуре пиков
    ref_segment = ref_segment - np.median(ref_segment)
    target_segment = target_segment - np.median(target_segment)

    # Вычисляем кросс-корреляцию. 'same' означает, что выходной массив будет того же
    # размера, что и входные, а пик будет в центре, если сдвига нет.
    correlation = correlate(ref_segment, target_segment, mode='same')

    # Находим, на сколько пикселей нужно сдвинуть спектры для лучшего совпадения.
    # Это позиция максимума в массиве корреляции.
    lag_array = np.arange(-len(correlation)//2, len(correlation)//2)
    global_shift_estimate = lag_array[np.argmax(correlation)]
    print(f" Global shift is {global_shift_estimate} pixels")
    # --- Проход 2: Уточненный поиск ---

    # Теперь мы знаем примерный сдвиг! Используем его для создания более точных
    # центров поиска для каждой линии.
    refined_search_centers = ref_positions + global_shift_estimate
    refined_search_window = 18

    target_peak_params = {
    'prominence_sigma': 4,
    'width_range': (2.1, 12),
    'distance_pixels': 8,
    }

    # Ищем пики в маленьких окнах вокруг этих УЖЕ СДВИНУТЫХ позиций.
    # Теперь `search_window` может быть маленьким, т.к. мы уже близко к цели.
    target_positions = find_peaks_for_order(np.arange(len(target_spectrum)),
        target_spectrum,
        target_peak_params
    )

    # --- Финальное сопоставление (теперь оно безопасное) ---
    matched_ref_pos = []
    shifts = []

    for ref_pos in ref_positions:
        # Для каждого опорного пика ищем ближайший найденный целевой пик.
        if len(target_positions) == 0: continue

        distances = np.abs(target_positions - (ref_pos + global_shift_estimate))
        closest_idx = np.argmin(distances)

        # Если ближайший пик находится в пределах нашего маленького окна,
        # считаем сопоставление успешным.
        if distances[closest_idx] < refined_search_window:
            target_pos = target_positions[closest_idx]
            matched_ref_pos.append(ref_pos)
            shifts.append(target_pos - ref_pos)

    return np.array(matched_ref_pos), np.array(shifts)

def main():    
    parser = argparse.ArgumentParser(description="Применяет решение от эталонного порядка ко всем остальным.")
    parser.add_argument("fits_file", type=str, help="Путь к исходному FITS-файлу.")
    parser.add_argument("trace_file", type=str, help="Путь к JSON-файлу с трассировкой порядков.")
    args = parser.parse_args()

    FITS_FILE_PATH = Path(args.fits_file)
    TRACED_ORDERS_JSON_PATH = Path(args.trace_file)

    # --- НАСТРОЙКИ ---
    peak_params = {
        'prominence_sigma': 4,
        'width_range': (2.1, 12),
        'distance_pixels': 8,
    }
    POLYNOMIAL_DEGREE = 5

    # --- ЗАГРУЗКА ДАННЫХ ---
    image_data, header, _ = fits_loader(FITS_FILE_PATH)
    traced_data = load_traced_orders(TRACED_ORDERS_JSON_PATH)

    x, all_spectra_1d = extract_all_orders(image_data, traced_data)
    print(f"Экстрагировано {len(all_spectra_1d)} 1D спектров.")

    # --- ВЫБОР ОПОРНОГО ПОРЯДКА ---
    visualize_spectrum_with_orders(image_data, traced_data, header)
    plt.show()
    ref_order_num = int(input("\nВведите номер опорного порядка: "))

    # Находим индекс и сам опорный спектр
    ref_order_index = -1
    for i, order_info in enumerate(traced_data['orders']):
        if order_info['order_number'] == ref_order_num:
            ref_order_index = i
            break
    
    if ref_order_index == -1:
        print(f"Ошибка: Порядок {ref_order_num} не найден.")
        return

    ref_spectrum = all_spectra_1d[ref_order_index]
    
    # Ищем пики на опорном порядке
    ref_peaks = find_peaks_for_order(x, ref_spectrum, peak_params)
    print(f"Найдено {len(ref_peaks)} опорных линий в порядке №{ref_order_num}.")

    # --- ОСНОВНОЙ ЦИКЛ ПО ПОРЯДКАМ ---
    distortion_model = {}  # Словарь для хранения коэффициентов полиномов

    for i, target_spectrum in enumerate(all_spectra_1d):
        target_order_num = traced_data['orders'][i]['order_number']
        if target_order_num == ref_order_num:
            continue

        print(f"\n--- Обработка порядка №{target_order_num} ---")
        
        matched_ref_pos, shifts = match_peaks_robustly(
            ref_spectrum=ref_spectrum,
            target_spectrum=target_spectrum,
            ref_positions=ref_peaks
        )
        print(f"Найдено {len(shifts)} совпадений.")

    #    ABS_SHIFT_LIMIT = 5.0

     #   pre_filter_mask = shifts < ABS_SHIFT_LIMIT
     #   x_prefiltered = matched_ref_pos[pre_filter_mask]
     #   y_prefiltered = shifts[pre_filter_mask]
     #   num_rejected = len(shifts) - len(x_prefiltered)

     #   if num_rejected > 0:
     #       print(f"  Предв. фильтр: отброшено {num_rejected} грубых выбросов (отклонение от медианы > {ABS_SHIFT_LIMIT} пикс).")

     #   if len(x_prefiltered) < POLYNOMIAL_DEGREE + 2:
     #       print("  После предварительного фильтра осталось недостаточно точек.")
     #       continue
#!!!!!!! Из ASTROPY фиттинг
        best_fit_model, inlier_mask_fine = fit_polynomial_robustly(
            x_data=matched_ref_pos, 
            y_data=shifts, 
            degree=POLYNOMIAL_DEGREE, 
            sigma_threshold=5.0,
           # abs_residual_limit=25 
        )

#        coeffs, inlier_mask = fit_polynomial_robustly(
#            x_prefiltered, y_prefiltered, POLYNOMIAL_DEGREE, sigma_threshold=3.0,abs_residual_limit=4.5)
        
        if best_fit_model is None:
            continue

        distortion_model[f'order_{target_order_num}'] = list(best_fit_model.parameters)
       # distortion_model[f'order_{target_order_num}'] = list(coeffs)

        # === НОВАЯ ВИЗУАЛИЗАЦИЯ (2 ГРАФИКА НА КАЖДЫЙ ПОРЯДОК) ===

        # --- График 1: Модель сдвига ---
        fig1, ax1 = plt.subplots(figsize=(12, 7), num=f'Shift Model - Order {target_order_num}')
        ax1.plot(matched_ref_pos, shifts, 'o', label='Совпавшие пики', ms=5)
        # Разделяем точки на "хорошие" и "выбросы" с помощью маски
        x_inliers = matched_ref_pos[inlier_mask_fine]
        y_inliers = shifts[inlier_mask_fine]
        x_outliers = matched_ref_pos[~inlier_mask_fine]
        y_outliers = shifts[~inlier_mask_fine]

        # Рисуем их разными стилями для наглядности
        ax1.plot(x_inliers, y_inliers, 'o', color='blue', label=f'Совпадения ({len(x_inliers)})')
        ax1.plot(x_outliers, y_outliers, 'x', color='red', ms=7, label=f'Выбросы ({len(x_outliers)})')

        
        # Строим кривую полинома
        x_min_plot = np.min(matched_ref_pos) if len(matched_ref_pos) > 0 else 0
        x_max_plot = np.max(matched_ref_pos) if len(matched_ref_pos) > 0 else 4096
        x_curve = np.linspace(x_min_plot, x_max_plot, 500)
        y_curve = best_fit_model(x_curve)

      #  x_curve = np.linspace(np.min(matched_ref_pos), np.max(matched_ref_pos), 500)
      #  y_curve = polynomial(x_curve, *coeffs)
        ax1.plot(x_curve, y_curve, 'k-', lw=2, label=f'Робастная модель (полином {POLYNOMIAL_DEGREE} ст.)')
        
        ax1.set_title(f'Модель сдвига для порядка №{target_order_num} (относительно №{ref_order_num})')
        ax1.set_xlabel('Координата пика в опорном порядке (пикс.)')
        ax1.set_ylabel('Сдвиг (пикс.)')
        ax1.legend()
        ax1.grid(True, linestyle=':')

        # --- График 2: Профили спектров ---
        fig2, ax2 = plt.subplots(figsize=(14, 8), num=f'Spectrum Profiles - Order {target_order_num}')
        ax2.plot(x, ref_spectrum, color='grey', label=f'Опорный спектр (№{ref_order_num})', alpha=0.9)
        ax2.plot(x, target_spectrum, color='c', label=f'Целевой спектр (№{target_order_num})', alpha=0.7)

        # Вычисляем позиции совпавших пиков на целевом спектре
        matched_target_pos = matched_ref_pos + shifts
        
        # Конвертируем позиции в int для индексации
        matched_ref_pos_int = matched_ref_pos.astype(int)
        matched_target_pos_int = matched_target_pos.astype(int)

        # Отмечаем совпавшие пики
        ax2.scatter(matched_ref_pos, ref_spectrum[matched_ref_pos_int], 
                    marker='x', s=100, color='blue', lw=1.5, label='Совпадения (опора)')
        ax2.scatter(matched_target_pos, target_spectrum[matched_target_pos_int], 
                    marker='o', s=100, facecolors='none', edgecolors='red', lw=1.5, label='Совпадения (цель)')

        ax2.set_title(f'Профили спектров и найденные совпадения')
        ax2.set_xlabel('Координата по оси дисперсии (пикс.)')
        ax2.set_ylabel('Интенсивность')
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        # --- ОСТАНОВКА ПРОГРАММЫ ---
        print("\nПоказаны графики для порядка №{0}. Закройте окна для продолжения...".format(target_order_num))
        plt.show() # Эта команда покажет окна и будет ждать их закрытия
        
        plt.close('all') # Закрываем все, чтобы подготовиться к следующей итерации

    # --- Сохранение и вывод финальных результатов ПОСЛЕ цикла ---
    distortion_model_path = 'distortion_model.json'
    with open(distortion_model_path, 'w') as f:
        json.dump(distortion_model, f, indent=4)
    print(f"\nМодель искажений сохранена в '{distortion_model_path}'")

    print("\n--- Коэффициенты модели искажений ---")
    for order_key, coeffs in distortion_model.items():
        print(f"{order_key}: {coeffs}")

if __name__ == '__main__':
    main()
