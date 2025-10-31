import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime

# --- Все ваши импорты и вспомогательные функции ---
from astropy.io import fits
from astropy.stats import mad_std
from scipy.signal import find_peaks, correlate
from astropy.modeling import models, fitting

from thar_calibration import *
from loaders import *
from visualize_trace import * 
from thar_rebin import *

def main():
    parser = argparse.ArgumentParser(description="Создает файл калибровочного решения по ThAr лампе.")
    parser.add_argument("lamp_fits", type=str, help="Путь к FITS-файлу с лампой.")
    parser.add_argument("trace_file", type=str, help="Путь к JSON-файлу с трассировкой.")
    parser.add_argument("atlas_file", type=str, help="Путь к файлу атласа линий.")
    parser.add_argument("output_json", type=str, help="Путь для сохранения итогового .json.")
    parser.add_argument("--load", type=str, default=None, help="Путь к .json для загрузки и продолжения калибровки.")
    args = parser.parse_args()

    GAIN, RON_E = 2.78, 5.6
    PEAK_PARAMS = {'prominence_sigma': 15, 'width_range': (1.8, 3.5), 'distance_pixels': 10}
    SHIFT_POLY_DEGREE = 5

    loaded_calib_points = None
    if args.load:
        try:
            with open(args.load, 'r') as f:
                loaded_solution = json.load(f)
            # Ищем точки в структуре, которую мы же и создаем
            points_from_json = loaded_solution.get('reference_dispersion_solution', {}).get('calib_points', {})
            if points_from_json:
                loaded_calib_points = {float(px): wl for px, wl in points_from_json.items()}
                print(f"\nЗагружены точки из '{args.load}'.")
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Не удалось корректно загрузить '{args.load}': {e}. Начинаем новую сессию.")

    print("--- 1. Загрузка данных лампы ---")
    image_data, _, _ = fits_loader(args.lamp_fits) 
    traced_data = load_traced_orders(args.trace_file) 
    atlas_lines = load_atlas_lines(args.atlas_file)
    x_coords_orig, all_spectra_1d = extract_all_orders(image_data, traced_data)

    print("\n--- 2. Интерактивная калибровка опорного порядка ---")
    ref_order_num = 11
    ref_order_index = next((i for i, o in enumerate(traced_data['orders']) if o['order_number'] == ref_order_num), -1)
    if ref_order_index == -1: print(f"Ошибка: опорный порядок {ref_order_num} не найден."); return
    ref_spectrum = all_spectra_1d[ref_order_index]

    fig_calib, ax_calib = plt.subplots(figsize=(17, 7)) 
    ax_calib.plot(x_coords_orig, ref_spectrum, label=f"Спектр порядка {ref_order_num}")
    ax_calib.set_title(f"Калибровка порядка {ref_order_num}")
    ax_calib.set_xlabel("Пиксели"); ax_calib.grid(True)
    # Находим тут пики один раз для отрисовки и передачи в интерактивную функцию
    initial_peaks = find_and_plot_lines(x_coords_orig, ref_spectrum, ax_calib, **PEAK_PARAMS)
    plt.show(block=False) 

    calibration_result = interactive_wavelength_calibration(
        found_peaks=initial_peaks, atlas_lines=atlas_lines, ax_spectrum=ax_calib,
        x_coords_orig=x_coords_orig,
        order_num=ref_order_num,
        initial_calib_points=loaded_calib_points
    )
    plt.close(fig_calib)

    if calibration_result is None: print("\nКалибровка отменена. Выход."); return
    
    # --- ИСПРАВЛЕНИЕ: Используем ключи из ВАШЕГО словаря ---
    ref_disp_coeffs = calibration_result['model']
    ref_calib_points = calibration_result['calib_points']
    ref_order_peaks = np.array(calibration_result['all_peaks_px'])
    
    print(f"\n   -> Опорное решение D(p_ref) получено. Найдено {len(ref_order_peaks)} пиков в опорном порядке.")
    
    print("\n--- 3. Поиск моделей искажений ---")
    distortion_models = {}
    for i, target_spectrum in enumerate(all_spectra_1d):
        target_order_num = traced_data['orders'][i]['order_number']
        if target_order_num == ref_order_num: distortion_models[target_order_num] = models.Polynomial1D(degree=0, c0=0.0); continue
        # ИСПРАВЛЕНИЕ: Используем пики, возвращенные из интерактивной функции
        matched_pos, shifts = match_peaks_robustly(ref_spectrum, target_spectrum, ref_order_peaks)
        shift_model, _ = fit_polynomial_robustly(matched_pos, shifts, SHIFT_POLY_DEGREE)
        distortion_models[target_order_num] = shift_model if shift_model is not None else np.poly1d([0.0])
        print(f"   -> Модель искажений для порядка {target_order_num} найдена.")

    print("\n--- 4. Формирование и сохранение файла калибровки ---")
    ref_disp_model = np.poly1d(ref_disp_coeffs)   
    # 1. Определяем полный диапазон длин волн опорного порядка
    w_start = ref_disp_model(x_coords_orig[0])
    w_end = ref_disp_model(x_coords_orig[-1])
    # 2. Определяем шаг (CDELT1) как минимальную дисперсию для сохранения разрешения
    dispersion_func = ref_disp_model.deriv()
    CDELT1 = np.min(np.abs(dispersion_func(x_coords_orig)))
    NAXIS1 = int(np.ceil((w_end - w_start) / CDELT1) + 1)
    CRVAL1 = w_start

    print(f"   -> Исходный диапазон опорного порядка: {w_start:.2f}Å - {w_end:.2f}Å")
    print(f"   -> Целевой шаг (CDELT1): {CDELT1:.4f} Å/пикс (для сохранения разрешения)")
    print(f"   -> Исходное кол-во точек: {len(x_coords_orig)}")
    print(f"   -> НОВОЕ (расчетное) кол-во точек для опорного порядка: {NAXIS1}")


    solution_data = {
        "metadata": { "description": "2D-to-1D Wavelength Calibration Solution", "lamp_fits_file": Path(args.lamp_fits).name, "trace_file": Path(args.trace_file).name, "atlas_file": Path(args.atlas_file).name, "reference_order": ref_order_num },
        "target_grid_parameters": { "crval1": CRVAL1, "cdelt1": CDELT1, "naxis1": NAXIS1 },
        "reference_dispersion_solution": {
            "coeffs": ref_disp_coeffs, 
            "degree": calibration_result['model_degree'],
            "calib_points": {str(k): v for k, v in ref_calib_points.items()} # Сохраняем точки
        },
        "distortion_models": { 
            str(order_num): { 
                "coeffs": model.parameters.tolist(), 
                "degree": model.degree 
                } for order_num, model in distortion_models.items() }
    }
    
    with open(args.output_json, 'w') as f: json.dump(solution_data, f, indent=4)
    print(f"\nГотово! Калибровочное решение сохранено в файл: '{args.output_json}'")

if __name__ == '__main__': main()    