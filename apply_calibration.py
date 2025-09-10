# Файл: apply_calibration.py (версия с улучшениями)

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.manipulation import resample
from astropy import units as u

from thar_calibration import *
from loaders import *
from visualize_trace import * 
from thar_rebin import *

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f: return json.load(f)
    except Exception as e: print(f"Ошибка при чтении файла '{file_path}': {e}"); return None


def main():
    parser = argparse.ArgumentParser(description="Применяет 2D калибровочное решение для линеаризации спектров.")
    parser.add_argument("science_fits", type=str, help="Путь к научному FITS-файлу.")
    parser.add_argument("trace_file", type=str, help="Путь к JSON-файлу с трассировкой.")
    parser.add_argument("solution_json", type=str, help="Путь к JSON-файлу с калибровочным решением.")
    parser.add_argument("output_dir", type=str, help="Директория для сохранения линеаризованных FITS-файлов.")
    args = parser.parse_args()

    # --- 1. Загрузка данных и настройка ---
    print("--- 1. Загрузка данных ---")
    science_data, header, _ = fits_loader(args.science_fits)
    trace_data = load_json_file(args.trace_file)
    solution = load_json_file(args.solution_json)
    if not trace_data or not solution: print("Ошибка: не удалось загрузить файлы. Выход."); return
    
    # Считываем параметры камеры или используем значения по умолчанию
    GAIN = header.get('GAIN', 2.78)
    RON_E = header.get('RON', 5.6)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"   -> Калибровка '{solution['metadata']['lamp_fits_file']}' загружена.")
    print(f"   -> Параметры CCD: GAIN={GAIN:.2f}, RON={RON_E:.2f}")

    # --- 2. Восстановление калибровочных моделей ---
    print("\n--- 2. Восстановление калибровочных моделей ---")
    # Восстанавливаем опорную модель как объект NumPy для удобства вычислений
    ref_disp_model_np = np.poly1d(solution['reference_dispersion_solution']['coeffs'])
    
    # Восстанавливаем модели искажений как объекты Astropy
    distortion_models = {
        int(order_num): models.Polynomial1D(degree=model_data['degree'], **{f'c{i}': c for i, c in enumerate(model_data['coeffs'])})
        for order_num, model_data in solution['distortion_models'].items()
    }
    print("   -> Модели успешно восстановлены.")

    # --- 3. Линеаризация каждого порядка ---
    print("\n--- 3. Выполнение линеаризации (Oversampling для каждого порядка) ---")
    resampler = resample.FluxConservingResampler(extrapolation_treatment='zero_fill')
    resampled_spectra_list = []

    for order_info in trace_data['orders']:
        order_num = order_info['order_number']
        
        # Шаг 3.1: Извлекаем 1D спектр (поток vs. пиксель)
        x_coords_orig, flux_orig = extract_order(science_data, trace_data, order_num)
        
        # Шаг 3.2: Строим полную нелинейную модель дисперсии для этого порядка
        distortion_model = distortion_models.get(order_num)
        if distortion_model is None:
            print(f" ! Пропуск порядка {order_num}: модель искажения не найдена.")
            continue
        
        # Находим обратную функцию для смещения итеративно, как вы и делали
        def get_p_ref(p):
            p_ref = p.copy()
            for _ in range(5): p_ref = p - distortion_model(p_ref)
            return p_ref
        
        # Полная модель: пиксель -> длина волны
        def effective_model_func(p): return ref_disp_model_np(get_p_ref(p))
        
        # Шаг 3.3: Вычисляем нелинейную ось длин волн и проверяем ее качество
        wavelengths_orig = effective_model_func(x_coords_orig)
        
        # Проверка на монотонность. Необходима для specutils.
        diffs = np.diff(wavelengths_orig)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
             print(f" ! ОШИБКА: Дисперсионная кривая для порядка {order_num} не монотонна! Пропуск.")
             continue
        if np.any(np.isinf(wavelengths_orig)) or np.any(np.isnan(wavelengths_orig)):
             print(f" ! ОШИБКА: Дисперсионная кривая для порядка {order_num} содержит NaN/inf! Пропуск.")
             continue

        # Шаг 3.4: Рассчитываем ошибку и создаем объект Spectrum1D
        error_adu = np.sqrt(np.maximum(flux_orig, 0) * GAIN + RON_E**2) / GAIN
        input_spectrum = Spectrum1D(
            flux=flux_orig * u.adu, 
            spectral_axis=wavelengths_orig * u.AA,
            uncertainty=StdDevUncertainty(error_adu * u.adu)
        )
        
        # Шаг 3.5: (ИСПРАВЛЕНО) Создаем НОВУЮ, УНИКАЛЬНУЮ сетку для ЭТОГО порядка
        w_start, w_end = input_spectrum.spectral_axis.min().value, input_spectrum.spectral_axis.max().value
        dw_min = np.min(np.abs(np.diff(input_spectrum.spectral_axis.value)))
        if dw_min == 0:
            print(f" ! ОШИБКА: Нулевой шаг по длинам волн в порядке {order_num}. Пропуск."); continue;
        n_new = int(np.ceil((w_end - w_start) / dw_min)) + 1
        target_grid = np.linspace(w_start, w_start + (n_new - 1) * dw_min, n_new) * u.AA

        print(f"   -> Порядок {order_num}: {len(x_coords_orig)} пикс. -> {n_new} пикс. (шаг {dw_min:.4f} Å)")
        
        # Шаг 3.6: Выполняем передискретизацию
        spectrum_resampled = resampler(input_spectrum, target_grid)
        resampled_spectra_list.append((order_num, spectrum_resampled))
        
        # Шаг 3.7: Сохраняем результат в отдельный FITS-файл
        hdu = fits.PrimaryHDU(spectrum_resampled.flux.value, header=header)
        hdu.header['CRVAL1'] = spectrum_resampled.spectral_axis.min().value
        hdu.header['CDELT1'] = dw_min
        hdu.header['CRPIX1'] = 1
        hdu.header['CUNIT1'] = 'Angstrom'
        # Добавляем информацию о происхождении
        hdu.header['HISTORY'] = 'Wavelength calibrated and resampled.'
        hdu.header['HISTORY'] = f"Source solution: {Path(args.solution_json).name}"
        hdu.header['HISTORY'] = f"Original order number: {order_num}"
        
        output_path = Path(args.output_dir) / f"order_{order_num:02d}.fits"
        hdu.writeto(output_path, overwrite=True)
        

    print("\n--- 4. Визуализация итогового спектра ---")
    if not resampled_spectra_list:
        print("Не удалось откалибровать ни одного спектра."); return

    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Находим опорный спектр для определения смещения
    ref_order_num = solution['metadata']['reference_order']
    ref_spec_tuple = next((s for s in resampled_spectra_list if s[0] == ref_order_num), None)
    offset_flux = ref_spec_tuple[1].flux.value if ref_spec_tuple else resampled_spectra_list[0][1].flux.value
    offset = np.nanmax(offset_flux) * 0.7 if np.any(offset_flux) else 1.0

    for i, (order_num, spec) in enumerate(resampled_spectra_list):
        ax.plot(spec.spectral_axis, spec.flux.value, lw=1, label=f"П.{order_num}")

    ax.set_title(f"Откалиброванный спектр: {Path(args.science_fits).name}", fontsize=16)
    ax.set_xlabel("Длина волны (Å)"); ax.set_ylabel("Поток (смещен)")
    ax.grid(True, linestyle=':'); ax.legend(ncol=7, loc='upper right')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()