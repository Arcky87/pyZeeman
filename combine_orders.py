# Файл: combine_orders.py
# Назначение: Объединяет линеаризованные спектральные порядки в один или несколько сегментов.

import numpy as np
import argparse
from pathlib import Path
import glob
import re

from astropy.io import fits
from specutils import Spectrum1D
from specutils.manipulation import resample
from astropy import units as u

# ==============================================================================
# ОСНОВНАЯ ЛОГИКА
# ==============================================================================

def combine_orders_in_group(spectra_to_combine, final_grid):
    """
    Выполняет передискретизацию и объединение группы спектров.

    Args:
        spectra_to_combine (list): Список объектов Spectrum1D для объединения.
        final_grid (astropy.units.Quantity): Финальная сетка длин волн.

    Returns:
        Spectrum1D: Объединенный спектр.
    """
    if not spectra_to_combine:
        return None

    # Создаем массивы для суммирования потока и весов
    total_flux = np.zeros(len(final_grid))
    total_weights = np.zeros(len(final_grid))
    
    # Ресемплер. LinearInterpolatedResampler - быстрый и хороший выбор.
    resampler = resample.LinearInterpolatedResampler(extrapolation_treatment='zero_fill')

    print(f"    -> Объединение {len(spectra_to_combine)} порядков...")

    for spec in spectra_to_combine:
        # Передискретизируем каждый спектр на финальную сетку
        resampled_spec = resampler(spec, final_grid)
        
        flux = resampled_spec.flux.value
        # Используем поток как вес. Где сигнал больше, там больше и вклад.
        # Добавляем небольшую константу, чтобы избежать деления на ноль в областях с нулевым потоком.
        weights = flux.copy()
        weights[weights < 0] = 0 # Отрицательный поток не должен давать веса
        
        # Находим, где у нас есть реальные данные (не нули от экстраполяции)
        valid_data_mask = weights > 1e-9
        
        # Добавляем взвешенный поток и веса в общие массивы
        total_flux[valid_data_mask] += flux[valid_data_mask] * weights[valid_data_mask]
        total_weights[valid_data_mask] += weights[valid_data_mask]

    # Вычисляем средневзвешенный поток
    # Там, где total_weights равен нулю, итоговый поток также будет равен нулю
    final_flux = np.divide(total_flux, total_weights, out=np.zeros_like(total_flux), where=total_weights!=0)
    
    # Создаем финальный объект Spectrum1D
    combined_spectrum = Spectrum1D(flux=final_flux*u.adu, spectral_axis=final_grid)
    
    return combined_spectrum


def main():
    parser = argparse.ArgumentParser(description="Объединяет линеаризованные FITS-файлы порядков.")
    parser.add_argument("resampled_dir", type=str, help="Директория с линеаризованными файлами (выход apply_calibration.py).")
    parser.add_argument("science_fits_basename", type=str, help="Базовое имя исходного научного файла для именования выходных файлов.")
    parser.add_argument("output_dir", type=str, help="Директория для сохранения объединенных спектров.")
    args = parser.parse_args()

    # --- 1. Поиск и загрузка всех линеаризованных порядков ---
    print(f"--- 1. Поиск и загрузка файлов из '{args.resampled_dir}' ---")
    resampled_files = sorted(glob.glob(str(Path(args.resampled_dir) / 'order_*.fits')))
    
    if not resampled_files:
        print(f"Ошибка: не найдено файлов 'order_*.fits' в директории '{args.resampled_dir}'.")
        return

    all_spectra = {}
    for f_path in resampled_files:
        try:
            # Spectrum1D.read автоматически считывает WCS из FITS заголовка
            spec = Spectrum1D.read(f_path)
            
            # Извлекаем номер порядка из имени файла
            match = re.search(r'order_(\d+)\.fits', Path(f_path).name)
            if match:
                order_num = int(match.group(1))
                all_spectra[order_num] = spec
                print(f"  - Загружен порядок {order_num} ({spec.spectral_axis.min():.2f} - {spec.spectral_axis.max():.2f})")
        except Exception as e:
            print(f" ! Не удалось прочитать файл '{f_path}': {e}")
    
    if not all_spectra:
        print("Не удалось загрузить ни одного спектра для объединения.")
        return

    # --- 2. Определение глобальной сетки длин волн ---
    print("\n--- 2. Определение глобальной сетки для объединения ---")
    
    # Сортируем ключи (номера порядков), чтобы работать с ними последовательно
    sorted_order_nums = sorted(all_spectra.keys())

    # Находим глобальные минимумы и максимумы
    w_min_global = min(all_spectra[order].spectral_axis.min() for order in sorted_order_nums)
    w_max_global = max(all_spectra[order].spectral_axis.max() for order in sorted_order_nums)
    
    # Находим самый маленький шаг (наилучшее разрешение), чтобы его сохранить
    dw_final = min(spec.spectral_axis[1] - spec.spectral_axis[0] for spec in all_spectra.values())

    # Создаем финальную сетку
    n_final = int(np.ceil((w_max_global.value - w_min_global.value) / dw_final.value)) + 1
    final_grid = np.linspace(w_min_global.value, w_min_global.value + (n_final - 1) * dw_final.value, n_final) * w_min_global.unit
    
    print(f"  Глобальный диапазон: {w_min_global:.2f} - {w_max_global:.2f}")
    print(f"  Финальный шаг: {dw_final:.4f}")
    print(f"  Размер финальной сетки: {len(final_grid)} точек")

    # --- 3. Группировка и объединение ---
    print("\n--- 3. Группировка и объединение спектров ---")
    
    # Разделяем на группы по 7 порядков
    group_1_keys = sorted_order_nums[:7]
    group_2_keys = sorted_order_nums[7:14]

    groups = {
        1: [all_spectra[k] for k in group_1_keys if k in all_spectra],
        2: [all_spectra[k] for k in group_2_keys if k in all_spectra]
    }
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for group_id, spectra_in_group in groups.items():
        if not spectra_in_group:
            print(f"\n-> Пропуск группы {group_id}: нет спектров для объединения.")
            continue
            
        print(f"\n-> Обработка группы {group_id}")
        combined_spec = combine_orders_in_group(spectra_in_group, final_grid)
        
        if combined_spec:
            flux_values = combined_spec.flux.value
            non_zero_indices = np.where(np.abs(flux_values) > 1e-9)[0]
            if len(non_zero_indices) == 0:
                print("  ! Внимание: объединенный спектр полностью нулевой. Сохранение отменено.")
                continue
            start_index = non_zero_indices[0]
            end_index = non_zero_indices[-1] + 1

            trimmed_spec = combined_spec[start_index:end_index]
            print(f"  - Спектр обрезан до диапазона: {trimmed_spec.spectral_axis.min():.2f} - {trimmed_spec.spectral_axis.max():.2f}")

            # Сохраняем результат
            output_basename = Path(args.science_fits_basename).stem
            output_filename = f"{output_basename}_{group_id}.fits"
            output_path = Path(args.output_dir) / output_filename
            
            hdu = fits.PrimaryHDU(trimmed_spec.flux.value)
            hdu.header['CRVAL1'] = trimmed_spec.spectral_axis.min().value
            hdu.header['CDELT1'] = (trimmed_spec.spectral_axis[1] - trimmed_spec.spectral_axis[0]).value
            hdu.header['CRPIX1'] = 1
            hdu.header['CUNIT1'] = 'Angstrom'
            hdu.header['HISTORY'] = f'Combined from {len(spectra_in_group)} orders.'
            
            # Записываем, какие именно порядки вошли в этот файл
            orders_included = sorted([k for k,v in all_spectra.items() if v in spectra_in_group])
            hdu.header['HISTORY'] = f'Source orders: {",".join(map(str, orders_included))}'
            
            hdu.writeto(output_path, overwrite=True)
            print(f"  ==> Результат сохранен в '{output_path}'")

    print("\nОбъединение завершено.")


if __name__ == '__main__':
    main()