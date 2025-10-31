#!/usr/bin/env python3
"""
apply_calibration.py - Применение дисперсионной калибровки к спектрам

Применяет калибровку в нативной сетке детектора (БЕЗ линеаризации)
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json
from astropy.io import fits

from extract_order_spectrum import extract_order_summed, load_trace_data


def load_json_file(file_path):
    """Загружает JSON файл"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при чтении файла '{file_path}': {e}")
        return None


def apply_calibration_native(
    science_file: Path,
    trace_file: Path,
    calibration_file: Path,
    output_dir: Path = None,
    gain: float = 2.78,
    ron_e: float = 5.6
):
    """
    Применяет калибровку к научному кадру (нативная сетка, БЕЗ линеаризации)
    """
    print("="*80)
    print("ПРИМЕНЕНИЕ КАЛИБРОВКИ (НАТИВНАЯ СЕТКА)")
    print("="*80)
    
    # Загрузка
    with fits.open(science_file) as hdul:
        science_data = hdul[0].data
        header = hdul[0].header
    
    trace_data = load_trace_data(trace_file)
    calibration = load_json_file(calibration_file)
    
    if not trace_data or not calibration:
        print("Ошибка загрузки данных")
        return None
    
    # Определить тип калибровки
    if 'slices' in calibration:
        print(f"Полная калибровка: {calibration['thar_file']}")
        print(f"Срезов: {len(calibration['slices'])}/14")
        slices = calibration['slices']
    elif 'model' in calibration:
        print(f"Калибровка одного среза: {calibration.get('order_num', 'unknown')}")
        slices = {str(calibration['order_num']): calibration}
    else:
        print("Неизвестный формат калибровки")
        return None
    
    # Создать output директорию если нужно
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Применить к каждому срезу
    results = {}
    
    for order_str in sorted(slices.keys(), key=lambda x: int(x)):
        order_num = int(order_str)
        slice_calib = slices[order_str]
        
        print(f"\nСрез {order_num}...")
        
        # Извлечь спектр
        x_pixels, flux = extract_order_summed(science_data, trace_data, order_num)
        
        if x_pixels is None:
            print(f"  ! Пропуск: не удалось извлечь")
            continue
        
        # Применить калибровку (нативная сетка!)
        model = np.poly1d(slice_calib['model'])
        wavelengths = model(x_pixels)
        
        # Вычислить ошибки
        error_adu = np.sqrt(np.maximum(flux, 1e-6) * gain + ron_e**2) / gain
        
        # Сохранить
        results[order_num] = {
            'wavelength': wavelengths,
            'flux': flux,
            'error': error_adu,
            'model_coeffs': slice_calib['model'],
            'rms': slice_calib.get('rms_angstrom', 0.0)
        }
        
        # Сохранить FITS если указана директория
        if output_dir:
            save_spectrum_native(
                results[order_num],
                output_dir / f"{science_file.stem}_order_{order_num:02d}.fits",
                header
            )
        
        print(f"  ✓ {len(wavelengths)} точек, "
              f"λ = {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} Å")
    
    print(f"\n✓ Обработано срезов: {len(results)}")
    if output_dir:
        print(f"  Сохранено в: {output_dir}")
    
    return results


def save_spectrum_native(spectrum_data, output_file, original_header=None):
    """
    Сохраняет спектр в FITS с нативной сеткой
    """
    # Primary: поток
    primary = fits.PrimaryHDU(spectrum_data['flux'])
    
    # Копировать исходный заголовок
    if original_header:
        for key in original_header:
            if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND']:
                try:
                    primary.header[key] = original_header[key]
                except:
                    pass
    
    # WCS заголовок (полиномиальный)
    primary.header['CTYPE1'] = 'WAVE'
    primary.header['CUNIT1'] = 'Angstrom'
    primary.header['DISPTYPE'] = 'POLYNOMIAL'
    primary.header['DC-FLAG'] = 0
    
    # Полином дисперсии
    coeffs = spectrum_data['model_coeffs']
    primary.header['CD_DEG'] = len(coeffs) - 1
    for i, coeff in enumerate(coeffs[::-1]):
        primary.header[f'CD1_{i}'] = coeff
    
    # Калибровочная информация
    primary.header['CAL_RMS'] = (spectrum_data.get('rms', 0.0), 'RMS of calibration (Angstrom)')
    primary.header['CRVAL1'] = spectrum_data['wavelength'][0]
    primary.header['CRPIX1'] = 1.0
    primary.header['WAVEMIN'] = spectrum_data['wavelength'].min()
    primary.header['WAVEMAX'] = spectrum_data['wavelength'].max()
    
    # Extension: таблица длин волн
    col_wave = fits.Column(name='WAVELENGTH', format='D', unit='Angstrom',
                          array=spectrum_data['wavelength'])
    col_flux = fits.Column(name='FLUX', format='D', unit='ADU',
                          array=spectrum_data['flux'])
    col_err = fits.Column(name='ERROR', format='D', unit='ADU',
                         array=spectrum_data['error'])
    
    table = fits.BinTableHDU.from_columns(
        [col_wave, col_flux, col_err],
        name='WAVELENGTH'
    )
    
    # Сохранить
    hdul = fits.HDUList([primary, table])
    hdul.writeto(output_file, overwrite=True)


def visualize_calibrated_spectra(calibrated_data, output_file, title=""):
    """
    Визуализирует все откалиброванные спектры на ОДНОМ графике
    
    Все 14 срезов наложены друг на друга с разными цветами
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Цветовая карта для 14 срезов
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    # Если передан dict, сортируем по номеру среза
    if isinstance(calibrated_data, dict):
        order_nums = sorted(calibrated_data.keys())
    else:
        order_nums = range(1, 15)
    
    # Найти глобальные мин/макс для осей
    all_wavelengths = []
    all_fluxes = []
    
    for order_num in order_nums:
        if order_num in calibrated_data:
            data = calibrated_data[order_num]
            all_wavelengths.extend(data['wavelength'])
            all_fluxes.extend(data['flux'])
    
    # Нарисовать все спектры
    for i, order_num in enumerate(order_nums):
        if order_num not in calibrated_data:
            continue
        
        data = calibrated_data[order_num]
        
        # Рисуем спектр
        label = f'Срез {order_num}'
        if 'rms' in data and data['rms'] > 0:
            label += f' (RMS={data["rms"]:.3f} Å)'
        
        ax.plot(data['wavelength'], data['flux'], 
               color=colors[i], lw=0.8, alpha=1.0, label=label)
    
    ax.set_xlabel('Длина волны (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Поток (ADU)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title('Откалиброванные спектры всех срезов (нативная сетка)',
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Сохранение: векторный формат (PDF/SVG) или растровый (PNG)
    # Определяется автоматически по расширению файла
    output_path = Path(output_file)
    if output_path.suffix.lower() in ['.pdf', '.svg']:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    print(f"\n✓ Визуализация сохранена: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Применяет дисперсионную калибровку к спектрам"
    )
    parser.add_argument("science_fits", help="Путь к научному FITS-файлу")
    parser.add_argument("trace_file", help="Путь к JSON-файлу с трассировкой")
    parser.add_argument("calibration", help="Путь к JSON-файлу с калибровкой")
    parser.add_argument("--output-dir", default=None,
                       help="Директория для сохранения FITS")
    parser.add_argument("--visualize", help="Создать визуализацию (PNG файл)")
    parser.add_argument("--gain", type=float, default=2.78,
                       help="Gain детектора (e-/ADU)")
    parser.add_argument("--ron", type=float, default=5.6,
                       help="Read-out noise (e-)")
    
    args = parser.parse_args()
    
    # Применить калибровку
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    results = apply_calibration_native(
        science_file=Path(args.science_fits),
        trace_file=Path(args.trace_file),
        calibration_file=Path(args.calibration),
        output_dir=output_dir,
        gain=args.gain,
        ron_e=args.ron
    )
    
    if not results:
        print("Ошибка: калибровка не выполнена")
        return
    
    # Визуализация
    if args.visualize:
        sci_name = Path(args.science_fits).stem
        visualize_calibrated_spectra(
            calibrated_data=results,
            output_file=Path(args.visualize),
            title=f'Откалиброванный спектр: {sci_name}'
        )
    
    print("\n" + "="*80)
    print("ГОТОВО!")
    print("="*80)


if __name__ == '__main__':
    main()
