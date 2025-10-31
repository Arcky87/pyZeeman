#!/usr/bin/env python3
"""
combine_orders.py - Объединение спектральных порядков для поляриметрии

Объединяет 7 верхних и 7 нижних срезов в два вектора на линейной сетке λ
"""

import numpy as np
import argparse
from pathlib import Path
from astropy.io import fits
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from astropy.nddata import StdDevUncertainty
from astropy import units as u
import spectres
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sum_spectra_irregular(spectrum_paths, min_step_limit=0.01, method='weighted'):
    """
    Суммирует спектры на их родных неравномерных сетках.
    
    Parameters:
    -----------
    spectrum_paths : list of Path
        Список путей к FITS файлам спектров
    method : str
        'weighted' - взвешенное по ошибкам, 'simple' - простое усреднение
    
    Returns:
    --------
    dict : {'wavelengths': [], 'flux': [], 'error': [], 'header': {}}
    """
    
    spectra = []
    
    for path in spectrum_paths:
        with fits.open(path) as hdul:
            # Читаем из таблицы WAVELENGTH
            table_data = hdul['WAVELENGTH'].data
            spectra.append({
                'wavelengths': table_data['WAVELENGTH'],
                'flux': table_data['FLUX'],
                'error': table_data['ERROR'],
                'header': hdul[0].header
            })
    # Создаем адаптивную равномерную сетку
    combined_wl = create_uniform_grid(spectra, min_step_limit)
    logger.info(f"Адаптивная сетка: {len(combined_wl)} точек")
    logger.info(f"Шаг сетки: {np.mean(np.diff(combined_wl)):.4f} Å")
    logger.info(f"Диапазон: {combined_wl[0]:.1f} - {combined_wl[-1]:.1f} Å")

    if method == 'weighted':
        return weighted_sum_irregular(spectra, combined_wl)
    else:
        return simple_sum_irregular(spectra, combined_wl)
    
def create_uniform_grid(spectra, min_step_limit=0.01):
    """
    Создает адаптивную равномерную сетку на основе самого мелкого шага в данных.
    """
    # Объединяем все длины волн для определения общего диапазона
    all_wavelengths = np.concatenate([spec['wavelengths'] for spec in spectra])
    global_min = np.min(all_wavelengths)
    global_max = np.max(all_wavelengths)
    
    # Находим самый мелкий шаг среди всех спектров
    min_steps = []
    for spec in spectra:
        diffs = np.diff(spec['wavelengths'])
        if len(diffs) > 0:
            min_steps.append(np.min(diffs))
    
    if not min_steps:
        # fallback: используем разумный шаг по умолчанию
        finest_step = 0.01
    else:
        finest_step = np.min(min_steps)
    
    # Защита от чрезмерного оверсэмплинга
    finest_step = max(finest_step, min_step_limit)
    
    logger.info(f"Самый мелкий шаг в данных: {np.min(min_steps):.4f} Å")
    logger.info(f"Используемый шаг сетки: {finest_step:.4f} Å")
    
    # Создаем равномерную сетку
    combined_wl = np.arange(global_min, global_max + finest_step, finest_step)
    
    return combined_wl

def weighted_sum_irregular(spectra,combined_wl):
    """
    Взвешенное суммирование на неравномерной сетке.
    """

    # Инициализируем массивы для взвешенного суммирования
    weighted_flux = np.zeros_like(combined_wl)
    total_weight = np.zeros_like(combined_wl)
    
    for i, spec in enumerate(spectra):
        # Пересэмплируем каждый спектр на адаптивную сетку с spectres
        resampled_flux, resampled_error = spectres.spectres(
            new_wavs =  combined_wl,
            spec_wavs = spec['wavelengths'],
            spec_fluxes = spec['flux'],
            spec_errs = spec['error'],
            fill=spec['flux'][1],
            verbose=True
        )
        
        # Вес = 1/σ² (избегаем деления на 0)
        weights = 1.0 / np.maximum(resampled_error**2, 1e-10)
        
        weighted_flux += resampled_flux * weights
        total_weight += weights
    
    # Взвешенное среднее
    result_flux = weighted_flux / total_weight
    result_error = 1.0 / np.sqrt(total_weight)

    return {
        'wavelengths': combined_wl,
        'flux': result_flux,
        'error': result_error,
        'header': spectra[0]['header']  # берем заголовок первого спектра
    }

def simple_sum_irregular(spectra,combined_wl):
    """
    Простое суммирование на неравномерной сетке.
    """  
    all_interpolated = []
    for i, spec in enumerate(spectra):
        resampled_flux, resampled_error =spectres.spectres(
            new_wavs=combined_wl,
            spec_wavs=spec['wavelengths'],
            spec_fluxes=spec['flux'],
            spec_errs=spec['error'],
            fill=spec['flux'][1],
            verbose=True
         )
        all_interpolated.append(resampled_flux)   

    all_interpolated = np.array(all_interpolated)
    
    return {
        'wavelengths': combined_wl,
        'flux': np.sum(all_interpolated, axis=0),
        'error': np.std(all_interpolated, axis=0),
        'header': spectra[0]['header']
    }

def resample_combined_spectrum(combined_spectrum, wavelength_step=0.1):
    """
    Пересэмплирует уже суммированный спектр на равномерную сетку.
    """
    # Создаем Spectrum1D из суммированных данных
    spectrum_obj = Spectrum1D(
        flux=combined_spectrum['flux'] * u.adu,
        spectral_axis=combined_spectrum['wavelengths'] * u.AA,
        uncertainty=StdDevUncertainty(combined_spectrum['error'] * u.adu)
    )
    
    # Создаем целевую равномерную сетку
    wl_min = np.min(combined_spectrum['wavelengths'])
    wl_max = np.max(combined_spectrum['wavelengths'])
    uniform_wl = np.linspace(wl_min, wl_max, 
                            int((wl_max - wl_min) / wavelength_step) + 1) * u.AA
    
    # Пересэмплирование
    resampler = FluxConservingResampler()
    final_spectrum = resampler(spectrum_obj, uniform_wl)
    
    return final_spectrum

def create_polarimetry_vectors(
    calibrated_dir: Path,
    output_base: Path,
    upper_orders=[1, 2, 3, 4, 5, 6, 7],
    lower_orders=[8, 9, 10, 11, 12, 13, 14]
):
    """
    Создаёт два вектора на линейной сетке λ для поляриметрии
    
    Алгоритм:
    1. Загружает все 14 срезов из FITS (нативная сетка)
    2. Определяет общую линейную сетку λ для каждой группы
    3. Интерполирует каждый срез (flux-conserving)
    4. Суммирует 7 верхних → вектор 1
    5. Суммирует 7 нижних → вектор 2
    6. Сохраняет с WCS: CRVAL1, CDELT1, CRPIX1
    
    Parameters:
    -----------
    calibrated_dir : Path
        Директория с откалиброванными срезами (order_01.fits ... order_14.fits)
    output_base : str
        Базовое имя выходных файлов (будет добавлено _1.fits и _2.fits)
    upper_orders : list
        Номера верхних срезов (ортогональная поляризация 1)
    lower_orders : list
        Номера нижних срезов (ортогональная поляризация 2)
    
    Returns:
    --------
    dict : Результаты для обоих векторов
    """
    print("="*80)
    print("ОБЪЕДИНЕНИЕ СРЕЗОВ ")
    print("="*80)
    
    # 1. Собрать пути к файлам для каждой группы
    upper_paths = []
    lower_paths = []
    
   # Проверить доступность файлов и собрать пути
    for order_num in range(1, 15):
        fits_file = calibrated_dir / f"{output_base.stem}_order_{order_num:02d}.fits"
        
        if not fits_file.exists():
           logger.warning(f"⚠️  Пропуск среза {order_num}: файл не найден")
           continue
            
        if order_num in upper_orders:
            upper_paths.append(fits_file)
        elif order_num in lower_orders:
            lower_paths.append(fits_file)
    
    logger.info(f"📦 Найдено файлов: верхняя группа - {len(upper_paths)}, нижняя группа - {len(lower_paths)}")
    
    # Обработка обеих групп
    results = {}

    for group_name, order_list in [('upper', upper_paths), ('lower', lower_paths)]:
        print(f"\n{'='*80}")
        print(f"2. Обработка {group_name} группы (срезов: {len(order_list)})")
        print("="*80)
        try:
            result_o = sum_spectra_irregular(
                spectrum_paths=order_list,
                min_step_limit=0.0001,
                method='simple' # weighted doesn't work
            )
          #  import matplotlib.pyplot as plt

            if group_name == 'upper':
                upper_output = output_base.with_name(output_base.stem + '_1.fits')
                save_polarimetry_vector(result_o, upper_output, upper_orders, 'upper')
                results['upper'] = {
                'output_file': upper_output,
                'orders_used': upper_orders,
                'wavelength_range': (result_o['wavelengths'][0], result_o['wavelengths'][-1]),
                'step': np.mean(np.diff(result_o['wavelengths'])),
                'total_points': len(result_o['wavelengths'])
                }
                logger.info(f"✅ Верхняя группа сохранена: {upper_output}")
            else:
                lower_output = output_base.with_name(output_base.stem + '_2.fits')
                save_polarimetry_vector(result_o, lower_output, lower_orders, 'lower')
                results['lower'] = {
                'output_file': lower_output,
                'orders_used': lower_orders,
                'wavelength_range': (result_o['wavelengths'][0], result_o['wavelengths'][-1]),
                'step': np.mean(np.diff(result_o['wavelengths'])),
                'total_points': len(result_o['wavelengths'])
                }
                logger.info(f"✅ Нижняя группа сохранена: {lower_output}")
        except Exception as e:
            logger.error(f"❌ Ошибка обработки группы: {group_name} {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 ИТОГИ СОЗДАНИЯ ВЕКТОРОВ ПОЛЯРИМЕТРИИ")
        logger.info('='*60)
    
        for group_name, result in results.items():
            logger.info(f"  {group_name.upper()}:")
            logger.info(f"    Файл: {result['output_file'].name}")
            logger.info(f"    Срезы: {result['orders_used']}")
            logger.info(f"    Точек: {result['total_points']}")
            logger.info(f"    Диапазон: {result['wavelength_range'][0]:.1f} - {result['wavelength_range'][1]:.1f} Å")
            logger.info(f"    Шаг: {result['step']:.4f} Å")
            logger.info("")

    return results

def save_polarimetry_vector(spectrum_data, output_path, orders_used, group_name):
    """
    Сохраняет вектор поляриметрии в FITS с правильным WCS.
    
    Parameters:
    -----------
    spectrum_data : dict
        Данные спектра из sum_spectra_irregular
    output_path : Path
        Путь для сохранения
    orders_used : list
        Список использованных порядков
    group_name : str
        Имя группы ('upper' или 'lower')
    """
    try:
        # Primary HDU с потоком
        primary = fits.PrimaryHDU(spectrum_data['flux'])
        wl_step = spectrum_data['wavelengths'][1] - spectrum_data['wavelengths'][0]
    except Exception as e:
        logger.error(f"❌ ОТЛАДКА: Ошибка создания PrimaryHDU: {e}")
        return

# КОПИРУЕМ ВЕСЬ ИСХОДНЫЙ ЗАГОЛОВОК
    if 'header' in spectrum_data and spectrum_data['header'] is not None:
        # Копируем все карточки из исходного заголовка
        for card in spectrum_data['header'].cards:
            try:
                # Пропускаем системные ключи, которые будут перезаписаны
                if card.keyword not in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 
                                    'CTYPE1', 'CUNIT1', 'CRVAL1', 'CRPIX1', 'CDELT1', 'CD1_1']:
                    primary.header[card.keyword] = (card.value, card.comment)
            except (ValueError, KeyError, fits.verify.VerifyError):
                # Пропускаем проблемные карточки
                continue
    
    # WCS заголовок для линейной дисперсии
    primary.header['CTYPE1'] = 'WAVE'
    primary.header['CUNIT1'] = 'Angstrom'
    primary.header['CRVAL1'] = spectrum_data['wavelengths'][0]
    primary.header['CRPIX1'] = 1.0
    primary.header['CDELT1'] = wl_step
    primary.header['CD1_1'] = wl_step
    primary.header['NAXIS1'] = len(spectrum_data['flux'])
    
    # История обработки
    primary.header['HISTORY'] = 'Polarimetry vector created'
    primary.header['HISTORY'] = f'Summed {len(orders_used)} orders: {orders_used}'
    primary.header['HISTORY'] = f'Resampled to uniform grid with spectres'
    
    # Таблица с длинами волн и ошибками (для совместимости)
    try:
        col_wave = fits.Column(name='WAVELENGTH', format='D', unit='Angstrom',
                            array=spectrum_data['wavelengths'])
        col_flux = fits.Column(name='FLUX', format='D', unit='ADU',
                            array=spectrum_data['flux'])
        col_err = fits.Column(name='ERROR', format='D', unit='ADU',
                            array=spectrum_data['error'])
        
        table = fits.BinTableHDU.from_columns([col_wave, col_flux, col_err], 
                                            name='WAVELENGTH')
    except Exception as e:
        logger.error(f"❌ ОТЛАДКА: Ошибка создания таблицы: {e}")
        return
    
    # Сохраняем
    try:
        hdul = fits.HDUList([primary, table])
        hdul.writeto(output_path, overwrite=True)  

        logger.info(f"✓ Вектор сохранен: {output_path}")
        logger.info(f"✓ CDELT1: {wl_step:.6f} Å/пиксель")
    except Exception as e:
        logger.error(f"❌ ОТЛАДКА: Ошибка сохранения файла: {e}")
        return


def main():
    parser = argparse.ArgumentParser(
        description="Объединяет срезы в два вектора для поляриметрии"
    )
    parser.add_argument("calibrated_dir", 
                       help="Директория с откалиброванными срезами (order_*.fits)")
    parser.add_argument("output_base",
                       help="Базовое имя выходных файлов (будет добавлено _1.fits и _2.fits)")
    parser.add_argument("--upper-orders", default="1,2,3,4,5,6,7",
                       help="Номера верхних срезов через запятую (по умолчанию: 1,2,3,4,5,6,7)")
    parser.add_argument("--lower-orders", default="8,9,10,11,12,13,14",
                       help="Номера нижних срезов через запятую (по умолчанию: 8,9,10,11,12,13,14)")
    
    args = parser.parse_args()
    
    # Парсинг списков срезов
    upper = [int(x.strip()) for x in args.upper_orders.split(',')]
    lower = [int(x.strip()) for x in args.lower_orders.split(',')]
    
    # Выполнить объединение
    results = create_polarimetry_vectors(
        calibrated_dir=Path(args.calibrated_dir),
        output_base=args.output_base,
        upper_orders=upper,
        lower_orders=lower
    )
    
    if results:
        print(f"\n✅ Созданы два вектора:")
        print(f"   - {args.output_base}_1.fits (верхний луч)")
        print(f"   - {args.output_base}_2.fits (нижний луч)")


if __name__ == '__main__':
    main()
