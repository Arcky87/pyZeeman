#!/usr/bin/env python3
"""
thar_auto_calibration.py - Модуль автоматической калибровки ThAr спектров

Выполняет:
1. Чтение ссылок на ThAr из FITS заголовков научных кадров (THAR_REF)
2. Извлечение спектров ThAr по трассировке
3. Интерактивную калибровку первого ThAr (reference)
4. Автоматическую калибровку остальных ThAr

Автор: pyZeeman pipeline
Дата: 2025-10-17
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from astropy.io import fits

# Импорты из существующих модулей
from extract_order_spectrum import extract_order_summed, load_trace_data
from thar_calibration import (
    find_and_plot_lines,
    interactive_wavelength_calibration,
    find_nearest_line,
    find_peaks_for_order,
    plot_extracted_spectrum,
    fit_dispersion_poly,
    match_peaks
)
from visualize_trace import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# РАБОТА С FITS ЗАГОЛОВКАМИ
# =============================================================================
def get_thar_reference(science_file: Path, keyword: str = 'THAR_REF') -> Optional[str]:
    """
    Читает имя ThAr файла из FITS заголовка научного кадра
    
    Parameters:
    -----------
    science_file : Path
        Путь к научному FITS файлу
    keyword : str
        Ключевое слово в заголовке
    
    Returns:
    --------
    str or None : Имя ThAr файла или None
    """
    try:
        with fits.open(science_file) as hdul:
            thar_ref = hdul[0].header.get(keyword, None)
            
            if thar_ref and thar_ref != 'NONE':
                return thar_ref
            else:
                return None
    except Exception as e:
        logger.error(f"Ошибка чтения {science_file}: {e}")
        return None

def collect_unique_thars(science_files: List[Path], 
                        raw_dir: Path,
                        keyword: str = 'THAR_REF') -> List[Path]:
    """
    Собирает уникальные ThAr файлы из заголовков научных кадров
    
    Parameters:
    -----------
    science_files : List[Path]
        Список научных FITS файлов
    raw_dir : Path
        Директория с сырыми данными (где лежат ThAr)
    keyword : str
        Ключевое слово в заголовке
    
    Returns:
    --------
    List[Path] : Список уникальных путей к ThAr файлам
    """
    thar_names = set()
    
    for science_file in science_files:
        thar_ref = get_thar_reference(science_file, keyword)
        if thar_ref:
            thar_names.add(thar_ref)
    
    # Преобразуем в полные пути
    thar_files = [raw_dir / name for name in sorted(thar_names)]
    
    logger.info(f"Найдено уникальных ThAr: {len(thar_files)}")
    for thar_file in thar_files:
        logger.info(f"  - {thar_file.name}")
    
    return thar_files

# =============================================================================
# ЗАГРУЗКА АТЛАСА
# =============================================================================
def load_thar_atlas(atlas_file: Path = Path('thar.dat')) -> np.ndarray:
    """Загружает атлас линий ThAr"""
    try:
        atlas_lines = np.loadtxt(atlas_file)
        logger.info(f"Загружен атлас: {len(atlas_lines)} линий из {atlas_file}")
        return atlas_lines
    except Exception as e:
        logger.error(f"Ошибка загрузки атласа {atlas_file}: {e}")
        return np.array([])


# =============================================================================
# КАЛИБРОВКА
# =============================================================================
def calibrate_first_thar_interactive(thar_file: Path,
                                     trace_file: Path,
                                     order_num: int,
                                     atlas_file: Path,
                                     gain: float = 2.0,
                                     ron_e: float = 5.6) -> Optional[dict]:
    """
    Интерактивная калибровка первого ThAr (reference)
    
    Parameters:
    -----------
    thar_file : Path
        Путь к FITS файлу ThAr
    trace_file : Path
        Путь к JSON файлу с трассировкой
    order_num : int
        Номер порядка для калибровки
    atlas_file : Path
        Путь к атласу линий
    gain, ron_e : float
        Параметры детектора
    
    Returns:
    --------
    dict or None : Калибровочное решение
    """
    logger.info("="*80)
    logger.info(f"ИНТЕРАКТИВНАЯ КАЛИБРОВКА: {thar_file.name}, порядок {order_num}")
    logger.info("="*80)
    
    # Загрузить ThAr изображение
    logger.info(f"Загрузка {thar_file}...")
    with fits.open(thar_file) as hdul:
        thar_data = hdul[0].data
    
    # Загрузить трассировку
    trace_data = load_trace_data(trace_file)
    if trace_data is None:
        return None
    
    # Извлечь спектр (с суммированием между границами)
    x_coords, flux = extract_order_summed(thar_data, trace_data, order_num)
  #  x_coords, flux = extract_order(thar_data, trace_data, order_num)
    if x_coords is None:
        return None
    
    # Загрузить атлас
    atlas_lines = load_thar_atlas(atlas_file)
    if len(atlas_lines) == 0:
        return None
    #TEST:
    ax=None
    fig=None
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plot_extracted_spectrum(x_coords, flux, order_num, ax=ax)
    
    # Найти и отрисовать пики (используем оригинальную функцию)
    logger.info("Поиск и отрисовка линий...")
    found_peaks = find_and_plot_lines(
        x_coords=x_coords,
        flux=flux,
        ax=ax,
        prominence_sigma=15.0,
        width_range=(1.8, 3.5),
        distance_pixels=10
    )
    logger.info(f"Найдено {len(found_peaks)} пиков")
    # Запустить интерактивную калибровку (используем готовую функцию из thar_calibration.py)
    result = interactive_wavelength_calibration(
        found_peaks=found_peaks,
        atlas_lines=atlas_lines,
        ax_spectrum=ax,  # Передаём axes для отрисовки
        x_coords_orig=x_coords,
        order_num=order_num,
        initial_calib_points=None
    )
    
    if result and 'model' in result:
        solution = {
            'thar_file': str(thar_file.name),
            'order_num': order_num,
            'model': result['model'],
            'model_degree': result['model_degree'],
            'calib_points': result['calib_points'],
            'found_peaks': result['all_peaks_px'],
            'rms': result.get('rms', 0.0),
            'n_points': len(result['calib_points'])
        }
        
        logger.info(f"\n✓ Калибровка завершена:")
        logger.info(f"  Степень полинома: {solution['model_degree']}")
        logger.info(f"  Опорных точек: {solution['n_points']}")
        logger.info(f"  RMS: {solution['rms']:.4f} Å")
        
        return solution
    else:
        logger.warning("Калибровка не завершена")
        return None


def calibrate_thar_auto(thar_file: Path,
                       trace_file: Path,
                       order_num: int,
                       reference_solution: dict,
                       atlas_file: Path,
                       max_shift_pixels: float = 3.0,
                       wavelength_tolerance: float = 0.5,
                       gain: float = 2.0,
                       ron_e: float = 5.6) -> Optional[dict]:
    """
    Автоматическая калибровка ThAr на основе reference решения
    
    Parameters:
    -----------
    thar_file : Path
        Путь к ThAr файлу
    trace_file : Path
        Путь к трассировке
    order_num : int
        Номер порядка
    reference_solution : dict
        Reference решение
    atlas_file : Path
        Атлас линий
    max_shift_pixels : float
        Максимальный допустимый сдвиг (пиксели)
    wavelength_tolerance : float
        Допуск при сопоставлении (Å)
    
    Returns:
    --------
    dict or None : Калибровочное решение
    """
    logger.info(f"Автоматическая калибровка: {thar_file.name}, порядок {order_num}")
    
    # Загрузить ThAr
    with fits.open(thar_file) as hdul:
        thar_data = hdul[0].data
    
    # Загрузить трассировку
    trace_data = load_trace_data(trace_file)
    if trace_data is None:
        return None
    
    # Извлечь спектр
    x_coords, flux = extract_order_summed(thar_data, trace_data, order_num)
    if x_coords is None:
        return None
    
    # Загрузить атлас
    atlas_lines = load_thar_atlas(atlas_file)
    if len(atlas_lines) == 0:
        return None
    
    # Найти пики (для auto режима используем find_peaks_for_order)
    peak_params = {
        'prominence_sigma': 15.0,
        'width_range': (1.8, 3.5),
        'distance_pixels': 10
    }
    found_peaks = find_peaks_for_order(x_coords, flux, peak_params)
    logger.info(f"  Найдено {len(found_peaks)} пиков")
    
    # Применить reference модель
    ref_model = np.poly1d(reference_solution['model'])
    predicted_wavelengths = ref_model(found_peaks)
    
    # Сопоставить с атласом
    matched_pairs = []
    for peak_pixel, pred_lambda in zip(found_peaks, predicted_wavelengths):
        atlas_lambda = find_nearest_line(pred_lambda, atlas_lines)
        
        if atlas_lambda is not None:
            delta_lambda = abs(atlas_lambda - pred_lambda)
            if delta_lambda < wavelength_tolerance:
                matched_pairs.append((peak_pixel, atlas_lambda))
    
    logger.info(f"  Сопоставлено {len(matched_pairs)}/{len(found_peaks)} линий")
    
    if len(matched_pairs) < 5:
        logger.warning(f"  Недостаточно линий ({len(matched_pairs)} < 5)")
        return None
    
    # Перефитить полином
    pixels, wavelengths = zip(*matched_pairs)
    poly_degree = reference_solution['model_degree']
    new_coeffs = np.polyfit(pixels, wavelengths, poly_degree)
    new_model = np.poly1d(new_coeffs)
    
    # RMS
    residuals = [wavelengths[i] - new_model(pixels[i]) for i in range(len(pixels))]
    rms = np.sqrt(np.mean(np.array(residuals)**2))
    
    # Сдвиг относительно reference
    mid_pixel = len(x_coords) // 2
    shift_angstrom = abs(new_model(mid_pixel) - ref_model(mid_pixel))
    dispersion = abs(new_model(mid_pixel + 1) - new_model(mid_pixel))
    shift_pixels = shift_angstrom / dispersion if dispersion > 0 else 0
    
    logger.info(f"  RMS: {rms:.4f} Å, сдвиг: {shift_pixels:.2f} пикс")
    
    if shift_pixels > max_shift_pixels:
        logger.warning(f"  ⚠ Сдвиг превышает лимит ({shift_pixels:.2f} > {max_shift_pixels})")
    
    solution = {
        'thar_file': str(thar_file.name),
        'order_num': order_num,
        'model': new_coeffs.tolist(),
        'model_degree': poly_degree,
        'calib_points': {str(float(p)): float(w) for p, w in matched_pairs},
        'rms': float(rms),
        'n_points': len(matched_pairs),
        'shift_from_reference_pixels': float(shift_pixels),
        'reference_used': reference_solution['thar_file']
    }
    
    return solution


# =============================================================================
# КАЛИБРОВКА ВСЕХ СРЕЗОВ
# =============================================================================

def calibrate_all_slices_auto(
    thar_file: Path,
    trace_file: Path,
    reference_order: int,
    reference_solution: dict,
    atlas_file: Path,
    poly_degree: int = 4,
    rms_excellent: float = 0.03,
    rms_good: float = 0.09,
    rms_warning: float = 0.25,
    pixel_tolerance: float = 0.5,
    wavelength_tolerance: float = 0.5,
    peak_params: dict = None,
    compare_template: bool = True
) -> Optional[dict]:
    """
    Калибрует все срезы (кроме reference) автоматически
    
    Parameters:
    -----------
    thar_file : Path
        Путь к ThAr файлу
    trace_file : Path
        Путь к трассировке
    reference_order : int
        Номер опорного среза (уже откалиброван)
    reference_solution : dict
        Решение для опорного среза
    atlas_file : Path
        Атлас линий
    poly_degree : int
        Степень полинома для каждого среза
    rms_excellent, rms_good, rms_warning : float
        Пороги качества в Ангстремах
    
    Returns:
    --------
    dict : Полное решение для всех 14 срезов
    """
    logger.info("="*80)
    logger.info(f"КАЛИБРОВКА ВСЕХ СРЕЗОВ: {thar_file.name}")
    logger.info(f"Опорный срез: {reference_order}")
    logger.info("="*80)

    import matplotlib.pyplot as plt # удалить после отладки
    plt.ion()
    
    # Загрузить ThAr
    with fits.open(thar_file) as hdul:
        thar_data = hdul[0].data
        header = hdul[0].header
        thar_data = np.maximum(thar_data, 1e-6)
    
    # Загрузить трассировку
    trace_data = load_trace_data(trace_file)
    if trace_data is None:
        return None
    
    # Загрузить атлас
    atlas_lines = load_thar_atlas(atlas_file)
    if len(atlas_lines) == 0:
        return None
    
    # Reference модель
  #  ref_model = np.poly1d(reference_solution['model'])
  #  calib_points = reference_solution['calib_points']
    
    # Результаты для всех срезов
    all_slices = {}
    
    # Используем переданные параметры или значения по умолчанию
    if peak_params is None:
        peak_params = {
            'prominence_sigma': 15.0,
            'width_range': (1.8, 3.5),
            'distance_pixels': 10
        }

    # ШАГ 1: Калибровать опорный срез ЭТОГО ThAr
    logger.info(f"\nШаг 1: Калибровка опорного среза {reference_order} для {thar_file.name}...")
    
    x_coords_ref, flux_ref = extract_order_summed(thar_data, trace_data, reference_order)
    if x_coords_ref is None:
        logger.error("Не удалось извлечь опорный срез")
        return None
    
    # Найти пики в опорном срезе
    ref_all_peaks = find_peaks_for_order(x_coords_ref, flux_ref, peak_params)
    logger.info(f"  Найдено {len(ref_all_peaks)} пиков в опорном срезе")

    #Сопоставить пики с интерактивным решением

    matched_pairs, unmatched = match_peaks(ref_all_peaks, reference_solution['calib_points'], pixel_tolerance)
  
    logger.info(f"  Сопоставлено {len(matched_pairs)} пиков с опорным решением. {len(unmatched)} не было сопоставлено!")

    matched_pixels = [pair['found_pixel'] for pair in matched_pairs]
    wavelengths = [pair['wavelength'] for pair in matched_pairs]
    disp_model = fit_dispersion_poly(matched_pixels, wavelengths, poly_degree)
      
 #   ref_wls_interactive = [float(w) for w in reference_solution['calib_points'].values()]
    ref_wls_full = sorted(set(wavelengths) | set(atlas_lines.tolist()))
        
    unmatched_lambda = disp_model(unmatched)
    unm_atlas_lmb = [find_nearest_line(l, ref_wls_full) for l in unmatched_lambda]
  #     Добавить фильтрацию по величине разницы м/у атласом и найденным 
  #       mask_false_wl = np.abs(unm_lambda_o - unm_lmb_o) < wavelength_tolerance
    all_pixels = np.array(matched_pixels + unmatched)
    all_wavelengths = np.array(wavelengths + unm_atlas_lmb)
    sort_indices = np.argsort(all_pixels)

    ref_model_current = fit_dispersion_poly(all_pixels[sort_indices], all_wavelengths[sort_indices], poly_degree)
    ref_coeffs_current = ref_model_current.c
    residuals_ref = [sorted(all_wavelengths)[i] - ref_model_current(sorted(all_pixels)[i]) for i in range(len(all_pixels))]
    rms_ref = np.sqrt(np.mean(np.array(residuals_ref)**2))
    
    logger.info(f"  В спектре найдено {len(all_wavelengths)} линий атласа ThAr для сопоставления")
    logger.info(f"  Опорный срез: RMS={rms_ref:.4f} Å, {len(all_wavelengths)} линий")
    
    # Сохранить откалиброванный опорный срез
    all_slices[str(reference_order)] = {
        'model': ref_coeffs_current.tolist(),
        'model_degree': poly_degree,
        'rms_angstrom': float(rms_ref),
        'rms_pixels': float(rms_ref / 0.12),
        'n_lines': len(all_pixels),
        'quality': 'excellent' if rms_ref <= rms_excellent else ('good' if rms_ref <= rms_good else 'warning'),
        'is_reference': True,
        'drift_from_template_pixels': None,
        'calib_points':{float(pixel): float(wavelength) for pixel, wavelength in zip(all_pixels[sort_indices], all_wavelengths[sort_indices])},
        'all_peaks': ref_all_peaks.tolist() if isinstance(ref_all_peaks, np.ndarray) else list(ref_all_peaks)
    }
#===============================
    if compare_template:
        thar_dir = thar_file.parent
        # Автоматически выбираем первый ThAr из reference_solution
        template_filename = reference_solution.get('thar_file')
        template_path = thar_dir / template_filename
        
        with fits.open(template_path) as hdut:
                templ_data = hdut[0].data
                header = hdut[0].header
                templ_data = np.maximum(templ_data,1e-6)
        
        x_coords_templ, flux_templ = extract_order_summed(templ_data, trace_data, reference_order)
        
        if plt.fignum_exists(1):
            plt.close(1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), num=1)
        fig.canvas.manager.window.move(1920, 100)

        pix_template, = ax1.plot(x_coords_templ, flux_templ,'b-')
        pix1, = ax1.plot(x_coords_ref, flux_ref,'r--')
    # plt.vlines(x=ref_wls_interactive, ymin=0,ymax=11000,ls=':',colors='red')
        ax1.set_xlim(1285,1389)

        ref_model_template = np.poly1d(reference_solution['model'])

        wl_template, = ax2.plot(ref_model_template(x_coords_templ), flux_templ,'b-')   
        wl_ref, = ax2.plot(ref_model_current(x_coords_ref),flux_ref, 'r-')
        ax2.set_xlim(4410,4480)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
 #==========================   
    # Использовать ОБНОВЛЁННЫЕ длины волн для остальных срезов
    ref_wls = all_wavelengths[sort_indices]
    logger.info(f"  Обновлённые ref_wls: {len(ref_wls)} линий")

    # Калибруем остальные срезы (используем те же peak_params)
    for order_num in range(1, 15):
        if order_num == reference_order:
            continue  # Уже добавлен
        
        logger.info(f"\nКалибровка среза {order_num}...")
        
        # Извлечь спектр
        x_coords, flux = extract_order_summed(thar_data, trace_data, order_num)
        # Тесты с визуализацией
#        fig1d = None
#        ax1d = None
#        fig1d, ax1d = plot_extracted_spectrum(x_coords, flux, order_num, ax=ax1d)
        if x_coords is None:
            logger.warning(f"  Пропуск среза {order_num}: не удалось извлечь")
            continue
        
        # Найти пики
        found_peaks = find_peaks_for_order(x_coords, flux, peak_params)
 #       found_peaks = find_and_plot_lines(x_coords, flux,ax1d, **peak_params)
 #       plt.vlines(x=pixels_ref, ymin=0,ymax=1500,ls=':')
        
        if len(found_peaks) < 6:
            logger.warning(f"  Пропуск среза {order_num}: мало пиков ({len(found_peaks)})")
            continue
        
        matched_pairs_o, unmatched_o = match_peaks(found_peaks, all_slices[str(reference_order)]['calib_points'], 3)
        pixls = [pair['found_pixel'] for pair in matched_pairs_o]
        wavelengths = [pair['wavelength'] for pair in matched_pairs_o]
        disp_model_o = fit_dispersion_poly(pixls, wavelengths, poly_degree)
        unm_lambda_o = np.array(disp_model_o(unmatched_o))
        unm_lmb_o = [find_nearest_line(l, ref_wls_full) for l in unm_lambda_o]
  #     Добавить фильтрацию по величине разницы м/у атласом и найденным 
  #       mask_false_wl = np.abs(unm_lambda_o - unm_lmb_o) < wavelength_tolerance

        # Сопоставить с интерактивно найденными точками, используя ОБНОВЛЁННУЮ модель
        # matched_pairs = []
        # for peak_pixel in found_peaks:
        #     pred_lambda = ref_model_current(peak_pixel)  # Используем НОВУЮ модель!
        #     atlas_lambda = find_nearest_line(pred_lambda, ref_wls)
            
        #     if atlas_lambda is not None:
        #         if abs(atlas_lambda - pred_lambda) < wavelength_tolerance:
        #             matched_pairs.append((peak_pixel, atlas_lambda))
    
        # if len(matched_pairs) < 5:
        #     logger.warning(f"  Пропуск среза {order_num}: мало сопоставлений ({len(matched_pairs)})")
        #     continue

    #    plt.figure(2)
        # ref_sp, = plt.plot(ref_model_current(x_coords), flux)
        # plt.draw()
        # input("check plots") 
        # ref_sp.remove()
    #    plt.vlines(x=[lmb[1] for lmb in matched_pairs], ymin=0,ymax=1100,ls=':',colors='red')
    #    plt.vlines(x=ref_model_current(found_peaks), ymin=0,ymax=1200,colors='green')
    #    plt.tight_layout()

               
        # Подогнать полином

        pixels_o = np.array(pixls + unmatched_o)
        wavelengths_o = np.array(wavelengths + unm_lmb_o)
        sort_indices_o = np.argsort(pixels_o)
        disp_model_o = fit_dispersion_poly(pixels_o[sort_indices_o], wavelengths_o[sort_indices_o], poly_degree)
        coeffs_o = disp_model_o.c
        # Вычислить RMS
        residuals_o = [sorted(wavelengths_o)[i] - disp_model_o(sorted(pixels_o)[i]) for i in range(len(pixels_o))]
        rms_o = np.sqrt(np.mean(np.array(residuals_o)**2))
        rms_pix_o = rms_o / 0.12  # Приблизительная дисперсия
        
        # Определить качество
        if rms_o <= rms_excellent:
            quality = 'excellent'
            symbol = '✅'
        elif rms_o <= rms_good:
            quality = 'good'
            symbol = '✓'
        elif rms_o <= rms_warning:
            quality = 'warning'
            symbol = '⚠'
        else:
            quality = 'critical'
            symbol = '✗'
      
        # Логировать
        log_func = logger.info if quality in ['excellent', 'good'] else logger.warning
        log_func(f"  Срез {order_num}: RMS={rms_o:.4f} Å ({rms_pix_o:.3f} пикс), "
                f"{len(wavelengths_o)} линий, качество: {quality} {symbol}")
        
        # Сохранить результат
        all_slices[str(order_num)] = {
            'model': coeffs_o.tolist(),
            'model_degree': poly_degree,
            'rms_angstrom': float(rms_o),
            'rms_pixels': float(rms_pix_o),
            'n_lines': len(wavelengths_o),
            'quality': quality,
            'is_reference': False,
            'drift_from_template_pixels': None,
            'calib_points': {float(pixel): float(wavelength) for pixel, wavelength in zip(pixels_o[sort_indices_o], wavelengths_o[sort_indices_o])}
        }
    
    # Формируем полное решение
    import datetime
    full_solution = {
        'thar_file': str(thar_file.name),
        'reference_order': reference_order,
        'poly_degree': poly_degree,
        'slices': all_slices
    }
    
    logger.info("\n" + "="*80)
    logger.info(f"Откалибровано срезов: {len(all_slices)}/14")
    logger.info("="*80)
    
    return full_solution


# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

def visualize_all_slices_calibration(calibration_data: dict, output_file: Path) -> bool:
    """
    Создаёт визуализацию сдвигов длин волн всех 14 срезов относительно опорного
    
    14 графиков друг под другом:
    - X: Пиксель
    - Y: Δλ = λ_slice(pixel) - λ_reference(pixel) [Ангстремы]
    
    Parameters:
    -----------
    calibration_data : dict
        Полное решение калибровки со всеми срезами
    output_file : Path
        Путь для сохранения PNG файла
    
    Returns:
    --------
    bool : True если визуализация создана успешно
    """
    try:
        import matplotlib.pyplot as plt
        
        slices = calibration_data['slices']
        reference_order = calibration_data['reference_order']
        reference_solution = load_calibration_solution(output_file.parent / "reference_solution.json")

        # Получить опорную модель
        ref_slice = slices[str(reference_order)]
        ref_model = np.poly1d(ref_slice['model'])
        
        # Создать 14 субплотов друг под другом
        fig, axes = plt.subplots(14, 1, figsize=(16, 20), sharex=True)
        
        x = np.linspace(0, 4600, 100)
        
        # Цвета для качества
        quality_colors = {
            'excellent': 'green',
            'good': 'blue',
            'warning': 'orange',
            'critical': 'red'
        }
        
        # Вычисляем сдвиги опорного спектра относительно референсного решения
        ref_shift_points_x = []
        ref_shift_points_y = []
        
        if 'calib_points' in ref_slice and 'calib_points' in reference_solution:
            ref_calib_points = ref_slice['calib_points']  # {пиксель: длина_волны}
            ref_solution_points = reference_solution['calib_points']  # {пиксель: длина_волны}
            
            # Находим общие пиксели в обоих решениях
            common_pixels = set(ref_calib_points.keys()) & set(ref_solution_points.keys())
            
            for pixel_str in common_pixels:
                pixel = float(pixel_str)
                # Длина волны в опорном решении
                ref_lambda = ref_calib_points[pixel_str]
                # Длина волны в референсном решении
                solution_lambda = ref_solution_points[pixel_str]
                # Сдвиг: опорное - референсное
                delta_lambda = ref_lambda - solution_lambda
                ref_shift_points_x.append(pixel)
                ref_shift_points_y.append(delta_lambda)
        
        print(f"Опорный срез: {len(ref_shift_points_x)} точек сдвига относительно референсного решения")

        for i, order_num in enumerate(sorted([int(k) for k in slices.keys()])):
            ax = axes[i]
            slice_data = slices[str(order_num)]
            slice_model = np.poly1d(slice_data['model'])
            
            # Вычислить полином сдвига
            if order_num == reference_order:
                # Опорный срез: сдвиг относительно референсного решения
                shift_values = ref_model(x) - np.poly1d(reference_solution['model'])(x)
                
                # Отождествленные точки (круги) - сдвиги относительно референсного
                shift_points_x = ref_shift_points_x
                shift_points_y = ref_shift_points_y
                
                # ВСЕ найденные пики (крестики) - для визуального контроля
                all_peaks_x = []
                all_peaks_y = []
                if 'all_peaks' in ref_slice:
                    all_peaks_x = ref_slice['all_peaks']
                    # Для всех найденных пиков вычисляем сдвиг относительно референсного
                    all_peaks_y = [ref_model(pixel) - np.poly1d(reference_solution['model'])(pixel) 
                                  for pixel in all_peaks_x]
            else:
                # Сдвиг: Δλ(pixel) = λ_slice(pixel) - λ_ref(pixel)
                shift_values = slice_model(x) - ref_model(x)
                
                # Точки сдвига: используем собственные calib_points этого среза
                shift_points_x = []
                shift_points_y = []
                
                if 'calib_points' in slice_data:
                    slice_calib_points = slice_data['calib_points']
                    
                    for pixel_str, slice_lambda in slice_calib_points.items():
                        pixel = float(pixel_str)
                        # Длина волны в этой точке по reference модели
                        ref_lambda = ref_model(pixel)
                        # Сдвиг
                        delta_lambda = slice_lambda - ref_lambda
                        shift_points_x.append(pixel)
                        shift_points_y.append(delta_lambda)
            
            # Цвет по качеству
            quality = slice_data.get('quality', 'good')
            color = quality_colors.get(quality, 'blue')
            
            # Вычислить индивидуальный диапазон Y для этого среза
            # Объединяем сдвиги кривой и точек
            all_y_values = list(shift_values)
            if shift_points_y:
                all_y_values.extend(shift_points_y)
            
            if all_y_values:
                y_min = np.min(all_y_values)
                y_max = np.max(all_y_values)
                y_range = y_max - y_min
                
                # Добавляем запас 20% для визуальной чистоты
                if y_range > 0:
                    margin = y_range * 1.15  # зумируем вглубь графика
                    ylim_min = y_min - margin
                    ylim_max = y_max + margin
                else:
                    # Если нет сдвига (reference с нулевыми значениями)
                    ylim_min, ylim_max = -0.03, 0.03
            else:
                ylim_min, ylim_max = -0.1, 0.1
            
            # Рисуем полином сдвига
            ax.plot(x, shift_values, color=color, linewidth=2, alpha=0.8)
            
            # Рисуем точки сдвига (отождествленные)
            if shift_points_x:
                ax.scatter(shift_points_x, shift_points_y, 
                          color=color, s=30, alpha=0.6, marker='o', label='Отождествленные')
            
            # Для reference рисуем ВСЕ найденные пики (крестиками)
            if order_num == reference_order and all_peaks_x:
                ax.scatter(all_peaks_x, all_peaks_y, 
                          color='gray', s=20, alpha=0.4, marker='x', label='Все найденные')
            
            # Горизонтальная линия на нуле
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Заголовок для каждого среза
            title = f"Срез {order_num}"
            if order_num == reference_order:
                title += " (REF)"
                # Добавляем информацию о сдвиге относительно референсного
                if ref_shift_points_y:
                    rms_ref_shift = np.sqrt(np.mean(np.array(ref_shift_points_y) ** 2))
                    title += f" | REF_shift={rms_ref_shift:.4f} Å"
            title += f" | RMS={slice_data['rms_angstrom']:.4f} Å"
            title += f" | {slice_data['n_lines']} линий"
            title += f" | {quality}"
            
            ax.text(0.01, 0.95, title, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            
            # Настройки осей
            ax.set_ylabel('Δλ (Å)', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_ylim(-0.6, 0.6)
            
            # Убираем метки X для всех кроме последнего
            if i < 13:
                ax.set_xticklabels([])
        
        # Метка X только для нижнего графика
        axes[-1].set_xlabel('Пиксель', fontsize=12, fontweight='bold')
        
        # Общий заголовок
        fig.suptitle(f'Сдвиги длин волн относительно среза №{reference_order}: {calibration_data["thar_file"]}\n'
                    f'Опорный срез показывает сдвиг относительно референсного решения',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Визуализация сохранена: {output_file}")
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Ошибка создания визуализации: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_reference_order_all_thars(
    calib_dir: Path,
    reference_file: Path,
    output_file: Path,
    data_dir: Path
) -> bool:
    """
    Создаёт визуализацию РЕАЛЬНЫХ экстрагированных профилей опорных срезов всех ThAr
    
    Показывает:
    - Реальный профиль опорного среза каждого ThAr в шкале длин волн
    - Вертикальные линии на длинах волн из интерактивной калибровки
    - Проверка компенсации сдвигов пиков автоматической калибровкой
    
    Parameters:
    -----------
    calib_dir : Path
        Директория с *_FULL.json файлами
    reference_file : Path
        Файл reference_solution.json
    output_file : Path
        Путь для сохранения PNG
    data_dir : Path
        Корневая директория с данными (для поиска ThAr и trace файлов)
    
    Returns:
    --------
    bool : True если успешно
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        
        # Загрузить reference решение
        ref_solution = load_calibration_solution(reference_file)
        if not ref_solution:
            logger.error("Не удалось загрузить reference решение")
            return False
        
        # Получить длины волн из интерактивной калибровки
        ref_calib_points = ref_solution.get('calib_points', {})
        manual_wavelengths = sorted([float(w) for w in ref_calib_points.values()])
        
        logger.info(f"Интерактивно отождествлено {len(manual_wavelengths)} линий")
        
        # Найти все *_FULL.json файлы
        full_files = sorted(calib_dir.glob('*_FULL.json'))
        if not full_files:
            logger.warning("Нет *_FULL.json файлов для визуализации")
            return False
        
        logger.info(f"Найдено {len(full_files)} откалиброванных ThAr")
        
        # Для каждого ThAr извлечь реальный профиль
        thar_profiles = []
        
        for full_file in full_files:
            solution = load_calibration_solution(full_file)
            if not solution or 'slices' not in solution:
                continue
            
            thar_filename = solution['thar_file']
            ref_order = solution['reference_order']
            ref_slice = solution['slices'].get(str(ref_order))
            
            if not ref_slice:
                continue
            
            # Найти ThAr FITS файл и trace файл
            thar_file = data_dir / thar_filename
            
            # Найти соответствующий trace файл (базируясь на имени ThAr)
            trace_dir = data_dir / 'TRACED_ORDERS'
            # Ищем trace файл который соответствует этому ThAr
            # Обычно это *_traced.json для того же кадра
            trace_file = None
            for trace_candidate in trace_dir.glob('*_traced.json'):
                # Попробуем найти по совпадению базового имени
                if thar_filename.replace('.fits', '') in trace_candidate.stem:
                    trace_file = trace_candidate
                    break
            
            if not trace_file or not trace_file.exists():
                # Попробуем альтернативный способ - ищем первый попавшийся trace
                traces = list(trace_dir.glob('*_traced.json'))
                if traces:
                    trace_file = traces[0]  # Используем первый (они все одинаковые для flat)
            
            if not thar_file.exists() or not trace_file:
                logger.warning(f"Пропуск {thar_filename}: файлы не найдены")
                continue
            
            # Загрузить ThAr изображение
            with fits.open(thar_file) as hdul:
                thar_data = hdul[0].data
            
            # Загрузить трассировку
            trace_data = load_trace_data(trace_file)
            if not trace_data:
                logger.warning(f"Пропуск {thar_filename}: нет трассировки")
                continue
            
            # Извлечь спектр опорного среза
            x_coords, flux = extract_order_summed(thar_data, trace_data, ref_order)
            if x_coords is None or flux is None:
                logger.warning(f"Пропуск {thar_filename}: не удалось извлечь срез {ref_order}")
                continue
            
            # Получить модель λ(pixel)
            model = np.poly1d(ref_slice['model'])
            
            # Преобразовать в шкалу длин волн
            wavelengths = model(x_coords)
            
            thar_profiles.append({
                'filename': thar_filename,
                'wavelengths': wavelengths,
                'flux': flux,
                'quality': ref_slice.get('quality', 'unknown'),
                'order': ref_order
            })
            
            logger.info(f"  Загружен {thar_filename}: {len(flux)} точек")
        
        if not thar_profiles:
            logger.warning("Не удалось загрузить ни одного профиля")
            return False
        
        # Создать субплоты (индивидуальные + 1 сводный)
        n_thars = len(thar_profiles)
        fig, axes = plt.subplots(n_thars + 1, 1, figsize=(16, 4*n_thars + 4), sharex=True)
        
        # axes всегда будет списком (даже если 1 ThAr)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Диапазон длин волн
        if manual_wavelengths:
            lambda_min = min(manual_wavelengths) - 10
            lambda_max = max(manual_wavelengths) + 10
        else:
            # Если нет ручных длин волн, используем диапазон первого профиля
            lambda_min = min(thar_profiles[0]['wavelengths'])
            lambda_max = max(thar_profiles[0]['wavelengths'])
        
        # Цвета качества
        quality_colors = {
            'excellent': 'green',
            'good': 'blue',
            'warning': 'orange',
            'critical': 'red',
            'unknown': 'gray'
        }
        
        # Построить каждый профиль
        for idx, profile in enumerate(thar_profiles):
            ax = axes[idx]
            
            # Нарисовать реальный профиль
            ax.plot(profile['wavelengths'], profile['flux'], 'b-', 
                   linewidth=0.5, alpha=0.8)
            
            # Вертикальные линии на интерактивных длинах волн
            for wl in manual_wavelengths:
                if lambda_min <= wl <= lambda_max:
                    ax.axvline(wl, color='red', linestyle=':', 
                              alpha=0.7, linewidth=0.4)
            
            # Метка
            color = quality_colors.get(profile['quality'], 'gray')
            label = f"{profile['filename']} (срез {profile['order']})"
            if idx == 0:
                label += " [ИНТЕРАКТИВНАЯ]"
            
            ax.text(0.01, 0.95, label, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            
            ax.set_ylabel('Поток (ADU)', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_xlim(lambda_min, lambda_max)
            
            if idx < n_thars:
                ax.set_xticklabels([])
        
        # СВОДНЫЙ ГРАФИК ВНИЗУ - все ThAr наложенные
        ax_combined = axes[-1]
        
        # Цвета для разных ThAr
        colors = plt.cm.tab10(np.linspace(0, 1, n_thars))
        
        for idx, profile in enumerate(thar_profiles):
            # Рисуем каждый спектр со своей шкалой λ (БЕЗ интерполяции)
            ax_combined.plot(profile['wavelengths'], profile['flux'], 
                           color=colors[idx], linewidth=0.7, alpha=0.6,
                           label=profile['filename'])
        
        # Вертикальные линии на интерактивных λ
        for wl in manual_wavelengths:
            if lambda_min <= wl <= lambda_max:
                ax_combined.axvline(wl, color='red', linestyle=':', 
                                  alpha=0.8, linewidth=0.7)
        
        # Настройки сводного графика
        ax_combined.set_ylabel('Поток (ADU)', fontsize=9)
        ax_combined.set_xlabel('Длина волны (Å)', fontsize=12, fontweight='bold')
        ax_combined.grid(True, alpha=0.3, linestyle=':')
        ax_combined.set_xlim(lambda_min, lambda_max)
       # ax_combined.legend(fontsize=8, loc='upper right', ncol=2)
        
        # Метка
        ax_combined.text(0.01, 0.95, 'ВСЕ ThAr НАЛОЖЕННЫЕ', 
                       transform=ax_combined.transAxes,
                       fontsize=11, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Заголовок
        title = f'Опорные срезы всех ThAr в шкале длин волн\n'
        title += f'Красные пунктиры = интерактивно отождествлённые длины волн ({len(manual_wavelengths)} линий)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Визуализация реальных профилей сохранена: {output_file}")
        plt.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка создания визуализации ThAr: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# СОХРАНЕНИЕ/ЗАГРУЗКА
# =============================================================================

def save_calibration_solution(solution: dict, output_file: Path) -> bool:
    """Сохраняет калибровочное решение в JSON"""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(solution, f, indent=2)
        logger.info(f"✓ Решение сохранено: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения {output_file}: {e}")
        return False


def load_calibration_solution(input_file: Path) -> Optional[dict]:
    """Загружает калибровочное решение из JSON"""
    try:
        with open(input_file, 'r') as f:
            solution = json.load(f)
        logger.info(f"✓ Решение загружено: {input_file}")
        return solution
    except Exception as e:
        logger.error(f"Ошибка загрузки {input_file}: {e}")
        return None


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    """Главная функция с поддержкой различных режимов калибровки"""
    import argparse
    
    # parser = argparse.ArgumentParser(description='Автоматическая калибровка ThAr')
    # subparsers = parser.add_subparsers(dest='mode', help='Режим работы')
    
    # # Режим 1: Интерактивная калибровка одного среза
    # interactive = subparsers.add_parser('interactive',
    #     help='Интерактивная калибровка одного среза')
    # interactive.add_argument('thar_file', help='ThAr FITS файл')
    # interactive.add_argument('trace_file', help='JSON трассировка')
    # interactive.add_argument('--order', type=int, default=11, help='Номер среза')
    # interactive.add_argument('--atlas', default='thar.dat', help='Файл атласа ThAr')
    # interactive.add_argument('--output', required=True, help='Выходной JSON файл')
    # interactive.add_argument('--gain', type=float, default=2.0)
    # interactive.add_argument('--ron', type=float, default=5.6)
    
    # # Режим 2: Распространение калибровки на все срезы
    # expand = subparsers.add_parser('expand-calibration',
    #     help='Распространить калибровку одного среза на все 14 срезов')
    # expand.add_argument('thar_file', help='ThAr FITS файл')
    # expand.add_argument('trace_file', help='JSON трассировка')
    # expand.add_argument('--reference', required=True,
    #     help='JSON с калибровкой одного среза (например, test.json)')
    # expand.add_argument('--reference-order', type=int, default=11,
    #     help='Номер опорного среза (по умолчанию 11)')
    # expand.add_argument('--poly-degree', type=int, default=4,
    #     help='Степень полинома (по умолчанию 4)')
    # expand.add_argument('--atlas', default='thar.dat', help='Файл атласа ThAr')
    # expand.add_argument('--output', required=True,
    #     help='Выходной JSON файл с калибровкой всех срезов')
    # expand.add_argument('--visualize', 
    #     help='Создать визуализацию дисперсионных кривых (PNG файл)')
    
    # # Режим 3: Автоматическая калибровка одного среза (старый режим)
    # auto = subparsers.add_parser('auto',
    #     help='Автоматическая калибровка одного среза на основе reference')
    # auto.add_argument('thar_file', help='ThAr FITS файл')
    # auto.add_argument('trace_file', help='JSON трассировка')
    # auto.add_argument('--order', type=int, default=11, help='Номер среза')
    # auto.add_argument('--reference', required=True, help='Reference калибровка')
    # auto.add_argument('--atlas', default='thar.dat', help='Файл атласа ThAr')
    # auto.add_argument('--output', help='Выходной JSON файл')
    # auto.add_argument('--gain', type=float, default=2.0)
    # auto.add_argument('--ron', type=float, default=5.6)
    
    # args = parser.parse_args()
    
    # if not args.mode:
    #     parser.print_help()
    #     return
    
    # # === РЕЖИМ: ИНТЕРАКТИВНАЯ КАЛИБРОВКА ===
    # if args.mode == 'interactive':
    #     solution = calibrate_first_thar_interactive(
    #         thar_file=Path(args.thar_file),
    #         trace_file=Path(args.trace_file),
    #         order_num=args.order,
    #         atlas_file=Path(args.atlas),
    #         gain=args.gain,
    #         ron_e=args.ron
    #     )
        
    #     if solution:
    #         save_calibration_solution(solution, Path(args.output))
    #     else:
    #         logger.error("Интерактивная калибровка не завершена")
    
    # # === РЕЖИМ: РАСПРОСТРАНЕНИЕ КАЛИБРОВКИ ===
    # elif args.mode == 'expand-calibration':
    #     logger.info("="*80)
    #     logger.info("РАСПРОСТРАНЕНИЕ КАЛИБРОВКИ НА ВСЕ СРЕЗЫ")
    #     logger.info("="*80)
        
    #     # Загрузить reference решение
    #     ref_solution = load_calibration_solution(Path(args.reference))
    #     if ref_solution is None:
    #         logger.error(f"Не удалось загрузить reference: {args.reference}")
    #         return
        
    #     logger.info(f"✓ Загружено решение для среза №{args.reference_order}")
    #     logger.info(f"  RMS: {ref_solution.get('rms', 0):.4f} Å")
    #     logger.info(f"  Точек: {ref_solution.get('n_points', 0)}")
    #     logger.info(f"  Степень полинома: {ref_solution.get('model_degree', 0)}")
        
    #     # Откалибровать все 14 срезов
    #     full_solution = calibrate_all_slices_auto(
    #         thar_file=Path(args.thar_file),
    #         trace_file=Path(args.trace_file),
    #         reference_order=args.reference_order,
    #         reference_solution=ref_solution,
    #         atlas_file=Path(args.atlas),
    #         poly_degree=args.poly_degree
    #     )
        
    #     if full_solution:
    #         save_calibration_solution(full_solution, Path(args.output))
    #         logger.info(f"\n✅ ГОТОВО! Откалибровано {len(full_solution['slices'])}/14 срезов")
    #         logger.info(f"   Результат сохранен: {args.output}")
            
    #         # Создать визуализацию, если указано
    #         if args.visualize:
    #             logger.info("\nСоздание визуализации...")
    #             visualize_all_slices_calibration(full_solution, Path(args.visualize))
    #     else:
    #         logger.error("Калибровка не удалась")
    
    # # === РЕЖИМ: АВТОМАТИЧЕСКАЯ КАЛИБРОВКА ===
    # elif args.mode == 'auto':
        # ref_solution = load_calibration_solution(Path(args.reference))
        # if ref_solution is None:
        #     return
        
        # solution = calibrate_thar_auto(
        #     thar_file=Path(args.thar_file),
        #     trace_file=Path(args.trace_file),
        #     order_num=args.order,
        #     reference_solution=ref_solution,
        #     atlas_file=Path(args.atlas),
        #     gain=args.gain,
        #     ron_e=args.ron
        # )
        
        # if solution and args.output:
        #     save_calibration_solution(solution, Path(args.output))
        # elif solution:
        #     print("\n" + "="*80)
        #     print("РЕШЕНИЕ:")
        #     print("="*80)
        #     print(json.dumps(solution, indent=2))

    full_solution = load_calibration_solution(Path('/data/Observations/test_pyzeeman_final/CALIBRATIONS/o031_CRR_bt_FULL.json'))
    visualize_all_slices_calibration(full_solution, Path('/data/Observations/test_pyzeeman_final/CALIBRATIONS/test_plot.pdf'))


if __name__ == '__main__':
    main()
