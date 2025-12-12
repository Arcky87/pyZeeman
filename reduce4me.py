#!/usr/bin/env python3
"""
reduce4me.py - Полный пайплайн обработки спектров pyZeeman

Этапы обработки:
1. Препроцессинг (lister, trimmer, cosmic ray removal, bias/flat)
2. Трассировка порядков
3. Сопоставление ThAr с научными кадрами
4. Калибровка ThAr (интерактивная + автоматическая)
5. Экстракция и объединение в векторы поляриметрии

Автор: pyZeeman team
Версия: 2.0
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from astropy.io import fits

# Импорт модулей препроцессинга
from lister import lister
from trimmer import trimmer
from list_astroscrappy import list_cosmo_cleaner
from medianer import medianer
from list_subtractor import list_subtractor

# Импорт модулей трассировки
from not_so_simple_tracer import trace_orders

# Импорт модулей новой калибровки
from match_thar_to_science import match_thar_to_science
from thar_auto_calibration import (
    calibrate_first_thar_interactive,
    calibrate_all_slices_auto,
    load_calibration_solution,
    save_calibration_solution
)
from apply_calibration import apply_calibration_native, visualize_calibrated_spectra
from combine_orders import create_polarimetry_vectors


# =============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================================================================

def setup_logging(data_dir: Path):
    """Настройка системы логирования"""
    log_file = data_dir / 'pyzeeman_pipeline.log'
    
    # Получаем root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие handlers (на случай повторного вызова)
    logger.handlers.clear()
    
    # Формат сообщений
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # FileHandler для записи в файл
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # StreamHandler для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# ЭТАП 1: ПРЕПРОЦЕССИНГ
# =============================================================================

def stage_1_preprocessing(config, logger):
    """
    Препроцессинг: сортировка, обрезка, очистка, создание мастер-кадров
    """
    logger.info("="*80)
    logger.info("ЭТАП 1: ПРЕПРОЦЕССИНГ")
    logger.info("="*80)
    
    data_dir = Path(config['data_dir'])
    raw_dir = data_dir / 'RAW'
    temp_dir = data_dir / 'temp'
    
    # Создать директории
    raw_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    # 1.1 Сортировка кадров по типам
    logger.info("\n1.1 Сортировка кадров...")
    files = lister(data_dir, raw_dir, temp_dir)
    if files is None:
        logger.error("Ошибка сортировки файлов")
        return False
    logger.info("✓ Списки созданы")
    
    # 1.2 Обрезка overscan
    logger.info("\n1.2 Обрезка overscan...")
    area = config['preprocessing']['trim_area']
    flip = config['preprocessing']['flip']
    
    for list_name in ['bias_list.txt', 'flat_list.txt', 'thar_list.txt', 'obj_list.txt']:
        list_file = temp_dir / list_name
        if list_file.exists():
            trimmer(data_dir, list_file, area, flip)
            logger.info(f"✓ {list_name} обрезан")
  
    # 1.3 Удаление космических частиц
    if config['preprocessing'].get('cosmic_ray_removal', True):
        logger.info("\n1.3 Удаление космических частиц...")
        status = list_cosmo_cleaner(
            dir_name=temp_dir,
            list_name='obj_list.txt',
            out_list_name='obj_CRR_list.txt',
            mask_list_name='obj_mask_list.txt',
            plot=False,
            rdn=config.get('ron', 5.6),
            gf=config.get('gain', 2.78),
            sigclip=config['preprocessing'].get('cosmic_ray_params', {}).get('sigclip', 4.5),
            sigfrac=config['preprocessing'].get('cosmic_ray_params', {}).get('sigfrac', 0.3),
            objlim=config['preprocessing'].get('cosmic_ray_params', {}).get('objlim', 5.0),
            niter=config['preprocessing'].get('cosmic_ray_params', {}).get('niter', 4)
        )
        logger.info(f"✓ {status}")
    
    # 1.4 Создание super bias
    logger.info("\n1.4 Создание super bias...")
    s_bias_name = config.get('s_bias_name', 's_bias.fits')
    sbias_data = medianer(data_dir, temp_dir / 'bias_list.txt', s_bias_name)
    logger.info(f"✓ Super bias: Mean={sbias_data[0]:.2f}, Median={sbias_data[1]:.2f}, Sigma={sbias_data[2]:.2f}")
    
    # 1.5 Вычитание bias
    logger.info("\n1.5 Вычитание bias...")
    for list_name in ['flat_list.txt', 'thar_list.txt', 'obj_CRR_list.txt']:
        list_file = temp_dir / list_name
        if list_file.exists():
            status = list_subtractor(list_file, data_dir / s_bias_name, 'Bias')
            logger.info(f"✓ {list_name}: {status}")
    
    # 1.6 Удаление рассеянного света (только для научных кадров)
    if config['preprocessing'].get('scatter_light', {}).get('enabled', False):
        logger.info("\n1.6 Удаление рассеянного света...")
        
        from backlong_zee import subtract_scattered_light
        
        border_width = config['preprocessing']['scatter_light'].get('border_width', 60)
        
        # Обработать только научные кадры (после вычитания bias)
        input_list = 'obj_CRR_list.txt'
        output_list = 'obj_CRR_bt_list.txt'
        
        if (temp_dir / input_list).exists():
            status = subtract_scattered_light(
                dir_name=temp_dir,
                list_name=input_list,
                out_list_name=output_list,
                border_width=border_width,
                plot=False  # Визуализация отключена
            )
            logger.info(f"✓ {status}")
            logger.info(f"  Фоновые изображения: *_background.fits")
        else:
            logger.warning(f"! Список {input_list} не найден")
    
    # 1.7 Создание super flat
    logger.info("\n1.7 Создание super flat...")
    s_flat_name = config.get('s_flat_name', 's_flat.fits')
    sflat_data = medianer(data_dir, temp_dir / 'flat_list.txt', s_flat_name)
    logger.info(f"✓ Super flat: Mean={sflat_data[0]:.2f}, Median={sflat_data[1]:.2f}, Sigma={sflat_data[2]:.2f}")
    
    logger.info("\n✓ Препроцессинг завершён")
    return True


# =============================================================================
# ЭТАП 2: ТРАССИРОВКА
# =============================================================================

def stage_2_tracing(config, logger):
    """
    Трассировка спектральных порядков
    """
    logger.info("\n" + "="*80)
    logger.info("ЭТАП 2: ТРАССИРОВКА ПОРЯДКОВ")
    logger.info("="*80)
    
    data_dir = Path(config['data_dir'])
    temp_dir = data_dir / 'temp'
    traced_dir = data_dir / 'TRACED_ORDERS'
    
    # Получить список обработанных научных кадров
    # Используем список после удаления рассеянного света, если оно было выполнено
    if config['preprocessing'].get('scatter_light', {}).get('enabled', False):
        obj_list = temp_dir / 'obj_CRR_bt_list.txt'
    else:
        obj_list = temp_dir / 'obj_CRR_list.txt'
    
    if not obj_list.exists():
        logger.error(f"Не найден список научных кадров: {obj_list.name}")
        return False
    
    logger.info(f"Используется список: {obj_list.name}")
    
    # Параметры трассировки
    s_flat_name = config.get('s_flat_name', 's_flat.fits')
    flat_file = data_dir / s_flat_name
    
    if not flat_file.exists():
        logger.error(f"! Super flat не найден: {flat_file}")
        return False
    
    # Запуск пакетной трассировки
    logger.info(f"Flat файл: {flat_file}")
    logger.info(f"Список спектров: {obj_list}")
    logger.info(f"Выходная директория: {traced_dir}")
    
    try:
        trace_orders(
            flat_file=str(flat_file),
            spec_list=str(obj_list),
            output_dir=str(traced_dir),
            n_orders=config.get('n_orders', 14),
            n_points_for_fit=config.get('tracing', {}).get('n_points_for_fit', 10),
            smooth=config.get('tracing', {}).get('smooth', False),
            smooth_sigma=config.get('tracing', {}).get('smooth_sigma',1.0),
            getxwd_gauss=config.get('tracing', {}).get('getxwd_gauss', True),
            show_point_fits=config.get('tracing', {}).get('visualization', {}).get('show_point_fits', False),
            show_final_traces=config.get('tracing', {}).get('visualization', {}).get('show_final_traces', False),
            save_plots=config.get('tracing', {}).get('visualization', {}).get('save_plots', False),
            overwrite=config.get('tracing', {}).get('overwrite', False),
            save_format=config.get('tracing', {}).get('save_format', json)
        )
        logger.info("✓ Трассировка завершена")
        return True
    except Exception as e:
        logger.error(f"! Ошибка трассировки: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# ЭТАП 3: СОПОСТАВЛЕНИЕ THAR
# =============================================================================

def stage_3_match_thar(config, logger):
    """
    Сопоставление ThAr с научными кадрами
    """
    logger.info("\n" + "="*80)
    logger.info("ЭТАП 3: СОПОСТАВЛЕНИЕ THAR")
    logger.info("="*80)
    
    data_dir = Path(config['data_dir'])
    temp_dir = data_dir / 'temp'
    
    # Получить пути к спискам
    # Используем тот же список, что будет использоваться для трассировки и калибровки
    if config['preprocessing'].get('scatter_light', {}).get('enabled', False):
        obj_list_file = temp_dir / 'obj_CRR_bt_list.txt'
        logger.info("Используется список после удаления рассеянного света")
    else:
        obj_list_file = temp_dir / 'obj_CRR_list.txt'
    
    thar_list_file = temp_dir / 'thar_list.txt'
    
    if not obj_list_file.exists() or not thar_list_file.exists():
        logger.error("Не найдены списки файлов")
        return False
    
    # Прочитать списки и создать List[Path]
    logger.info("Загрузка списков файлов...")
    with open(obj_list_file, 'r') as f:
        obj_files = [Path(line.strip()) for line in f if line.strip()]
    
    with open(thar_list_file, 'r') as f:
        thar_files = [Path(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Загружено: {len(obj_files)} объектных, {len(thar_files)} ThAr")
    
    # Запустить сопоставление (передаём списки Path объектов)
    logger.info("Запуск match_thar_to_science...")
    try:
        match_thar_to_science(obj_files, thar_files)
        logger.info("✓ THAR_REF добавлены в заголовки")
    except Exception as e:
        logger.error(f"! Ошибка сопоставления: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


# =============================================================================
# ЭТАП 4: КАЛИБРОВКА THAR
# =============================================================================

def stage_4_thar_calibration(config, logger):
    """
    Калибровка ThAr: интерактивная (первый раз) + автоматическая (остальные)
    """
    logger.info("\n" + "="*80)
    logger.info("ЭТАП 4: КАЛИБРОВКА THAR")
    logger.info("="*80)
    
    data_dir = Path(config['data_dir'])
    temp_dir = data_dir / 'temp'
    traced_dir = data_dir / 'TRACED_ORDERS'
    calib_dir = data_dir / 'CALIBRATIONS'
    calib_dir.mkdir(exist_ok=True)
    
    reference_file = calib_dir / 'reference_solution.json'
    reference_order = config.get('reference_order', 11)
    atlas_file = Path(config.get('atlas_file', 'thar.dat'))
    
    # Получить список научных кадров
    # Используем тот же список, что и для трассировки
    if config['preprocessing'].get('scatter_light', {}).get('enabled', False):
        obj_list = temp_dir / 'obj_CRR_bt_list.txt'
        logger.info("Используется список после удаления рассеянного света")
    else:
        obj_list = temp_dir / 'obj_CRR_list.txt'
    
    with open(obj_list, 'r') as f:
        science_files = [Path(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Загружено научных кадров: {len(science_files)}")
    
    # 4.1 Проверка наличия reference решения
    if not reference_file.exists():
        logger.info("\n4.1 Reference решение не найдено. Требуется интерактивная калибровка.")
        logger.info("="*80)
        logger.info("ИНТЕРАКТИВНАЯ КАЛИБРОВКА (первый раз за ночь)")
        logger.info("="*80)
        
        # Взять первый кадр для примера
        first_science = science_files[4]
        
        # Прочитать THAR_REF
        with fits.open(first_science) as hdul:
            thar_ref = hdul[0].header.get('THAR_REF', None)
        
        if not thar_ref:
            logger.error("! THAR_REF не найден в заголовке первого кадра")
            return False
        
        thar_file = data_dir / thar_ref
        trace_file = traced_dir / f"{first_science.stem}_traced.json"
        
        logger.info(f"ThAr файл: {thar_file.name}")
        logger.info(f"Trace файл: {trace_file.name}")
        logger.info(f"Опорный срез: {reference_order}")
        logger.info("\nЗапуск интерактивной калибровки...")
        logger.info("Следуйте инструкциям в графическом окне!")
        
        # Интерактивная калибровка
        try:
            reference_solution = calibrate_first_thar_interactive(
                thar_file=thar_file,
                trace_file=trace_file,
                order_num=reference_order,
                atlas_file=atlas_file,
                gain=config.get('gain', 2.78),
                ron_e=config.get('ron', 5.6)
            )
            
            if reference_solution:
                save_calibration_solution(reference_solution, reference_file)
                logger.info(f"\n✓ Reference решение сохранено: {reference_file}")
            else:
                logger.error("! Интерактивная калибровка не завершена")
                return False
        except Exception as e:
            logger.error(f"! Ошибка интерактивной калибровки: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.info(f"✓ Reference решение найдено: {reference_file}")
    
    # Загрузить reference
    reference_solution = load_calibration_solution(reference_file)
    if not reference_solution:
        logger.error("! Не удалось загрузить reference")
        return False
    
    # 4.2 Автоматическая калибровка для каждого кадра
    logger.info("\n4.2 Автоматическая калибровка всех кадров...")
    logger.info("="*60)
    
    for science_file in science_files:
        logger.info(f"\nКадр: {science_file.name}")
        
        # Прочитать THAR_REF
        with fits.open(science_file) as hdul:
            thar_ref = hdul[0].header.get('THAR_REF', None)
        
        if not thar_ref:
            logger.warning(f"! THAR_REF не найден в {science_file.name}, пропуск")
            continue
        
        thar_file = data_dir / thar_ref
        trace_file = traced_dir / f"{science_file.stem}_traced.json"
        calib_output = calib_dir / f"{science_file.stem}_FULL.json"
        
        if not trace_file.exists():
            logger.warning(f"! Trace не найден: {trace_file.name}, пропуск")
            continue
        
        logger.info(f"  THAR_REF: {thar_ref}")
        logger.info(f"  Trace: {trace_file.name}")
        
        # Параметры калибровки из конфига
        calib_config = config.get('calibration', {})
        
        peak_params = calib_config.get('peak_detection', {
            'prominence_sigma': 15.0,
            'width_range': (1.8, 3.5),
            'distance_pixels': 10
        })
        
        matching_params = calib_config.get('matching', {
            'wavelength_tolerance': 0.5,
            'max_shift_pixels': 3.0
        })
        
        quality_params = calib_config.get('quality_thresholds', {
            'rms_excellent': 0.03,
            'rms_good': 0.09,
            'rms_warning': 0.25
        })
        
        viz_params = calib_config.get('visualization', {
            'save_plots': False,
            'output_format': 'png',
            'dpi': 150
        })
        
        # Создать FULL калибровку
        try:
            full_calibration = calibrate_all_slices_auto(
                thar_file=thar_file,
                trace_file=trace_file,
                reference_order=reference_order,
                reference_solution=reference_solution,
                atlas_file=atlas_file,
                poly_degree=config.get('poly_degree', 4),
                rms_excellent=quality_params['rms_excellent'],
                rms_good=quality_params['rms_good'],
                rms_warning=quality_params['rms_warning'],
                pixel_tolerance=matching_params['max_shift_pixels'],
                wavelength_tolerance=matching_params['wavelength_tolerance'],
                peak_params=peak_params,
                compare_template=False
            )
            
            if full_calibration:
                save_calibration_solution(full_calibration, calib_output)
                logger.info(f"  ✓ Калибровка: {calib_output.name}")
                
                # Визуализация если включена
                if viz_params['save_plots']:
                    from thar_auto_calibration import visualize_all_slices_calibration
                    viz_file = calib_dir / f"{science_file.stem}_calibration.{viz_params['output_format']}"
                    visualize_all_slices_calibration(full_calibration, viz_file)
            else:
                logger.warning(f"  ! Калибровка не удалась для {science_file.name}")
        except Exception as e:
            logger.error(f"  ! Ошибка: {e}")
    
    logger.info("\n✓ Калибровка ThAr завершена")
    
    # # Создать сводную визуализацию всех ThAr
    # logger.info("\nСоздание сводной визуализации опорных срезов...")
    # from thar_auto_calibration import visualize_reference_order_all_thars
        
    # viz_all_thars = calib_dir / "all_thars_reference_order.pdf"
    # visualize_reference_order_all_thars(
    #     calib_dir=calib_dir,
    #     reference_file=reference_file,
    #     output_file=viz_all_thars,
    #     data_dir=data_dir
    # )
    
    return True


# =============================================================================
# ЭТАП 5: ЭКСТРАКЦИЯ И ОБЪЕДИНЕНИЕ
# =============================================================================

def stage_5_extraction(config, logger):
    """
    Экстракция спектров и объединение в векторы поляриметрии
    """
    logger.info("\n" + "="*80)
    logger.info("ЭТАП 5: ЭКСТРАКЦИЯ И ОБЪЕДИНЕНИЕ")
    logger.info("="*80)
    
    data_dir = Path(config['data_dir'])
    temp_dir = data_dir / 'temp'
    traced_dir = data_dir / 'TRACED_ORDERS'
    calib_dir = data_dir / 'CALIBRATIONS'
    calibrated_dir = data_dir / 'CALIBRATED'
    final_dir = data_dir / 'FINAL'
    
    calibrated_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)
    
    # Получить список научных кадров
    # Используем файлы после удаления рассеянного света для экстракции
    if config['preprocessing'].get('scatter_light', {}).get('enabled', False):
        obj_list = temp_dir / 'obj_CRR_bt_list.txt'
        logger.info("Используется список после удаления рассеянного света")
    else:
        obj_list = temp_dir / 'obj_CRR_list.txt'
    
    with open(obj_list, 'r') as f:
        science_files = [Path(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Научных кадров для экстракции: {len(science_files)}")
    
    upper_orders = config.get('upper_orders', [1,2,3,4,5,6,7])
    lower_orders = config.get('lower_orders', [8,9,10,11,12,13,14])
    
    # Обработать каждый кадр
    for science_file in science_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Кадр: {science_file.name}")
        logger.info("="*60)
        
        trace_file = traced_dir / f"{science_file.stem}_traced.json"
        calib_file = calib_dir / f"{science_file.stem}_FULL.json"
        output_dir = calibrated_dir / science_file.stem
        
        if not trace_file.exists():
            logger.warning(f"! Trace не найден, пропуск")
            continue
        
        if not calib_file.exists():
            logger.warning(f"! Калибровка не найдена, пропуск")
            continue
        
        # 5.1 Применение калибровки
        logger.info("5.1 Применение калибровки...")
        try:
            results = apply_calibration_native(
                science_file=science_file,
                trace_file=trace_file,
                calibration_file=calib_file,
                output_dir=output_dir,
                gain=config.get('gain', 2.78),
                ron_e=config.get('ron', 5.6)
            )
            
            if results:
                logger.info(f"✓ Извлечено {len(results)} срезов")
                sci_name = Path(science_file).stem
                visualize_calibrated_spectra(calibrated_data=results,
                                             output_file=Path(science_file.parent/"CALIBRATED"/f"calib_orders_{sci_name}.pdf"),
                                             title=f'Откалиброванный спектр: {sci_name}')
            else:
                logger.warning("! Экстракция не удалась")
                continue
        except Exception as e:
            logger.error(f"! Ошибка экстракции: {e}")
            continue
        
        # 5.2 Объединение в векторы
        logger.info("\n5.2 Объединение в векторы...")
        try:
            output_base = final_dir / science_file.stem
            
            vectors = create_polarimetry_vectors(
                calibrated_dir=output_dir,
                output_base=output_base,
                upper_orders=upper_orders,
                lower_orders=lower_orders
            )
            
            if vectors:
                logger.info(f"✓ Векторы созданы:")
                logger.info(f"  {output_base}_1.fits")
                logger.info(f"  {output_base}_2.fits")
            else:
                logger.warning("! Объединение не удалось")
                continue
        except Exception as e:
            logger.error(f"! Ошибка объединения: {e}")
            continue
        
        # 5.3 Барицентрическая коррекция финальных векторов
        if config.get('barycentric_correction', {}).get('enabled', True):
            logger.info("\n5.3 Барицентрическая коррекция...")
            
            try:
                from barycorr import process_vector_files, OBSERVATORY
                
                # Получить координаты обсерватории из конфига
                obs_config = config.get('observatory', OBSERVATORY)
                
                # Применить коррекцию к финальным векторам в директории FINAL
                success = process_vector_files(
                    final_dir=final_dir,
                    science_file=science_file,
                    observatory=obs_config
                )
                
                if success:
                    logger.info("  ✓ Барицентрическая коррекция применена к финальным векторам")
                else:
                    logger.warning("  ! Барицентрическая коррекция не удалась")
            except Exception as e:
                logger.error(f"  ! Ошибка барицентрической коррекции: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info("\n5.3 Барицентрическая коррекция отключена в конфигурации")
    
    logger.info("\n✓ Экстракция и объединение завершены")
    return True


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="pyZeeman Pipeline - Полная обработка спектров"
    )
    parser.add_argument('config', help='Файл конфигурации (JSON)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Пропустить препроцессинг')
    parser.add_argument('--skip-tracing', action='store_true',
                       help='Пропустить трассировку')
    parser.add_argument('--skip-matching', action='store_true',
                       help='Пропустить сопоставление ThAr')
    parser.add_argument('--skip-calibration', action='store_true',
                       help='Пропустить калибровку ThAr')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Пропустить экстракцию и объединение')
    parser.add_argument('--only-calibration', action='store_true',
                       help='Только калибровка (пропустить всё до калибровки)')
    parser.add_argument('--only-extraction', action='store_true',
                       help='Только экстракция (пропустить всё до экстракции)')
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_dir = Path(config['data_dir'])
    logger = setup_logging(data_dir)
    
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("pyZeeman Pipeline - Полная обработка спектров")
    logger.info("="*80)
    logger.info(f"Время старта: {start_time}")
    logger.info(f"Директория данных: {data_dir}")
    logger.info(f"Конфигурация: {args.config}")
    logger.info("="*80)
    
    try:
        # Этап 1: Препроцессинг
        if not args.skip_preprocessing and not args.only_calibration and not args.only_extraction:
            if not stage_1_preprocessing(config, logger):
                logger.error("! Препроцессинг завершился с ошибкой")
                return 1
        else:
            logger.info("Пропуск препроцессинга")
        
        # Этап 2: Трассировка
        if not args.skip_tracing and not args.only_calibration and not args.only_extraction:
            if not stage_2_tracing(config, logger):
                logger.error("! Трассировка завершилась с ошибкой")
                return 1
        else:
            logger.info("Пропуск трассировки")
        
        # Этап 3: Сопоставление ThAr
        if not args.skip_matching and not args.only_calibration and not args.only_extraction:
            if not stage_3_match_thar(config, logger):
                logger.error("! Сопоставление ThAr завершилось с ошибкой")
                return 1
        else:
            logger.info("Пропуск сопоставления ThAr")
        
        # Этап 4: Калибровка ThAr
        if not args.skip_calibration and not args.only_extraction:
            if not stage_4_thar_calibration(config, logger):
                logger.error("! Калибровка ThAr завершилась с ошибкой")
                return 1
        else:
            logger.info("Пропуск калибровки ThAr")
        
        # Этап 5: Экстракция и объединение
        if not args.skip_extraction:
            if not stage_5_extraction(config, logger):
                logger.error("! Экстракция завершилась с ошибкой")
                return 1
        else:
            logger.info("Пропуск экстракции и объединения")
        
        # Завершение
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("="*80)
        logger.info(f"Время окончания: {end_time}")
        logger.info(f"Длительность: {duration}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n! КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
