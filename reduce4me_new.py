#!/usr/bin/env python3
"""
reduce4me_new.py - Главный скрипт обработки для пайплайна pyZeeman

Пайплайн для обработки двумерных астрономических спектров, полученных
с образ-срезателя и анализатора круговой поляризации (ОЗСП).

Автор: Рефакторинг для pyZeeman
Дата: 2025-10-14
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Импорт модулей пайплайна
from lister import lister
from trimmer import trimmer
from list_astroscrappy import list_cosmo_cleaner
from medianer import medianer
from list_subtractor import list_subtractor

# Импорт новых модулей для pyZeeman
from not_so_simple_tracer import trace_orders
# Примечание: thar_calibration.py используется интерактивно отдельно
# from apply_calibration import main as apply_calibration
# from combine_orders import main as combine_orders

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# ==============================================================================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ==============================================================================

DEFAULT_CONFIG = {
    "data_path": "/data/Observations/test_pyzeeman",
    "gain": 2.78,
    "ron": 5.6,
    "n_orders": 14,
    "trim_area": [0, 2048, 0, 2048],  # [x1, x2, y1, y2]
    "flip": False,
    "cosmic_ray_params": {
        "sigclip": 4.5,
        "sigfrac": 0.3,
        "objlim": 5.0,
        "niter": 4
    },
    "tracer_params": {
        "n_points_for_fit": 10,
        "smooth": True,
        "smooth_sigma": 1.0,
        "getxwd_gauss": False
    }
}


# ==============================================================================
# ФУНКЦИИ ПАЙПЛАЙНА
# ==============================================================================

def setup_logging(log_path):
    """Настройка системы логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def setup_directories(data_path):
    """Создание необходимых директорий"""
    data_path = Path(data_path)
    
    raw_dir = data_path / 'RAW'
    temp_dir = data_path / 'TEMP'
    reduced_dir = data_path / 'REDUCED'
    traced_dir = data_path / 'TRACED_ORDERS'
    calibrated_dir = data_path / 'CALIBRATED'
    combined_dir = data_path / 'COMBINED'
    
    for directory in [temp_dir, reduced_dir, traced_dir, calibrated_dir, combined_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        'data': data_path,
        'raw': raw_dir,
        'temp': temp_dir,
        'reduced': reduced_dir,
        'traced': traced_dir,
        'calibrated': calibrated_dir,
        'combined': combined_dir
    }


def step_01_create_lists(dirs, logger):
    """Шаг 1: Создание списков кадров по типам"""
    logger.info("="*80)
    logger.info("ШАГ 1: Создание списков кадров")
    logger.info("="*80)
    
    result = lister(dirs['data'], dirs['raw'], dirs['temp'])
    
    if result:
        logger.info("✓ Списки успешно созданы")
        
        # Проверка созданных списков
        lists = ['bias_list.txt', 'flat_list.txt', 'thar_list.txt', 'obj_list.txt']
        for list_file in lists:
            list_path = dirs['temp'] / list_file
            if list_path.exists():
                with open(list_path, 'r') as f:
                    count = len(f.readlines())
                logger.info(f"  - {list_file}: {count} файлов")
        return True
    else:
        logger.error("✗ Ошибка при создании списков")
        return False


def step_02_trim_images(dirs, config, logger):
    """Шаг 2: Обрезка области интереса"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 2: Обрезка области интереса (ROI)")
    logger.info("="*80)
    
    area = config['trim_area']
    flip = config['flip']
    
    lists_to_trim = ['bias_list.txt', 'flat_list.txt', 'thar_list.txt', 'obj_list.txt']
    
    for list_name in lists_to_trim:
        list_path = dirs['temp'] / list_name
        if list_path.exists():
            logger.info(f"Обрезка {list_name}...")
            try:
                trimmer(dirs['data'], list_path, area, flip)
                logger.info(f"  ✓ {list_name} обрезан")
            except Exception as e:
                logger.error(f"  ✗ Ошибка при обрезке {list_name}: {e}")
                return False
    
    return True


def step_03_remove_cosmic_rays(dirs, config, logger):
    """Шаг 3: Удаление космических частиц"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 3: Удаление космических частиц")
    logger.info("="*80)
    
    obj_list = 'obj_list.txt'
    obj_crr_list = 'obj_CRR_list.txt'
    mask_list = 'obj_mask_list.txt'
    
    cr_params = config['cosmic_ray_params']
    
    try:
        status = list_cosmo_cleaner(
            dirs['temp'],
            obj_list,
            obj_crr_list,
            mask_list,
            plot=False,
            rdn=config['ron'],
            gf=config['gain'],
            **cr_params
        )
        logger.info(f"✓ Космические частицы удалены: {status}")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка при удалении космических частиц: {e}")
        return False


def step_04_create_master_calibrations(dirs, config, logger):
    """Шаг 4: Создание мастер-калибровок"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 4: Создание мастер-калибровок")
    logger.info("="*80)
    
    # Мастер-BIAS
    logger.info("Создание мастер-BIAS...")
    try:
        bias_stats = medianer(
            dirs['data'],
            dirs['temp'] / 'bias_list.txt',
            's_bias.fits'
        )
        logger.info(f"  ✓ Мастер-BIAS: Mean={bias_stats[0]:.2f}, Median={bias_stats[1]:.2f}, Sigma={bias_stats[2]:.2f}")
    except Exception as e:
        logger.error(f"  ✗ Ошибка при создании мастер-BIAS: {e}")
        return False
    
    # Мастер-FLAT
    logger.info("Создание мастер-FLAT...")
    try:
        flat_stats = medianer(
            dirs['data'],
            dirs['temp'] / 'flat_list.txt',
            's_flat.fits'
        )
        logger.info(f"  ✓ Мастер-FLAT: Mean={flat_stats[0]:.2f}, Median={flat_stats[1]:.2f}, Sigma={flat_stats[2]:.2f}")
    except Exception as e:
        logger.error(f"  ✗ Ошибка при создании мастер-FLAT: {e}")
        return False
    
    return True


def step_05_subtract_calibrations(dirs, logger):
    """Шаг 5: Вычитание калибровок"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 5: Вычитание калибровок")
    logger.info("="*80)
    
    s_bias_path = dirs['data'] / 's_bias.fits'
    
    # Вычитание BIAS из FLAT
    logger.info("Вычитание BIAS из FLAT...")
    try:
        status = list_subtractor(
            dirs['temp'] / 'flat_list.txt',
            s_bias_path,
            'Bias'
        )
        logger.info(f"  ✓ FLAT: {status}")
    except Exception as e:
        logger.error(f"  ✗ Ошибка: {e}")
        return False
    
    # Вычитание BIAS из ThAr
    logger.info("Вычитание BIAS из ThAr...")
    try:
        status = list_subtractor(
            dirs['temp'] / 'thar_list.txt',
            s_bias_path,
            'Bias'
        )
        logger.info(f"  ✓ ThAr: {status}")
    except Exception as e:
        logger.error(f"  ✗ Ошибка: {e}")
        return False
    
    # Вычитание BIAS из объектов
    logger.info("Вычитание BIAS из объектов...")
    try:
        status = list_subtractor(
            dirs['temp'] / 'obj_CRR_list.txt',
            s_bias_path,
            'Bias'
        )
        logger.info(f"  ✓ Объекты: {status}")
    except Exception as e:
        logger.error(f"  ✗ Ошибка: {e}")
        return False
    
    return True


def step_06_trace_orders(dirs, config, logger):
    """Шаг 6: Трассировка спектральных порядков"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 6: Трассировка спектральных порядков")
    logger.info("="*80)
    
    flat_file = str(dirs['data'] / 's_flat.fits')
    spec_list = str(dirs['temp'] / 'obj_CRR_bt_list.txt')  # bt = bias-trimmed
    output_dir = str(dirs['traced'])
    
    tracer_params = config['tracer_params']
    
    logger.info("Запуск трассировки...")
    logger.info(f"  - Флэт-файл: {flat_file}")
    logger.info(f"  - Список спектров: {spec_list}")
    logger.info(f"  - Количество порядков: {config['n_orders']}")
    
    try:
        trace_orders(
            flat_file=flat_file,
            spec_list=spec_list,
            output_dir=output_dir,
            n_orders=config['n_orders'],
            n_points_for_fit=tracer_params['n_points_for_fit'],
            smooth=tracer_params['smooth'],
            smooth_sigma=tracer_params['smooth_sigma'],
            getxwd_gauss=tracer_params['getxwd_gauss'],
            plot=False,
            save_plots=True,
            overwrite=True,
            save_format='json'
        )
        logger.info("✓ Трассировка завершена")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка при трассировке: {e}")
        return False


def step_07_wavelength_calibration(dirs, logger):
    """Шаг 7: Калибровка по длинам волн (интерактивный шаг)"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 7: Калибровка по длинам волн")
    logger.info("="*80)
    logger.info("")
    logger.info("ВНИМАНИЕ: Этот шаг требует интерактивной работы!")
    logger.info("")
    logger.info("Для выполнения калибровки по длинам волн:")
    logger.info("1. Запустите скрипт thar_calibration.py интерактивно")
    logger.info("2. Выберите опорный спектр ThAr")
    logger.info("3. Выполните калибровку и сохраните решение")
    logger.info("")
    logger.info("Команда:")
    logger.info("  python thar_calibration.py")
    logger.info("")
    logger.info("Калибровочный файл должен быть сохранен в формате:")
    logger.info("  <filename>.distortion_model.json")
    logger.info("")
    
    return True


def step_08_apply_calibration(dirs, logger):
    """Шаг 8: Применение калибровки (требует ручного запуска)"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 8: Применение калибровки к спектрам")
    logger.info("="*80)
    logger.info("")
    logger.info("После создания калибровочного решения, примените его:")
    logger.info("")
    logger.info("Для каждого научного спектра выполните:")
    logger.info("  python apply_calibration.py \\")
    logger.info("    <science_file>.fits \\")
    logger.info("    TRACED_ORDERS/<trace_file>.json \\")
    logger.info("    <calibration>.distortion_model.json \\")
    logger.info("    CALIBRATED/")
    logger.info("")
    
    return True


def step_09_combine_orders(dirs, logger):
    """Шаг 9: Объединение порядков (требует ручного запуска)"""
    logger.info("\n" + "="*80)
    logger.info("ШАГ 9: Объединение порядков в поляризационные компоненты")
    logger.info("="*80)
    logger.info("")
    logger.info("После калибровки всех спектров, объедините порядки:")
    logger.info("")
    logger.info("  python combine_orders.py \\")
    logger.info("    CALIBRATED/ \\")
    logger.info("    <science_basename> \\")
    logger.info("    COMBINED/")
    logger.info("")
    logger.info("Результат: два файла для ортогональных поляризаций")
    logger.info("  - <basename>_1.fits (порядки 1-7)")
    logger.info("  - <basename>_2.fits (порядки 8-14)")
    logger.info("")
    
    return True


def create_obj_bt_list(temp_dir):
    """Создает список obj_CRR_bt_list.txt из obj_CRR_list.txt"""
    obj_crr_list = temp_dir / 'obj_CRR_list.txt'
    obj_bt_list = temp_dir / 'obj_CRR_bt_list.txt'
    
    if not obj_crr_list.exists():
        return False
    
    with open(obj_crr_list, 'r') as f_in, open(obj_bt_list, 'w') as f_out:
        for line in f_in:
            filename = Path(line.strip())
            # Добавляем суффикс _bt (bias-trimmed) если его еще нет
            if '_bt' not in filename.stem:
                new_filename = filename.parent / f"{filename.stem}_bt{filename.suffix}"
                f_out.write(str(new_filename) + '\n')
            else:
                f_out.write(line)
    
    return True


# ==============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='pyZeeman - Пайплайн обработки спектров с ОЗСП',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Полная обработка с конфигурацией по умолчанию:
  python reduce4me_new.py /data/Observations/test_pyzeeman

  # С пользовательским конфигом:
  python reduce4me_new.py /data/Observations/test_pyzeeman --config my_config.json

  # Только определенные шаги:
  python reduce4me_new.py /data/Observations/test_pyzeeman --steps 1 2 3
        """
    )
    
    parser.add_argument(
        'data_path',
        type=str,
        help='Путь к директории с данными'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Путь к JSON файлу конфигурации'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=None,
        help='Номера шагов для выполнения (по умолчанию: все автоматические шаги)'
    )
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = DEFAULT_CONFIG.copy()
    config['data_path'] = args.data_path
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                user_config = json.load(f)
            config.update(user_config)
            print(f"Загружена конфигурация из {args.config}")
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            sys.exit(1)
    
    # Настройка директорий
    dirs = setup_directories(config['data_path'])
    
    # Настройка логирования
    log_path = dirs['data'] / 'pyzeeman_pipeline.log'
    logger = setup_logging(log_path)
    
    # Начало обработки
    start_time = datetime.now()
    logger.info("\n" + "="*80)
    logger.info("pyZeeman - Пайплайн обработки спектров с ОЗСП")
    logger.info("="*80)
    logger.info(f"Начало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Директория данных: {dirs['data']}")
    logger.info(f"Конфигурация: GAIN={config['gain']}, RON={config['ron']}")
    
    # Определение шагов для выполнения
    all_steps = {
        1: ("Создание списков кадров", step_01_create_lists),
        2: ("Обрезка области интереса", step_02_trim_images),
        3: ("Удаление космических частиц", step_03_remove_cosmic_rays),
        4: ("Создание мастер-калибровок", step_04_create_master_calibrations),
        5: ("Вычитание калибровок", step_05_subtract_calibrations),
        6: ("Трассировка порядков", step_06_trace_orders),
        7: ("Калибровка по длинам волн (интерактивно)", step_07_wavelength_calibration),
        8: ("Применение калибровки (вручную)", step_08_apply_calibration),
        9: ("Объединение порядков (вручную)", step_09_combine_orders),
    }
    
    # Автоматические шаги (не требуют интерактивности)
    auto_steps = [1, 2, 3, 4, 5, 6]
    steps_to_run = args.steps if args.steps else auto_steps
    
    # Выполнение шагов
    for step_num in steps_to_run:
        if step_num not in all_steps:
            logger.warning(f"Неизвестный шаг: {step_num}")
            continue
        
        step_name, step_func = all_steps[step_num]
        
        try:
            if step_num <= 1:
                success = step_func(dirs, logger)
            elif step_num <= 3:
                success = step_func(dirs, config, logger)
            elif step_num <= 5:
                success = step_func(dirs, logger)
            elif step_num == 6:
                # Создаем список obj_CRR_bt_list.txt перед трассировкой
                create_obj_bt_list(dirs['temp'])
                success = step_func(dirs, config, logger)
            else:
                success = step_func(dirs, logger)
            
            if not success:
                logger.error(f"Ошибка на шаге {step_num}: {step_name}")
                if step_num in auto_steps:
                    logger.error("Прерывание пайплайна")
                    sys.exit(1)
        
        except Exception as e:
            logger.error(f"Исключение на шаге {step_num} ({step_name}): {e}")
            if step_num in auto_steps:
                logger.error("Прерывание пайплайна")
                sys.exit(1)
    
    # Завершение
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*80)
    logger.info("ОБРАБОТКА ЗАВЕРШЕНА")
    logger.info("="*80)
    logger.info(f"Окончание: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Длительность: {duration}")
    logger.info(f"Лог сохранен: {log_path}")
    logger.info("")
    logger.info("Следующие шаги (выполняются вручную):")
    logger.info("  7. Калибровка по длинам волн (thar_calibration.py)")
    logger.info("  8. Применение калибровки (apply_calibration.py)")
    logger.info("  9. Объединение порядков (combine_orders.py)")
    logger.info("")


if __name__ == '__main__':
    main()
