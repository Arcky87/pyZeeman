#!/usr/bin/env python3
"""
match_thar_to_science.py - Модуль сопоставления ThAr калибровок с научными кадрами

Этот модуль выполняет сопоставление калибровочных кадров ThAr с научными кадрами
на основе времени наблюдения. Учитывает, что научные кадры снимаются парами,
и каждая пара должна быть привязана к одному и тому же ThAr кадру.

Автор: pyZeeman pipeline
Дата: 2025-10-17
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_time_obs(time_str: str) -> timedelta:
    """
    Парсит TIME-OBS из FITS заголовка в timedelta
    
    Parameters:
    -----------
    time_str : str
        Время в формате 'hh:mm:ss.sss'
    
    Returns:
    --------
    timedelta : Время от начала суток
    """
    try:
        # Разбираем формат hh:mm:ss.sss
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        # Секунды могут содержать миллисекунды
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        microseconds = int(seconds_parts[1]) * 1000 if len(seconds_parts) > 1 else 0
        
        return timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    except Exception as e:
        logger.error(f"Ошибка парсинга времени '{time_str}': {e}")
        return timedelta(0)


def get_obs_time(fits_file: Path) -> Tuple[Optional[str], Optional[timedelta]]:
    """
    Извлекает время наблюдения из FITS файла
    
    Parameters:
    -----------
    fits_file : Path
        Путь к FITS файлу
    
    Returns:
    --------
    tuple : (DATE-OBS, TIME-OBS as timedelta) или (None, None) при ошибке
    """
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            date_obs = header.get('DATE-OBS', None)
            time_obs_str = header.get('TIME-OBS', None)
            
            if time_obs_str:
                time_obs = parse_time_obs(time_obs_str)
                return date_obs, time_obs
            else:
                logger.warning(f"TIME-OBS не найден в {fits_file}")
                return None, None
    except Exception as e:
        logger.error(f"Ошибка чтения {fits_file}: {e}")
        return None, None


def find_nearest_thar(obj_time: timedelta, 
                      thar_times: List[Tuple[Path, timedelta]], 
                      max_separation_hours: float = 3.0) -> Optional[Path]:
    """
    Находит ближайший ThAr кадр по времени
    
    Parameters:
    -----------
    obj_time : timedelta
        Время наблюдения объектного кадра
    thar_times : List[Tuple[Path, timedelta]]
        Список (путь_к_ThAr, время_ThAr)
    max_separation_hours : float
        Максимальное допустимое расстояние во времени (часы)
    
    Returns:
    --------
    Path or None : Путь к ближайшему ThAr или None
    """
    if not thar_times:
        return None
    
    max_separation = timedelta(hours=max_separation_hours)
    
    # Находим минимальное расстояние
    min_distance = None
    nearest_thar = None
    
    for thar_file, thar_time in thar_times:
        # Вычисляем разницу во времени
        distance = abs((obj_time - thar_time).total_seconds())
        
        if min_distance is None or distance < min_distance:
            min_distance = distance
            nearest_thar = thar_file
    
    # Проверяем, что расстояние не превышает максимального
    if min_distance and min_distance < max_separation.total_seconds():
        return nearest_thar
    else:
        logger.warning(f"Ближайший ThAr находится на расстоянии {min_distance/3600:.2f} часов (превышает лимит)")
        return None


def match_thar_to_science(obj_files: List[Path], 
                          thar_files: List[Path],
                          pair_size: int = 2,
                          max_separation_hours: float = 3.0,
                          write_to_header: bool = True,
                          thar_keyword: str = 'THAR_REF') -> Dict[Path, Path]:
    """
    Сопоставляет ThAr калибровки с научными кадрами
    
    Алгоритм:
    1. Загружает времена наблюдений из всех файлов
    2. Группирует объектные кадры парами (по умолчанию pair_size=2)
    3. Для каждой пары находит ближайший ThAr кадр
    4. Записывает имя ThAr файла в заголовок объектных кадров
    
    Parameters:
    -----------
    obj_files : List[Path]
        Список путей к объектным FITS файлам
    thar_files : List[Path]
        Список путей к ThAr FITS файлам
    pair_size : int
        Количество объектных кадров, использующих один ThAr (обычно 2)
    max_separation_hours : float
        Максимальное допустимое расстояние между obj и ThAr (часы)
    write_to_header : bool
        Записывать ли результат в FITS заголовок
    thar_keyword : str
        Ключевое слово для записи в FITS заголовок
    
    Returns:
    --------
    dict : {obj_file: thar_file} - словарь сопоставлений
    """
    
    logger.info(f"Сопоставление {len(obj_files)} объектных кадров с {len(thar_files)} ThAr кадрами")
    logger.info(f"Размер пары: {pair_size}, макс. разделение: {max_separation_hours} час(ов)")
    
    # Шаг 1: Загружаем времена наблюдений
    obj_times = []
    for obj_file in obj_files:
        date_obs, time_obs = get_obs_time(obj_file)
        if time_obs is not None:
            obj_times.append((obj_file, date_obs, time_obs))
        else:
            logger.warning(f"Пропуск {obj_file.name}: не удалось получить время")
    
    thar_times = []
    for thar_file in thar_files:
        date_obs, time_obs = get_obs_time(thar_file)
        if time_obs is not None:
            thar_times.append((thar_file, time_obs))
        else:
            logger.warning(f"Пропуск {thar_file.name}: не удалось получить время")
    
    # Шаг 2: Сортируем по времени
    obj_times.sort(key=lambda x: x[2])  # По time_obs
    thar_times.sort(key=lambda x: x[1])
    
    logger.info(f"Успешно загружено времен: {len(obj_times)} объектных, {len(thar_times)} ThAr")
    
    # Шаг 3: Сопоставление
    mapping = {}
    
    # Обрабатываем объектные кадры парами
    for i in range(0, len(obj_times), pair_size):
        pair = obj_times[i:i+pair_size]
        
        if len(pair) == 0:
            continue
        
        # Берем среднее время для пары (или время первого кадра)
        pair_time = pair[0][2]  # Время первого кадра в паре
        
        # Находим ближайший ThAr
        nearest_thar = find_nearest_thar(pair_time, thar_times, max_separation_hours)
        
        if nearest_thar is None:
            logger.warning(f"Не найден подходящий ThAr для пары начиная с {pair[0][0].name}")
            # Все равно записываем в mapping как None
            for obj_file, date_obs, time_obs in pair:
                mapping[obj_file] = None
        else:
            # Записываем одинаковый ThAr для всей пары
            logger.info(f"Пара {[p[0].name for p in pair]} → {nearest_thar.name}")
            for obj_file, date_obs, time_obs in pair:
                mapping[obj_file] = nearest_thar
    
    # Шаг 4: Запись в FITS заголовки
    if write_to_header:
        logger.info(f"Запись соответствий в FITS заголовки (ключ: {thar_keyword})")
        for obj_file, thar_file in mapping.items():
            try:
                with fits.open(obj_file, mode='update') as hdul:
                    if thar_file is not None:
                        # Записываем имя файла ThAr
                        hdul[0].header[thar_keyword] = thar_file.name
                        hdul[0].header.comments[thar_keyword] = 'Reference ThAr calibration file'
                    else:
                        hdul[0].header[thar_keyword] = 'NONE'
                        hdul[0].header.comments[thar_keyword] = 'No suitable ThAr found'
                    hdul.flush()
                logger.debug(f"Записано: {obj_file.name} → {thar_file.name if thar_file else 'NONE'}")
            except Exception as e:
                logger.error(f"Ошибка записи в {obj_file}: {e}")
    
    # Статистика
    successful = sum(1 for v in mapping.values() if v is not None)
    logger.info(f"\nРезультат сопоставления:")
    logger.info(f"  Успешно сопоставлено: {successful}/{len(mapping)}")
    logger.info(f"  Уникальных ThAr использовано: {len(set(v for v in mapping.values() if v))}")
    
    return mapping


def verify_mapping(mapping: Dict[Path, Path], verbose: bool = True) -> bool:
    """
    Проверяет корректность сопоставления
    
    Parameters:
    -----------
    mapping : dict
        Словарь {obj_file: thar_file}
    verbose : bool
        Выводить ли детальную информацию
    
    Returns:
    --------
    bool : True если все OK, False если есть проблемы
    """
    issues = []
    
    # Проверка 1: Все ли объектные файлы имеют ThAr
    none_count = sum(1 for v in mapping.values() if v is None)
    if none_count > 0:
        issues.append(f"{none_count} объектных кадров без ThAr")
    
    # Проверка 2: Используются ли все ThAr файлы
    unique_thars = set(v for v in mapping.values() if v is not None)
    
    if verbose:
        logger.info("\n=== Проверка сопоставления ===")
        logger.info(f"Всего сопоставлений: {len(mapping)}")
        logger.info(f"Уникальных ThAr: {len(unique_thars)}")
        
        # Статистика использования каждого ThAr
        thar_usage = {}
        for thar_file in mapping.values():
            if thar_file is not None:
                thar_usage[thar_file] = thar_usage.get(thar_file, 0) + 1
        
        logger.info("\nИспользование ThAr файлов:")
        for thar_file, count in sorted(thar_usage.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {thar_file.name}: {count} объектных кадров")
    
    if issues:
        logger.warning("\nОбнаружены проблемы:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("\n✓ Сопоставление корректно")
    return True


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

def main():
    """Тестовый запуск модуля"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Сопоставление ThAr с научными кадрами')
    parser.add_argument('obj_list', type=str, help='Путь к списку объектных файлов')
    parser.add_argument('thar_list', type=str, help='Путь к списку ThAr файлов')
    parser.add_argument('--pair-size', type=int, default=2, help='Размер пары объектных кадров')
    parser.add_argument('--max-hours', type=float, default=3.0, help='Макс. разделение в часах')
    parser.add_argument('--no-write', action='store_true', help='Не записывать в FITS заголовки')
    parser.add_argument('--keyword', type=str, default='THAR_REF', help='Ключевое слово FITS')
    
    args = parser.parse_args()
    
    # Загружаем списки файлов
    obj_list_path = Path(args.obj_list)
    thar_list_path = Path(args.thar_list)
    
    if not obj_list_path.exists():
        logger.error(f"Файл не найден: {obj_list_path}")
        return
    
    if not thar_list_path.exists():
        logger.error(f"Файл не найден: {thar_list_path}")
        return
    
    # Читаем пути из списков
    with open(obj_list_path, 'r') as f:
        obj_files = [Path(line.strip()) for line in f if line.strip()]
    
    with open(thar_list_path, 'r') as f:
        thar_files = [Path(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Загружено: {len(obj_files)} объектных, {len(thar_files)} ThAr")
    
    # Выполняем сопоставление
    mapping = match_thar_to_science(
        obj_files=obj_files,
        thar_files=thar_files,
        pair_size=args.pair_size,
        max_separation_hours=args.max_hours,
        write_to_header=not args.no_write,
        thar_keyword=args.keyword
    )
    
    # Проверяем результат
    verify_mapping(mapping, verbose=True)
    
    # Выводим детальную информацию
    print("\n" + "="*80)
    print("ДЕТАЛЬНОЕ СОПОСТАВЛЕНИЕ:")
    print("="*80)
    for obj_file, thar_file in mapping.items():
        thar_name = thar_file.name if thar_file else "NONE"
        print(f"{obj_file.name:40s} → {thar_name}")


if __name__ == '__main__':
    main()
