#!/usr/bin/env python3
"""
extract_order_spectrum.py - Модуль извлечения спектральных порядков

Извлекает 1D спектры из 2D изображений с использованием трассировки,
суммируя поток между границами порядка (y_lower и y_upper).

Автор: pyZeeman pipeline
Дата: 2025-10-17
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trace_data(trace_file: Path) -> Optional[dict]:
    """Загружает данные трассировки из JSON"""
    try:
        with open(trace_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки {trace_file}: {e}")
        return None


def extract_order_summed(image_data: np.ndarray,
                        trace_data: dict,
                        order_num: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Извлекает 1D спектр порядка с суммированием потока между границами.
    Для граничных пикселей используются весовые коэффициенты
    
    Parameters:
    -----------
    image_data : np.ndarray
        2D массив изображения
    trace_data : dict
        Данные трассировки из JSON
    order_num : int
        Номер порядка для извлечения
    
    Returns:
    --------
    tuple : (x_coords, flux_summed) или (None, None)
    
    Notes:
    ------
    Суммирует поток между y_upper и y_lower для каждого столбца x.
    y_upper < y_lower (координата Y растет вниз на изображении).
    """
    # Найти нужный порядок
    order_info = None
    for order in trace_data.get('orders', []):
        if order['order_number'] == order_num:
            order_info = order
            break
    
    if order_info is None:
        logger.error(f"Порядок {order_num} не найден в трассировке")
        return None, None
    
    # Извлечь границы
    trace_full = order_info.get('trace_full', {})
    y_upper = np.array(trace_full.get('y_upper', []))
    y_lower = np.array(trace_full.get('y_lower', []))
    x_coords = np.array(trace_full.get('x', []))
    
    if len(y_upper) == 0 or len(y_lower) == 0:
        logger.error(f"Порядок {order_num}: отсутствуют границы")
        return None, None
    
    # Проверка длин
    if not (len(y_upper) == len(y_lower) == len(x_coords)):
        logger.error(f"Порядок {order_num}: несовпадающие длины массивов")
        return None, None
    
    logger.info(f"Извлечение порядка {order_num}:")
    logger.info(f"  Длина: {len(x_coords)} пикселей")
    logger.info(f"  y_upper: [{y_upper.min():.1f}, {y_upper.max():.1f}]")
    logger.info(f"  y_lower: [{y_lower.min():.1f}, {y_lower.max():.1f}]")
    logger.info(f"  Средняя ширина: {abs(y_lower - y_upper).mean():.1f} пикселей")
    
    # Извлечение с суммированием
    flux_summed = np.zeros(image_data.shape[1])
    
    for i, x in enumerate(x_coords):
        x_int = int(x)
        if not (0 <= x_int < image_data.shape[1]):
            continue

        y_start = int(np.floor(y_upper[i]))
        y_end = int(np.ceil(y_lower[i]))

        if y_start < 0 or y_end >= image_data.shape[0]:
            continue

        weight_top = 1.0 - (y_upper[i] - y_start)
        weight_bottom = y_lower[i] - (y_end - 1)
                      
        # Суммируем поток в столбце x между y_min и y_max
        column_flux = 0.0

        # for y in range(y_min, y_max + 1):
        #     if 0 <= y < image_data.shape[0]:
        #         column_flux += image_data[y, x_int]
        column_flux += image_data[y_start, x_int] * weight_top
        if y_end - y_start > 1:
            column_flux += np.sum(image_data[y_start+1:y_end, x_int])
        column_flux += image_data[y_end, x_int] * weight_bottom
        
        flux_summed[x_int] = column_flux
    
    logger.info(f"  Извлеченный поток: [{flux_summed.min():.1f}, {flux_summed.max():.1f}]")
    logger.info(f"  Медиана: {np.median(flux_summed):.1f}")
    
    return x_coords, flux_summed


def extract_all_orders(image_data: np.ndarray,
                      trace_data: dict) -> dict:
    """
    Извлекает все порядки из изображения
    
    Parameters:
    -----------
    image_data : np.ndarray
        2D изображение
    trace_data : dict
        Данные трассировки
    
    Returns:
    --------
    dict : {order_num: (x_coords, flux_summed)}
    """
    extracted_orders = {}
    
    for order_info in trace_data.get('orders', []):
        order_num = order_info['order_number']
        x_coords, flux = extract_order_summed(image_data, trace_data, order_num)
        
        if x_coords is not None:
            extracted_orders[order_num] = (x_coords, flux)
    
    logger.info(f"\nВсего извлечено порядков: {len(extracted_orders)}")
    return extracted_orders


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def main():
    """Тестовая функция"""
    import argparse
    from astropy.io import fits
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Извлечение спектральных порядков')
    parser.add_argument('image', type=str, help='FITS файл с изображением')
    parser.add_argument('trace', type=str, help='JSON файл с трассировкой')
    parser.add_argument('--order', type=int, help='Номер порядка (опционально)')
    parser.add_argument('--plot', action='store_true', help='Показать график')
    
    args = parser.parse_args()
    
    # Загрузка данных
    logger.info(f"Загрузка изображения: {args.image}")
    with fits.open(args.image) as hdul:
        image_data = hdul[0].data
    
    logger.info(f"Загрузка трассировки: {args.trace}")
    trace_data = load_trace_data(Path(args.trace))
    
    if trace_data is None:
        return
    
    # Извлечение
    if args.order:
        # Один порядок
        x, flux = extract_order_summed(image_data, trace_data, args.order)
        
        if x is not None and args.plot:
            plt.figure(figsize=(14, 6))
            plt.plot(x, flux, 'b-', linewidth=0.5)
            plt.xlabel('X (пиксели)')
            plt.ylabel('Суммарный поток (ADU)')
            plt.title(f'Порядок {args.order}')
            plt.grid(True, alpha=0.3)
            plt.show()
    else:
        # Все порядки
        extracted = extract_all_orders(image_data, trace_data)
        
        if args.plot and len(extracted) > 0:
            n_orders = len(extracted)
            ncols = 3
            nrows = (n_orders + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*3))
            axes = axes.flatten() if n_orders > 1 else [axes]
            
            for idx, (order_num, (x, flux)) in enumerate(sorted(extracted.items())):
                ax = axes[idx]
                ax.plot(x, flux, 'b-', linewidth=0.5)
                ax.set_title(f'Порядок {order_num}')
                ax.set_xlabel('X')
                ax.set_ylabel('Поток')
                ax.grid(True, alpha=0.3)
            
            # Скрыть неиспользуемые subplot'ы
            for idx in range(len(extracted), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()
