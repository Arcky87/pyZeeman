from pathlib import Path
import numpy as np
from astropy.io import fits
import json
from numpy.polynomial.chebyshev import chebval

def list_loader(file_list_path, check_exist=True):
    """
    Загружает список файлов из текстового файла
    
    Parameters:
        file_list_path: str - путь к файлу со списком
        check_exist: bool - проверять ли существование файлов
        
    Returns:
        list - список путей к файлам
    """
    file_list_path = Path(file_list_path)
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list {file_list_path} not found")
    
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    if check_exist:
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing[:3]}... ({len(missing)} total)")
    
    return files

def fits_loader(file_path, hdu_index=0, dtype=np.float32):
    """
    Загружает данные из FITS-файла
    
    Parameters:
        file_path: str - путь к FITS-файлу
        hdu_index: int - индекс HDU с данными
        dtype: type - тип данных для преобразования
        
    Returns:
        tuple: (data, header, file_path)
    """
    file_path = Path(file_path)
    with fits.open(file_path) as hdul:
        data = hdul[hdu_index].data.astype(dtype)
        header = hdul[hdu_index].header
    
    return data, header, str(file_path)

def text_loader(file_path, delimiter=None, skip_rows=0, dtype=np.float32):
    """
    Загружает данные из текстового файла
    
    Parameters:
        file_path: str - путь к текстовому файлу
        delimiter: str - разделитель столбцов (None для автоопределения)
        skip_rows: int - количество пропускаемых строк в начале файла
        dtype: type - тип данных для преобразования
        
    Returns:
        tuple: (data, None, file_path)  # header=None для совместимости
    """
    file_path = Path(file_path)
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_rows, dtype=dtype)
    return data, str(file_path)


def load_traced_orders(json_file):
    """
    Загружает результаты трассировки спектральных порядков из JSON файла
    
    Parameters:
    -----------
    json_file : str or Path
        Путь к JSON файлу с результатами трассировки
    
    Returns:
    --------
    dict : Словарь с метаданными и данными трассировки
        {
            'metadata': dict,
            'orders': list of dict,
            'n_orders': int
        }
    """
    json_path = Path(json_file)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Файл трассировки не найден: {json_file}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON файла {json_file}: {e}")
    
    # Проверяем структуру данных
    if 'metadata' not in data or 'orders' not in data:
        raise ValueError(f"Неправильная структура файла трассировки: {json_file}")
    
    # Добавляем удобные поля
    data['n_orders'] = len(data['orders'])
    
    return data

def get_order_trace(traced_data, order_number, x_range=None):
    """
    Извлекает трассировку для конкретного порядка
    
    Parameters:
    -----------
    traced_data : dict
        Данные трассировки от load_traced_orders()
    order_number : int
        Номер порядка для извлечения
    x_range : tuple, optional
        (x_start, x_end) - диапазон X для ограничения трассы
    
    Returns:
    --------
    dict : Данные трассировки порядка
        {
            'order_number': int,
            'x': np.array,
            'y_center': np.array,
            'y_upper': np.array,
            'y_lower': np.array,
            'width': float,
            'coeffs': np.array
        }
    """
    # Ищем нужный порядок
    order_data = None
    for order in traced_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        available_orders = [o['order_number'] for o in traced_data['orders']]
        raise ValueError(f"Порядок {order_number} не найден. "
                        f"Доступные порядки: {available_orders}")
    
    # Извлекаем координаты
    x = np.array(order_data['trace_full']['x'])
    y_center = np.array(order_data['trace_full']['y_center'])
    y_upper = np.array(order_data['trace_full']['y_upper'])
    y_lower = np.array(order_data['trace_full']['y_lower'])
    
    # Ограничиваем диапазон если нужно
    if x_range is not None:
        x_start, x_end = x_range
        mask = (x >= x_start) & (x <= x_end)
        x = x[mask]
        y_center = y_center[mask]
        y_upper = y_upper[mask]
        y_lower = y_lower[mask]
    
    return {
        'order_number': order_data['order_number'],
        'x': x,
        'y_center': y_center,
        'y_upper': y_upper,
        'y_lower': y_lower,
        'width': order_data['median_width'],
        'coeffs': np.array(order_data['chebyshev_coeffs'])
    }

def get_all_order_traces(traced_data, x_range=None):
    """
    Извлекает трассировки для всех порядков
    
    Parameters:
    -----------
    traced_data : dict
        Данные трассировки от load_traced_orders()
    x_range : tuple, optional
        (x_start, x_end) - диапазон X для ограничения трасс
    
    Returns:
    --------
    dict : Словарь с трассировками всех порядков
        {order_number: trace_data, ...}
    """
    all_traces = {}
    
    for order in traced_data['orders']:
        order_num = order['order_number']
        all_traces[order_num] = get_order_trace(traced_data, order_num, x_range)
    
    return all_traces

def evaluate_trace_at_x(traced_data, order_number, x_positions):
    """
    Вычисляет положение трассы порядка в заданных X координатах
    используя коэффициенты Чебышева
    
    Parameters:
    -----------
    traced_data : dict
        Данные трассировки от load_traced_orders()
    order_number : int
        Номер порядка
    x_positions : array-like
        X координаты для вычисления
    
    Returns:
    --------
    dict : Вычисленные координаты
        {
            'x': np.array,
            'y_center': np.array,
            'y_upper': np.array,
            'y_lower': np.array
        }
    """
    # Получаем данные порядка
    order_data = None
    for order in traced_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        raise ValueError(f"Порядок {order_number} не найден")
    
    x_positions = np.asarray(x_positions)
    coeffs = np.array(order_data['chebyshev_coeffs'])
    width = order_data['median_width']
    
    # Вычисляем центральную линию
    y_center = chebval(x_positions, coeffs)
    
    # Вычисляем границы
    y_upper = y_center - width / 2
    y_lower = y_center + width / 2
    
    return {
        'x': x_positions,
        'y_center': y_center,
        'y_upper': y_upper,
        'y_lower': y_lower
    }

def create_order_mask(traced_data, image_shape, order_numbers=None):
    """
    Создает маску порядков для изображения
    
    Parameters:
    -----------
    traced_data : dict
        Данные трассировки от load_traced_orders()
    image_shape : tuple
        (height, width) изображения
    order_numbers : list, optional
        Список номеров порядков для включения в маску.
        Если None, включаются все порядки.
    
    Returns:
    --------
    np.array : Маска размером image_shape, где каждый пиксель
               содержит номер порядка (0 = фон)
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.int16)
    
    # Определяем какие порядки включать
    if order_numbers is None:
        orders_to_process = traced_data['orders']
    else:
        orders_to_process = [o for o in traced_data['orders'] 
                           if o['order_number'] in order_numbers]
    
    for order_data in orders_to_process:
        order_num = order_data['order_number']
        
        # Получаем координаты трассы
        x_coords = np.array(order_data['trace_full']['x'])
        y_upper = np.array(order_data['trace_full']['y_upper'])
        y_lower = np.array(order_data['trace_full']['y_lower'])
        
        # Заполняем маску
        for i, x in enumerate(x_coords):
            if 0 <= x < width:
                y_start = max(0, int(np.floor(y_upper[i])))
                y_end = min(height, int(np.ceil(y_lower[i])))
                mask[y_start:y_end, x] = order_num
    
    return mask

def get_trace_summary(traced_data):
    """
    Возвращает сводную информацию о трассировке
    
    Parameters:
    -----------
    traced_data : dict
        Данные трассировки от load_traced_orders()
    
    Returns:
    --------
    dict : Сводная информация
    """
    orders = traced_data['orders']
    
    if not orders:
        return {'n_orders': 0}
    
    order_numbers = [o['order_number'] for o in orders]
    widths = [o['median_width'] for o in orders]
    
    # Получаем диапазон Y координат
    all_y_centers = []
    for order in orders:
        all_y_centers.extend(order['trace_full']['y_center'])
    
    summary = {
        'source_file': traced_data['metadata'].get('source_file', 'unknown'),
        'n_orders': len(orders),
        'order_numbers': order_numbers,
        'width_stats': {
            'min': float(np.min(widths)),
            'max': float(np.max(widths)),
            'mean': float(np.mean(widths)),
            'median': float(np.median(widths))
        },
        'y_range': {
            'min': float(np.min(all_y_centers)),
            'max': float(np.max(all_y_centers))
        },
        'processing_time': traced_data['metadata'].get('processing_time', 0)
    }
    
    return summary

def load_order_boundaries(boundaries_file):
    """
    Загружает границы порядков из JSON файла
    
    Parameters:
    -----------
    boundaries_file : str or Path
        Путь к файлу с границами порядков
    
    Returns:
    --------
    dict : Данные границ порядков
    """
    boundaries_path = Path(boundaries_file)
    
    if not boundaries_path.exists():
        raise FileNotFoundError(f"Файл границ не найден: {boundaries_file}")
    
    with open(boundaries_path, 'r') as f:
        boundaries_data = json.load(f)
    
    # Преобразуем обратно в numpy массивы
    if 'metadata' in boundaries_data:
        metadata = boundaries_data['metadata']
        metadata['peaks'] = np.array(metadata['peaks'])
        metadata['boundaries'] = np.array(metadata['boundaries'])
    
    return boundaries_data

# Пример использования:
if __name__ == '__main__':
    # Загрузка данных трассировки
    traced = load_traced_orders('traced_orders/o012_CRR_bt_traced.json')
    
    # Сводка
    summary = get_trace_summary(traced)
    print(f"Обработано порядков: {summary['n_orders']}")
    print(f"Средняя ширина: {summary['width_stats']['mean']:.2f}")
    
    # Получение конкретного порядка
    order_5 = get_order_trace(traced, order_number=5)
    print(f"Порядок 5: ширина = {order_5['width']:.2f}")
    
    # Создание маски
    mask = create_order_mask(traced, (2048, 4096))
    print(f"Создана маска размером {mask.shape}")
    
    # Вычисление трассы в новых точках
    x_new = np.linspace(100, 4000, 100)
    trace_new = evaluate_trace_at_x(traced, order_number=5, x_positions=x_new)
    print(f"Вычислена трасса для {len(x_new)} точек")
