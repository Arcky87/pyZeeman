import json
from pathlib import Path
import numpy as np

def load_traced_orders(json_file, order_numbers=None):
    """
    Загрузка данных трассировки спектральных порядков из JSON файла
    
    Parameters:
    -----------
    json_file : str or Path
        Путь к JSON файлу с результатами трассировки
    order_numbers : list, int, slice, or None
        Номера порядков для загрузки:
        - None: загрузить все порядки
        - int: загрузить один порядок по номеру
        - list: загрузить порядки по списку номеров [1, 3, 5]
        - slice: загрузить диапазон порядков slice(1, 10) (по номерам)
    
    Returns:
    --------
    dict : Словарь с данными трассировки:
        {
            'metadata': {
                'source_file': str,
                'processing_time': float,
                'n_orders_traced': int,
                'n_orders_loaded': int,  # количество загруженных порядков
                'parameters': dict
            },
            'orders': list,  # список выбранных порядков
            'order_numbers': list,  # номера загруженных порядков
            'available_orders': list  # все доступные номера порядков
        }
    
    Raises:
    -------
    FileNotFoundError: если файл не найден
    ValueError: если указаны неверные номера порядков
    """
    
    json_file = Path(json_file)
    
    # Проверка существования файла
    if not json_file.exists():
        raise FileNotFoundError(f"Файл {json_file} не найден")
    
    # Загрузка данных из JSON
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON файла: {e}")
    
    # Проверка структуры данных
    if 'metadata' not in data or 'orders' not in data:
        raise ValueError("Неверная структура JSON файла: отсутствуют 'metadata' или 'orders'")
    
    metadata = data['metadata']
    all_orders = data['orders']
    
    # Получение списка доступных номеров порядков
    available_orders = [order['order_number'] for order in all_orders]
    available_orders.sort()
    
    # Обработка выбора порядков
    if order_numbers is None:
        # Все порядки
        selected_orders = all_orders
        selected_order_numbers = available_orders
    else:
        # Нормализация номеров порядков
        if isinstance(order_numbers, int):
            order_numbers = [order_numbers]
        elif isinstance(order_numbers, slice):
            # Для slice используем доступные номера порядков
            start = order_numbers.start if order_numbers.start is not None else min(available_orders)
            stop = order_numbers.stop if order_numbers.stop is not None else max(available_orders) + 1
            step = order_numbers.step if order_numbers.step is not None else 1
            order_numbers = list(range(start, stop, step))
        elif not isinstance(order_numbers, (list, tuple)):
            raise ValueError("order_numbers должен быть int, list, slice или None")
        
        order_numbers = list(order_numbers)
        
        # Проверка корректности номеров порядков
        invalid_orders = [num for num in order_numbers if num not in available_orders]
        if invalid_orders:
            raise ValueError(f"Неверные номера порядков: {invalid_orders}. "
                           f"Доступные номера: {available_orders}")
        
        # Фильтрация порядков
        selected_orders = [order for order in all_orders 
                          if order['order_number'] in order_numbers]
        selected_order_numbers = sorted(order_numbers)
    
    # Преобразование массивов в numpy arrays для удобства
    processed_orders = []
    for order in selected_orders:
        processed_order = order.copy()
        
        # Преобразование trace_points
        if 'trace_points' in processed_order:
            for key in ['x_measured', 'y_measured', 'widths_measured']:
                if key in processed_order['trace_points']:
                    processed_order['trace_points'][key] = np.array(
                        processed_order['trace_points'][key]
                    )
        
        # Преобразование trace_full
        if 'trace_full' in processed_order:
            for key in ['x', 'y_center', 'y_upper', 'y_lower']:
                if key in processed_order['trace_full']:
                    processed_order['trace_full'][key] = np.array(
                        processed_order['trace_full'][key]
                    )
        
        # Преобразование коэффициентов Чебышева
        if 'chebyshev_coeffs' in processed_order:
            processed_order['chebyshev_coeffs'] = np.array(
                processed_order['chebyshev_coeffs']
            )
        
        processed_orders.append(processed_order)
    
    # Формирование результата
    result = {
        'metadata': {
            **metadata,
            'n_orders_loaded': len(selected_orders),
            'original_file': str(json_file)
        },
        'orders': processed_orders,
        'order_numbers': selected_order_numbers,
        'available_orders': available_orders
    }
    
    print(f"Загружено {len(selected_orders)} из {len(all_orders)} порядков из {json_file}")
    if order_numbers is not None:
        print(f"Выбранные порядки: {selected_order_numbers}")
    
    return result


def get_trace_info(json_file):
    """
    Получение краткой информации о файле трассировки
    
    Parameters:
    -----------
    json_file : str or Path
        Путь к JSON файлу с результатами трассировки
    
    Returns:
    --------
    dict : Краткая информация о файле
    """
    json_file = Path(json_file)
    
    if not json_file.exists():
        raise FileNotFoundError(f"Файл {json_file} не найден")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    available_orders = [order['order_number'] for order in data['orders']]
    
    return {
        'source_file': metadata.get('source_file'),
        'n_orders_traced': metadata.get('n_orders_traced'),
        'processing_time': metadata.get('processing_time'),
        'parameters': metadata.get('parameters', {}),
        'available_orders': sorted(available_orders),
        'order_range': f"{min(available_orders)}-{max(available_orders)}",
        'file_path': str(json_file)
    }


def extract_order_data(traced_data, order_number, data_type='full'):
    """
    Извлечение данных конкретного порядка
    
    Parameters:
    -----------
    traced_data : dict
        Результат функции load_traced_orders()
    order_number : int
        Номер порядка
    data_type : str
        Тип данных: 'full', 'trace_points', 'trace_full', 'coeffs'
    
    Returns:
    --------
    dict or np.array : Данные указанного типа для порядка
    """
    # Поиск порядка
    order_data = None
    for order in traced_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        raise ValueError(f"Порядок {order_number} не найден в загруженных данных")
    
    if data_type == 'full':
        return order_data
    elif data_type == 'trace_points':
        return order_data.get('trace_points', {})
    elif data_type == 'trace_full':
        return order_data.get('trace_full', {})
    elif data_type == 'coeffs':
        return order_data.get('chebyshev_coeffs', np.array([]))
    else:
        raise ValueError("data_type должен быть 'full', 'trace_points', 'trace_full' или 'coeffs'")
    
if __name__ == '__main__':
    all_traced = load_traced_orders("/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json")
    print(f"Загружено {all_traced['metadata']['n_orders_loaded']} порядков")
    print(f"Доступные порядки: {all_traced['available_orders']}")
    # 2. Загрузить один порядок
    single_order = load_traced_orders("/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json", order_numbers=5)

# 3. Загрузить несколько порядков по номерам
    selected_orders = load_traced_orders("/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json", 
                                   order_numbers=[1, 3, 5, 7])

# 4. Загрузить диапазон порядков (с 2 по 10)
    range_orders = load_traced_orders("/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json", 
                                order_numbers=slice(2, 11))

# 5. Загрузить каждый второй порядок из диапазона
    every_second = load_traced_orders("/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json", 
                                order_numbers=slice(1, 15, 2))

# 6. Получить информацию о файле
    info = get_trace_info("/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json")
    print(f"Исходный файл: {info['source_file']}")
    print(f"Доступные порядки: {info['order_range']}")

# 7. Извлечь данные конкретного порядка
    order_5_full = extract_order_data(all_traced, 5, 'full')
    order_5_trace = extract_order_data(all_traced, 5, 'trace_full')

# 8. Работа с данными трассировки
    for order in all_traced['orders']:
        order_num = order['order_number']
        x_coords = order['trace_full']['x']
        y_upper = order['trace_full']['y_upper']
        y_lower = order['trace_full']['y_lower']
    
        print(f"Порядок {order_num}: {len(x_coords)} точек трассировки")