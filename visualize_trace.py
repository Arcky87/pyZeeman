import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from loaders import fits_loader, load_traced_orders, get_order_trace

def visualize_spectrum_with_orders(image_data, traced_data, header=None):
    fig, ax = plt.subplots(figsize=(15, 8))

    vmin, vmax = np.percentile(image_data, [1, 99])
    ax.imshow(image_data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    if 'orders' in traced_data:
        for order_info in traced_data['orders']:
            x = order_info['trace_full']['x']
            y_center = order_info['trace_full']['y_center']
            y_upper = order_info['trace_full']['y_upper']
            y_lower = order_info['trace_full']['y_lower']

            ax.plot(x, y_center, '-', color='cyan', lw=1, alpha=0.8)
            ax.plot(x, y_upper, '--', color='red', lw=0.8, alpha=0.7)
            ax.plot(x, y_lower, '--', color='red', lw=0.8, alpha=0.7)
            
            text_index = len(x) // 10
            ax.text(x[text_index], y_center[text_index], str(order_info['order_number']), 
                    color='yellow', fontsize=8, ha='center', va='center')
            
    ax.set_xlabel("Ось X (пиксели)")
    ax.set_ylabel("Ось Y (пиксели)")
    plt.tight_layout()
   # plt.show()
    return fig, ax

def extract_order(image_data, traced_data, order_number):
    img_height, img_width = image_data.shape
    trace = get_order_trace(traced_data, order_number)

    x_coords = trace['x']
    y_upper_trace = trace['y_upper']
    y_lower_trace = trace['y_lower']
    #y_center_trace = trace['y_center']

    extracted_flux = np.zeros(img_width)

    for i in range(len(x_coords)):
        x_col = int(x_coords[i])

        if not (0 <= x_col < img_width):
            continue

        y_start = int(np.floor(y_upper_trace[i]))
        y_end = int(np.ceil(y_lower_trace[i]))
        #y_center = int(np.rint(y_center_trace[i]))

        weight_top = 1.0 - (y_upper_trace[i] - y_start)
        weight_bottom = y_lower_trace[i] - (y_end - 1)
        column_flux = 0.0

        column_flux += image_data[y_start, x_col] * weight_top

        if y_start <= y_end:
            column_flux += np.sum(image_data[y_start+1:y_end, x_col])
        
        column_flux += image_data[y_end, x_col] * weight_bottom
        extracted_flux[x_col] = column_flux
        
       # extracted_flux[x_col] = image_data[y_center,x_col]

    return np.arange(img_width), extracted_flux

def plot_extracted_spectrum(x_coords, flux, order_number,ax=None):
    if ax is None:
        # Если нет, создаем новую фигуру и оси
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
        ax.cla()

    ax.plot(x_coords, flux)
    ax.set_title(f"Order number {order_number}")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Flux, ADU")
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig, ax

def extract_all_orders(image_data, traced_data):
    """
    Экстрагирует все порядки из изображения, используя данные трассировки.

    Args:
        image_data (np.ndarray): 2D массив FITS-изображения.
        traced_data (dict): Словарь с данными трассировки порядков.

    Returns:
        list: Список, где каждый элемент - это 1D np.ndarray с экстрагированным
              потоком для соответствующего порядка. Порядки идут в той же
              последовательности, что и в traced_data.
    """
    all_fluxes = []
    
    # Получаем список словарей, описывающих каждый порядок
    orders_to_extract = traced_data.get('orders', [])
   
    # Итерируемся по каждому порядку
    for order_info in orders_to_extract:
        order_num = order_info['order_number']        
        x, flux_array = extract_order(image_data, traced_data, order_num)
        all_fluxes.append(flux_array)
    print("\nЭкстракция всех порядков завершена.")
    return x, all_fluxes


if __name__ == '__main__':

    FITS_FILE_PATH = Path('/data/Observations/test_pyzeeman/o018.fts')
    TRACED_ORDERS_JSON_PATH = Path('/data/Observations/test_pyzeeman/traced_orders/o015_CRR_bt_traced.json')
    
    image_data, header, _ = fits_loader(FITS_FILE_PATH)
    traced_data = load_traced_orders(TRACED_ORDERS_JSON_PATH)
    
    plt.ion()
    visualize_spectrum_with_orders(image_data, traced_data, header)

    fig1d, ax1d = None, None

    while True:
        prompt = "\nВведите номер порядка для извлечения (или нажмите Enter для выхода): "
        user_input = input(prompt)
        if not user_input:
            break

        try:
            order_to_extract = int(user_input)
            x, flux = extract_order(image_data, traced_data, order_to_extract)
            fig1d, ax1d = plot_extracted_spectrum(x, flux, order_to_extract, ax=ax1d)
            if fig1d.canvas.manager.get_window_title() == '':
                fig1d.canvas.manager.set_window_title('Профиль выделенного порядка')
        
            fig1d.canvas.draw()
            fig1d.canvas.flush_events()
        except ValueError:
            print("Ошибка: Введите целое число.")
        except Exception as e:
            print(f"Не удалось извлечь порядок {user_input}: {e}")
    plt.ioff()  
    plt.show()

