import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from astropy.io import fits

def _load_image_data(image_data):
    """Универсальная загрузка изображения"""
    if isinstance(image_data, (str, Path)):
        with fits.open(image_data) as hdul:
            image = hdul[0].data.astype(float)
            header = hdul[0].header
        return image, header
    else:
        return image_data.astype(float), None


def diagnose_trace_quality(image_data, trace_data, order_number, 
                          x_positions=None, show_profiles=True, 
                          show_residuals=True):
    """
    Диагностика качества трассировки порядка
    
    Parameters:
    -----------
    image_data : np.array
        2D изображение
    trace_data : dict
        Данные трассировки
    order_number : int
        Номер порядка
    x_positions : list or None
        Конкретные X позиции для детального анализа
    show_profiles : bool
        Показать профили в разных позициях
    show_residuals : bool
        Показать остатки фитирования
    """
    
    
    # Загрузка изображения (путь или массив)
    image, header = _load_image_data(image_data)
    
    # Получение данных порядка
    order_data = None
    for order in trace_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        raise ValueError(f"Порядок {order_number} не найден")
    
    x_trace = order_data['trace_full']['x']
    y_center = order_data['trace_full']['y_center']
    y_upper = order_data['trace_full']['y_upper']
    y_lower = order_data['trace_full']['y_lower']
    
    # Если позиции не заданы, выберем равномерно распределенные
    if x_positions is None:
        n_positions = 5
        x_positions = np.linspace(np.min(x_trace), np.max(x_trace), n_positions).astype(int)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Общий вид трассировки
    ax1 = plt.subplot(3, 3, (1, 3))
    
    # Показать область вокруг порядка
    x_min, x_max = int(np.min(x_trace)), int(np.max(x_trace))
    y_min = int(np.min(y_lower)) - 20
    y_max = int(np.max(y_upper)) + 20
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)
    
    im_section = image[y_min:y_max, x_min:x_max]
    im = ax1.imshow(im_section, aspect='auto', origin='lower', 
                    extent=[x_min, x_max, y_min, y_max],
                    cmap='viridis', vmin=np.percentile(im_section, 5),
                    vmax=np.percentile(im_section, 95))
    
    ax1.plot(x_trace, y_center, 'r-', linewidth=2, label='Центр трассировки')
    ax1.plot(x_trace, y_upper, 'lime', linewidth=1.5, label='Верхняя граница')
    ax1.plot(x_trace, y_lower, 'lime', linewidth=1.5, label='Нижняя граница')
    
    # Отметить выбранные позиции
    for i, x_pos in enumerate(x_positions):
        ax1.axvline(x_pos, color='orange', linestyle='--', alpha=0.7)
        ax1.text(x_pos, y_max-5, f'{i+1}', color='orange', ha='center', 
                fontweight='bold', fontsize=12)
    
    ax1.set_title(f'Трассировка порядка {order_number}')
    ax1.set_xlabel('X (пиксели)')
    ax1.set_ylabel('Y (пиксели)')
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Интенсивность')
    
    # 2-6. Профили в выбранных позициях
    if show_profiles:
        for i, x_pos in enumerate(x_positions[:5]):
            ax = plt.subplot(3, 3, 4 + i)
            
            if x_pos < 0 or x_pos >= image.shape[1]:
                continue
            
            # Интерполяция для точной позиции
            y_center_interp = np.interp(x_pos, x_trace, y_center)
            y_upper_interp = np.interp(x_pos, x_trace, y_upper)
            y_lower_interp = np.interp(x_pos, x_trace, y_lower)
            
            # Извлечение профиля
            profile_range = 30  # пикселей вокруг центра
            y_start = max(0, int(y_center_interp - profile_range))
            y_end = min(image.shape[0], int(y_center_interp + profile_range))
            
            profile = image[y_start:y_end, x_pos]
            y_coords = np.arange(y_start, y_end)
            
            ax.plot(y_coords, profile, 'k-', linewidth=1, label='Профиль')
            
            # Отметить границы трассировки
            ax.axvline(y_center_interp, color='r', linewidth=2, label='Центр')
            ax.axvline(y_upper_interp, color='lime', linewidth=1.5, label='Границы')
            ax.axvline(y_lower_interp, color='lime', linewidth=1.5)
            
            # Заштриховать область экстракции
            ax.axvspan(y_lower_interp, y_upper_interp, alpha=0.2, color='green')
            
            # Попытка фитирования гауссианом для сравнения
            try:
                from scipy.optimize import curve_fit
                
                def gaussian(x, a, mu, sigma, bg):
                    return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg
                
                # Начальные приближения
                p0 = [np.max(profile), y_center_interp, 
                      order_data.get('median_width', 10) / 2.35, np.min(profile)]
                
                popt, _ = curve_fit(gaussian, y_coords, profile, p0=p0)
                fitted_profile = gaussian(y_coords, *popt)
                ax.plot(y_coords, fitted_profile, 'r--', alpha=0.7, label='Гауссов фит')
                
                # Показать FWHM
                fwhm = 2.35 * abs(popt[2])
                ax.text(0.05, 0.95, f'FWHM: {fwhm:.1f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except:
                pass
            
            ax.set_title(f'X = {x_pos}')
            ax.set_xlabel('Y (пиксели)')
            ax.set_ylabel('Интенсивность')
            if i == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # 7. Анализ ширины порядка
    ax7 = plt.subplot(3, 3, 7)
    widths = y_upper - y_lower
    ax7.plot(x_trace, widths, 'b-', linewidth=1, label='Ширина порядка')
    ax7.axhline(order_data.get('median_width', np.median(widths)), 
                color='r', linestyle='--', label=f'Медианная ширина')
    ax7.set_title('Изменение ширины порядка')
    ax7.set_xlabel('X (пиксели)')
    ax7.set_ylabel('Ширина (пиксели)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Остатки Чебышевского фитирования
    if show_residuals and 'chebyshev_coeffs' in order_data:
        ax8 = plt.subplot(3, 3, 8)
        
        coeffs = order_data['chebyshev_coeffs']
        if len(coeffs) > 0:
            # Восстановление фитированной кривой
            x_norm = 2 * (x_trace - np.min(x_trace)) / (np.max(x_trace) - np.min(x_trace)) - 1
            y_fitted = np.polynomial.chebyshev.chebval(x_norm, coeffs)
            
            residuals = y_center - y_fitted
            ax8.plot(x_trace, residuals, 'ro', markersize=2, label='Остатки')
            ax8.axhline(0, color='k', linestyle='-', alpha=0.5)
            ax8.axhline(np.std(residuals), color='r', linestyle='--', alpha=0.5)
            ax8.axhline(-np.std(residuals), color='r', linestyle='--', alpha=0.5)
            
            ax8.set_title(f'Остатки фитирования (σ = {np.std(residuals):.3f})')
            ax8.set_xlabel('X (пиксели)')
            ax8.set_ylabel('Остатки (пиксели)')
            ax8.grid(True, alpha=0.3)
            
            # Статистика
            rms = np.sqrt(np.mean(residuals**2))
            ax8.text(0.05, 0.95, f'RMS: {rms:.3f} пикс', transform=ax8.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 9. Статистика трассировки
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""Статистика порядка {order_number}:
    
Количество точек: {len(x_trace)}
X диапазон: {np.min(x_trace):.0f} - {np.max(x_trace):.0f}
Y диапазон: {np.min(y_center):.1f} - {np.max(y_center):.1f}

Ширина порядка:
  Медианная: {np.median(widths):.1f} пикс
  Мин/Макс: {np.min(widths):.1f} / {np.max(widths):.1f}
  Стд. откл.: {np.std(widths):.2f}

Наклон порядка:
  Средний: {np.mean(np.gradient(y_center)):.4f} пикс/пикс
  Кривизна: {np.mean(np.gradient(np.gradient(y_center))):.6f}
"""
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'x_positions_analyzed': x_positions,
        'order_widths': widths,
        'median_width': np.median(widths),
        'width_variation': np.std(widths),
        'slope': np.mean(np.gradient(y_center)),
        'curvature': np.mean(np.gradient(np.gradient(y_center)))
    }


def diagnose_extraction_step_by_step(image_data, trace_data, order_number, 
                                   x_position, extraction_method='sum',
                                   aperture_width=None, background_regions=None):
    """
    Пошаговая диагностика экстракции в конкретной X позиции
    """
    
       # Загрузка изображения (путь или массив)
    image, header = _load_image_data(image_data)
 
    # Получение данных порядка
    order_data = None
    for order in trace_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        raise ValueError(f"Порядок {order_number} не найден")
    
    x_trace = order_data['trace_full']['x']
    y_center = order_data['trace_full']['y_center']
    y_upper = order_data['trace_full']['y_upper']
    y_lower = order_data['trace_full']['y_lower']
    
    # Интерполяция для заданной позиции
    y_center_interp = np.interp(x_position, x_trace, y_center)
    y_upper_interp = np.interp(x_position, x_trace, y_upper)
    y_lower_interp = np.interp(x_position, x_trace, y_lower)
    
    # Применение ширины апертуры
    if aperture_width is not None:
        half_width = aperture_width / 2
        y_upper_interp = y_center_interp + half_width
        y_lower_interp = y_center_interp - half_width
    
    # Определение границ экстракции
    y_low = int(np.floor(y_lower_interp))
    y_high = int(np.ceil(y_upper_interp))
    y_low = max(0, y_low)
    y_high = min(image.shape[0], y_high)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Общий вид области
    ax = axes[0, 0]
    context = 50
    y_start = max(0, y_low - context)
    y_end = min(image.shape[0], y_high + context)
    x_start = max(0, x_position - context)
    x_end = min(image.shape[1], x_position + context)
    
    im_section = image[y_start:y_end, x_start:x_end]
    im = ax.imshow(im_section, aspect='auto', origin='lower',
                   extent=[x_start, x_end, y_start, y_end], cmap='viridis')
    
    # Показать область экстракции
    rect = patches.Rectangle((x_position-0.5, y_low), 1, y_high-y_low,
                           linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    ax.axhline(y_center_interp, color='yellow', linewidth=2, label='Центр')
    ax.axhline(y_upper_interp, color='lime', linewidth=1.5, label='Границы')
    ax.axhline(y_lower_interp, color='lime', linewidth=1.5)
    
    # Показать области фона
    if background_regions:
        for bg_start, bg_end in background_regions:
            ax.axhspan(bg_start, bg_end, alpha=0.3, color='orange', label='Фон')
    
    ax.set_title(f'Область экстракции в X={x_position}')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')
    ax.legend()
    plt.colorbar(im, ax=ax)
    
    # 2. Профиль в данной позиции
    ax = axes[0, 1]
    
    profile_range = 60
    y_prof_start = max(0, int(y_center_interp - profile_range))
    y_prof_end = min(image.shape[0], int(y_center_interp + profile_range))
    
    column = image[y_prof_start:y_prof_end, x_position]
    y_coords = np.arange(y_prof_start, y_prof_end)
    
    ax.plot(y_coords, column, 'k-', linewidth=1, label='Профиль')
    
    # Выделить область экстракции
    extraction_mask = (y_coords >= y_low) & (y_coords < y_high)
    ax.fill_between(y_coords[extraction_mask], 0, column[extraction_mask], 
                   alpha=0.3, color='green', label='Область экстракции')
    
    # Показать области фона
    if background_regions:
        bg_values = []
        for bg_start, bg_end in background_regions:
            bg_mask = (y_coords >= bg_start) & (y_coords < bg_end)
            if np.any(bg_mask):
                ax.fill_between(y_coords[bg_mask], 0, column[bg_mask], 
                              alpha=0.3, color='orange', label='Фон' if not bg_values else '')
                bg_values.extend(column[bg_mask])
        
        if bg_values:
            bg_level = np.median(bg_values)
            ax.axhline(bg_level, color='orange', linestyle='--', 
                      label=f'Уровень фона: {bg_level:.1f}')
    
    ax.axvline(y_center_interp, color='red', linewidth=2, label='Центр')
    ax.axvline(y_upper_interp, color='lime', linewidth=1.5, alpha=0.7)
    ax.axvline(y_lower_interp, color='lime', linewidth=1.5, alpha=0.7)
    
    ax.set_title('Профиль и области')
    ax.set_xlabel('Y (пиксели)')
    ax.set_ylabel('Интенсивность')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Веса для экстракции
    ax = axes[0, 2]
    
    extraction_column = column[extraction_mask]
    extraction_y = y_coords[extraction_mask]
    
    if extraction_method == 'sum':
        weights = np.ones_like(extraction_column)
    elif extraction_method == 'mean':
        weights = np.ones_like(extraction_column) / len(extraction_column)
    elif extraction_method == 'gaussian_weighted':
        sigma = order_data.get('median_width', 10) / 2.35
        weights = np.exp(-0.5 * ((extraction_y - y_center_interp) / sigma)**2)
        weights /= np.sum(weights)
    elif extraction_method == 'optimal':
        # Упрощенные оптимальные веса
        sigma = order_data.get('median_width', 10) / 2.35
        profile_weights = np.exp(-0.5 * ((extraction_y - y_center_interp) / sigma)**2)
        variance = np.abs(extraction_column) + 1
        weights = profile_weights / variance
        weights /= np.sum(weights)
    
    bars = ax.bar(extraction_y, weights, width=0.8, alpha=0.7, color='blue')
    ax.set_title(f'Веса для метода "{extraction_method}"')
    ax.set_xlabel('Y (пиксели)')
    ax.set_ylabel('Вес')
    ax.grid(True, alpha=0.3)
    
    # 4. Вклад каждого пикселя
    ax = axes[1, 0]
    
    bg_level = 0
    if background_regions:
        bg_values = []
        for bg_start, bg_end in background_regions:
            bg_mask = (y_coords >= bg_start) & (y_coords < bg_end)
            if np.any(bg_mask):
                bg_values.extend(column[bg_mask])
        if bg_values:
            bg_level = np.median(bg_values)
    
    contributions = (extraction_column - bg_level) * weights
    bars = ax.bar(extraction_y, contributions, width=0.8, alpha=0.7, color='green')
    
    # Подсветить отрицательные вклады
    negative_mask = contributions < 0
    if np.any(negative_mask):
        ax.bar(extraction_y[negative_mask], contributions[negative_mask], 
               width=0.8, alpha=0.7, color='red', label='Отрицательный вклад')
    
    ax.set_title('Вклад каждого пикселя в итоговый сигнал')
    ax.set_xlabel('Y (пиксели)')
    ax.set_ylabel('Взвешенная интенсивность')
    ax.grid(True, alpha=0.3)
    if np.any(negative_mask):
        ax.legend()
    
    # 5. Статистика экстракции
    ax = axes[1, 1]
    ax.axis('off')
    
    total_signal = np.sum(contributions)
    raw_signal = np.sum(extraction_column - bg_level)
    n_pixels = len(extraction_column)
    
    stats_text = f"""Статистика экстракции:

Позиция X: {x_position}
Центр Y: {y_center_interp:.2f}
Границы Y: {y_lower_interp:.2f} - {y_upper_interp:.2f}

Пикселей в апертуре: {n_pixels}
Уровень фона: {bg_level:.1f}

Сырой сигнал: {raw_signal:.1f}
Взвешенный сигнал: {total_signal:.1f}
Средняя интенсивность: {np.mean(extraction_column):.1f}

Метод экстракции: {extraction_method}
"""
    
    if aperture_width:
        stats_text += f"Ширина апертуры: {aperture_width:.1f}\n"
    else:
        stats_text += f"Ширина (трассировка): {y_upper_interp - y_lower_interp:.1f}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. Сравнение методов экстракции
    ax = axes[1, 2]
    
    methods = ['sum', 'mean', 'gaussian_weighted', 'optimal']
    results = []
    
    for method in methods:
        if method == 'sum':
            w = np.ones_like(extraction_column)
        elif method == 'mean':
            w = np.ones_like(extraction_column) / len(extraction_column)
        elif method == 'gaussian_weighted':
            sigma = order_data.get('median_width', 10) / 2.35
            w = np.exp(-0.5 * ((extraction_y - y_center_interp) / sigma)**2)
            w /= np.sum(w)
        elif method == 'optimal':
            sigma = order_data.get('median_width', 10) / 2.35
            profile_w = np.exp(-0.5 * ((extraction_y - y_center_interp) / sigma)**2)
            variance = np.abs(extraction_column) + 1
            w = profile_w / variance
            w /= np.sum(w)
        
        result = np.sum((extraction_column - bg_level) * w)
        results.append(result)
    
    bars = ax.bar(methods, results, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    
    # Выделить текущий метод
    current_idx = methods.index(extraction_method)
    bars[current_idx].set_color('darkblue')
    bars[current_idx].set_alpha(1.0)
    
    ax.set_title('Сравнение методов экстракции')
    ax.set_ylabel('Извлеченный сигнал')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Добавить значения на столбцы
    for i, (bar, value) in enumerate(zip(bars, results)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'x_position': x_position,
        'y_center': y_center_interp,
        'y_bounds': (y_lower_interp, y_upper_interp),
        'n_pixels': n_pixels,
        'background_level': bg_level,
        'extracted_signal': total_signal,
        'method_comparison': dict(zip(methods, results))
    }


def check_interpolation_quality(trace_data, order_number, plot=True):
    """
    Проверка качества интерполяции трассировки
    """
    
    order_data = None
    for order in trace_data['orders']:
        if order['order_number'] == order_number:
            order_data = order
            break
    
    if order_data is None:
        raise ValueError(f"Порядок {order_number} не найден")
    
    # Исходные данные трассировки
    x_measured = order_data['trace_points']['x_measured']
    y_measured = order_data['trace_points']['y_measured']
    
    # Полная интерполированная трассировка
    x_full = order_data['trace_full']['x']
    y_full = order_data['trace_full']['y_center']
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Исходные точки vs интерполяция
        ax = axes[0, 0]
        ax.plot(x_full, y_full, 'b-', linewidth=1, label='Интерполированная')
        ax.plot(x_measured, y_measured, 'ro', markersize=4, label='Измеренные точки')
        ax.set_title('Трассировка: измеренные точки vs интерполяция')
        ax.set_xlabel('X (пиксели)')
        ax.set_ylabel('Y (пиксели)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Остатки интерполяции
        ax = axes[0, 1]
        y_interp_at_measured = np.interp(x_measured, x_full, y_full)
        residuals = y_measured - y_interp_at_measured
        
        ax.plot(x_measured, residuals, 'ro-', markersize=3)
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        ax.axhline(np.std(residuals), color='r', linestyle='--', alpha=0.5, 
                  label=f'±σ = ±{np.std(residuals):.3f}')
        ax.axhline(-np.std(residuals), color='r', linestyle='--', alpha=0.5)
        ax.set_title('Остатки интерполяции')
        ax.set_xlabel('X (пиксели)')
        ax.set_ylabel('Остатки (пиксели)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Производная (наклон)
        ax = axes[1, 0]
        if len(x_full) > 1:
            slope = np.gradient(y_full, x_full)
            ax.plot(x_full, slope, 'g-', linewidth=1)
            ax.set_title('Наклон трассировки (dy/dx)')
            ax.set_xlabel('X (пиксели)')
            ax.set_ylabel('Наклон')
            ax.grid(True, alpha=0.3)
        
        # 4. Статистика
        ax = axes[1, 1]
        ax.axis('off')
        
        rms_residuals = np.sqrt(np.mean(residuals**2))
        max_residual = np.max(np.abs(residuals))
        
        stats_text = f"""Качество интерполяции:

Измеренных точек: {len(x_measured)}
Интерполированных точек: {len(x_full)}
Шаг интерполяции: {np.median(np.diff(x_full)):.1f} пикс

Остатки интерполяции:
  RMS: {rms_residuals:.4f} пикс
  Максимальный: {max_residual:.4f} пикс
  Стандартное отклонение: {np.std(residuals):.4f} пикс

Наклон трассировки:
  Средний: {np.mean(slope) if len(x_full) > 1 else 0:.4f}
  Диапазон: {np.min(slope) if len(x_full) > 1 else 0:.4f} - {np.max(slope) if len(x_full) > 1 else 0:.4f}
"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    return {
        'n_measured_points': len(x_measured),
        'n_interpolated_points': len(x_full),
        'interpolation_rms': np.sqrt(np.mean(residuals**2)),
        'max_residual': np.max(np.abs(residuals)),
        'residuals_std': np.std(residuals)
    }