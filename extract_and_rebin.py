import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.models import Polynomial1D
from scipy.interpolate import interp1d
from specutils import Spectrum1D

# --- 1. Подготовка данных (симуляция) ---

# Определим общую опорную сетку, на которую все будет пересчитываться
reference_grid = np.arange(2048)

# Загрузим нашу модель искажений (вместо чтения JSON, создадим ее в коде)
# Коэффициенты от старшей степени к младшей (c2, c1, c0 для 2-й степени)
distortion_models = {
    # Опорный порядок: сдвиг равен нулю, модель = константа 0.
    'order_60': Polynomial1D(degree=1, c0=0.0, c1=0.0), # Модель y = 0*x + 0
    
    # Соседний порядок: сдвиг ~ -15 пикселей, с небольшой кривизной
    'order_59': Polynomial1D(degree=2, c0=-15.0, c1=-2e-4, c2=1e-7),
    
    # Другой соседний порядок: сдвиг ~ +16 пикселей, с другой кривизной
    'order_61': Polynomial1D(degree=2, c0=16.5, c1=1.5e-4, c2=-1.2e-7)
}

# Создадим несколько симулированных спектров (объектов Spectrum1D)
input_spectra = []
true_line_pos = 1024.5  # Положение "настоящей" линии в опорной системе координат

for order_name, model in distortion_models.items():
    # Исходная сетка каждого порядка - своя собственная
    original_grid = np.arange(2048)
    
    # Где бы находилась линия на этой сетке, чтобы ПОСЛЕ сдвига попасть в true_line_pos?
    # pos_distorted = pos_original + shift(pos_original) => pos_original ≈ pos_distorted - shift(pos_distorted)
    # Это лишь для симуляции, нам не нужно так точно.
    # Проще: сдвинем истинную позицию и создадим линию там.
    shift_at_center = model(true_line_pos)
    line_pos_on_this_order = true_line_pos - shift_at_center
    
    # Создаем поток с линией
    flux_values = 1.0 - 0.8 * np.exp(-(original_grid - line_pos_on_this_order)**2 / (2 * 4**2))
    flux_values += np.random.normal(0, 0.03, flux_values.shape)
    
    # Создаем объект Spectrum1D
    spectrum = Spectrum1D(
        flux=flux_values * u.adu,  # Используем единицы потока
        spectral_axis=original_grid * u.pix # И единицы пикселей
    )
    spectrum.meta['order_name'] = order_name # Сохраним имя для удобства
    input_spectra.append(spectrum)


# --- 2. Функция для пересэмплирования (сердце алгоритма) ---

def resample_spectrum_modern(spectrum: Spectrum1D, model: Polynomial1D, target_grid: np.ndarray):
    """
    Пересэмплирует спектр на новую сетку, используя модель искажений.
    
    Args:
        spectrum (Spectrum1D): Входной спектр для обработки.
        model (Polynomial1D): Модель искажений astropy.
        target_grid (np.ndarray): Целевая регулярная сетка (в пикселях).
        
    Returns:
        Spectrum1D: Новый спектр, живущий на целевой сетке.
    """
    # Текущая сетка спектра (в виде простого numpy-массива)
    original_grid_values = spectrum.spectral_axis.value

    # Вычисляем сдвиг для каждого пикселя
    shifts = model(original_grid_values)

    # Создаем новую, нерегулярную сетку, куда "переехали" значения потока
    distorted_grid_values = original_grid_values + shifts
    
    # Создаем функцию-интерполятор.
    # Мы используем .value для передачи голых numpy-массивов в SciPy.
    # 'linear' - самый надежный и быстрый вариант, аналог MIDAS.
    # fill_value=np.nan полезен для отладки, чтобы видеть, где нет данных.
    interpolator = interp1d(
        distorted_grid_values,
        spectrum.flux.value,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Вычисляем поток на новой, ЦЕЛЕВОЙ регулярной сетке
    resampled_flux_values = interpolator(target_grid)
    
    # Собираем результат обратно в объект Spectrum1D, уже на новой сетке
    resampled_spectrum = Spectrum1D(
        flux=resampled_flux_values * spectrum.unit,
        spectral_axis=target_grid * u.pix
    )
    
    return resampled_spectrum
    

# --- 3. Выполняем обработку ---

resampled_spectra = []
for spectrum in input_spectra:
    order_name = spectrum.meta['order_name']
    model = distortion_models[order_name]
    
    print(f"Пересэмплирование {order_name}...")
    resampled_spec = resample_spectrum_modern(spectrum, model, reference_grid)
    resampled_spectra.append(resampled_spec)

# Объединяем спектры.
# Просто складываем потоки, так как они теперь на одной сетке.
# np.nansum игнорирует NaN на краях, где не было перекрытия.
final_flux = np.nansum([spec.flux.value for spec in resampled_spectra], axis=0)

# Создаем финальный объект Spectrum1D
final_combined_spectrum = Spectrum1D(
    flux=final_flux * u.adu,
    spectral_axis=reference_grid * u.pix
)


# --- 4. Визуализация для проверки ---

plt.style.use('default')
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# График 1: Исходные, невыровненные спектры
axes[0].set_title("1. Исходные спектры (не выровнены)")
for spec in input_spectra:
    axes[0].plot(spec.spectral_axis, spec.flux, label=spec.meta['order_name'])
axes[0].axvline(true_line_pos, color='k', linestyle=':', label=f'Истинное положение ({true_line_pos} pix)')
axes[0].legend()
axes[0].set_ylabel("Поток (ADU)")

# График 2: Спектры после пересэмплирования (выровнены)
axes[1].set_title("2. Спектры после пересэмплирования (выровнены)")
for spec in resampled_spectra:
    axes[1].plot(spec.spectral_axis, spec.flux)
axes[1].axvline(true_line_pos, color='k', linestyle=':')
axes[1].set_ylabel("Поток (ADU)")
axes[1].set_ylim(axes[0].get_ylim())

# График 3: Финальный объединенный спектр
axes[2].set_title("3. Финальный объединенный спектр")
axes[2].plot(final_combined_spectrum.spectral_axis, final_combined_spectrum.flux, color='black')
axes[2].axvline(true_line_pos, color='k', linestyle=':')
axes[2].set_ylabel("Суммарный поток (ADU)")
axes[2].set_xlabel("Общая пиксельная сетка")

plt.tight_layout()
plt.show()