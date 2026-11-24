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
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from astropy.io import fits
from scipy.optimize import OptimizeWarning, curve_fit

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
# ИЗВЛЕЧЕНИЕ 14 СРЕЗОВ И СУБПИКСЕЛЬНЫЙ АНАЛИЗ ЛИНИЙ
# =============================================================================

def extract_all_14_slices(thar_file: Path,
                          trace_file: Path,
                          n_slices: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлекает 1D спектры всех 14 срезов ThAr кадра по трассировке.

    Returns:
    --------
    tuple : (x_coords, slices_flux) где slices_flux имеет форму [n_slices, Nx]
    """
    with fits.open(thar_file) as hdul:
        thar_data = np.maximum(hdul[0].data, 1e-6)

    trace_data = load_trace_data(trace_file)
    if trace_data is None:
        raise ValueError(f"Не удалось загрузить файл трассировки: {trace_file}")

    flux_list = []
    x_coords = None

    for order_num in range(1, n_slices + 1):
        x, fl = extract_order_summed(thar_data, trace_data, order_num)
        if fl is None:
            raise ValueError(f"Не удалось извлечь срез {order_num} из {thar_file}")
        flux_list.append(fl)
        if x_coords is None and x is not None:
            x_coords = x

    if x_coords is None:
        x_coords = np.arange(thar_data.shape[1], dtype=np.float64)

    return x_coords, np.array(flux_list, dtype=np.float64)


def _gauss_bg(x: np.ndarray, a: float, c: float, s: float, b: float) -> np.ndarray:
    return a * np.exp(-0.5 * ((x - c) / s) ** 2) + b


def gauss_center(v: np.ndarray,
                 x0: float,
                 fit_half: int = 8,
                 sigma_max: float = 4.0,
                 sigma0: float = 1.2) -> Optional[float]:
    """
    Субпиксельное уточнение центра спектральной линии по гауссовой подгонке с фоном.
    """
    if not np.isfinite(x0):
        return None
    i = round(x0)
    i0, i1 = i - fit_half, i + fit_half
    if i0 < 0 or i1 >= v.size:
        return None
    x = np.arange(i0, i1 + 1, dtype=np.float64)
    y = v[i0:i1 + 1]
    p0 = [float(y.max() - y.min()), float(x0), sigma0, float(y.min())]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p, _ = curve_fit(_gauss_bg, x, y, p0=p0, maxfev=2000)
    except (RuntimeError, ValueError):
        return None

    a, c, s = p[0], p[1], abs(p[2])
    if not (a > 0 and 0.4 <= s <= sigma_max and i0 <= c <= i1):
        return None
    return float(c)


def channel_shift(a: np.ndarray, b: np.ndarray, max_lag: int = 30) -> int:
    """
    Целочисленный сдвиг профиля b относительно профиля a по максимуму кросскорреляции.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    cc = [np.dot(a[max_lag:-max_lag], np.roll(b, -lag)[max_lag:-max_lag]) for lag in lags]
    return int(lags[int(np.argmax(cc))])


def prepare_display_spectra(slices_flux: np.ndarray,
                            gamma: float = 0.4,
                            bg_half: int = 5) -> np.ndarray:
    """
    Подготовка спектров для интерактивного отображения (I^gamma + вычет фона).
    """
    out = np.empty_like(slices_flux)
    x = np.arange(slices_flux.shape[1])
    w = bg_half
    for j, v in enumerate(slices_flux):
        v_clipped = np.where(v < 1.0, 1.0, v) ** gamma
        mins = [i for i in range(w, v_clipped.size - w) if v_clipped[i] == v_clipped[i - w:i + w + 1].min()]
        bg = np.interp(x, mins, v_clipped[mins]) if len(mins) > 2 else np.full_like(v_clipped, v_clipped.min())
        out[j] = np.clip(v_clipped - bg, 0, None)
    return out


def locate_line_across_slices(slices_flux: np.ndarray,
                              x_click: float,
                              click_slice: int,
                              shifts: List[int],
                              eps: int = 15) -> Optional[List[float]]:
    """
    Определяет субпиксельные координаты линии во всех 14 срезах по клику в click_slice.
    """
    n_slices = slices_flux.shape[0]
    nx = slices_flux.shape[1]
    x_base = x_click - shifts[click_slice]
    xs = []
    for k in range(n_slices):
        g = x_base + shifts[k]
        lo, hi = int(g) - eps, int(g) + eps
        if lo < 2 or hi >= nx - 2:
            return None
        pk = lo + int(np.argmax(slices_flux[k, lo:hi + 1]))
        c = gauss_center(slices_flux[k], float(pk))
        if c is None:
            p_sub = np.polyfit(np.arange(pk - 2, pk + 3, dtype=float),
                               slices_flux[k, pk - 2:pk + 3], 2)
            c = float(-p_sub[1] / (2.0 * p_sub[0])) if p_sub[0] < 0 else float(pk)
        xs.append(float(c))
    return xs

def load_partial_marks(partial_path: Path) -> List[Dict]:
    """Загружает сохраненную разметку из partial JSON файла"""
    if not partial_path.exists():
        return []
    try:
        with open(partial_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get('marks', [])
    except Exception as e:
        logger.warning(f"Не удалось загрузить {partial_path}: {e}")
        return []


def save_partial_marks(partial_path: Path, marks: List[Dict]) -> None:
    """Сохраняет разметку в partial JSON файл для защиты от сбоев"""
    try:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, 'w') as f:
            json.dump({'marks': marks}, f, indent=2)
    except Exception as e:
        logger.warning(f"Ошибка автосохранения partial: {e}")


def remove_nearest_mark(marks: List[Dict],
                        x_click: float,
                        slice_idx: int) -> Optional[float]:
    """Удаляет из списка разметки ближайшую к клику линию"""
    if not marks:
        return None
    distances = [abs(m['pixels'][slice_idx] - x_click) for m in marks]
    min_idx = int(np.argmin(distances))
    if distances[min_idx] < 25.0:
        removed = marks.pop(min_idx)
        return removed['wavelength']
    return None


def build_reference_solution_dict(marks: List[Dict],
                                  thar_file: Path,
                                  poly_degree: int = 5,
                                  reference_order: int = 4) -> Dict:
    """
    Строит полное полиномиальное решение для всех 14 срезов по отождествленным линиям.
    Формат полностью совместим со структурой FULL.json и apply_calibration.py.
    """
    n_slices = len(marks[0]['pixels'])
    slices_dict = {}

    for s_idx in range(n_slices):
        order_num = s_idx + 1
        x_pts = np.array([m['pixels'][s_idx] for m in marks], dtype=np.float64)
        lam_pts = np.array([m['wavelength'] for m in marks], dtype=np.float64)
        sort_idx = np.argsort(x_pts)
        x_pts = x_pts[sort_idx]
        lam_pts = lam_pts[sort_idx]

        deg = min(poly_degree, len(x_pts) - 1)
        poly = fit_dispersion_poly(x_pts, lam_pts, deg)
        coeffs = poly.c
        residuals = lam_pts - poly(x_pts)
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        calib_points = {str(float(px)): float(wl) for px, wl in zip(x_pts, lam_pts)}

        slices_dict[str(order_num)] = {
            'model': coeffs.tolist(),
            'model_degree': int(deg),
            'rms_angstrom': float(rms),
            'rms_pixels': float(rms / 0.12),
            'n_lines': len(x_pts),
            'quality': 'excellent' if rms <= 0.05 else ('good' if rms <= 0.09 else 'warning'),
            'is_reference': (order_num == reference_order),
            'drift_from_template_pixels': 0.0,
            'calib_points': calib_points,
            'all_peaks': x_pts.tolist()
        }

    return {
        'thar_file': thar_file.name,
        'reference_order': reference_order,
        'poly_degree': poly_degree,
        'marks': marks,
        'slices': slices_dict
    }


def run_interactive_14_slices_gui(slices_raw: np.ndarray,
                                  shifts: List[int],
                                  atlas_lines: np.ndarray,
                                  out_solution: Path,
                                  partial_path: Path,
                                  thar_file: Path,
                                  poly_degree: int = 5,
                                  initial_gamma: float = 0.4) -> Optional[Dict]:
    """
    Интерактивный GUI Matplotlib: 14 субплотов спектров со связанной осью X.
    Левый клик по линии в любом срезе находит ее положение во всех 14 срезах.
    Ввод длины волны в текстовое поле фиксирует линию в эталоне.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    n_slices = slices_raw.shape[0]
    nx = slices_raw.shape[1]

    state = {
        'gamma': initial_gamma,
        'pending_xs': None,
        'pending_slice': None,
        'marks': load_partial_marks(partial_path),
        'saved': False,
        'result': None,
        'busy_clear': False
    }

    fig, axes = plt.subplots(n_slices, 1, figsize=(16, 9), sharex=True)
    fig.canvas.manager.set_window_title(f"pyZeeman - Отождествление линий ThAr: {thar_file.name} (14 срезов)")
    plt.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.14, hspace=0.08)

    curves = []
    for k, ax in enumerate(axes):
        order_num = k + 1
        beam = "U" if order_num <= 7 else "L"
        color = "#1f77b4" if beam == "U" else "#2ca02c"
        ln, = ax.plot(np.arange(nx), np.zeros(nx), color=color, lw=0.6)
        curves.append(ln)
        ax.set_ylabel(f"С{order_num:02d} ({beam})", fontsize=7, labelpad=2)
        ax.tick_params(axis='both', which='both', labelsize=6, pad=1)
        ax.grid(True, alpha=0.25, linestyle=':')

    axes[-1].set_xlabel("X (пиксели дисперсии)", fontsize=9, labelpad=2)

    ax_tb_lam = fig.add_axes([0.08, 0.04, 0.13, 0.045])
    tb_lam = TextBox(ax_tb_lam, 'λ, Å: ', initial='')

    ax_tb_gam = fig.add_axes([0.27, 0.04, 0.06, 0.045])
    tb_gam = TextBox(ax_tb_gam, 'γ: ', initial=f"{state['gamma']:.2g}")

    ax_btn_gup = fig.add_axes([0.34, 0.04, 0.065, 0.045])
    btn_gup = Button(ax_btn_gup, 'γ ×1.25')

    ax_btn_gdn = fig.add_axes([0.41, 0.04, 0.065, 0.045])
    btn_gdn = Button(ax_btn_gdn, 'γ ÷1.25')

    ax_btn_save = fig.add_axes([0.79, 0.04, 0.09, 0.045])
    btn_save = Button(ax_btn_save, 'Записать')

    ax_btn_done = fig.add_axes([0.89, 0.04, 0.09, 0.045])
    btn_done = Button(ax_btn_done, 'Готово')

    status_text = fig.text(0.08, 0.015, "", fontsize=9, color="#004d40")
    art = {'pending': [], 'marks': []}

    def say(msg: str):
        status_text.set_text(msg)
        logger.info(msg)
        fig.canvas.draw_idle()

    def draw_marks():
        for a in art['marks']:
            try:
                a.remove()
            except Exception:
                pass
        art['marks'] = []

        top_ax = axes[0]
        y_max = top_ax.get_ylim()[1]

        for m in state['marks']:
            lam = m['wavelength']
            xs = m['pixels']
            for k, ax in enumerate(axes):
                vl = ax.axvline(xs[k], color='red', lw=0.7, alpha=0.85)
                art['marks'].append(vl)
            txt = top_ax.text(xs[0], y_max * 0.95, f"{lam:.2f}",
                              rotation=90, fontsize=6, va='top', ha='center', color='red')
            art['marks'].append(txt)

        fig.canvas.draw_idle()

    def draw_pending():
        for a in art['pending']:
            try:
                a.remove()
            except Exception:
                pass
        art['pending'] = []

        if state['pending_xs'] is not None:
            for k, ax in enumerate(axes):
                vl = ax.axvline(state['pending_xs'][k], color='orange', lw=1.2, ls='--')
                art['pending'].append(vl)

        fig.canvas.draw_idle()

    def redraw_spectra():
        disp_data = prepare_display_spectra(slices_raw, gamma=state['gamma'])
        for k, (ax, ln) in enumerate(zip(axes, curves)):
            ln.set_ydata(disp_data[k])
            ax.relim()
            ax.autoscale_view(scalex=False)

        axes[0].set_title(
            f"Калибровка 14 срезов | ЛКМ — выбрать линию во всех 14 срезах | Ввод λ в поле внизу + Enter | ПКМ — удалить | Линий: {len(state['marks'])}",
            fontsize=9, pad=3
        )
        draw_marks()
        draw_pending()

    def on_click(ev):
        if ev.inaxes not in axes or ev.xdata is None:
            return
        tb = fig.canvas.toolbar
        if tb is not None and getattr(tb, "mode", ""):
            return

        k = list(axes).index(ev.inaxes)

        if ev.button == 1:
            xs = locate_line_across_slices(slices_raw, ev.xdata, k, shifts, eps=15)
            if xs is None:
                say("Линия слишком близко к краю спектра")
                return
            state['pending_xs'] = xs
            state['pending_slice'] = k
            draw_pending()

            suggest_str = ""
            if len(state['marks']) >= 3 and len(atlas_lines) > 0:
                ref_xs = [m['pixels'][0] for m in state['marks']]
                ref_lams = [m['wavelength'] for m in state['marks']]
                deg = min(2, len(ref_xs) - 1)
                p_cur = np.polyfit(ref_xs, ref_lams, deg)
                pred_lam = float(np.polyval(p_cur, xs[0]))
                nearest = find_nearest_line(pred_lam, atlas_lines)
                if nearest is not None and abs(nearest - pred_lam) < 5.0:
                    suggest_str = f" [подсказка атласа: {nearest:.4f} Å]"

            say(f"Выбрана линия в срезе {k + 1}: x={xs[k]:.2f}. Введите λ в поле внизу и нажмите Enter{suggest_str}")

        elif ev.button == 3:
            lam = remove_nearest_mark(state['marks'], ev.xdata, k)
            if lam is not None:
                save_partial_marks(partial_path, state['marks'])
                say(f"Удалена линия λ={lam:.2f}. Всего линий: {len(state['marks'])}")
                redraw_spectra()

    def clear_lambda():
        state['busy_clear'] = True
        tb_lam.set_val("")
        state['busy_clear'] = False
        tb_lam.stop_typing()

    def on_lambda_submit(text: str):
        if state['busy_clear'] or not text.strip():
            return
        if state['pending_xs'] is None:
            say("Сначала выберите линию кликом мыши на графике")
            clear_lambda()
            return

        text = text.strip().replace(",", ".")
        try:
            lam = float(text)
        except ValueError:
            say(f"Ошибка ввода: '{text}' не является числом")
            clear_lambda()
            return

        if not (1000.0 <= lam <= 30000.0):
            say(f"Длина волны {lam} вне диапазона 1000...30000 Å")
            clear_lambda()
            return

        for m in state['marks']:
            if abs(m['pixels'][0] - state['pending_xs'][0]) < 1.0 or abs(m['wavelength'] - lam) < 0.05:
                say(f"Линия около x={state['pending_xs'][0]:.1f} уже добавлена как λ={m['wavelength']:.2f}")
                state['pending_xs'] = None
                clear_lambda()
                draw_pending()
                return

        state['marks'].append({
            'wavelength': lam,
            'pixels': [round(float(v), 4) for v in state['pending_xs']]
        })
        state['pending_xs'] = None
        state['pending_slice'] = None
        clear_lambda()
        save_partial_marks(partial_path, state['marks'])
        say(f"Добавлена линия λ={lam:.4f}. Всего линий: {len(state['marks'])}")
        redraw_spectra()

    def on_gamma_submit(text: str):
        if state['busy_clear']:
            return
        tb_gam.stop_typing()
        try:
            g = float(text.strip().replace(",", "."))
            if g > 0:
                state['gamma'] = g
                redraw_spectra()
                say(f"Гамма изменена: {g:.2g}")
        except ValueError:
            pass

    def change_gamma(factor: float):
        state['gamma'] = max(0.05, min(2.0, state['gamma'] * factor))
        state['busy_clear'] = True
        tb_gam.set_val(f"{state['gamma']:.2g}")
        state['busy_clear'] = False
        redraw_spectra()

    def do_save(_=None) -> bool:
        if len(state['marks']) < poly_degree + 1:
            say(f"Недостаточно линий ({len(state['marks'])} < {poly_degree + 1}). Добавьте еще линии.")
            return False

        sol = build_reference_solution_dict(state['marks'], thar_file, poly_degree=poly_degree)
        save_calibration_solution(sol, out_solution)
        state['saved'] = True
        state['result'] = sol
        say(f"✓ Решение успешно сохранено в {out_solution.name}")
        return True

    def do_done(_=None):
        if do_save():
            plt.close(fig)

    tb_lam.on_submit(on_lambda_submit)
    tb_gam.on_submit(on_gamma_submit)
    btn_gup.on_clicked(lambda _: change_gamma(1.25))
    btn_gdn.on_clicked(lambda _: change_gamma(0.80))
    btn_save.on_clicked(do_save)
    btn_done.on_clicked(do_done)
    fig.canvas.mpl_connect("button_press_event", on_click)

    fig._pyzeeman_widgets = (tb_lam, tb_gam, btn_gup, btn_gdn, btn_save, btn_done)

    redraw_spectra()
    say(f"Сдвиги 14 срезов относительно опорного: {shifts}. Загружено линий: {len(state['marks'])}")
    plt.show()

    return state['result']



# =============================================================================
# ИНТЕРАКТИВНАЯ КАЛИБРОВКА (ТОЧКА ВХОДА)
# =============================================================================

def calibrate_first_thar_interactive(thar_file: Path,
                                     trace_file: Path,
                                     order_num: int = 4,
                                     atlas_file: Path = Path('thar.dat'),
                                     gain: float = 2.78,
                                     ron_e: float = 5.6,
                                     poly_degree: int = 5,
                                     peak_params: Optional[dict] = None) -> Optional[Dict]:
    """
    Интерактивная калибровка первого ThAr кадра одновременно для всех 14 срезов.
    """
    logger.info("=" * 80)
    logger.info(f"ИНТЕРАКТИВНАЯ КАЛИБРОВКА 14 СРЕЗОВ: {thar_file.name}")
    logger.info("=" * 80)

    atlas_lines = load_thar_atlas(atlas_file)
    x_coords, slices_raw = extract_all_14_slices(thar_file, trace_file)

    ref_idx = max(0, min(order_num - 1, 13))
    shifts = [channel_shift(slices_raw[ref_idx], slices_raw[k], max_lag=30) for k in range(14)]
    logger.info(f"Относительные сдвиги срезов к срезу {order_num}: {shifts}")

    out_solution = thar_file.parent / "CALIBRATIONS" / "reference_solution.json"
    partial_path = thar_file.parent / "CALIBRATIONS" / "reference_solution.partial.json"

    solution = run_interactive_14_slices_gui(
        slices_raw=slices_raw,
        shifts=shifts,
        atlas_lines=atlas_lines,
        out_solution=out_solution,
        partial_path=partial_path,
        thar_file=thar_file,
        poly_degree=poly_degree
    )

    if solution is not None:
        logger.info(f"✓ Интерактивная калибровка 14 срезов завершена успешно")
    else:
        logger.warning("! Интерактивная калибровка не завершена")

    return solution



# =============================================================================
# АВТОМАТИЧЕСКАЯ КАЛИБРОВКА СРЕЗОВ (ТОЧКА ВХОДА)
# =============================================================================

def calibrate_all_slices_auto(thar_file: Path,
                             trace_file: Path,
                             reference_order: int,
                             reference_solution: dict,
                             atlas_file: Path,
                             poly_degree: int = 5,
                             rms_excellent: float = 0.05,
                             rms_good: float = 0.09,
                             rms_warning: float = 0.35,
                             pixel_tolerance: float = 4.0,
                             wavelength_tolerance: float = 0.5,
                             peak_params: Optional[dict] = None,
                             compare_template: bool = False) -> Optional[Dict]:
    """
    Автоматическая калибровка всех 14 срезов текущего ThAr на основе 14-срезового эталона.

    Алгоритм:
    1. Извлекаются все 14 срезов нового кадра.
    2. Оценивается инструментальный дрейф относительно опорного кадра через кросскорреляцию.
    3. Для каждой линии эталона в каждом срезе находится точный субпиксельный центр
       гауссовой подгонкой в узком окне вокруг x_ref + shift.
    4. Строится полиномиальное дисперсионное решение с 3-sigma фильтрацией выбросов.
    """
    logger.info("=" * 80)
    logger.info(f"АВТОМАТИЧЕСКАЯ КАЛИБРОВКА 14 СРЕЗОВ: {thar_file.name}")
    logger.info("=" * 80)

    marks = reference_solution.get('marks')
    if not marks:
        marks = []
        slices_ref = reference_solution.get('slices', {})
        if slices_ref and str(reference_order) in slices_ref:
            pts_ref = slices_ref[str(reference_order)].get('calib_points', {})
            for px_str, wl in pts_ref.items():
                p_val = float(px_str)
                px_all = []
                for s in range(1, 15):
                    s_pts = slices_ref.get(str(s), {}).get('calib_points', {})
                    best_p = p_val
                    min_dw = 999.0
                    for spx, swl in s_pts.items():
                        if abs(float(swl) - wl) < min_dw:
                            min_dw = abs(float(swl) - wl)
                            best_p = float(spx)
                    px_all.append(best_p)
                marks.append({'wavelength': float(wl), 'pixels': px_all})

    if not marks or len(marks) < poly_degree + 1:
        logger.error("! В reference_solution нет достаточного количества линий эталона")
        return None

    x_coords, current_slices = extract_all_14_slices(thar_file, trace_file)
    n_slices = current_slices.shape[0]

    all_slices = {}

    for s_idx in range(n_slices):
        order_num = s_idx + 1
        ref_x_pts = [m['pixels'][s_idx] for m in marks]
        ref_wls = [m['wavelength'] for m in marks]

        local_shifts = []
        for x0 in ref_x_pts:
            i0 = max(2, int(x0) - 10)
            i1 = min(current_slices.shape[1] - 3, int(x0) + 10)
            if i1 > i0:
                pk = i0 + int(np.argmax(current_slices[s_idx, i0:i1 + 1]))
                local_shifts.append(pk - x0)

        drift = float(np.median(local_shifts)) if local_shifts else 0.0

        tracked_pixels = []
        tracked_wavelengths = []

        for x0, wl in zip(ref_x_pts, ref_wls):
            x_guess = x0 + drift
            c = gauss_center(current_slices[s_idx], x_guess, fit_half=8, sigma_max=4.5)
            if c is not None and abs(c - x_guess) <= 6.0:
                tracked_pixels.append(float(c))
                tracked_wavelengths.append(float(wl))

        if len(tracked_pixels) < poly_degree + 1:
            logger.warning(f"  Срез {order_num}: мало отождествленных линий ({len(tracked_pixels)}), снижаем степень полинома")
            deg = max(2, len(tracked_pixels) - 1)
        else:
            deg = poly_degree

        tracked_pixels = np.array(tracked_pixels, dtype=np.float64)
        tracked_wavelengths = np.array(tracked_wavelengths, dtype=np.float64)
        sort_idx = np.argsort(tracked_pixels)
        tracked_pixels = tracked_pixels[sort_idx]
        tracked_wavelengths = tracked_wavelengths[sort_idx]

        disp_model = fit_dispersion_poly(tracked_pixels, tracked_wavelengths, deg)
        coeffs = disp_model.c
        residuals = tracked_wavelengths - disp_model(tracked_pixels)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        rms_pix = rms / 0.12

        if rms <= rms_excellent:
            quality = 'excellent'
            symbol = '✅'
        elif rms <= rms_good:
            quality = 'good'
            symbol = '✓'
        elif rms <= rms_warning:
            quality = 'warning'
            symbol = '⚠'
        else:
            quality = 'critical'
            symbol = '✗'

        logger.info(f"  Срез {order_num:2d}: RMS={rms:.4f} Å ({rms_pix:.3f} px), дрейф={drift:+.2f} px, {len(tracked_pixels)}/{len(marks)} линий {symbol}")

        all_slices[str(order_num)] = {
            'model': coeffs.tolist(),
            'model_degree': int(deg),
            'rms_angstrom': float(rms),
            'rms_pixels': float(rms_pix),
            'n_lines': len(tracked_pixels),
            'quality': quality,
            'is_reference': (order_num == reference_order),
            'drift_from_template_pixels': float(drift),
            'calib_points': {str(float(px)): float(wl) for px, wl in zip(tracked_pixels, tracked_wavelengths)},
            'all_peaks': tracked_pixels.tolist()
        }

    full_solution = {
        'thar_file': thar_file.name,
        'reference_order': reference_order,
        'poly_degree': poly_degree,
        'slices': all_slices
    }

    logger.info(f"\n✓ Откалибровано срезов: {len(all_slices)}/14")
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
        
        if reference_solution and 'calib_points' in ref_slice and 'calib_points' in reference_solution:
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
                # Опорный срез: сдвиг относительно референсного решения если оно передано
                if reference_solution and 'model' in reference_solution:
                    shift_values = ref_model(x) - np.poly1d(reference_solution['model'])(x)
                else:
                    shift_values = np.zeros_like(x)
                
                # Отождествленные точки (круги) - сдвиги относительно референсного
                shift_points_x = ref_shift_points_x
                shift_points_y = ref_shift_points_y
                
                # ВСЕ найденные пики (крестики) - для визуального контроля
                all_peaks_x = []
                all_peaks_y = []
                if 'all_peaks' in ref_slice:
                    all_peaks_x = ref_slice['all_peaks']
                    if reference_solution and 'model' in reference_solution:
                        all_peaks_y = [ref_model(pixel) - np.poly1d(reference_solution['model'])(pixel) 
                                      for pixel in all_peaks_x]
                    else:
                        all_peaks_y = [0.0 for _ in all_peaks_x]
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
