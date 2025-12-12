#!/usr/bin/env python3
"""
barycorr.py - Барицентрическая коррекция спектров для pyZeeman

Применяет барицентрическую коррекцию к откалиброванным спектрам:
1. Извлекает координаты объекта из SIMBAD
2. Вычисляет барицентрическую поправку
3. Применяет релятивистскую коррекцию Доплера к λ
4. Сохраняет с суффиксом _barycorr

Автор: pyZeeman team
Дата: 2025-12-12
"""

import numpy as np
import logging
from pathlib import Path
from astropy import time, units as u
from astropy.io import fits
from astropy.coordinates import EarthLocation, SkyCoord, FK5

# Попытка импортировать astroquery (опционально)
try:
    from astroquery.simbad import Simbad
    SIMBAD_AVAILABLE = True
except ImportError:
    SIMBAD_AVAILABLE = False
    logging.warning("astroquery не установлен, координаты нужно вводить вручную")

logger = logging.getLogger(__name__)

# Координаты Кавказской горной обсерватории
OBSERVATORY = {
    'lon': 41.44044,      # Восточная долгота (градусы)
    'lat': 43.646825,     # Северная широта (градусы)
    'alt': 2070.0         # Высота (метры)
}


def doppler_corrections(ra, dec, mean_time, obs_lat=43.646825, 
                       obs_lon=41.44044, obs_alt=2070.):
    """
    Вычисляет проекцию скорости телескопа относительно четырех систем координат: 
    geo, helio, bary, lsr.
    
    Для коррекции оси скоростей к LSR:
    vcorr = doppler_corrections(ra, dec, mean_time)
    rv = rv + vcorr[3] + rv * vcorr[3] / c
    где rv - наблюдаемая лучевая скорость.
    
    Parameters:
    -----------
    ra : float
        Прямое восхождение в градусах, система J2000
    dec : float
        Склонение в градусах, система J2000
    mean_time : str
        Среднее время наблюдения в формате isot. Пример: "2017-01-15T01:59:58.99"
    obs_lon : float
        Восточная долгота обсерватории в градусах
    obs_lat : float
        Широта обсерватории в градусах
    obs_alt : float
        Высота обсерватории в метрах
    
    Returns:
    --------
    list : [geo, helio, bary, lsr] в км/с
        geo - геоцентрическая коррекция (вращение Земли)
        helio - гелиоцентрическая коррекция
        bary - барицентрическая коррекция
        lsr - коррекция к LSR (Local Standard of Rest)
    """
    # Инициализация координат источника
    src = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    
    # Локальные параметры
    mytime = time.Time(mean_time, format='isot', scale='utc')
    location = EarthLocation.from_geodetic(lat=obs_lat*u.deg, lon=obs_lon*u.deg, height=obs_alt*u.m)
 
    # Орбитальная скорость Земли относительно Солнца
    # helio = проекция скорости орбиты Земли относительно центра Солнца
    # bary = проекция скорости орбиты Земля+Луна относительно центра Солнца
    barycorr = src.radial_velocity_correction(obstime=mytime, location=location)  
    barycorr = barycorr.to(u.km/u.s)
    heliocorr = src.radial_velocity_correction('heliocentric', obstime=mytime, location=location)  
    heliocorr = heliocorr.to(u.km/u.s)
        
    # Скорость вращения Земли
    # Из chdoppler.pro, "Spherical Astronomy" R. Green p.270 
    lst = mytime.sidereal_time('apparent', obs_lon)
    obs_lat = obs_lat * u.deg
    obs_lon = obs_lon * u.deg
    hour_angle = lst - src.ra
    v_spin = -0.465 * np.cos(obs_lat) * np.cos(src.dec) * np.sin(hour_angle)

    # LSR определяется как: Солнце движется со скоростью 20.0 км/с 
    # в направлении RA=18.0h и dec=30.0d в координатах 1900J
    # Относительно объектов вблизи нас в Млечном Пути (не вращение Солнца 
    # относительно галактического центра!)
    lsr_coord = SkyCoord('18h', '30d', frame='fk5', equinox='J1900')
    lsr_coord = lsr_coord.transform_to(FK5(equinox='J2000'))

    lsr_comp = np.array([np.cos(lsr_coord.dec.rad) * np.cos(lsr_coord.ra.rad),
                         np.cos(lsr_coord.dec.rad) * np.sin(lsr_coord.ra.rad),
                         np.sin(lsr_coord.dec.rad)])

    src_comp = np.array([np.cos(src.dec.rad) * np.cos(src.ra.rad),
                         np.cos(src.dec.rad) * np.sin(src.ra.rad),
                         np.sin(src.dec.rad)])

    k = np.array([lsr_comp[0]*src_comp[0], lsr_comp[1]*src_comp[1], lsr_comp[2]*src_comp[2]])
    v_lsr = 20. * np.sum(k, axis=0) * u.km/u.s
    
    geo = -v_spin
    helio = heliocorr
    bary = barycorr
    lsr = barycorr + v_lsr
    vtotal = [geo, helio, bary, lsr]
    
    return vtotal


def get_object_coordinates(object_name: str, retry_on_fail: bool = True) -> tuple:
    """
    Получает координаты объекта из каталога SIMBAD.
    
    Parameters:
    -----------
    object_name : str
        Имя объекта (из FITS заголовка OBJECT)
    retry_on_fail : bool
        Если True и объект не найден, запросить альтернативное имя у пользователя
    
    Returns:
    --------
    tuple : (ra, dec) в градусах, или (None, None) при ошибке
    """
    if not SIMBAD_AVAILABLE:
        logger.error("astroquery не установлен! Установите: pip install astroquery")
        return None, None
    
    try:
        # Запрос к SIMBAD
        result_table = Simbad.query_object(object_name)
        
        if result_table is None or len(result_table) == 0:
            logger.warning(f"Объект '{object_name}' не найден в SIMBAD")
            
            if retry_on_fail:
                print(f"\n❌ Объект '{object_name}' не найден в каталоге SIMBAD")
                alt_name = input("Введите альтернативное имя объекта (или Enter для пропуска): ").strip()
                
                if alt_name:
                    logger.info(f"Повторная попытка с именем: {alt_name}")
                    return get_object_coordinates(alt_name, retry_on_fail=False)
            
            return None, None
        
        # Получить координаты из таблицы результатов
        ra = result_table['ra'].value[0]
        dec = result_table['dec'].value[0]
        coord = SkyCoord(f'{ra} {dec}', unit=(u.degree, u.degree))
        
        # Извлечь координаты в градусах
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
        
        logger.info(f"✓ Координаты '{object_name}': RA={ra_deg:.6f}°, DEC={dec_deg:.6f}°")
        return ra_deg, dec_deg
        
    except Exception as e:
        logger.error(f"Ошибка запроса SIMBAD: {e}")
        return None, None


def get_observation_time(fits_file: Path) -> tuple:
    """
    Извлекает время наблюдения из FITS заголовка.
    
    Parameters:
    -----------
    fits_file : Path
        Путь к FITS файлу
    
    Returns:
    --------
    tuple : (date_obs, exptime) или (None, None)
        date_obs : str - время начала экспозиции (ISO формат)
        exptime : float - время экспозиции в секундах
    """
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            date_obs = header.get('DATE-OBS', None)
            exptime = header.get('EXPTIME', None)
            
            if date_obs is None or exptime is None:
                logger.error(f"Отсутствуют DATE-OBS или EXPTIME в {fits_file.name}")
                return None, None
            
            return date_obs, float(exptime)
    except Exception as e:
        logger.error(f"Ошибка чтения {fits_file}: {e}")
        return None, None


def calculate_mean_time(date_obs: str, exptime: float) -> str:
    """
    Вычисляет среднее время наблюдения.
    
    Parameters:
    -----------
    date_obs : str
        Время начала экспозиции (ISO формат)
    exptime : float
        Время экспозиции (сек)
    
    Returns:
    --------
    str : Среднее время в формате ISOT
    """
    start_time = time.Time(date_obs, format='isot', scale='utc')
    mid_time = start_time + time.TimeDelta(exptime / 2.0, format='sec')
    return mid_time.isot


def apply_barycentric_correction_to_wavelengths(wavelengths: np.ndarray,
                                                barycorr_kms: float) -> np.ndarray:
    """
    Применяет барицентрическую коррекцию к длинам волн.
    
    Использует полную релятивистскую формулу Доплера:
    λ_bary = λ_obs × √[(1 + β)/(1 - β)]
    где β = v_bary/c
    
    Эта формула обеспечивает точность ~0.015 м/с для типичных 
    барицентрических скоростей (~30 км/с), что критично для 
    высокоточной спектроскопии и поиска экзопланет.
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Наблюдаемые длины волн (Å)
    barycorr_kms : float
        Барицентрическая коррекция (км/с)
    
    Returns:
    --------
    np.ndarray : Скорректированные длины волн (Å)
    """
    c_kms = 299792.458  # Скорость света км/с
    beta = barycorr_kms / c_kms
    
    # Релятивистская формула Доплера
    correction_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    
    return wavelengths * correction_factor


def process_single_order(input_fits: Path, 
                        output_fits: Path,
                        barycorr_kms: float) -> bool:
    """
    Обрабатывает один FITS файл порядка.
    
    Структура входного FITS:
    - HDU[0]: Primary с flux данными
    - HDU[1]: таблица с колонками wavelength, flux, error
    
    Parameters:
    -----------
    input_fits : Path
        Входной откалиброванный FITS
    output_fits : Path
        Выходной FITS с барицентрической коррекцией
    barycorr_kms : float
        Барицентрическая коррекция (км/с)
    
    Returns:
    --------
    bool : True если успешно
    """
    try:
        with fits.open(input_fits) as hdul:
            # Прочитать данные из таблицы
            data_table = hdul[1].data
            wavelengths = data_table['wavelength']
            flux = data_table['flux']
            error = data_table['error']
            
            # Применить коррекцию к wavelengths
            corrected_wl = apply_barycentric_correction_to_wavelengths(
                wavelengths, barycorr_kms
            )
            
            # Primary HDU с flux данными (как в save_polarimetry_vector)
            primary = fits.PrimaryHDU(flux)
            
            # Копировать исходный заголовок
            if hdul[0].header is not None:
                for card in hdul[0].header.cards:
                    try:
                        # Пропускаем системные ключи
                        if card.keyword not in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 
                                               'EXTEND', 'CTYPE1', 'CUNIT1', 'CRVAL1', 'CRPIX1', 
                                               'CDELT1', 'CD1_1']:
                            primary.header[card.keyword] = (card.value, card.comment)
                    except (ValueError, KeyError, fits.verify.VerifyError):
                        continue
            
            # Добавить информацию о барицентрической коррекции
            primary.header['BARYCORR'] = (barycorr_kms, 'Barycentric correction (km/s)')
            primary.header['HIERARCH BARY_APPLIED'] = (True, 'Barycentric correction applied')
            
            # WCS для скорректированных длин волн
            wl_step = corrected_wl[1] - corrected_wl[0] if len(corrected_wl) > 1 else 1.0
            primary.header['CTYPE1'] = 'WAVE'
            primary.header['CUNIT1'] = 'Angstrom'
            primary.header['CRVAL1'] = corrected_wl[0]
            primary.header['CRPIX1'] = 1.0
            primary.header['CDELT1'] = wl_step
            primary.header['CD1_1'] = wl_step
            
            # Таблица с данными (для совместимости)
            col_wave = fits.Column(name='wavelength', format='D', unit='Angstrom',
                                  array=corrected_wl)
            col_flux = fits.Column(name='flux', format='D', unit='ADU',
                                  array=flux)
            col_err = fits.Column(name='error', format='D', unit='ADU',
                                 array=error)
            
            table = fits.BinTableHDU.from_columns([col_wave, col_flux, col_err])
            
            # Сохранить
            hdul_out = fits.HDUList([primary, table])
            hdul_out.writeto(output_fits, overwrite=True)
            
        logger.info(f"  ✓ {output_fits.name}")
        return True
        
    except Exception as e:
        logger.error(f"  ! Ошибка обработки {input_fits.name}: {e}")
        return False


def process_calibrated_directory(calibrated_dir: Path,
                                science_file: Path,
                                observatory: dict = OBSERVATORY) -> bool:
    """
    Обрабатывает всю директорию с откалиброванными порядками.
    
    Parameters:
    -----------
    calibrated_dir : Path
        Директория вида CALIBRATED/o123_CRR_bt/
    science_file : Path
        Исходный научный FITS для чтения OBJECT, DATE-OBS
    observatory : dict
        Координаты обсерватории {'lon', 'lat', 'alt'}
    
    Returns:
    --------
    bool : True если успешно
    """
    logger.info(f"\nБарицентрическая коррекция: {calibrated_dir.name}")
    
    # 1. Прочитать OBJECT из заголовка
    try:
        with fits.open(science_file) as hdul:
            object_name = hdul[0].header.get('OBJECT', None)
            
        if not object_name:
            logger.error(f"! OBJECT не найден в {science_file.name}")
            return False
            
        logger.info(f"  Объект: {object_name}")
    except Exception as e:
        logger.error(f"! Ошибка чтения {science_file.name}: {e}")
        return False
    
    # 2. Получить координаты объекта
    ra, dec = get_object_coordinates(object_name)
    if ra is None or dec is None:
        logger.error(f"! Не удалось определить координаты для '{object_name}'")
        return False
    
    # 3. Получить время наблюдения
    date_obs, exptime = get_observation_time(science_file)
    if date_obs is None:
        return False
    
    mean_time = calculate_mean_time(date_obs, exptime)
    logger.info(f"  Время (среднее): {mean_time}")
    logger.info(f"  Экспозиция: {exptime} сек")
    
    # 4. Вычислить барицентрическую коррекцию
    try:
        corrections = doppler_corrections(
            ra=ra,
            dec=dec,
            mean_time=mean_time,
            obs_lat=observatory['lat'],
            obs_lon=observatory['lon'],
            obs_alt=observatory['alt']
        )
        
        barycorr_kms = corrections[2].value  # bary correction
        
        logger.info(f"  Барицентрическая коррекция: {barycorr_kms:.3f} км/с")
        
    except Exception as e:
        logger.error(f"! Ошибка расчета коррекции: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Применить к каждому order файлу
    order_files = sorted(calibrated_dir.glob('order_*_calibrated.fits'))
    
    if len(order_files) == 0:
        logger.warning(f"! Нет откалиброванных order файлов в {calibrated_dir}")
        return False
    
    logger.info(f"  Найдено {len(order_files)} order файлов")
    
    success_count = 0
    for order_file in order_files:
        output_file = order_file.parent / order_file.name.replace(
            '_calibrated.fits', '_calibrated_barycorr.fits'
        )
        
        if process_single_order(order_file, output_file, barycorr_kms):
            success_count += 1
    
    logger.info(f"  Обработано: {success_count}/{len(order_files)}")
    
    return success_count == len(order_files)


def process_vector_files(final_dir: Path,
                        science_file: Path,
                        observatory: dict = OBSERVATORY) -> bool:
    """
    Обрабатывает финальные векторы поляриметрии.
    
    Ищет файлы вида: {stem}_1.fits и {stem}_2.fits в директории FINAL
    и применяет барицентрическую коррекцию.
    
    Parameters:
    -----------
    final_dir : Path
        Директория FINAL с векторами поляриметрии
    science_file : Path
        Исходный научный FITS для чтения OBJECT, DATE-OBS
    observatory : dict
        Координаты обсерватории
    
    Returns:
    --------
    bool : True если успешно
    """
    logger.info(f"\nБарицентрическая коррекция векторов: {science_file.stem}")
    
    # 1. Прочитать OBJECT из заголовка
    try:
        with fits.open(science_file) as hdul:
            object_name = hdul[0].header.get('OBJECT', None)
            
        if not object_name:
            logger.error(f"! OBJECT не найден в {science_file.name}")
            return False
            
        logger.info(f"  Объект: {object_name}")
    except Exception as e:
        logger.error(f"! Ошибка чтения {science_file.name}: {e}")
        return False
    
    # 2. Получить координаты объекта
    ra, dec = get_object_coordinates(object_name)
    if ra is None or dec is None:
        logger.error(f"! Не удалось определить координаты для '{object_name}'")
        return False
    
    # 3. Получить время наблюдения
    date_obs, exptime = get_observation_time(science_file)
    if date_obs is None:
        return False
    
    mean_time = calculate_mean_time(date_obs, exptime)
    logger.info(f"  Время (среднее): {mean_time}")
    logger.info(f"  Экспозиция: {exptime} сек")
    
    # 4. Вычислить барицентрическую коррекцию
    try:
        corrections = doppler_corrections(
            ra=ra,
            dec=dec,
            mean_time=mean_time,
            obs_lat=observatory['lat'],
            obs_lon=observatory['lon'],
            obs_alt=observatory['alt']
        )
        
        barycorr_kms = corrections[2].value  # bary correction
        
        logger.info(f"  Барицентрическая коррекция: {barycorr_kms:.3f} км/с")
        
    except Exception as e:
        logger.error(f"! Ошибка расчета коррекции: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Найти векторы поляриметрии
    stem = science_file.stem
    vector_files = []
    
    for suffix in ['_1.fits', '_2.fits']:
        vector_file = final_dir / f"{stem}{suffix}"
        if vector_file.exists():
            vector_files.append(vector_file)
    
    if len(vector_files) == 0:
        logger.warning(f"! Не найдены векторы для {stem} в {final_dir}")
        return False
    
    logger.info(f"  Найдено {len(vector_files)} векторов")
    
    # 6. Применить коррекцию к каждому вектору
    success_count = 0
    for vector_file in vector_files:
        output_file = vector_file.parent / vector_file.name.replace(
            '.fits', '_barycorr.fits'
        )
        
        if process_single_order(vector_file, output_file, barycorr_kms):
            success_count += 1
    
    logger.info(f"  Обработано векторов: {success_count}/{len(vector_files)}")
    
    return success_count == len(vector_files)


if __name__ == '__main__':
    # Тестовый запуск
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    print("="*80)
    print("Тестовый модуль барицентрической коррекции")
    print("="*80)
    
    # Пример использования для order файлов
    calibrated_dir = Path('CALIBRATED/o123_CRR_bt')
    science_file = Path('temp/o123_CRR_bt.fits')
    
    if calibrated_dir.exists() and science_file.exists():
        print("\nТест для order файлов:")
        process_calibrated_directory(calibrated_dir, science_file)
    
    # Пример использования для векторов
    final_dir = Path('FINAL')
    
    if final_dir.exists() and science_file.exists():
        print("\nТест для векторов:")
        process_vector_files(final_dir, science_file)
    else:
        print("\nДля тестирования укажите существующие пути к файлам")
        print(f"final_dir = {final_dir}")
        print(f"science_file = {science_file}")
