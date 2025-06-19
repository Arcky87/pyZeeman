from pathlib import Path
import numpy as np
from astropy.io import fits

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