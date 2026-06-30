import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

def median_filtrs(source_name, f):
    # Загружаем изображение в массив NumPy (высота, ширина, каналы)
    img = np.array(Image.open(source_name))
    h, w, _ = img.shape

    for _ in range(f):
        # Копия для записи результата текущего прохода
        new_img = img.copy()

        # Обрабатываем каждый канал отдельно
        for ch in range(3):
            # Три вертикальных среза: строки [0:h-2], [1:h-1], [2:h]
            top = img[0:h-2, :, ch]
            mid = img[1:h-1, :, ch]
            bot = img[2:h, :, ch]

            # Стек размером (3, h-2, w), медиана по первому измерению
            stacked = np.stack([top, mid, bot], axis=0)
            median = np.median(stacked, axis=0)  # (h-2, w)

            # Записываем медиану в первые h-2 строк
            new_img[0:h-2, :, ch] = median

        # Переходим к следующей итерации с обновлённым изображением
        img = new_img

    # Сохраняем результат
    Image.fromarray(img.astype('uint8')).save("median_filtrs.jpg", "JPEG")

def filter_matrixSvertki(source_name, result_name, div):
    img = np.array(Image.open(source_name))
    h, w, c = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    out[:] = img
    out[:, -1, :] = 0
    out[-1, :, :] = 0

    kernel = np.array([[0.5, 0.75, 0.5],
                       [0.75, 1.0, 0.75],
                       [0.5, 0.75, 0.5]], dtype=np.float64)

    rows = slice(2, w - 1)
    cols = slice(2, h - 1)

    for ch in range(c):
        center = img[rows, cols, ch]
        top_left = img[rows - 1, cols - 1, ch]
        top_mid = img[rows - 1, cols, ch]
        top_right = img[rows - 1, cols + 1, ch]
        mid_left = img[rows, cols - 1, ch]
        mid_right = img[rows, cols + 1, ch]
        bottom_left = img[rows + 1, cols - 1, ch]
        bottom_mid = img[rows + 1, cols, ch]
        bottom_right = img[rows + 1, cols + 1, ch]

        conv = (top_left * 0.5 + top_mid * 0.75 + top_right * 0.5 +
                mid_left * 0.75 + center * 1.0 + mid_right * 0.75 +
                bottom_left * 0.5 + bottom_mid * 0.75 + bottom_right * 0.5) / div

        out[rows, cols, ch] = np.clip(conv, 0, 255).astype(np.uint8)

    Image.fromarray(out.astype('uint8')).save(result_name, "JPEG"

def filter_Kuwahara(source_name, result_name):
    img = np.array(Image.open(source_name))
    h, w, c = img.shape
    out = img.copy()

    # Область обрабатываемых пикселей (i от 4 до w-4, j от 4 до h-4)
    rows = slice(4, h - 3)   # j
    cols = slice(4, w - 3)   # i
    n_rows = rows.stop - rows.start
    n_cols = cols.stop - cols.start
    total = n_rows * n_cols

    # Функция для получения блока 3x3 со смещениями (dr, dc) относительно (i,j)
    def get_block(dr_start, dr_end, dc_start, dc_end):
        # dr_start..dr_end-1, dc_start..dc_end-1
        # Возвращаем массив (total, 9, c)
        slices = []
        for dr in range(dr_start, dr_end):
            for dc in range(dc_start, dc_end):
                slices.append(img[rows + dr, cols + dc])
        # объединяем по оси 1 (9 срезов)
        return np.stack(slices, axis=1)

    # Четыре блока: верхний-левый, верхний-правый, нижний-левый, нижний-правый
    blocks = [
        get_block(-3, 0, -3, 0),   # блок1: rows-3..-1, cols-3..-1
        get_block(1, 4, -3, 0),    # блок2: rows+1..+3, cols-3..-1
        get_block(-3, 0, 1, 4),    # блок3: rows-3..-1, cols+1..+3
        get_block(1, 4, 1, 4)      # блок4: rows+1..+3, cols+1..+3
    ]

    # Для каждого блока вычисляем среднее и дисперсию по 9 значениям
    means = []
    variances = []
    for b in blocks:
        mean = np.mean(b, axis=1)          # (total, c)
        var = np.var(b, axis=1)            # (total, c)
        means.append(mean)
        variances.append(var)

    # Объединяем в массивы (total, 4, c)
    means_all = np.stack(means, axis=1)
    vars_all = np.stack(variances, axis=1)

    # Для каждого канала выбираем индекс блока с минимальной дисперсией
    indices = np.argmin(vars_all, axis=1)   # (total, c)

    # Заполняем выходное изображение для обработанной области
    for ch in range(c):
        idx = indices[:, ch]                # (total,)
        # Выбираем соответствующие средние
        selected = means_all[np.arange(total), idx, ch]
        # Записываем в out по срезам rows, cols, ch
        out[rows, cols, ch] = selected.reshape(n_rows, n_cols)

    Image.fromarray(out.astype('uint8')).save(result_name, "JPEG")

def filter_Kuan(source_name, result_name, ch_vib):
    # Загрузка изображения
    img = np.array(Image.open(source_name)).astype(np.float64)
    h, w, c = img.shape

    # Выходное изображение – копия оригинала
    out = img.copy()

    # Если изображение слишком мало, фильтр не применяется
    if h < 7 or w < 7:
        Image.fromarray(out.astype('uint8')).save(result_name, "JPEG")
        return

    # Глобальные дисперсии для каждого канала
    disp_obsh = np.var(img, axis=(0, 1))  # массив (c,)

    # Внутренняя область (центры окон 7x7, исключая 4 пикселя по краям)
    rows = slice(4, h - 3)   # индексы 4..h-4 включительно
    cols = slice(4, w - 3)   # индексы 4..w-4 включительно
    n_rows = h - 7
    n_cols = w - 7

    # Параметр CU (скаляр для каждого канала, но зависит от ch_vib и дисперсии)
    # В исходном коде: CU = 1 / sqrt(ch_vib) * dispObsh
    # Если CU == 0, то CU = 1 (но это маловероятно)
    sqrt_ch = np.sqrt(ch_vib)
    CU = (1.0 / sqrt_ch) * disp_obsh
    CU = np.where(CU == 0, 1.0, CU)   # защита от деления на ноль

    # Обработка каждого канала
    for ch in range(c):
        # Исходный канал
        img_ch = img[:, :, ch]

        # Скользящие окна 7x7
        windows = sliding_window_view(img_ch, (7, 7))  # форма (h-6, w-6, 7, 7)

        # Локальное среднее и дисперсия для всех окон
        mean_all = np.mean(windows, axis=(2, 3))      # (h-6, w-6)
        var_all = np.var(windows, axis=(2, 3))        # (h-6, w-6)

        # Берём только окна, соответствующие внутренним центрам
        # Индексы окон: r = центр - 3, центр от 4 до h-4 => r от 1 до h-7 = (h-6)-1
        # slice(1, h-6) даст индексы 1..h-7, что соответствует центрам 4..h-4
        mean_loc = mean_all[1:h-6, 1:w-6]   # (n_rows, n_cols)
        var_loc = var_all[1:h-6, 1:w-6]     # (n_rows, n_cols)

        # Значения пикселей в центрах
        PC = img[rows, cols, ch]            # (n_rows, n_cols)

        # Локальное среднее и дисперсия
        LM = mean_loc
        LV = var_loc

        # Защита от деления на ноль
        PC = np.where(PC == 0, 1.0, PC)
        LM = np.where(LM == 0, 1.0, LM)
        LV = np.where(LV == 0, 1.0, LV)

        # Вычисление CI = sqrt(LV) / LM * disp_obsh[ch]
        # В исходном коде: CI = math.sqrt(LV) / LM * dispObsh (без скобок)
        CI = (np.sqrt(LV) / LM) * disp_obsh[ch]
        CI = np.where(CI == 0, 1.0, CI)

        # Коэффициент K
        # CU – скаляр для канала, поэтому используем его как число
        cu2 = CU[ch] * CU[ch]
        # ci2 = CI^2
        ci2 = CI * CI
        K = (1 - (cu2 / ci2)) / (1 + cu2)
        K = np.where(K == 0, 1.0, K)

        # Результирующее значение
        R = PC * K + LM * (1 - K)
        R = np.where(R == 0, 1.0, R)

        # Обрезка и запись в выходной массив
        out[rows, cols, ch] = np.clip(R, 0, 255)

    # Сохранение результата
    Image.fromarray(out.astype('uint8')).save(result_name, "JPEG")
