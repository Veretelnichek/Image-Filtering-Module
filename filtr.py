import numpy as np
from PIL import Image

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
