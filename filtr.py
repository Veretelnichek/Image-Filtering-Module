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
