import numpy as np
import cv2
import depthai as dai
import os
import argparse
from fps_counter import FPSCounter

# --- 1. Настройка параметров через аргументы командной строки ---
parser = argparse.ArgumentParser(description="Потоковая коррекция дисторсии, ректификация и построение карты глубины с камеры DepthAI.")
parser.add_argument("--calib_file", type=str, default="cam_stereo.yml",
                    help="Путь к файлу с параметрами стереокалибровки (.yaml).")
parser.add_argument("--fps", type=int, default=30,
                    help="Частота кадров для монохромных камер.")
parser.add_argument("--img_width", type=int, required=True,
                    help="Ширина изображения, для которой была выполнена калибровка (например, 1280).")
parser.add_argument("--img_height", type=int, required=True,
                    help="Высота изображения, для которой была выполнена калибровка (например, 800).")

# Параметры SGBM
parser.add_argument("--sgbm_num_disp", type=int, default=96, # Должно быть кратно 16
                    help="Количество смещений для SGBM.")
parser.add_argument("--sgbm_block_size", type=int, default=5, # Нечетное 3-11
                    help="Размер блока для SGBM.")
parser.add_argument("--sgbm_p1", type=int, default=8 * 3 * 5**2, # 8 * 3 * blockSize**2
                    help="Параметр P1 для SGBM.")
parser.add_argument("--sgbm_p2", type=int, default=32 * 3 * 5**2, # 32 * 3 * blockSize**2
                    help="Параметр P2 для SGBM.")
parser.add_argument("--sgbm_uniqueness_ratio", type=int, default=10,
                    help="Параметр uniquenessRatio для SGBM.")
parser.add_argument("--sgbm_speckle_ws", type=int, default=100,
                    help="Параметр speckleWindowSize для SGBM.")
parser.add_argument("--sgbm_speckle_range", type=int, default=32,
                    help="Параметр speckleRange для SGBM.")

# Параметры WLS фильтра (требует opencv-contrib-python)
parser.add_argument("--use_wls_filter", action="store_true",
                    help="Использовать фильтр взвешенных наименьших квадратов (WLS filter) для постобработки карты глубины.")
parser.add_argument("--wls_lambda", type=float, default=80000.0,
                    help="Параметр lambda для WLS фильтра (сглаживание).")
parser.add_argument("--wls_sigma", type=float, default=1.5,
                    help="Параметр sigmaColor для WLS фильтра (чувствительность к краям).")

# Дополнительная постобработка
parser.add_argument("--median_blur_size", type=int, default=0, # 0 = не применять
                    help="Размер ядра медианного фильтра для постобработки (нечетное число, >0 для активации).")

args = parser.parse_args()

calibration_file = args.calib_file
img_size = (args.img_width, args.img_height)


# --- Вспомогательная функция для чтения YAML из OpenCV ---
def read_yaml_opencv(filepath):
    """Читает параметры калибровки из YAML-файла, как в C++ примере."""
    cv_file = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    if not cv_file.isOpened():
        raise FileNotFoundError(f"Не удалось открыть файл калибровки: {filepath}")

    params = {}
    
    keys = ["K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2", "Q", "E", "F"] 
    
    for key_name in keys:
        node = cv_file.getNode(key_name)
        if not node.empty():
            if key_name == "T":
                if node.isSeq():
                    t_list = [node.at(i).real() for i in range(node.size())]
                    params[key_name] = np.array(t_list, dtype=np.float64).reshape(3, 1)
                else:
                    params[key_name] = node.mat()
            else:
                params[key_name] = node.mat()
        else:
            print(f"Предупреждение: Параметр '{key_name}' не найден в файле {filepath}.")
            params[key_name] = None 
    cv_file.release()
    return params

# --- 2. Загрузка параметров калибровки ---
print(f"Загрузка параметров калибровки из '{calibration_file}'...")
if not os.path.exists(calibration_file):
    print(f"ОШИБКА: Файл калибровки '{calibration_file}' не найден!")
    exit()

calib_params = None
try:
    calib_params = read_yaml_opencv(calibration_file)
    
    # Используем правильные ключи из calib_params (K1, D1 и т.д.)
    K1 = calib_params["K1"]
    D1 = calib_params["D1"]
    K2 = calib_params["K2"]
    D2 = calib_params["D2"]
    R1 = calib_params["R1"]
    P1 = calib_params["P1"]
    R2 = calib_params["R2"]
    P2 = calib_params["P2"]
    Q = calib_params["Q"] # Q не используется для ремаппинга или SGBM, но загружено
    
    # Проверка на наличие всех необходимых параметров
    if any(param is None for param in [K1, D1, K2, D2, R1, P1, R2, P2]):
        raise ValueError("Один или несколько обязательных параметров калибровки не были загружены или являются пустыми.")

except FileNotFoundError as e:
    print(f"ОШИБКА: {e}")
    exit()
except Exception as e:
    print(f"ОШИБКА при чтении или обработке файла калибровки: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("Параметры калибровки успешно загружены.")
print(f"Размер изображений для калибровки (из аргументов): {img_size}")

# --- 3. Инициализация карт для ремаппинга ---
# CV_32F - тип данных для карт, как в C++ примере
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32F)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32F)
print("Карты для ремаппинга инициализированы.")

# --- 4. Настройка Stereo Matcher ---
min_disp = 0
num_disp = args.sgbm_num_disp
block_size = args.sgbm_block_size
p1 = args.sgbm_p1
p2 = args.sgbm_p2
uniqueness_ratio = args.sgbm_uniqueness_ratio
speckle_ws = args.sgbm_speckle_ws
speckle_range = args.sgbm_speckle_range
median_blur_size = args.median_blur_size

# Проверки параметров SGBM
if block_size % 2 == 0:
    print(f"ПРЕДУПРЕЖДЕНИЕ: SGBM blockSize ({block_size}) должен быть нечетным. Устанавливаю {block_size + 1}.")
    block_size += 1
if num_disp % 16 != 0:
    print(f"ПРЕДУПРЕЖДЕНИЕ: SGBM numDisparities ({num_disp}) должен быть кратным 16. Устанавливаю {num_disp - (num_disp % 16)}.")
    num_disp -= (num_disp % 16)

stereo_bm_left = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=p1,
    P2=p2,
    disp12MaxDiff=1, # Обычно 1, можно попробовать 0-2
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_ws,
    speckleRange=speckle_range,
    preFilterCap=63, # Обычно 63
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # или STEREO_SGBM_MODE_HH для большей точности (медленнее)
)
print(f"Stereo Matcher (StereoSGBM) инициализирован с numDisparities={num_disp}, blockSize={block_size}")

wls_filter = None
if args.use_wls_filter:
    # Убедитесь, что opencv-contrib-python установлен для ximgproc
    try:
        stereo_bm_right = cv2.ximgproc.createRightMatcher(stereo_bm_left) 
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_bm_left)
        wls_filter.setLambda(args.wls_lambda)
        wls_filter.setSigmaColor(args.wls_sigma)
        print(f"WLS фильтр инициализирован (Lambda: {args.wls_lambda}, Sigma: {args.wls_sigma}).")
    except AttributeError:
        print("ОШИБКА: Для использования WLS фильтра необходим 'opencv-contrib-python'. Установите его: pip install opencv-contrib-python")
        print("WLS фильтр будет отключен.")
        args.use_wls_filter = False # Отключаем, если модуль не найден

if median_blur_size > 0:
    if median_blur_size % 2 == 0:
        median_blur_size += 1 # Медианный фильтр требует нечетного размера ядра
    print(f"Медианный фильтр для постобработки будет применен с ядром размера {median_blur_size}.")


# --- 5. Конфигурация пайплайна DepthAI ---
def create_pipeline(img_size_calib_h, fps):
    pipeline = dai.Pipeline()

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)

    xoutLeft.setStreamName("left")
    xoutRight.setStreamName("right")

    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    
    # Установка разрешения, соответствующего калибровке
    if img_size_calib_h == 400:
        resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
    elif img_size_calib_h == 720:
        resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    elif img_size_calib_h == 800:
        resolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
    else:
        print(f"Предупреждение: Разрешение {img_size[0]}x{img_size[1]} из аргументов не соответствует стандартным разрешениям DepthAI.")
        print(f"Будет использовано THE_720_P. Изображение будет изменено до {img_size[0]}x{img_size[1]} перед ремаппингом.")
        resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P # Fallback
        # Note: Если resolution DepthAI не совпадет с img_size, то cv2.resize будет работать.

    monoLeft.setResolution(resolution)
    monoRight.setResolution(resolution)
    monoLeft.setFps(fps)
    monoRight.setFps(fps)

    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)

    return pipeline

# --- 6. Основной цикл работы с устройством ---
print("\n--- ЗАПУСК ПОТОКА СКОРРЕКТИРОВАННЫХ ИЗОБРАЖЕНИЙ И КАРТЫ ГЛУБИНЫ ---")
pipeline = create_pipeline(img_size[1], args.fps) # img_size[1] - это высота


fps=FPSCounter()
with dai.Device(pipeline) as device:
    # device.setIrLaserDotProjectorBrightness(800) #laser
    print("Камера OAK подключена. Нажмите 'q' для выхода.")
    print("Окна: Скорректированный стереопоток (без дисторсии и выпрямленный), Карта глубины.")

    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        fps.update()
        print(fps.get_fps_text())
        inLeft = qLeft.get()
        inRight = qRight.get()
        
        frameL = inLeft.getCvFrame() # Получаем монохромный кадр
        frameR = inRight.getCvFrame() # Получаем монохромный кадр

        # Проверяем и изменяем размер кадра, если он не соответствует размеру калибровки
        current_img_size_stream = frameL.shape[::-1] # (width, height)
        if current_img_size_stream != img_size:
            if current_img_size_stream[0] != img_size[0] or current_img_size_stream[1] != img_size[1]:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Текущий размер камеры {current_img_size_stream} не соответствует размеру калибровки {img_size}. Изменение размера.", end='\r')
                frameL = cv2.resize(frameL, img_size)
                frameR = cv2.resize(frameR, img_size)
        
        # Применение коррекции дисторсии и ректификации
        imgU1 = cv2.remap(frameL, map1_l, map2_l, cv2.INTER_LINEAR)
        imgU2 = cv2.remap(frameR, map1_r, map2_r, cv2.INTER_LINEAR)

        # --- Расчет карты глубины ---
        # Для Stereo SGBM используем полные выпрямленные изображения (imgU1, imgU2),
        # так как ROI1/ROI2 отсутствуют в вашем файле калибровки и C++ код их не использовал для обрезки
        disparity_left_raw = stereo_bm_left.compute(imgU1, imgU2)

        filtered_disparity_map = None
        if args.use_wls_filter:
            disparity_right_raw = stereo_bm_right.compute(imgU2, imgU1) # Правый матчер
            filtered_disparity_map = wls_filter.filter(disparity_left_raw, imgU1, disparity_map_right=disparity_right_raw)
        else:
            filtered_disparity_map = disparity_left_raw.astype(np.float32) / 16.0 
            if median_blur_size > 0:
                filtered_disparity_map = cv2.medianBlur(filtered_disparity_map, median_blur_size)

        # --- Нормализация для визуализации ---
        normalized_grayscale_disparity = np.zeros_like(imgU1, dtype=np.uint8)
        valid_mask = (filtered_disparity_map > 0) 

        if np.any(valid_mask):
            min_disp_val = filtered_disparity_map[valid_mask].min()
            max_disp_val = filtered_disparity_map[valid_mask].max()
            
            if max_disp_val > min_disp_val:
                scaled_valid_disparity = np.interp(filtered_disparity_map[valid_mask], 
                                                     (min_disp_val, max_disp_val), 
                                                     (0, 255))
                normalized_grayscale_disparity[valid_mask] = scaled_valid_disparity.astype(np.uint8)
            else: 
                normalized_grayscale_disparity[valid_mask] = 127 

        normalized_disparity_map = cv2.applyColorMap(normalized_grayscale_disparity, cv2.COLORMAP_JET)

        # --- Подготовка для отображения ---
        # Кадры imgU1 и imgU2 уже одного размера (img_size)
        display_imgU1_bgr = cv2.cvtColor(imgU1, cv2.COLOR_GRAY2BGR)
        display_imgU2_bgr = cv2.cvtColor(imgU2, cv2.COLOR_GRAY2BGR)
        combined_undistorted_rectified = np.hstack((display_imgU1_bgr, display_imgU2_bgr))
        
        # Изменяем размер карты глубины, чтобы она соответствовала высоте склеенных кадров
        depth_map_display = cv2.resize(normalized_disparity_map, 
                                       (combined_undistorted_rectified.shape[1], combined_undistorted_rectified.shape[0]), 
                                       interpolation=cv2.INTER_AREA)

        # --- Отображение окон ---
        cv2.imshow("Undistorted & Rectified Stereo Stream", cv2.resize(combined_undistorted_rectified, (1080, 480)))
        cv2.imshow("Depth Map (Normalized)", cv2.resize(depth_map_display, (1080, 480)))


        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Выход из потока.")
            break

cv2.destroyAllWindows()
print("Поток завершен.")