import cv2
import numpy as np
import pprint

calib_file = "../result/cam_stereo.yml"
img_width = 1280
img_height = 800 
fps = 30
use_wls_filter =True 
wls_lambda = 80000
wls_sigma = 1.5
sgbm_num_disp = 80
sgbm_block_size = 5
sgbm_uniqueness_ratio = 10
img_size = (img_width, img_height)
min_disp = 0
p1=8 * 3 * sgbm_block_size**2
p2=32 * 3 * sgbm_block_size**2
sgbm_uniqueness_ratio=10
sgbm_speckle_ws=100
sgbm_speckle_range = 32
median_blur_size=0

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


calib_params = read_yaml_opencv("../result/cam_stereo.yml")
K1 = calib_params["K1"]
D1 = calib_params["D1"]
K2 = calib_params["K2"]
D2 = calib_params["D2"]
R1 = calib_params["R1"]
P1 = calib_params["P1"]
R2 = calib_params["R2"]
P2 = calib_params["P2"]
Q = calib_params["Q"]
# pprint.pprint(f"{K1=}; {D1=}; {K2=}; {D2=}; {R1=}; {P1=}; {R2=}; {P2=}; {Q=}")

map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32F)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32F)

stereo_bm_left = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=sgbm_num_disp,
    blockSize=sgbm_block_size,
    P1=p1,
    P2=p2,
    disp12MaxDiff=2, # Обычно 1, можно попробовать 0-2
    uniquenessRatio=sgbm_uniqueness_ratio,
    speckleWindowSize=sgbm_speckle_ws,
    speckleRange=sgbm_speckle_range,
    preFilterCap=63, # Обычно 63
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # или STEREO_SGBM_MODE_HH для большей точности (медленнее)
)

stereo_bm_right = cv2.ximgproc.createRightMatcher(stereo_bm_left) 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_bm_left)
wls_filter.setLambda(wls_lambda)
wls_filter.setSigmaColor(wls_sigma)

if median_blur_size > 0:
    if median_blur_size % 2 == 0:
        median_blur_size += 1 # Медианный фильтр требует нечетного размера ядра

left_cam=cv2.VideoCapture(2, cv2.CAP_V4L2)
right_cam=cv2.VideoCapture(4, cv2.CAP_V4L2)

while True:
    ret_left, im_left = left_cam.read()
    ret_right, im_right = right_cam.read()

    im_left = cv2.resize(im_left, img_size)
    im_right = cv2.resize(im_right, img_size)

    imgU1 = cv2.remap(im_left, map1_l, map2_l, cv2.INTER_LINEAR)
    imgU2 = cv2.remap(im_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # Конвертируем в градации серого для вычисления диспаратности
    grayU1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    grayU2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)

    disparity_left_raw = stereo_bm_left.compute(grayU1, grayU2)
    disparity_right_raw = stereo_bm_right.compute(grayU2, grayU1) # Правый матчер
    filtered_disparity_map = wls_filter.filter(disparity_left_raw, grayU1, disparity_map_right=disparity_right_raw)

    normalized_grayscale_disparity = np.zeros((img_height, img_width), dtype=np.uint8)
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

    # ИСПРАВЛЕНО: imgU1 и imgU2 уже BGR, конвертация не нужна
    combined_undistorted_rectified = np.hstack((imgU1, imgU2))

    depth_map_display = cv2.resize(normalized_disparity_map, 
                                       (combined_undistorted_rectified.shape[1], combined_undistorted_rectified.shape[0]), 
                                       interpolation=cv2.INTER_AREA)

    cv2.imshow("Undistorted & Rectified Stereo Stream", cv2.resize(combined_undistorted_rectified, (1080, 480)))
    cv2.imshow("Depth Map (Normalized)", cv2.resize(depth_map_display, (1080, 480)))

    # if ret_left:
    #     cv2.imshow("Left", im_left)
    # if ret_right:
    #     cv2.imshow("Right", im_right)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()