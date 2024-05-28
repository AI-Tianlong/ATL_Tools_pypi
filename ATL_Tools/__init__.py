"""
ATL_Tools
========
包含了 `ATL_path`和`ATL_gdal`两大工具

用法：
----
    >>> # 在开头复制这一句
    >>> from ATL_Tools import mkdir_or_exist, find_data_list # 

    >>> from ATL_Tools.ATL_gdal import (
            read_img_to_array_with_info, # ✔读取影像为数组并返回信息
            read_img_to_array,  # ✔读取影像为数组
            save_ds_to_tif,     # ✔将GDAL dataset数据格式写入tif保存
            save_array_to_tif,  # 将数组格式写入tif保存
            read_img_get_geo,   # ✔计算影像角点的地理坐标或投影坐标
            ds_get_img_geo,     # 读取dataset格式，计算影像角点的地理坐标或投影坐标
            pix_to_geo,         # 计算影像某一像素点的地理坐标或投影坐标
            geo_to_pix,         # 根据GDAL的仿射变换参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
            Mosaic_all_imgs,    # ✔将指定路径文件夹下的tif影像全部镶嵌到一张影像上
            Mosaic_2img_to_one, # 将两幅影像镶嵌至同一幅影像
            raster_overlap,     # 两个栅格数据集取重叠区或求交集（仅测试方形影像）
            crop_tif_with_json_zero, # ✔将带有坐标的图像按照json矢量进行裁切,无数据区域为0
            crop_tif_with_json_nan,  # ✔将带有坐标的图像按照json矢量进行裁切,无数据区域为nan
            Merge_multi_json,   # ✔将多个小的json合并为一个大的json,
            resample_image,      # ✔使用GDAL对图像进行重采样
            shp_to_geojson,      # ✔将shp文件转为geojson文件
            clip_big_image, # 将大图裁切为小图，支持重叠
        )

---
示例程序：
----       
示例1-根据json批量裁切影像根据矢量批量裁切影像

    >>> from ATL_Tools import mkdir_or_exist, find_data_list
    >>> from ATL_Tools.ATL_gdal import crop_tif_with_json_nan
    >>> from tqdm import tqdm
    >>> import os 

    >>> img_path_all = '../推理出的结果_24类_RGB/'
    >>> output_path_all = '../推理出的结果_24类_RGB_crop'
    >>> json_path_all = '../要推理的json/'
    >>> mkdir_or_exist(output_path_all)
    >>> img_list = find_data_list(img_path_all, suffix='.tif')


    >>> for img_path in tqdm(img_list, colour='Green'):

    >>>     img_output_path = os.path.join(output_path_all, os.path.basename(img_path))
    >>>     json_path = os.path.join(json_path_all, os.path.basename(img_path).split('_')[-1].replace('.tif', '.json'))
    >>>     print(f'正在裁切: {img_output_path},json: {json_path}')
    >>>     crop_tif_with_json_nan(img_path, img_output_path, json_path) 

"""
from .ATL_path import *
# from .ATL_gdal import * # 为了可以不安装gdal使用 `conda install gdal`
