# ATL_Tools 使用指南
## 0. 简介
ATL_Tools 是一个由 [AI-Tianlong【GitHub】](https://github.com/AI-Tianlong)开发的工具集合，包含一些便利的小工具。
如果您有新的模块添加，或者对现有模块有改进意见，欢迎提交 PR 至  [ATL_Tools_pypi 【GitHub Repo】](https://github.com/AI-Tianlong/ATL_Tools_pypi).
## 1. 文件夹/数据绝对路径搜索工具
加载方式:

```python
from ATL_Tools import mkdir_or_exist, find_data_list
```

## 2. 遥感图像处理工具:
加载方式:
```python
from ATL_Tools.ATL_gdal import (
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
    resample_image      # ✔使用GDAL对图像进行重采样
                        )
```
## 3. 使用方法

### 3.1 ATL_path 文件夹工具

使用程序示例:  
1. 创建文件夹
    ```python
    from ATL_Tools import mkdir_or_exist, find_data_list
    #创建文件夹
    mkdir_or_exist('新文件夹名称')
    #获取文件夹内所有后缀为.jpg的文件绝对路径
    ```
2. 获取文件夹内所有后缀为 `.jpg` 的文件绝对路径
    ```python
    img_lists = find_data_list(img_root_path='数据集文件夹路径', suffix ='.jpg')
    ```
### 3.2 ATL_gdal 遥感图像处理工具
使用程序示例：
1. 根据矢量批量裁切影像
    ```python
    from ATL_Tools import mkdir_or_exist, find_data_list
    from ATL_Tools.ATL_gdal import crop_tif_with_json_nan
    from tqdm import tqdm
    import os 

    img_path_all = '../推理出的结果_24类_RGB/'
    output_path_all = '../推理出的结果_24类_RGB_crop'
    json_path_all = '../要推理的json/'
    mkdir_or_exist(output_path_all)
    img_list = find_data_list(img_path_all, suffix='.tif')


    for img_path in tqdm(img_list, colour='Green'):

        img_output_path = os.path.join(output_path_all, os.path.basename(img_path))
        json_path = os.path.join(json_path_all, os.path.basename(img_path).split('_')[-1].replace('.tif', '.json'))
        print(f'正在裁切: {img_output_path},json: {json_path}')
        crop_tif_with_json_nan(img_path, img_output_path, json_path)
    ```

## 4. 版本更新日志
- 2023-12-06 v1.0.2 修复README中显示问题。
- 2023-12-06 v1.0.3 修改项目名称为ATL_Tools。
- 2024-04-03 v1.0.6 增加ATL_gdal模块，用于处理遥感影像。
- 2024-04-09 v1.0.7 修复ATL_gdal模块中对于ATL_path的引用，`__init__.py` 注释掉`from ATL_gdal import *`, 可不安装gdal使用ATL_Tools
- 2024-04-16 v1.0.8 修复 `ValueError: cannot convert float NaN to integer` in ATL_gdal Line 371
- 2024-04-16 v1.0.9 修复 修复`Mosaic_all_images()`对于mosaic RGB uint8标签的支持，优化`find_data_list()`函数显示，优化`_init_.py`, 优化`Readme.md`显示
- 2024-04-16 v1.1.0 pypi页面增加`ATL_Tools`Github贡献地址。
- 2024-04-16 v1.1.1 `crop_tif_with_json_nan()`增加可选参数`add_alpha_chan: bool`控制是否为RGB标签增加 alpha 通道
- 2024-04-16 v1.1.2 修复 ATL_gdal Line 397 变量使用错误
- 2024-04-18 v1.1.3 修复 ATL_gdal Mosaic中对float32图像背景设置为nan的支持
