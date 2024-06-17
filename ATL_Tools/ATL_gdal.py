
"""
ATL_GDAL
---
包含了使用`GDAL`对遥感图像进行处理的一些工具

用法：
----
    >>> # 在开头复制这一句
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
            crop_tif_with_json_nan,  # ✔将带有坐标的图像按照json矢量进行裁切,无数据区域为nan,支持alpha
            Merge_multi_json,   # ✔将多个小的json合并为一个大的json,
            resample_image,     # ✔使用GDAL对图像进行重采样
            shp_to_geojson,     # ✔将shp文件转为geojson文件
            clip_big_image,     # 将大图裁切为小图，支持重叠

        )
                            
    ________________________________________________________________
    >>> # 示例1-读取影像为数组并返回信息
    >>> img = read_img_to_array_with_info(img_path) # 读取图片，并输出图片信息
    ________________________________________________________________
    >>> # 示例2-读取影像为数组
    >>> img = read_img_to_array(img_path) # 读取图片为数组
    ________________________________________________________________
    >>> # 示例3:
    

"""
#!/usr/bin/env python
# coding: utf-8

from osgeo import gdal, ogr
import os
import glob
import numpy as np
import math
from typing import Dict, List, Optional, Sequence, Union
from .ATL_path import mkdir_or_exist, find_data_list
from tqdm import tqdm 
import json
import cv2

def read_img_to_array_with_info(filename: str, 
             convert_HWC: Optional[bool] = False) -> np.ndarray:   
    '''✔读取影像为数组并返回信息.

    Args:
        filename (str): 输入的影像路径
        convert_HWC (bool): 是否转换为 H*W*C 格式，默认为False 

    Returns: 
        影像的numpy数组格式，并显示影像的基本信息

    '''
    DataTypeList = ['GDT_Byte','GDT_UInt16','GDT_Int16','GDT_UInt32',
                    'GDT_Int32','GDT_Float32','GDT_Float64']

    dataset = gdal.Open(filename) #打开文件    
    print('栅格矩阵的行数:', dataset.RasterYSize)
    print('栅格矩阵的列数:', dataset.RasterXSize)
    print('波段数:', dataset.RasterCount)
    print('数据类型:', DataTypeList[dataset.GetRasterBand(1).DataType])
    print('仿射矩阵(左上角像素的大地坐标和像素分辨率)', dataset.GetGeoTransform())
    print('地图投影信息:', dataset.GetProjection())
    im_data = dataset.ReadAsArray()
    if convert_HWC:
        im_data = np.transpose(im_data, (1, 2, 0))

    del dataset 
    return im_data

def read_img_to_array(filename: str, 
             convert_HWC: Optional[bool] = False) -> np.ndarray:     # 读取影像为数组
    '''✔读取影像为数组.

    Args:
        filenam (str):输入的影像路径 
        convert_HWC (bool): 是否转换为 H*W*C 格式，默认为False 

    Returns: 
        影像的numpy数组格式
    '''

    dataset = gdal.Open(filename) #打开文件
    im_data = dataset.ReadAsArray()
    if convert_HWC:
        im_data = np.transpose(im_data, (1, 2, 0))
    
    del dataset 
    return im_data

def save_ds_to_tif(GDAL_dataset: gdal.Dataset,
           out_path: str,
           bands = None) -> None:
    """✔将GDAL dataset数据格式写入tif保存.

    Args:
        GDAL_dataset (gdal.Dataset)：输入的GDAL影像数据格式
        out_path (str)：输出的文件路径
        bands(List[str]): 输出的波段数，默认为None(输出所有波段

    Returns: 
        无 Return，输出影像文件至`out_path`
    """

    # 读取dataset信息
    im_array = GDAL_dataset.ReadAsArray()
    im_array = np.transpose(im_array, (1, 2, 0))
    print(im_array.shape)
    im_width = GDAL_dataset.RasterXSize
    im_height = GDAL_dataset.RasterYSize
    im_bands = GDAL_dataset.RasterCount   
    im_geotrans = GDAL_dataset.GetGeoTransform()  
    im_proj = GDAL_dataset.GetProjection()
    im_datatype = GDAL_dataset.GetRasterBand(1).DataType
    
    # 将dataset 写入 tif
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path ,im_width, im_height,im_bands,im_datatype)
    ds.SetGeoTransform(im_geotrans) 
    ds.SetProjection(im_proj)
    
    for band_num in range(im_bands):
        band = ds.GetRasterBand(band_num + 1)
        band.WriteArray(im_array[:, :, band_num]) 
    # del ds
    
def save_array_to_tif(
        img_array: np.ndarray,
        out_path: str, 
        Transform = None, 
        Projection = None, 
        Band: int=3, 
        Datatype: int = 6):
    """×将数组格式写入tif保存.

    Args:
        img_array (np.ndarry): 待保存的影像数组
        out_path (str): 输出的文件路径
        Transform：仿射矩阵六参数数组，默认为空,详细格式见GDAL。
        Projection ：投影，默认为空,详细格式见GDA
        Band (int): 波段数，默认为1
        Datatype (int): 保存数据格式（位深），默认为6，GDT_Float32

    Returns: 
        输出影像文件
    """

    h,w,c= img_array.shape

    print(img_array.shape)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path, w, h, c, Datatype)
    if not Transform==None:
        ds.SetGeoTransform(Transform) 
    if not Projection==None:
        ds.SetProjection(Projection)  
    if not Band == None:
        Band = c

    for band_num in range(Band):
        band = ds.GetRasterBand(band_num + 1)
        band.WriteArray(img_array[:, :, band_num]) 
    del ds
    
def read_img_get_geo(img_path: str):
    '''计算影像角点的地理坐标或投影坐标

    Args:
        img_path (str): 影像路径

    Returns: 
        min_x: x方向最小值
        max_y: y方向最大值
        max_x: x方向最大值
        min_y: y方向最小值
    '''

    ds=gdal.Open(img_path)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize 
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    
    return min_x, max_y, max_x, min_y

def ds_get_img_geo(GDAL_dataset: gdal.Dataset):
    '''读取dataset格式，计算影像角点的地理坐标或投影坐标

    Args:
        GDAL_dataset： GDAL dataset格式数据

    Returns: 
        min_x： x方向最小值
        max_y： y方向最大值
        max_x： x方向最大值
        min_y:  y方向最小值
    '''
    geotrans=list(GDAL_dataset.GetGeoTransform())
    xsize=GDAL_dataset.RasterXSize 
    ysize=GDAL_dataset.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    GDAL_dataset=None
    
    return min_x,max_y,max_x,min_y

def pix_to_geo(Xpixel: int, Ypixel: int, GeoTransform)->List[int]:
    '''计算影像某一像素点的地理坐标或投影坐标

    Args:
        Xpixel (int): 像素坐标x
        Ypixel (int): 像素坐标y
        GeoTransform：仿射变换参数

    Returns: 
        XGeo： 地理坐标或投影坐标X
        YGeo： 地理坐标或投影坐标Y
    '''

    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo

def geo_to_pix(dataset, x, y):
    '''根据GDAL的仿射变换参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）

    Args:
        dataset: GDAL地理数据
        x: 投影或地理坐标x
        y: 投影或地理坐标y 

    Returns: 
        影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''

    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    
    return np.linalg.solve(a, b)
    
def Mosaic_all_imgs(img_file_path: str,
                    output_path: str,
                    add_alpha_chan: bool=False,
                    nan_or_zero: str='zero',
                    output_band_chan: int=1,
                    img_list:List[str]=None) -> None:
    
    '''✔将指定路径文件夹下的tif影像全部镶嵌到一张影像上
        细节测试:
        mosaic图像：
        ✔ 镶嵌根据矢量裁切后的图像，无数据区域为nan
        ✔ 镶嵌矩形的图像，无数据区域为nan
        mosaic标签：
        x 彩色标签是uint8的，不能用nan，只能用0


    注：将多个图合并之后，再进行裁切的话，nan就是白的，zero就是黑的
        如果单独裁切一个的话，不管nan还是zero都是白的
        如果将裁切后的进行合并的话，会把nan的部分也合并进去，需要单独处理

        需要进行优化
    
    Args:
        img_file_path (str)：tif 影像存放路径
        output_path (str): 输出镶嵌后 tif 的路径
        add_alpha_chan (bool): 是否添加alpha通道，将无数据区域显示为空白，默认为False
        Nan_or_Zero (str): 'nan'或'zero'镶嵌后的无效数据nan或0,默认为0
                           'nan'更适合显示，'0更适合训练'
        output_band_chan (int): 对于多光谱图像，如果只想保存前3个通道的话，指定通道数
        img_list (List[str]):  需要镶嵌的影像列表，默认为None
        
        例子: Mosaic_all_imgs(img_path_all, output_path, add_alpha_chan=True) # 对于RGB标签，添加alpha通道
              Mosaic_all_imgs(img_path_all, output_path, Nan_or_Zero='zero') # 对于float32 img，mosaic为zero
              Mosaic_all_imgs(img_path_all, output_path, Nan_or_Zero='zero') # 对于float32 img，mosaic为nan #展示用

    Returns: 
        镶嵌合成的整体影像
    '''

    # os.chdir(img_file_path) # 切换到指定路径
    #如果存在同名影像则先删除
    if os.path.exists(output_path):
        print(f"  【ATL-LOG】存在{output_path}, 已覆盖")
        os.remove(output_path)

    # 如果不指定的话，就找所有的tif文件
    if img_list == None:
        print("  【ATL-LOG】未指定img_list, 寻找所有tif文件...")
        all_files = find_data_list(img_file_path, suffix='.tif') # 寻找所有tif文件
       
    elif img_list != None:
        print("  【ATL-LOG】指定img_list, 寻找指定的tif文件...")
        all_files = img_list
    assert all_files!=None, 'No tif files found in the path'
    print(f"  【ATL-LOG】本次镶嵌的影像有{len(all_files)}张")

    #获取待镶嵌栅格的最大最小的坐标值
    min_x, max_y, max_x, min_y = read_img_get_geo(all_files[0]) 
    for in_fn in all_files:
        minx, maxy, maxx, miny = read_img_get_geo(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)


    #计算镶嵌后影像的行列号
    in_ds=gdal.Open(all_files[0])
    geotrans=list(in_ds.GetGeoTransform())
    
    # 这一行代码获取了数据集的地理转换信息，包括地理坐标系的变换参数。
    # in_ds.GetGeoTransform() 返回一个包含六个浮点数的元组，
    # 分别表示左上角的X坐标、水平像素分辨率、X方向的旋转（通常为0）、
    # 左上角的Y坐标、Y方向的旋转（通常为0）、垂直像素分辨率。
    # 这一行代码将获取的元组转换为列表形式，并赋值给 geotrans。

    width_geo_resolution = geotrans[1]
    heigh_geo_resolution = geotrans[5]
    # print(f'width_geo_resolution:{width_geo_resolution}, heigh_geo_resolution:{heigh_geo_resolution}')

    columns = math.ceil((max_x-min_x) / width_geo_resolution) + 50 # 有几个像素的偏差，所以会导致小图超出大图范围！！！ 
    rows = math.ceil((max_y-min_y) / (-heigh_geo_resolution)) + 50 # 有几个像素的偏差，所以会导致小图超出大图范围！！！ 

    bands = in_ds.RasterCount  # 读进来的bands
    if not output_band_chan==None:
        bands = output_band_chan
        print("  【ATL-LOG】人为指定输出波段数为:", bands)
    print(f'  【ATL-LOG】新合并图像的尺寸: {rows, columns, bands} (高，宽，波段)')

    in_band_DataType = in_ds.GetRasterBand(1).DataType

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, columns, rows, bands, in_band_DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)


    #定义仿射逆变换
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    #开始逐渐写入
    for in_fn in tqdm(all_files, desc='正在镶嵌图像ing...', colour='GREEN'):
        print('正在镶嵌:', os.path.abspath(in_fn))
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        # print(f'  【ATL-LOG】读入图像的尺寸：{in_ds.RasterYSize, in_ds.RasterXSize} (高，宽)')

        #仿射逆变换
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        # print(f'逆变换后的像素：x:{x}, y:{y}')
        # 该函数返回一个转换器对象 trans，可以使用这个对象执行从输入数据集到输出数据集的坐标转换
        trans = gdal.Transformer(in_ds, out_ds, [])       # in_ds是源栅格，out_ds是目标栅格
        success, xyz = trans.TransformPoint(False, 0, 0)  # 计算in_ds中左上角像元对应out_ds中的行列号
        x, y, z = map(int, xyz)
        # print(f'  【ATL-LOG】小图(0, 0)变换到大图的像素：(({y},{x}), ({y+in_ds.RasterYSize},{x+in_ds.RasterXSize}))')

        for band_num in range(bands):
            # 小图的单通道，(h,w),无数据全是nan
            in_ds_array = in_ds.GetRasterBand(band_num + 1).ReadAsArray() #(h,w)
            # 无效数据的地方全是nan,这也符合下载下来的图，空缺的地方是nan、
            # 把nan的地方，用0替代
            in_ds_array = np.nan_to_num(in_ds_array, nan=0.)

            # 最后合并的大图的波段通道，(h,w)，没数据的地方全是0
            big_out_band = out_ds.GetRasterBand(band_num + 1)
            # 大图中，小图区域的数据
            Tiny_in_BigOut_data = big_out_band.ReadAsArray(x, y, in_ds_array.shape[1], in_ds_array.shape[0])
            Tiny_in_BigOut_data = np.nan_to_num(Tiny_in_BigOut_data, nan=0.)
            # 最后要写入大图的数据：如果是根据矢量裁切完的应该不会有重合，直接相加就行
            # 但是如果是矩形大图的话，有重叠的话，则需要舍弃小图的重叠部分。
            # 第一步：找到大图中有数据的区域：即大图不为零的地方
            # 第二步：利用大图中不为零的地方，把小图中的值设置为0 
            # 第三部：把两个图相加，得到最后的结果
            zero_mask_in_tiny_of_big = Tiny_in_BigOut_data!=0.
            in_ds_array[zero_mask_in_tiny_of_big] = 0.
            # print(f'小图的尺寸{in_ds_array.shape}')
            # print(f'大图中小图尺寸：{Tiny_in_BigOut_data.shape}')
            
            in_ds_array = Tiny_in_BigOut_data + in_ds_array

            # 写入大图
            big_out_band.WriteArray(in_ds_array, x, y)
    del in_ds, out_ds # 必须要有这个

    if nan_or_zero == 'zero' and add_alpha_chan == False:
        print(f"  【ATL-LOG】空缺部分为'zero', 不添加alpha通道, 支持float32-img、uint8-label")
        pass
    # 最后把所有为0.的地方都变成nan
    # 如果是float32图像的话,nan是可以work的,则会让无数据的地方变成nan,显示的时候就是透明的
    elif nan_or_zero == 'nan' and add_alpha_chan ==  False:
        print(f"  【ATL-LOG】空缺部分为'nan', 不添加alpha通道,支持-float32img")
        output_img_ds = gdal.Open(output_path)
        Transform = output_img_ds.GetGeoTransform()
        Projection = output_img_ds.GetProjection()
        
        img_nan = output_img_ds.ReadAsArray()
        img_nan = img_nan.transpose(1,2,0)
        img_nan[img_nan==0.] = np.nan

        save_array_to_tif(img_array = img_nan,
                          out_path = output_path, # 覆盖图像
                          Transform = Transform,
                          Projection = Projection,
                          Datatype = output_img_ds.GetRasterBand(1).DataType,
                          Band = bands)

    # 如果创建的图像是uint8的话，nan是不行的，只能用0,添加alpha通道
    elif add_alpha_chan == True:
        print(f"  【ATL-LOG】添加alpha通道,支持uint8-rgb-img uint8-RGB-label")
        output_img_ds = gdal.Open(output_path)
        Transform = output_img_ds.GetGeoTransform()
        Projection = output_img_ds.GetProjection()

        output_img = output_img_ds.ReadAsArray()
        output_img = np.transpose(output_img, (1, 2, 0))
        h,w,c = output_img.shape

        alpha_posi_array = np.zeros((h, w), dtype=np.uint8)
        alpha_band_sum = np.zeros((h, w), dtype=np.uint16)
        alpha_band_sum = output_img.sum(2)
        
        alpha_posi_array[alpha_band_sum==0]=0
        alpha_posi_array[alpha_band_sum!=0]=255

        alpha_image_array = cv2.merge((output_img, alpha_posi_array))
        # 保存带有Alpha通道的图像
        save_array_to_tif(img_array = alpha_image_array,
                          out_path = output_path, # 覆盖图像
                          Transform = Transform,
                          Projection = Projection,
                          Datatype = output_img_ds.GetRasterBand(1).DataType,
                          Band = bands)

    else:
        print(f'--> 暂不支持此数据组合的合Mosaic')
        print(f'--> 如数据为 float32, 请使用 nan_or_zero="nan"/"zero", add_alpha_chan=False')
        print(f'--> 如数据为 uint8 RGB, 请使用 nan_or_zero="nan"/"zero", add_alpha_chan=True/False')

      
    print(f'-->镶嵌图像已完成，输出至 {output_path}')
     

def Mosaic_2img_to_one(ds1 , ds2, path):
    '''将两幅影像镶嵌至同一幅影像

    Args:
        ds1：镶嵌数据集1
        ds2：镶嵌数据集1

    Returns: 
        镶嵌合成的整体影像
    '''
    band1 = ds1.GetRasterBand(1)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize
    
    band2 = ds2.GetRasterBand(1)
    rows2 = ds2.RasterYSize
    cols2 = ds2.RasterXSize
    
    (minX1,maxY1,maxX1,minY1) = ds_get_img_geo(ds1)
    (minX2,maxY2,maxX2,minY2) = ds_get_img_geo(ds2)


    transform1 = ds1.GetGeoTransform()
    pixelWidth1 = transform1[1]
    pixelHeight1 = transform1[5] #是负值（important）
    
    transform2 = ds2.GetGeoTransform()
    pixelWidth2 = transform1[1]
    pixelHeight2 = transform1[5] 
    
    # 获取输出图像坐标
    minX = min(minX1, minX2)
    maxX = max(maxX1, maxX2)
    minY = min(minY1, minY2)
    maxY = max(maxY1, maxY2)

    #获取输出图像的行与列
    cols = int((maxX - minX) / pixelWidth1)
    rows = int((maxY - minY) / abs(pixelHeight1))

    # 计算图1左上角的偏移值（在输出图像中）
    xOffset1 = int((minX1 - minX) / pixelWidth1)
    yOffset1 = int((maxY1 - maxY) / pixelHeight1)

    # 计算图2左上角的偏移值（在输出图像中）
    xOffset2 = int((minX2 - minX) / pixelWidth1)
    yOffset2 = int((maxY2 - maxY) / pixelHeight1)

    # 创建一个输出图像
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create( path, cols, rows, 1, band1.DataType)#1是bands，默认
    out_band = out_ds.GetRasterBand(1)

    # 读图1的数据并将其写到输出图像中
    data1 = band1.ReadAsArray(0, 0, cols1, rows1)
    out_band.WriteArray(data1, xOffset1, yOffset1)

    #读图2的数据并将其写到输出图像中
    data2 = band2.ReadAsArray(0, 0, cols2, rows2)
    out_band.WriteArray(data2, xOffset2, yOffset2)
    ''' 写图像步骤'''
    
    #第二个参数是1的话：整幅图像重度，不需要统计
    # 设置输出图像的几何信息和投影信息
    geotransform = [minX, pixelWidth1, 0, maxY, 0, pixelHeight1]
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(ds1.GetProjection())
    
    del ds1,ds2,out_band,out_ds,driver
   
    return 0

def raster_overlap(ds1, ds2, nodata1=None, nodata2=None):
    '''两个栅格数据集取重叠区或求交集（仅测试方形影像）

    Args:
        ds1 (GDAL dataset) - GDAL dataset of an image
        ds2 (GDAL dataset) - GDAL dataset of an image
        nodata1 (number) - nodata value of image 1
        nodata2 (number) - nodata value of image 2
        
    Returns: 
        ds1c (GDAL dataset), ds2c (GDAL dataset): 011
    '''

##Setting nodata
    nodata = 0
    ###Check if images NoData is set
    if nodata2 is not None:
        nodata = nodata2
        ds2.GetRasterBand(1).SetNoDataValue(nodata)
    else:
        if ds2.GetRasterBand(1).GetNoDataValue() is None:
            ds2.GetRasterBand(1).SetNoDataValue(nodata)

    if nodata1 is not None:
        nodata = nodata1
        ds1.GetRasterBand(1).SetNoDataValue(nodata1)
    else:
        if ds1.GetRasterBand(1).GetNoDataValue() is None:
            ds1.GetRasterBand(1).SetNoDataValue(nodata)

    ### Get extent from ds1
    projection = ds1.GetProjection()
    geoTransform = ds1.GetGeoTransform()

    ###Get minx and max y
    
    [minx, maxy, maxx, miny] = ds_get_img_geo(ds1)
    [minx_2, maxy_2, maxx_2, miny_2] = ds_get_img_geo(ds2)
    
    min_x = sorted([maxx,minx_2,minx,maxx_2])[1]    # 对边界值排序，第二三个为重叠区边界
    max_y = sorted([maxy,miny_2,miny,maxy_2])[2]
    max_x = sorted([maxx,minx_2,minx,maxx_2])[2]
    min_y = sorted([maxy,miny_2,miny,maxy_2])[1]
    
    ###Warp to same spatial resolution
    gdaloptions = {'format': 'MEM', 'xRes': geoTransform[1], 'yRes': 
    geoTransform[5], 'dstSRS': projection}
    ds2w = gdal.Warp('', ds2, **gdaloptions)
    ds2 = None

    ###Translate to same projection
    ds2c = gdal.Translate('', ds2w, format='MEM', projWin=[min_x, max_y, max_x, min_y], 
    outputSRS=projection)
    ds2w = None
    ds1c = gdal.Translate('', ds1, format='MEM', projWin=[min_x, max_y, max_x, min_y], 
    outputSRS=projection)
    ds1 = None

    return ds1c,ds2c

def crop_tif_with_json_zero(img_path: Union[str, gdal.Dataset],
                            output_path: str,
                            geojson_path: str):
    '''✔将带有坐标的图像按照json矢量进行裁切,矢量外无数据区域为0,适合训练
        也可以裁切mask标签。

    Args:
        img_path (str or gdal.Dataset): 输入图像的路径 or GDAL dataset
        output_path (str): 输出图像的路径
        geojson_path (str): Geojson文件的路径
        
    Returns: 
        保存裁切后的图像至本地

    Example:
        >>> from ATL_Tools import mkdir_or_exist, find_data_list
        >>> from ATL_Tools.ATL_gdal import crop_tif_with_json_zero
        >>> from tqdm import tqdm
        >>> import os 
        >>> from osgeo import gdal

        >>> label_path_all = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/大庆/推理结果-24类/推理结果-24-mask'
        >>> output_path_all = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/大庆/推理结果-24类/推理结果-24-mask-crop'
        >>> img_path_all = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/大庆/要推理的images-矢量裁切'
        >>> json_path_all = '../要推理的json/'
        >>> mkdir_or_exist(output_path_all)
        >>> label_list = find_data_list(label_path_all, suffix='.png')

        >>> for label_path in tqdm(label_list, colour='Green'):
        >>>     # 添加坐标
        >>>     IMG_file_path = os.path.join(img_path_all, os.path.basename(label_path).replace('.png', '.tif'))
        >>>     IMG_gdal = gdal.Open(IMG_file_path, gdal.GA_ReadOnly)
        >>>     assert  IMG_gdal is not None, f"无法打开 {IMG_file_path}"
        >>>     trans = IMG_gdal.GetGeoTransform()
        >>>     proj = IMG_gdal.GetProjection()
        >>>     label_gdal = gdal.Open(label_path, gdal.GA_ReadOnly)
        >>>     label_gdal.SetGeoTransform(trans)
        >>>     # 裁切
        >>>     label_output_path = os.path.join(output_path_all, os.path.basename(label_path).replace('.png', '.tif'))
        >>>     json_path = os.path.join(json_path_all, os.path.basename(label_path).split('_')[-1].replace('.png', '.json'))
        >>>     print(f'正在裁切: {label_output_path},json: {json_path}')
        >>>     crop_tif_with_json_zero(label_gdal, label_output_path, json_path)
    '''
    if os.path.exists(output_path):
        print(f"存在{output_path}, 已覆盖")
        os.remove(output_path)

    # 打开栅格文件
    if isinstance(img_path, str):
        raster_ds = gdal.Open(img_path)
    elif isinstance(img_path, gdal.Dataset):
        raster_ds = img_path
    assert raster_ds!=None, f'打开 {raster_ds} 失败'

    # 打开GeoJSON文件
    geojson_ds = ogr.Open(geojson_path)
    geojson_layer = geojson_ds.GetLayer()

    # 获取GeoJSON文件的范围
    xmin, xmax, ymin, ymax = geojson_layer.GetExtent()
    # 设置裁剪范围
    warp_options = gdal.WarpOptions(cutlineDSName=geojson_path,
                                    cutlineWhere=None,
                                    cropToCutline=None,
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    dstSRS='EPSG:4326')  # 设置输出投影，这里使用EPSG:4326，即WGS84经纬度坐标系

    # 执行裁剪
    gdal.Warp(output_path, raster_ds, options=warp_options)

    # 关闭数据源
    raster_ds = None
    geojson_ds = None
    if isinstance(img_path, str):
        print(f'根据矢量裁切{img_path}完成！无数据区域为0')
    elif isinstance(img_path, gdal.Dataset):
        print(f'根据矢量裁切完成！无数据区域为0')

def crop_tif_with_json_nan(img_path: Union[str, gdal.Dataset],
                           output_path: str,
                           geojson_path: str,
                           add_alpha_chan: bool = False) -> None:
    '''✔将带有坐标的图像按照json矢量进行裁切
    使无数据区域的值为nan,优先使用这个, 矢量外无数据区域为nan

    Args:
        img_path (str or gdal.Dataset): 输入图像的路径 or GDAL dataset
        output_path (str): 输出图像的路径
        geojson_path (str): Geojson文件的路径
        add_alpha_chan (bool): 是否给RGB添加alpha通道
        
    Returns: 
        保存裁切后的图像至本地
    '''

    if os.path.exists(output_path):
        print(f"存在{output_path}, 已覆盖")
        os.remove(output_path)
        
    # 打开栅格文件
    if isinstance(img_path, str):
        raster_ds = gdal.Open(img_path)
    elif isinstance(img_path, gdal.Dataset):
        raster_ds = img_path
    assert raster_ds!=None, f'打开 {raster_ds} 失败'

    # 打开GeoJSON文件
    geojson_ds = ogr.Open(geojson_path)
    geojson_layer = geojson_ds.GetLayer()

    # 获取GeoJSON文件的范围
    xmin, xmax, ymin, ymax = geojson_layer.GetExtent()
    # 设置裁剪范围
    warp_options = gdal.WarpOptions(cutlineDSName=geojson_path,
                                    cutlineWhere=None,
                                    cropToCutline=None,
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    dstSRS='EPSG:4326',# 设置输出投影，这里使用EPSG:4326，即WGS84经纬度坐标系
                                    creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS', 'ALPHA=YES'],
                                    dstNodata=float('nan')  # 设置裁剪后的无数据值为 NaN
                                    )  

    # 执行裁剪
    gdal.Warp(output_path, raster_ds, options=warp_options)
    # 关闭数据源
    raster_ds = None
    geojson_ds = None
    
    if isinstance(img_path, str):
        print(f'根据矢量裁切{img_path}完成！无数据区域为0')
    elif isinstance(img_path, gdal.Dataset):
        print(f'根据矢量裁切完成！无数据区域为0')
    
    if add_alpha_chan==True:
        print(f'正在添加 alpha 通道...')

        output_img_ds = gdal.Open(output_path)
        Transform = output_img_ds.GetGeoTransform()
        Projection = output_img_ds.GetProjection()

        output_img = output_img_ds.ReadAsArray()
        output_img = np.transpose(output_img, (1, 2, 0))
        h,w,c = output_img.shape

        alpha_posi_array = np.zeros((h, w), dtype=np.uint8)
        alpha_band_sum = np.zeros((h, w), dtype=np.uint16)
        alpha_band_sum = output_img.sum(2)
        
        alpha_posi_array[alpha_band_sum==0]=0
        alpha_posi_array[alpha_band_sum!=0]=255

        alpha_image_array = cv2.merge((output_img, alpha_posi_array))
        # 保存带有Alpha通道的图像
        save_array_to_tif(img_array = alpha_image_array,
                            out_path = output_path, # 覆盖图像
                            Transform = Transform,
                            Projection = Projection,
                            Datatype = 1)




def Merge_multi_json(input_json_file: str,
                     output_json: str) -> None:
    """✔将多个小的json合并为一个大的json

    Args:
        input_json_file (str): 要合并的json文件的路径
        output_json (str): 合并后输出的json文件名
    """
    
    # 读取json文件
    def read_json(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    json_file = find_data_list(input_json_file, '.json')
    assert json_file != None, f"{json_file} 下 未找到 .json 文件"

    new_json_features = []
    for tiny_json_name in tqdm(json_file):
        tiny_json = read_json(tiny_json_name)
        tiny_features =  tiny_json['features'][0]
        new_json_features.append(tiny_features)
    # print(new_json_features)
        # print(tiny_features)

    new_json_content = {
        "type": "FeatureCollection",
        "features": new_json_features
    }

    with open(output_json, 'w') as f:
        json.dump(new_json_content, f)
    print(f"合并json {output_json} 文件完成！")

def resample_image(input_path: str, 
                   output_path: str, 
                   scale_factor: str):
    """✔使用GDAL对图像进行重采样
    
    Args:
        input_path (str): 输入图像路径
        output_path (str): 输出图像路径
        scale_factor (float): 缩放因子

    Returns:
        输出重采样后的图像
    """

    # 打开输入图像
    input_ds = gdal.Open(input_path)

    # 获取输入图像的宽度和高度
    cols = input_ds.RasterXSize
    rows = input_ds.RasterYSize

    # 计算输出图像的新宽度和新高度
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)

    # 使用gdal.Warp函数进行重采样
    gdal.Warp(output_path, input_ds, format='GTiff', width=new_cols, height=new_rows)

    # 关闭数据集
    input_ds = None

def shp_to_geojson(input_shp: str, output_geojson: str) -> None:
    """把 Shapefile 文件转换为 GeoJSON 文件
    
    Args:
        input_shp (str): 输入的shp路径xxx.shp 
        output_geojson (str): 输出的json路径xxx.json
        
    Returns:
        保存输入的shp为Geojson
    
    示例：
    >>> # 将一个shp文件转换为geojson文件
    >>> shp_path = './nan值的shp文件/'
    >>> shp_list = find_data_list(shp_path, '.shp')
    >>> for shp_ in tqdm(shp_list):
    >>> output_geojson = os.path.join('./nan的json/', os.path.basename(shp_).split('.')[0]+'.json')
    >>> shp_to_geojson(shp_, output_geojson)

    """
    # 构建ogr2ogr命令
    ogr2ogr_cmd = [
        'ogr2ogr',   # 命令名称
        '-f', 'GeoJSON',   # 输出格式为GeoJSON
        output_geojson,    # 输出文件路径
        input_shp   # 输入Shapefile文件路径
    ]

    # 调用ogr2ogr工具执行转换
    subprocess.run(ogr2ogr_cmd)

def clip_big_image(image_path: str, 
                   save_path: str, 
                   crop_size: int = 5000):
    """把大图裁切成指定尺寸的小图，以供 GPU 可以进行推理（不带坐标）
    
    Args:
        image_path (str): 输入的大图路径
        save_path (str): 输出的小图路径
        crop_size (int): 裁切的尺寸，默认为5000
    
    Returns:
        None, 保存小图至指定的文件夹

    Examples:
       完整的使用示例见：
       裁切：
       https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-Sentinel-2-%E5%A4%84%E7%90%86%E4%BB%A3%E7%A0%81/8-%E5%9B%BE%E7%89%87%E5%A4%AA%E5%A4%A7%E7%88%86%E6%98%BE%E5%AD%98-%E8%A3%81%E5%88%87%E6%88%905000%E5%B0%8F%E5%9B%BE%E6%8E%A8%E7%90%86.py
       推理之后合并:
       https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-Sentinel-2-%E5%A4%84%E7%90%86%E4%BB%A3%E7%A0%81/9-%E9%87%8D%E6%96%B0%E5%B0%86%E6%AD%A5%E9%AA%A48%E7%9A%84%E5%B0%8F%E5%9B%BE%E5%90%88%E5%B9%B6%E6%88%90%E5%A4%A7%E5%9B%BE.py
   
    """



    img_gdal = gdal.Open(image_path)
    img_bit = img_gdal.GetRasterBand(1).DataType

    img_basename = os.path.basename(image_path).split('.')[0] #nangangqu.tif
    
    # 对图像进行裁切,分为8000*8000的地方和512*512的地方
    image_gdal = gdal.Open(image_path)
    img = image_gdal.ReadAsArray()
    img = img.transpose((1,2,0))

    h, w, c = img.shape
    rows, cols, bands = img.shape

    if h < crop_size or w < crop_size:
        print(f'--- 当前 {img_basename} 图像尺寸小于 {crop_size}，不进行裁切')

        out_path = os.path.join(save_path, os.path.basename(image_path))
        Driver = gdal.GetDriverByName("Gtiff")
        new_img = Driver.Create(out_path, w,h,c, img_bit)
        for band_num in range(bands):
            band = new_img.GetRasterBand(band_num+1)
            band.WriteArray(img[:, :, band_num])
        return None
    
    else:
    
        hang = h - (h//crop_size)*crop_size
        lie =  w - (w//crop_size)*crop_size
        print(f'图像尺寸：{img.shape}')
        print(f'可裁成{h//crop_size+1}行...{hang}')
        print(f'可裁成{w//crop_size+1}列...{lie}')
        print(f'共{crop_size}：{((h//crop_size+1)*(w//crop_size+1))+1}张')

        # 8000的部分 xxxxx._0_0_8000.tif
        for i in range(h//crop_size):
            for j in range(w//crop_size):
                out_path = os.path.join(save_path, img_basename+'_'+str(i)+'_'+str(j)+'_'+str(crop_size)+'.tif')
                Driver = gdal.GetDriverByName("Gtiff")

                new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
                new_img = Driver.Create(out_path, crop_size, crop_size, c, img_bit)
            
                new_512 = img[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size,:]   #横着来       

                for band_num in range(bands):
                    band = new_img.GetRasterBand(band_num + 1)
                    band.WriteArray(new_512[:, :, band_num])

        #以外的部分

        # 下边的 xxxxx._xia_0_8000.tif
        for j in range(w//crop_size):
            new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
            new_512 = img[h-crop_size:h, j*crop_size:j*crop_size+crop_size, :]   #横着来
            
            out_path = os.path.join(save_path, img_basename+'_'+str('xia')+'_'+str(j)+'_'+str(crop_size)+'.tif')
            # cv2.imwrite(out_path,new_512)
            Driver = gdal.GetDriverByName("Gtiff")
            new_img = Driver.Create(out_path, crop_size,crop_size, c, img_bit)
            for band_num in range(bands):
                band = new_img.GetRasterBand(band_num+1)
                band.WriteArray(new_512[:, :, band_num])

        #右边的 xxxxx._you_0_8000.tif
        for j in range(h//crop_size):
            new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
            new_512 = img[j*crop_size:j*crop_size+crop_size, w-crop_size:w, :]   #横着来
            
            out_path = os.path.join(save_path,img_basename+'_'+str('you')+'_'+str(j)+'_'+str(crop_size)+'.tif')
            # cv2.imwrite(out_path,new_512)
            Driver = gdal.GetDriverByName("Gtiff")
            new_img = Driver.Create(out_path, crop_size,crop_size, c, img_bit)
            for band_num in range(bands):
                band = new_img.GetRasterBand(band_num+1)
                band.WriteArray(new_512[:, :, band_num])

        #右下角的
        new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
        new_512 = img[h-crop_size:h, w-crop_size:w, :]   #横着来
        out_path = os.path.join(save_path, img_basename+'_'+str('youxia')+'_'+str(crop_size)+'.tif')
        # cv2.imwrite(out_path,new_512)
        Driver = gdal.GetDriverByName("Gtiff")
        new_img = Driver.Create(out_path, crop_size,crop_size, c, img_bit)
        for band_num in range(bands):
            band = new_img.GetRasterBand(band_num+1)
            band.WriteArray(new_512[:, :, band_num])

    print(f'--- 当前 {img_basename} 图像裁切完成')



