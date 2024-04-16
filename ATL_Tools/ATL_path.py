
"""
ATL_path
========
包含了 `mkdir_or_exist()`, `find_data_list()` 创建文件夹和寻找数据集名称两大函数

用法：
----
    >>> # 在开头复制这一句
    >>> from ATL_path import mkdir_or_exist, find_data_list, scandir
    ________________________________________________________________
    >>> # 示例1-创建文件夹：
    >>> mkdir_or_exist(xxxx) # 创建文件夹, 存在则不创建
    ________________________________________________________________
    >>> # 示例2-寻找所有符合后缀文件夹名称(绝对路径):
    >>> # 寻找尾缀为'.jpg'的数据集列表
    >>> img_list = find_data_list(img_root_path: str, suffix: str ='.jpg') 
        for img_path in tqdm(img_list):
            print(img_path)
"""

import os
import os.path as osp
from pathlib import Path
from typing import List

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    """
    若不存在，则创建文件夹，代替os.path.exist()

    示例-创建文件夹：
    >>> mkdir_or_exist(xxxx) # 创建文件夹, 存在则不创建
    """
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def find_data_list(img_root_path: str, suffix: str ='.jpg') -> List:
    
    """根据给定的数据集根目录，找出其下所有符合后缀的数据集图片完整路径

    Args:
        img_root_path (str): 数据集根目录.
        suffix (str): 后缀名. Defaults to '.jpg'.

    Returns:
        List: 所有符合后缀的数据集图片完整路径(经过sorted()排序).
    """     
    print('--------------------------------------------------------------')
    print(f'-- 正在读取数据集列表... "{img_root_path}" ')

    img_list = []
    for img_name in scandir(img_root_path, suffix=suffix, recursive=True):
        if suffix in img_name:
            img_path = os.path.join(img_root_path, img_name)
            img_list.append(img_path)
    print(f'-- 共在 "{img_root_path}" 下寻找到 *{suffix} {len(img_list)} 项')
    print('-------------------------------------------------------------')
    return sorted(img_list)