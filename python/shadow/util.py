from __future__ import print_function

import errno
import os
import platform
import zipfile

def mkdir_p(path):
    try:
        os.makedirs(path)
    except Exception, e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise ValueError('mkdir failed', path, e.errno)

def rmfile_p(path):
    try:
        os.remove(path)
    except Exception as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise ValueError('rmfile failed', path, e.errno)

def check_os():
    os_str = platform.system().lower()
    if 'linux' in os_str:
        return 'linux'
    elif 'windows' in os_str:
        return 'windows'
    else:
        raise ValueError('Unknown OS type!', os_str)

def handle_zip(file_path):
    uzip_dir = os.path.dirname(file_path)
    mkdir_p(uzip_dir)
    zfile = zipfile.ZipFile(file_path)
    for name in zfile.namelist():
        zfile.extract(name, uzip_dir)
    zfile.close()
    rmfile_p(file_path)
