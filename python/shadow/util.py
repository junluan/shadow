from __future__ import print_function

import errno
import os
import platform
import zipfile


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise OSError('mkdir failed', path)


def rmfile_p(path):
    try:
        os.remove(path)
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise OSError('rmfile failed', path)


def check_os():
    os_str = platform.system().lower()
    if 'linux' in os_str:
        return 'linux'
    elif 'windows' in os_str:
        return 'windows'
    else:
        raise ValueError('Unknown OS type', os_str)


def handle_zip(file_path, extract_path=''):
    if extract_path == '':
        extract_path = os.path.dirname(file_path)
    mkdir_p(extract_path)
    with zipfile.ZipFile(file_path) as zfile:
        zfile.extractall(extract_path)
