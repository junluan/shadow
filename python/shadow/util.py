from __future__ import division
from __future__ import print_function

import errno
import hashlib
import os
import platform
import shutil
import sys
import zipfile


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise OSError('mkdir failed', path)


def rmdir_p(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        pass


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


def get_file_md5(file_path):
    with open(file_path) as f:
        m = hashlib.md5()
        while True:
            data = f.read(10240)
            if not data:
                break
            m.update(data)
        return m.hexdigest()


def progress(percentage, prefix=''):
    max_columns = 20
    full = int(max_columns * percentage / 100)
    left_columns = max_columns - full - 1
    if full < max_columns:
        bar = full * '=' + '>' + left_columns * ' '
    else:
        bar = full * '=' + left_columns * ' '
    sys.stdout.write(u'\u001b[1000D' + prefix +
                     '[' + bar + '] ' + str(percentage) + '%')
    sys.stdout.flush()
