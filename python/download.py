from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import urllib
import zipfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/shadow')
import util as util


def update_file(file_disk_path, file_download_path):
    if os.path.isfile(file_disk_path):
        file_disk_md5 = util.get_file_md5(file_disk_path)
        file_download_md5 = util.get_file_md5(file_download_path)
        if file_disk_md5 != file_download_md5:
            shutil.copy(file_download_path, file_disk_path)
    else:
        file_disk_dir = os.path.dirname(file_disk_path)
        if not os.path.isdir(file_disk_dir):
            util.mkdir_p(file_disk_dir)
        shutil.copy(file_download_path, file_disk_path)


def update_zipfile(file_disk_dir, file_download_path):
    extract_path = os.path.dirname(file_download_path)
    with zipfile.ZipFile(file_download_path) as zfile:
        for file_name in zfile.namelist():
            zfile.extract(file_name, extract_path)
            file_disk_path = file_disk_dir + '/' + file_name
            file_extract_path = extract_path + '/' + file_name
            if os.path.isfile(file_extract_path):
                update_file(file_disk_path, file_extract_path)


def download_file(file_obj, ftp_root, disk_work_root):
    download_dir = disk_work_root + '/.download'

    file_name = file_obj['name']
    file_ext = os.path.splitext(file_name)[1]
    file_folder = file_obj['folder']

    file_disk_dir = disk_work_root + '/' + file_folder

    file_disk_path = file_disk_dir + '/' + file_name
    file_ftp_path = ftp_root + '/' + file_folder + '/' + file_name

    file_download_path = download_dir + '/' + file_folder + '/' + file_name
    util.mkdir_p(os.path.dirname(file_download_path))

    print('Downloading ' + file_name + ' ... ', end='')
    urllib.urlretrieve(file_ftp_path, file_download_path)
    urllib.urlcleanup()

    if file_ext == '.zip':
        update_zipfile(file_disk_dir, file_download_path)
    else:
        update_file(file_disk_path, file_download_path)

    util.rmdir_p(download_dir)
    print('Done!')


def download(ftp_ip, project_name, files_name):
    os_type = util.check_os()
    disk_work_root = os.path.dirname(os.path.abspath(__file__)) + '/..'

    ftp_url = 'ftp://' + ftp_ip
    ftp_root = ftp_url + '/' + project_name + '/' + os_type

    json_file_path = ftp_url + '/' + project_name + '/download_' + os_type + '.json'
    json_file = urllib.urlopen(json_file_path)
    json_obj = json.load(json_file)
    json_file.close()

    if project_name != json_obj['Project']:
        raise ValueError('Project mismatch!',
                         project_name, json_obj['Project'])

    if len(files_name) == 0:
        for file_obj in json_obj['Files']:
            if file_obj['type'] != 'required':
                continue
            download_file(file_obj, ftp_root, disk_work_root)
    else:
        for name in files_name:
            is_download = False
            for file_obj in json_obj['Files']:
                if name == file_obj['name']:
                    download_file(file_obj, ftp_root, disk_work_root)
                    is_download = True
                    break
            if not is_download:
                print('File ' + name + ' is not on the server!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download files from ftp server!')
    parser.add_argument('--ftp', default='172.17.122.193',
                        help='The ftp server\'s ip address, default is 172.17.122.193.')
    parser.add_argument('--project', default='shadow',
                        help='The project\'s name, default is shadow.')
    parser.add_argument('--files', nargs='*', default=[],
                        help='The files to be downloaded manually.')
    args = parser.parse_args()

    ftp_ip = args.ftp
    project_name = args.project
    files_name = args.files

    download(ftp_ip, project_name, files_name)
