from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import zipfile

try:
    import urllib.error as urlliberror
    import urllib.request as urllib
    HTTPError = urlliberror.HTTPError
    URLError = urlliberror.URLError
except ImportError:
    import urllib2 as urllib
    HTTPError = urllib.HTTPError
    URLError = urllib.URLError

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


def download_file(file_obj, ftp_root, disk_work_root, show_progress):
    download_dir = disk_work_root + '/.download'

    file_name = file_obj['Name']
    file_ext = os.path.splitext(file_name)[1]
    file_folder = file_obj['Folder']

    file_disk_dir = disk_work_root + '/' + file_folder

    file_disk_path = file_disk_dir + '/' + file_name
    file_ftp_path = ftp_root + '/' + file_folder + '/' + file_name

    file_download_path = download_dir + '/' + file_folder + '/' + file_name
    util.mkdir_p(os.path.dirname(file_download_path))

    try:
        response = urllib.urlopen(file_ftp_path)
        size = int(response.info().getheader('Content-Length').strip())
        downloaded_size = 0
        chunk = min(size, 8192)
        if show_progress:
            util.progress(0, 'Downloading ' + file_name + ' ')
        else:
            print('Downloading ' + file_name + ' ...', end='')
        with open(file_download_path, "wb") as local_file:
            while True:
                data_chunk = response.read(chunk)
                if not data_chunk:
                    break
                local_file.write(data_chunk)
                downloaded_size += len(data_chunk)
                if show_progress:
                    util.progress(int(100 * downloaded_size / size),
                                  'Downloading ' + file_name + ' ')
        print(' Done!')
    except HTTPError as e:
        raise Exception("Could not download file. [HTTP Error] {code}: {reason}."
                        .format(code=e.code, reason=e.reason))
    except URLError as e:
        raise Exception("Could not download file. [URL Error] {reason}."
                        .format(reason=e.reason))
    except Exception as e:
        raise e

    if file_ext == '.zip':
        update_zipfile(file_disk_dir, file_download_path)
    else:
        update_file(file_disk_path, file_download_path)

    util.rmdir_p(download_dir)


def download(args):
    os_type = util.check_os()
    disk_work_root = os.path.dirname(os.path.abspath(__file__)) + '/..'

    ftp_url = 'ftp://' + args.ftp
    ftp_root = ftp_url + '/' + args.project + '/' + os_type

    json_file_path = ftp_url + '/' + args.project + '/download_' + os_type + '.json'
    json_file = urllib.urlopen(json_file_path)
    json_obj = json.load(json_file)
    json_file.close()

    if args.project != json_obj['Project']:
        raise ValueError('Project mismatch!',
                         args.project, json_obj['Project'])

    if len(args.files) == 0:
        for file_obj in json_obj['Files']:
            if file_obj['Type'] != 'required':
                continue
            download_file(file_obj, ftp_root, disk_work_root, args.progress)
    else:
        for name in args.files:
            is_download = False
            for file_obj in json_obj['Files']:
                if name == file_obj['Name']:
                    download_file(file_obj, ftp_root,
                                  disk_work_root, args.progress)
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
    parser.add_argument('--files', '-f', nargs='*', default=[],
                        help='The files to be downloaded manually.')
    parser.add_argument('--progress', '-p', nargs='?', const=True, default=False,
                        help='Show progress bar when downloading files.')
    args = parser.parse_args()

    download(args)
