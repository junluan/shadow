from __future__ import print_function

import argparse
import json
import os
import sys
import urllib
import urllib2

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/shadow')
import util as Util


def download_file(file_obj, ftp_root, disk_work_root):
    file_name = file_obj['name']
    file_ext = os.path.splitext(file_name)[1]
    file_folder = file_obj['folder']
    
    file_ftp_path = ftp_root + '/' + file_folder + '/' + file_name
    file_disk_dir = disk_work_root + '/' + file_folder
    file_disk_path = file_disk_dir + '/' + file_name

    Util.mkdir_p(file_disk_dir)
    print('Downloading ' + file_name + ' ... ', end='')
    urllib.urlretrieve(file_ftp_path, file_disk_path)
    urllib.urlcleanup()
    print('Done!')

    if file_ext == '.zip':
        Util.handle_zip(file_disk_path)

def download(ftp_ip, project_name, files_name):
    os_type = Util.check_os()
    disk_work_root = os.path.dirname(os.path.abspath(__file__)) + '/..'

    ftp_url = 'ftp://' + ftp_ip
    ftp_root = ftp_url + '/' + project_name + '/' + os_type

    json_file_path = ftp_url + '/' + project_name + '/download_' + os_type + '.json'
    json_file = urllib2.urlopen(json_file_path)
    json_obj = json.load(json_file)
    json_file.close()

    if project_name != json_obj['Project']:
        raise ValueError('Project mismatch!', project_name, json_obj['Project'])

    if len(files_name) == 0:
        for file_obj in json_obj['Files']:
            if file_obj['type'] != 'required':
                continue
            download_file(file_obj, ftp_root, disk_work_root)
    else:
        for name in files_name:
            file_obj = None
            for file in json_obj['Files']:
                if name == file['name']:
                    file_obj = file
            if file_obj != None:
                download_file(file_obj, ftp_root, disk_work_root)
            else:
                print('File ' + name + ' is not on the server!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download files from ftp server!')
    parser.add_argument('--ftp', default='172.17.122.193', help='The ftp server\'s ip address, default is 172.17.122.193.')
    parser.add_argument('--project', default='shadow', help='The project\'s name, default is shadow.')
    parser.add_argument('--files', nargs='*', default=[], help='The files to be downloaded manually.')
    args = parser.parse_args()

    ftp_ip = args.ftp
    project_name = args.project
    files_name = args.files

    download(ftp_ip, project_name, files_name)
