import os
import pathlib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
from google_images_download import google_images_download

# Dataset Helper Functions ___________________________________________________


def downloader(search_kws, dpath, limit=40):
    response = google_images_download.googleimagesdownload()
    for kw in search_kws:
        args = {'keywords': kw, 'limit': limit, 'output_directory': dpath, 'image_directory': kw.replace(' ','_'), 'format': 'jpg', 'chromedriver': os.path.join(os.path.expanduser('~'),'chromedriver')}
        response.download(args)
        
    # check every downloaded file and delete non-PIL openable ones
    to_delete_paths = []
    for dirpath, dirnames, filenames in os.walk(dpath):
        if dirpath == dpath:
            for dr in dirnames:
                files = os.listdir(os.path.join(dirpath,dr))
                for file in files:
                    fpath = os.path.join(dirpath,dr,file)
                    try:
                        PIL.Image.open(fpath)
                    except OSError:
                        to_delete_paths.append(fpath)
    print('Deleting the following as PIL failed to open')
    for file in to_delete_paths:
        print(file)
        os.remove(file)
        
def ignore_files(folder, files):
    return [f for f in files if not os.path.isdir(os.path.join(folder, f))]

def make_folder_imagedatasets(dpath, test=True, pct_valid = .2, pct_test = .5):
    '''This utility function '''
    dpath = pathlib.Path(dpath)
    existing_dirs = os.listdir(dpath)
    ignore_dirs = ['train', 'valid', 'test']
    if any(findir in existing_dirs for findir in ignore_dirs):
        print('There are already dir(s) named train/valid/test in dpath. Investigate')
        return
    else:
        print('Going to make train/valid and possibly test folder structure')
        
        # setup dirpaths
        trpath = os.path.join(dpath, 'train')
        vapath = os.path.join(dpath, 'valid')

        # make train
        print('Creating train dir and moving all dpath dirs into train')
        for i, dr in enumerate(existing_dirs):
            if i == 0:
                os.makedirs(trpath)
            shutil.move(os.path.join(dpath,dr), trpath)
        
        # make valid dir structure
        print('Ceating valid dir and copy train structure into valid')
        shutil.copytree(trpath,vapath,symlinks=False,ignore=ignore_files)

        # move over files from train to valid
        for dirpath, dirnames, filenames in os.walk(trpath):
            if dirpath == trpath:
                for dr in dirnames:
                    files = os.listdir(os.path.join(dirpath, dr))
                    n_move = int(len(files)*pct_valid)
                    if n_move <= 0:
                        print('number to move to valid is less than 0. Fix')
                        return
                    move_files = np.random.choice(files, n_move, replace=False)
                    for file in move_files:
                        shutil.move(os.path.join(dirpath,dr,file), os.path.join(vapath,dr))
                        
        # make test dir structure
        if test:
            tepath = os.path.join(dpath, 'test')
            print('Ceating test dir and copy valid structure into test')
            shutil.copytree(vapath,tepath,symlinks=False,ignore=ignore_files)
            
            # move over files from train to valid
            for dirpath, dirnames, filenames in os.walk(vapath):
                if dirpath == vapath:
                    for dr in dirnames:
                        files = os.listdir(os.path.join(dirpath, dr))
                        n_move = int(len(files)*pct_test)
                        if n_move <= 0:
                            print('number to move to test is less than 0. Fix')
                            return
                        move_files = np.random.choice(files, n_move, replace=False)
                        for file in move_files:
                            shutil.move(os.path.join(dirpath,dr,file), os.path.join(tepath,dr))
