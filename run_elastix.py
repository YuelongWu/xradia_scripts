import os
import glob
from scipy import ndimage
import numpy as np
import subprocess
import time

rootpath = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02234'
threads = 20
mode = 'align'

subfolder = 'Pyrogallol'

imgpath = os.path.join(rootpath, 'data_dump')
workpath = os.path.join(rootpath, 'workdir')
maskpath = os.path.join(workpath, 'masks')
ext = '.nrrd'

outpath = os.path.join(workpath, 'elastix_out')
tform0 = os.path.join(workpath, 'tf_{}.txt'.format(subfolder))
if not os.path.isfile(tform0):
    tform0 = None
pfile = os.path.join(workpath, 'Parameters.txt')

imglist = glob.glob(os.path.join(imgpath, subfolder, '*' + ext))
imglist.sort()

fMask = None # os.path.join(maskpath, 'No2_ReOs_fMask.mha')
mMask = None # os.path.join(maskpath, 'No2_ReOs_mMask.mha')

def get_next_image_pair(resultpath, imglist):
    resultlist0 = glob.glob(os.path.join(resultpath, '*/'))
    resultlist0.sort()
    resultnames0 = [os.path.basename(t.rstrip('/')) for t in resultlist0]
    imgnames = [os.path.basename(t).replace(ext, '') for t in imglist]
    resultlist = [t[0] for t in zip(resultlist0, resultnames0) if t[1] in imgnames]
    resultnames = [os.path.basename(t.rstrip('/')) for t in resultlist]
    to_align = [not (t in resultnames) for t in imgnames]
    if not any(to_align):
        return None
    dis, indx = ndimage.distance_transform_edt(to_align,return_distances=True,return_indices=True)
    indx = np.squeeze(indx)
    dis[dis == 0] = dis.max() + 1
    idxt = np.argmin(dis)
    imgout = imglist[idxt]
    refname = imgnames[indx[idxt]]
    if refname in resultnames:
        idxt = resultnames.index(refname)
        refout = resultlist[idxt]
    else:
        refout = ''
    return imgout, refout

def get_result_file_path(fdir, fname=None):
    if fname:
        fdir = os.path.join(fdir, fname)
    if not fdir:
        return ''
    if os.path.isfile(os.path.join(fdir, 'result.0' + ext)):
        return os.path.join(fdir, 'result.0' + ext)
    elif os.path.isfile(os.path.join(fdir, 'result' + ext)):
        return os.path.join(fdir, 'result' + ext)
    else:
        raise OSError('{} has NOT been aligned. Delete folder.'.format(fdir))

cnt0 = 0
while cnt0 < len(imglist):
    stm0 = time.time()
    if mode == 'render':
        img = imglist[cnt0]
        refdir = ''
    else:
        next_pairs = get_next_image_pair(outpath, imglist)
        if not next_pairs:
            print('finished all volumns')
            break
        img, refdir = next_pairs
        ref = get_result_file_path(refdir)
    cnt0 += 1

    imgoutdir = os.path.join(outpath, os.path.basename(img).replace(ext, ''))
    if os.path.isfile(os.path.join(imgoutdir, 'processed')):
        continue
    tform1 = os.path.join(refdir, 'TransformParameters.0.txt')
    if os.path.isfile(tform1):
        tform0 = tform1
    if not os.path.isdir(imgoutdir):
        # print('mkdir: ' + imgoutdir)
        os.makedirs(imgoutdir, exist_ok=True)
    if mode == 'align':
        print('{}  ->  {}'.format(os.path.basename(img), os.path.basename(os.path.dirname(ref))))
        cmd = 'elastix -threads {0} -f "{1}" -m "{2}" -out "{3}" -p "{4}"'.format(
            threads, ref, img, imgoutdir, pfile)
        if tform0:
            cmd = cmd + ' -t0 "{}"'.format(tform0)
        if fMask:
            cmd = cmd + ' -fMask "{}"'.format(fMask)
        if mMask:
            cmd = cmd + ' -mMask "{}"'.format(mMask)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    elif mode == 'transform':
        cmd = 'transformix -threads {0} -in "{1}" -out "{2}" -tp "{3}"'.format(
            threads, img, imgoutdir, tform0)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    elif mode == 'render':
        tform1 = os.path.join(imgoutdir, 'TransformParameters.0.txt')
        if not os.path.isfile(tform1):
            raise OSError('{} not exist'.format(tform1))
        cmd = 'transformix -threads {0} -in "{1}" -out "{2}" -tp "{3}"'.format(
            threads, img, imgoutdir, tform1)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        raise ValueError('{} not valide mode'.format(mode))
    with open(os.path.join(imgoutdir, 'processed'), 'w') as ftrump:
        pass
    print('Finished {}: {} sec'.format(os.path.basename(img), round(time.time()-stm0)))
