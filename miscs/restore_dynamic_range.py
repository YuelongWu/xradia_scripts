import os
import SimpleITK as sitk
import numpy as np
import h5py
import glob

rootpath = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02232'

ext = '.nrrd'
moveonly = True
if not moveonly:
    max_file = os.path.join(rootpath, 'workdir', 'blk_max.h5')
    max_scl = 0.60
    with h5py.File(max_file, mode='r') as f:
        max_dict = {t:f[t][0] for t in f}

srcpath = os.path.join(rootpath, 'workdir', 'elastix_out')
outpath = os.path.join(rootpath, 'mha_files')

if not os.path.isdir(outpath):
    os.makedirs(outpath, exist_ok=True)

imglist = glob.glob(os.path.join(srcpath,'**','result.0' + ext),recursive=True)
imglist.sort()
imglist.reverse()

for imgdir in imglist:
    imgname = os.path.dirname(imgdir)
    imgname = os.path.basename(imgname)
    if moveonly:
        os.replace(imgdir, os.path.join(outpath, imgname + ext))
    else:
        img = sitk.ReadImage(imgdir)
        img_np = sitk.GetArrayFromImage(img)
        img_np_f = img_np.astype(np.float32)
        img_np_f = img_np_f * min(1, max_dict[imgname]) / max_scl
        img_np_f = img_np_f.clip(0,255)
        imgout = sitk.GetImageFromArray(img_np_f.astype(np.uint8), isVector=False)
        imgout.CopyInformation(img)
        sitk.WriteImage(imgout, os.path.join(outpath, imgname + 'ext'),
            useCompression=True)
