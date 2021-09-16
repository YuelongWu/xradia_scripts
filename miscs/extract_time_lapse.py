import os
import SimpleITK as sitk
import numpy as np
import h5py
import glob
from skimage.io import imsave, imread
from scipy.io import savemat
import time

rootpath = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02234'
max_scl0 = 1
ext = '.nrrd'
max_scl = 0.5
offst = 0
subfolder = None

z0 = 600
x0 = 320
y0 = 250

if subfolder is None:
    srcpath = os.path.join(rootpath, 'mha_files')
    outpath = os.path.join(rootpath, 'time_lapses')
    outhpath = os.path.join(rootpath, 'histogram')
else:
    srcpath = os.path.join(rootpath, 'mha_files', subfolder)
    outpath = os.path.join(rootpath, 'time_lapses', subfolder)
    outhpath = os.path.join(rootpath, 'histogram', subfolder)

if not os.path.isdir(outpath):
    os.makedirs(outpath, exist_ok=True)

if not os.path.isdir(outhpath):
    os.makedirs(outhpath, exist_ok=True)

outz = os.path.join(outpath, 'Z{}'.format(z0))
outx = os.path.join(outpath, 'X{}'.format(x0))
outy = os.path.join(outpath, 'Y{}'.format(y0))
outp = os.path.join(outpath, 'Projection')
outdirs = (outz, outx, outy, outp)
for odir in outdirs:
    if not os.path.isdir(odir):
        os.mkdir(odir)

imglist = glob.glob(os.path.join(srcpath,'*' + ext))
imglist.sort()

# mask0 = (imread(os.path.join(rootpath, 'mask_No2_ReOs.png')) < 128).astype(np.float32)
# mask0 = sitk.ReadImage(os.path.join(rootpath, 'No4_ReOs_fMask.mha'))
# mask0 = sitk.GetArrayFromImage(mask0) < 128

for imgdir in imglist:
    t0 = time.time()
    # if 'Pyro' not in imgdir:
        # continue
    imgname = os.path.basename(imgdir).replace(ext,  '.png')
    # if all(os.path.isfile(os.path.join(odir,imgname)) for odir in outdirs):
    #     print('skipping {}: {} sec'.format(imgname, time.time()-t0))
    #     continue
    img = sitk.ReadImage(imgdir)
    img_np = sitk.GetArrayFromImage(img)

    # if '1stOs' in imgdir:
    #     img_np = img_np * (0.0025 / 0.006)
    # elif 'Reducing' in imgdir:
    #     img_np = img_np * (0.002 / 0.006)
    # elif 'Pyro' in imgdir:
    #     img_np = img_np * (0.00125 / 0.006)
    # elif '2ndOs' in imgdir:
    #     img_np = img_np * (0.0045 / 0.006)
    # elif 'UA' in imgdir:
    #     pass
    # else:
    #     raise RuntimeError('No grouping')

    imgz = 255 - img_np[z0] / max_scl0
    imsave(os.path.join(outz, imgname), imgz.clip(0,255).astype(np.uint8))
    imgy = 255 - img_np[:,y0,:] / max_scl0
    imsave(os.path.join(outy, imgname), imgy.clip(0,255).astype(np.uint8))
    imgx = 255 - img_np[:,:,x0] / max_scl0
    imsave(os.path.join(outx, imgname), imgx.clip(0,255).astype(np.uint8))
    img_np_f = img_np.astype(np.float32)
    img_np_f = (img_np_f - offst).clip(0,255)
    # img_np_f = img_np_f * (1-mask0) + img_np_f[:,:,::-1] * mask0
    imgp = img_np_f.mean(axis=1) / max_scl
    imgp = 255 - imgp.clip(0, 255)
    imsave(os.path.join(outp, imgname), imgp.astype(np.uint8))
    hist, _ = np.histogram(img_np, bins=np.arange(-0.5,256,1))
    savemat(os.path.join(outhpath, imgname.replace('.png', '.mat')), {'h':hist})
    print('{}: {} sec'.format(imgname, time.time() - t0))
