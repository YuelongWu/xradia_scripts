import SimpleITK as sitk
import glob
from scipy.io import loadmat
import os

rootpath = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210104/data_dump'

ext = '.nrrd'

matlist = glob.glob(os.path.join(rootpath, '**', '*.mat'), recursive=True)
flist = [m.replace('.mat', ext) for m in matlist]

for mname, fname in zip(matlist,flist):
    if os.path.isfile(fname):
        img = sitk.ReadImage(fname)
        t = loadmat(mname)
        psz = t['pixel_size'].item()
        pz0 = img.GetSpacing()
        if pz0[0] != psz:
            print('rescaling {}'.format(fname))
            img.SetSpacing((psz, psz, psz))
            sitk.WriteImage(img, fname, useCompression=True)
    
    fnamet = fname.replace(ext, '_normalized'+ext)
    if os.path.isfile(fnamet):
        img = sitk.ReadImage(fnamet)
        t = loadmat(mname)
        psz = t['pixel_size'].item()
        pz0 = img.GetSpacing()
        if pz0[0] != psz:
            print('rescaling {}'.format(fname))
            img.SetSpacing((psz, psz, psz))
            sitk.WriteImage(img, fnamet, useCompression=True)

