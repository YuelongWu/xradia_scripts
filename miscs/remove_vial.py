import os
import SimpleITK as sitk
import numpy as np
import glob
from skimage.io import imsave, imread
from skimage.filters import threshold_triangle, gaussian
from skimage.morphology import reconstruction, binary_dilation, disk, remove_small_objects

def max_projection(imgdir, dim=0):
    img = sitk.ReadImage(imgdir)
    img_np = sitk.GetArrayFromImage(img)
    return img_np.max(axis=dim)


def remove_based_on_mask(vol, mask):
    mask = mask > 0
    modified = False
    if not np.any(mask):
        return vol, modified
    else:
        modified = True
    vol_min = np.copy(vol)
    vol_min = np.minimum(vol_min, np.flip(vol, axis=1))
    vol_min = np.minimum(vol_min, np.flip(vol, axis=2))
    vol_min = np.minimum(vol_min, np.flip(vol, axis=(1,2)))
    mask = mask.astype(np.float32)
    mask = gaussian(mask, sigma=1)
    vol_out = vol * (1-mask) + vol_min * mask
    return vol_out.astype(vol.dtype), modified


def main_max_projection_drive(rootpath):
    ext = '.mha'
    imglist = glob.glob(os.path.join(rootpath, '**', '*' + ext), recursive=True)
    imglist.sort()

    for imgdir in imglist:
        foldername = os.path.join(os.path.dirname(imgdir),'max_proj')
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        imgname = os.path.basename(imgdir).replace(ext, '.png')
        outname = os.path.join(foldername, imgname)
        if os.path.isfile(outname):
            continue
        print(outname)
        img_proj = max_projection(imgdir)
        imsave(outname, img_proj)


def main_thresholding_mask(rootpath):
    imglist = glob.glob(os.path.join(rootpath, '**', 'max_proj', '*.png'))
    imglist.sort()

    for imgdir in imglist:
        foldername = os.path.join(os.path.dirname(os.path.dirname(imgdir)), 'mask')
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        outname = os.path.join(foldername, os.path.basename(imgdir))
        # if os.path.isfile(outname):
        #    continue
        print(outname)
        img = imread(imgdir)
        tmpval = img[img < img.mean()]
        thresh = tmpval.mean() + 2.5 * tmpval.std()
        mask = img > thresh
        seed = np.copy(mask)
        seed[1:-1,1:-1] = 0
        mask = reconstruction(seed, mask)
        mask = remove_small_objects(mask > 0, min_size=64)
        se = disk(3)
        mask = binary_dilation(mask, selem=se)
        mask = mask.astype(np.uint8) * 255
        imsave(outname, mask)


def main_remove_vial(rootpath, outpath=None):
    ext = '.mha'
    exto = '.nrrd'
    imglist = glob.glob(os.path.join(rootpath, '**', '*' + ext), recursive=True)
    imglist.sort()
    # ffilt = ['P0_02234_1stOs_32_recon.mha', 'P0_02234_UA_1_recon.mha', 'P0_02234_UA_2_Recon.mha']
    for imgdir in imglist:
        imgfolder = os.path.dirname(imgdir)
        imgname = os.path.basename(imgdir)
        # if ffilt is not None and imgname not in ffilt:
        #     continue
        if outpath is not None:
            in_place = False
            relfolder = os.path.relpath(imgfolder, start=rootpath)
            outfolder = os.path.join(outpath, relfolder)
            if not os.path.isdir(outfolder):
                os.makedirs(outfolder)
            outname = os.path.join(outfolder, imgname.replace(ext, exto))
            # if os.path.isfile(outname):
            #     continue
            print(outname)
        else:
            in_place = True
            outname = imgdir
            print('In place:' + outname)
        maskdir = os.path.join(rootpath, 'mask', imgname.replace(ext, '.png'))
        if not os.path.isfile(maskdir):
            print('No mask: {}'.format(imgname))
            continue
        mask = imread(maskdir)
        img = sitk.ReadImage(imgdir)
        img_np = sitk.GetArrayFromImage(img)
        img_npout, modified = remove_based_on_mask(img_np, mask)

        if in_place and not modified:
            continue

        se = disk(15)
        maskt = binary_dilation(mask == 0, se)
        idxt = np.nonzero(np.any(maskt, axis=0))
        x1 = idxt[0].min()
        x2 = idxt[0].max()
        idxt = np.nonzero(np.any(maskt, axis=1))
        y1 = idxt[0].min()
        y2 = idxt[0].max()
        img_npout = img_npout[:, y1:y2,x1:x2]
        img_npout = img_npout[:,::-1,::-1]

        imgout = sitk.GetImageFromArray(img_npout, isVector=False)
        imgout.SetSpacing(img.GetSpacing())
        sitk.WriteImage(imgout, outname, useCompression=True)



if __name__ == '__main__':
    rootpath = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02234/data_dump/'
    outpath = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02234/data_dump/'
    # main_max_projection_drive(rootpath)
    # main_thresholding_mask(rootpath)
    main_remove_vial(rootpath, outpath)
