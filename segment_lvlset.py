import glob
import math
from functools import partial
import os
import numpy as np
import SimpleITK as sitk
from skimage import measure, filters
from scipy import ndimage
import time
import pickle
import multiprocessing as mp


def getLargestCC(segmentation):
    # from https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    # print(segmentation.max())
    if segmentation.max() > 1:
        labels = segmentation
    else:
        labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def segment_main(parent_dir):
    imgdir = os.path.join(parent_dir, 'mha_files')
    outdir = os.path.join(parent_dir, 'seg', 'lvlset')

    imglist = glob.glob(os.path.join(imgdir, '*.nrrd'))
    if not imglist:
        raise RuntimeError('No image found')
    imglist.sort()
    seg_one_img = partial(seg_one_img_lvlset, outdir=outdir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with mp.pool.Pool(20, maxtasksperchild=10) as p:
        p.map(seg_one_img, imglist)



def seg_one_img_lvlset(mname, outdir):
    outname = os.path.join(outdir, os.path.basename(mname))
    t0 = time.time()
    if os.path.isfile(outname):
        return
    scl = 2 # downsample to speed up
    img0 = sitk.ReadImage(mname)
    spc0 = img0.GetSpacing()
    img0.SetSpacing((1.0,1.0,1.0))
    tform = sitk.Similarity3DTransform()
    tform.SetScale(scl)
    imgsz = img0.GetSize()
    ds_filter = sitk.ResampleImageFilter()
    ds_filter.SetSize(tuple([int(t/2) for t in imgsz]))
    ds_filter.SetTransform(tform)
    img = ds_filter.Execute(img0)

    grad_filter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    grad_filter.SetSigma(1.0 / scl)
    sigmoid_filter = sitk.SigmoidImageFilter()
    sigmoid_filter.SetOutputMinimum(0.0)
    sigmoid_filter.SetOutputMaximum(1.0)
    sigmoid_filter.SetAlpha(-5)
    sigmoid_filter.SetBeta(0.5)
    img_g = grad_filter.Execute(img)
    img_g = sigmoid_filter.Execute(img_g)
    img_g = sitk.Cast(img_g ** 0.5, sitk.sitkFloat32)
    img_g_np = sitk.GetArrayFromImage(img_g)

    ls_filter = sitk.GeodesicActiveContourLevelSetImageFilter()
    ls_filter.SetPropagationScaling(-1.0)
    ls_filter.SetCurvatureScaling(1.0)
    ls_filter.SetAdvectionScaling(1.0)
    ls_filter.SetMaximumRMSError(1e-3)
    ls_filter.SetNumberOfIterations(50)

    erode_filter = sitk.GrayscaleErodeImageFilter()
    erode_filter.SetKernelRadius(math.ceil(4/scl))
    gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian_filter.SetSigma(4.0 / scl)
    clip_filter = sitk.RescaleIntensityImageFilter()
    clip_filter.SetOutputMaximum(1)
    clip_filter.SetOutputMinimum(0)
    thresh_filter = sitk.OtsuThresholdImageFilter()
    thresh_filter.SetInsideValue(0)
    thresh_filter.SetOutsideValue(1)

    imgf = erode_filter.Execute(img)
    imgf = gaussian_filter.Execute(imgf)
    imgf = clip_filter.Execute(imgf) ** 0.5
    seg0 = thresh_filter.Execute(imgf)
    seg0 = erode_filter.Execute(seg0)
    seg0_np = sitk.GetArrayFromImage(seg0) > 0

    g0 = np.inf
    gg = []
    for _ in range(10):
        init_ls = sitk.SignedMaurerDistanceMap(seg0 > 0, insideIsPositive=True, useImageSpacing=True)
        seg = ls_filter.Execute(sitk.Cast(init_ls, sitk.sitkFloat32), img_g)
        seg_np = sitk.GetArrayFromImage(seg) > 0
        g = (img_g_np[seg_np > seg0_np]**2).mean()
        gg.append(g)
        if g > g0 or np.all(seg_np == seg0_np):
            print('{}:{}'.format(os.path.basename(mname), gg))
            break
        else:
            g0 = g
            seg0 = seg > 0
            seg0_np = seg_np
    
    fill_hole_filter = sitk.BinaryFillholeImageFilter()
    fill_hole_filter.SetForegroundValue(127)

    seg_out = 127 * sitk.Cast(seg0 > 0, sitk.sitkUInt8)
    seg_out_fill = fill_hole_filter.Execute(seg_out)
    seg_out = seg_out + seg_out_fill

    tform0 = sitk.Similarity3DTransform()
    tform0.SetScale(1/scl)
    us_filter = sitk.ResampleImageFilter()
    us_filter.SetSize(img0.GetSize())
    us_filter.SetTransform(tform0)
    seg_out = us_filter.Execute(seg_out)
    seg_out.SetSpacing(spc0)
    sitk.WriteImage(seg_out, outname, useCompression=True)
    V0 = round((seg0_np > 0).sum() * (scl * spc0[0]/1000)**3, 3)
    print('{}: {} sec | V {}'.format(os.path.basename(outname), time.time()-t0, V0))


def fill_hole_template(seclist, outdir):
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(18)
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(10)
    min_filter = sitk.MinimumImageFilter()
    max_filter = sitk.MaximumImageFilter()
    for k, mname in enumerate(seclist):
        if k == 0:
            seg0 = sitk.ReadImage(mname)
            spc0 = seg0.GetSpacing()
            seg0.SetSpacing((1,1,1))
        else:
            seg = sitk.ReadImage(mname)
            seg.SetSpacing((1,1,1))
            seg_diff = erode_filter.Execute(seg0 > 0)
            seg_diff = min_filter.Execute(seg_diff, seg < seg0)
            seg_diff_np = sitk.GetArrayFromImage(seg_diff)
            if np.sum(seg_diff_np > 0) < 500:
                seg0 = seg
                continue
            seg_diff = dilate_filter.Execute(seg_diff)
            seg0 = max_filter.Execute(seg, seg_diff * seg0)
            seg0.SetSpacing(spc0)
            sitk.WriteImage(seg0, os.path.join(outdir, os.path.basename(mname)), useCompression=True)
            seg0.SetSpacing((1,1,1))


def fill_hole_template_main(parent_dir, subgroups):
    segdir = os.path.join(parent_dir, 'seg', 'direct')
    outdir = os.path.join(parent_dir, 'seg', 'filled')
    seglist0 = glob.glob(os.path.join(segdir, '*.nrrd'))
    if not seglist0:
        raise RuntimeError('No image found')
    seglist0.sort()
    seglist0.reverse()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for grp in subgroups:
        seglist = [t for t in seglist0 if grp in os.path.basename(t)]
        seglist = seglist[-10:]
        fill_hole_template(seglist, outdir)


def measure_seg_volume(parent_dir):
    segdir = os.path.join(parent_dir, 'seg', 'direct')
    seglist0 = glob.glob(os.path.join(segdir, '*.nrrd'))
    seglist0.sort()
    fv = open(os.path.join(segdir, 'volumes.csv'), 'w')
    for mname in seglist0:
        img = sitk.ReadImage(mname)
        spc = img.GetSpacing()
        img_np = sitk.GetArrayFromImage(img)
        seg = img_np > 0
        v0 = seg.sum() * (spc[0] / 1000) ** 3
        bname = os.path.basename(mname).replace('.nrrd', '')
        idxt = bname.find('_')
        bname = bname[idxt+1:]
        vstr = bname + ', ' + str(round(v0, 3)) + '\n'
        print(vstr)
        fv.write(vstr)
    fv.close()


def measure_seg_volume(parent_dir):
    segdir = os.path.join(parent_dir, 'seg', 'direct')
    seglist0 = glob.glob(os.path.join(segdir, '*.nrrd'))
    seglist0.sort()
    fv = open(os.path.join(segdir, 'volumes.csv'), 'w')
    for mname in seglist0:
        img = sitk.ReadImage(mname)
        spc = img.GetSpacing()
        img_np = sitk.GetArrayFromImage(img)
        seg = img_np > 0
        v0 = seg.sum() * (spc[0] / 1000) ** 3
        bname = os.path.basename(mname).replace('.nrrd', '')
        idxt = bname.find('_')
        bname = bname[idxt+1:]
        vstr = bname + ', ' + str(round(v0, 3)) + '\n'
        print(vstr)
        fv.write(vstr)
    fv.close()


def average_seg_main(parent_dir, subgroups):
    segdir = os.path.join(parent_dir, 'seg', 'direct')
    seglist0 = glob.glob(os.path.join(segdir, '*.nrrd'))
    outdir = os.path.join(parent_dir, 'seg', 'group')
    if not seglist0:
        raise RuntimeError('No image found')
    seglist0.sort()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for grp in subgroups:
        outname = os.path.join(outdir, grp + '.nrrd')
        # if os.path.isfile(outname):
        #     continue
        seglist = [t for t in seglist0 if grp in os.path.basename(t)]
        seg_m = None
        for mname in seglist:
            img = sitk.ReadImage(mname)
            img_np = sitk.GetArrayFromImage(img)
            if seg_m is None:
                seg_m = img_np.astype(np.float32)
            else:
                seg_m += img_np
        seg_m = 255 * seg_m / seg_m.max()
        seg_out = sitk.GetImageFromArray(seg_m.astype(np.uint8), isVector=False)
        seg_out.SetSpacing(img.GetSpacing())
        sitk.WriteImage(seg_out, outname, useCompression=True)


if __name__ == '__main__':
    parent_dir = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02234'
    # subgroups = ['1stOs', 'KFeCN', 'Pyrogallol', '2ndOs', 'UA']
    segment_main(parent_dir)
    # average_seg_main(parent_dir, subgroups)
    # subgroups = ['1stOs', '2ndOs']
    # fill_hole_template_main(parent_dir, subgroups)
    # smooth_seg_main(parent_dir)
    # measure_seg_volume(parent_dir)