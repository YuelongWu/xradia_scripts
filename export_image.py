import os
import sys
import numpy as np
import xradia_data as xd
from scipy.io import savemat
from imageio import mimwrite, imwrite
import glob
from datetime import datetime, timedelta
import SimpleITK as sitk
import time

metaonly = False
parent_dir = '/n/boslfs/LABS/lichtman_lab/uCT/P0_20210223'
if metaonly:
    out_parent_dir = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02230/meta'
else:
    out_parent_dir = '/n/boslfs02/LABS/lichtman_lab/uct/P0_20210223/P0_02230/data_dump'
    maxproj_dir = os.path.join(out_parent_dir, 'maxproj')
    if not os.path.isdir(maxproj_dir):
        os.makedirs(maxproj_dir)

if not os.path.isdir(out_parent_dir):
    os.makedirs(out_parent_dir)


subfolders = ['P0_02230']

ext = '.txm'
t0 = datetime.strptime('01/01/2021 00:00:00.000', r'%m/%d/%Y %H:%M:%S.%f')

max_scl = 0.005
rel_max_scl = 0.85
rotation = 0
# zlim = (110, 1000)
# ylim = (240, 240 + 550)
# xlim = (160, 160 + 660)
bit_depth = 8
medfilt_rad = 2
threads_num = 20

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads_num)

refdefined = False
for subfolder in subfolders:
    flist = glob.glob(os.path.join(parent_dir, subfolder,'**','*' + ext), recursive=True)
    # print(os.path.join(parent_dir, subfolder,'**','*' + ext))
    # flist = [ft for ft in flist if "Reducing__2020-10-05_085028" in ft]
    print('{}: {} files'.format(subfolder, len(flist)))
    outdir = os.path.join(out_parent_dir, subfolder.replace(' ', '_'))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for k, txmname in enumerate(flist):
        tt0 = time.time()
        outname = os.path.basename(txmname)
        # if '1stOs_46_recon' not in outname:
        #     continue
        imgoutdir = os.path.join(outdir, outname.replace(ext, '.tiff'))
        mhaoutdir =  os.path.join(outdir, outname.replace(ext, '.mha'))
        metaoutdir =  os.path.join(outdir, outname.replace(ext, '.mat'))

        if not metaonly and os.path.isfile(metaoutdir) and k > 0:
            continue
        txrmnames = glob.glob(os.path.join(os.path.dirname(txmname), '*.txrm'))
        for txrmname in txrmnames:
            if 'Drift' not in txrmname:
                break
        try:
            txm = xd.Zeiss_TXM(txmname)
            txrm = xd.Zeiss_TXRM(txrmname)


            metadata = {}
            metadata['pixel_size'] = txm.PixelSize()
            metadata['mid_time'] = (txrm.DateMean() - t0).total_seconds() / 60
            metadata['start_time'] = (txrm.Date() - t0).total_seconds() / 60
            metadata['y_sz'] = txm.ReconImageDepth()
            metadata['x_sz'] = txm.ReconImageWidth()
            metadata['z_sz'] = txm.ReconImageHeight()
            metadata['y'] = np.mean(txm.YPosition())
            metadata['x'] = np.mean(txm.XPosition())
            metadata['z'] = np.mean(txm.ZPosition())
            metadata['global_min'] = txm.GlobalMin()
            metadata['global_max'] = txm.GlobalMax()
            metadata['proj_num'] = txm.NumOfProjects()
            metadata['angles'] = np.array(txrm.Angles())
            metadata['exptime'] = txm.ExpTimes()
            metadata['voltage'] = txm.Voltage()
            metadata['beam_hardening'] = txm['ReconSettings/BeamHardening','f'][0]
            metadata['smth_factor'] = txm['ReconSettings/ReconFilterSmoothFactor','f'][0]

            if metaonly:
                savemat(metaoutdir, metadata)
                continue

            if not refdefined:
                y0 = metadata['y']
                x0 = metadata['x']
                z0 = metadata['z']
                refdefined = True

            if not metaonly and os.path.isfile(metaoutdir):
                continue
            # dy = (y0 - metadata['y']) / metadata['pixel_size']
            # dx = (x0 - metadata['x']) / metadata['pixel_size']
            # dz = (z0 - metadata['z']) / metadata['pixel_size']

            txm.set_rotation(rotation)

            # txm.set_offset((-dx, -dz))
            # dy = round(dy)
            # blk = txm[(zlim[0]+dy):(zlim[1]+dy), ylim[0]:ylim[1], xlim[0]:xlim[1]]

            blk = txm[:,:,:]
            offsets = (0, 0, 0)


            blk = blk.astype(np.float32)
            print('{} ptp: {}'.format(outname, blk.ptp()))
            metadata['local_min'] = blk.min()
            metadata['local_max'] = blk.max()
            metadata['local_avg'] = blk.mean()

            blk = (blk / 65535) * (metadata['global_max'] - metadata['global_min']) + metadata['global_min']
            blk = blk * 100 / metadata['proj_num']
            blk = blk.clip(0, None)

            if medfilt_rad > 0 and np.any(blk > 0):
                img = sitk.GetImageFromArray(blk, isVector=False)
                medfilter = sitk.MedianImageFilter()
                # medfilter.SetNumberOfThreads(threads_num)
                medfilter.SetRadius((medfilt_rad,medfilt_rad,medfilt_rad))
                img = medfilter.Execute(img)
                blk = sitk.GetArrayFromImage(img)

            upper_grayscale = np.quantile(blk[blk>0], 0.995) / 0.75
            lower_grayscale = np.quantile(blk[blk>0], 0.4)
            blk = (blk - lower_grayscale) / (upper_grayscale - lower_grayscale)
            blk = blk.clip(0, 1)
            metadata['upper_grayscale'] = upper_grayscale
            metadata['lower_grayscale'] = lower_grayscale
            savemat(metaoutdir, metadata)

            if bit_depth == 16:
                blkt = (65535 * blk).astype(np.uint16)
            elif bit_depth == 8:
                blkt = (255 * blk).astype(np.uint8)
            else:
                blkt = blk

            mxproj = blkt.max(axis=0)
            imwrite(os.path.join(maxproj_dir, outname.replace(ext, '.png')), mxproj)

            img = sitk.GetImageFromArray(blkt, isVector=False)
            psz = metadata['pixel_size']
            img.SetSpacing((psz, psz, psz))
            offsets = tuple([float(t) * psz for t in offsets])
            img.SetOrigin(offsets)

            sitk.WriteImage(img, mhaoutdir, useCompression=True)

        except KeyboardInterrupt:
            raise

        except:
            e = sys.exc_info()[0]
            print('{} error: {}'.format(outname, e))
        # print('{}: {} sec'.format(outname, time.time() - tt0))
