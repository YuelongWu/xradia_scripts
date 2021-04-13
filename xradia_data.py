import olefile
import numpy as np
import re
import struct
from datetime import datetime, timedelta
import time
import os
import gc
import cv2


ImageDataPatternSub = re.compile('^ImageData\d*/Image')
ImageDataPattern = re.compile('^ImageData\d*/Image\d+$')
RefImagePatternSub = re.compile('^MultiReferenceData/Image')
RefImagePattern = re.compile('^MultiReferenceData/Image\d+$')
AMCImagePatternSub = re.compile('^AMC/ImageData\d*/Image')
AMCImagePattern = re.compile('^AMC/ImageData\d*/Image\d+$')


class Metadata_Registry:
    # helper class that keeps lists of metadata to dump
    metadata_set = {}
    def __init__(self, decorated = None):
        self.func = decorated

    def __set_name__(self, owner, name):
        if owner.__name__ not in Metadata_Registry.metadata_set:
            Metadata_Registry.metadata_set[owner.__name__] = set()
            # inherit metadata
            for parent in owner.__bases__:
                if parent.__name__ in Metadata_Registry.metadata_set:
                    Metadata_Registry.metadata_set[owner.__name__].update(
                        Metadata_Registry.metadata_set[parent.__name__])
        if self.func is None:
            setattr(owner, name, True)
        else:
            Metadata_Registry.metadata_set[owner.__name__].add(self.func.__name__)
            setattr(owner, name, self.func)



class Zeiss_OLE:
    registered = Metadata_Registry()
    def __init__(self, filedir, endian='<'):
        self.file_dir = os.path.normpath(filedir)
        self._endian = endian
        assert olefile.isOleFile(filedir), 'Input file should be an OLE container'
        self.ole = olefile.OleFileIO(filename=filedir, write_mode=False)

        self.clear_cache()

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, indx):
        if isinstance(indx, str):
            return self.read_stream(indx)
        if isinstance(indx, tuple) and len(indx) == 2 and isinstance(indx[0],str):
            return self.read_stream(indx[0], fmt=indx[1])
        if not indx and not isinstance(indx, int):
            return None
        if self._image_loaded:
            return self._image_block[indx]
        
        image_dict = self.image_dict
        z_indices = list(image_dict.keys())
        z_indices.sort()
        if isinstance(indx, int):
            img = self.imagedata(z_indices[indx])
            img = self.init_tform_image(img)
            return img
        elif isinstance(indx, slice):
            z_indices = z_indices[indx]
            imgdpt = len(z_indices)
            for z, idx in enumerate(z_indices):
                img = self.imagedata(idx)
                img = self.init_tform_image(img)
                if z == 0:
                    imgblock = np.empty_like(img, shape=(imgdpt, *img.shape))
                imgblock[z] = img
        elif isinstance(indx, list) and all(isinstance(t, int) for t in indx):
            z_indices_n = [z_indices[t] for t in indx]
            imgdpt = len(z_indices_n)
            for z, idx in enumerate(z_indices_n):
                img = self.imagedata(idx)
                img = self.init_tform_image(img)
                if z == 0:
                    imgblock = np.empty_like(img, shape=(imgdpt, *img.shape))
                imgblock[z] = img
        elif isinstance(indx, tuple):
            if isinstance(indx[0], (int, slice)):
                z_indices = z_indices[indx[0]]
            else:
                z_indices_n = [z_indices[t] for t in indx[0]]
                z_indices = z_indices_n
            imgdpt = len(z_indices)
            for z, idx in enumerate(z_indices):
                img = self.imagedata(idx)
                img = self.init_tform_image(img)
                img = img[indx[1], indx[2]]
                if z == 0:
                    imgblock = np.empty_like(img, shape=(imgdpt, *img.shape))
                imgblock[z] = img
        else:
            raise ValueError('unknown indexing type {}')
        if imgblock.shape[0] == 1:
            imgblock = imgblock[0]
        return imgblock

    def __iter__(self):
        self._crnt_img = 0
        return self

    def __next__(self):
        if self._crnt_img < len(self):
            img = self[self._crnt_img]
            self._crnt_img += 1
            return img
        else:
            raise StopIteration

    
    @staticmethod
    def _tform_image_cv(img_np, tforms):
        rows, cols = img_np.shape
        M = np.eye(3)
        modified = False
        for tfm in tforms:
            if np.isscalar(tfm) and tfm != 0: # rotation
                R = cv2.getRotationMatrix2D((cols/2, rows/2), tfm, 1)
                A = np.concatenate((R, [[0,0,1]]), axis=0)
                M = A @ M
                modified = True
            if isinstance(tfm, (list, tuple, np.ndarray)) and any(t for t in tfm):
                T = np.float32([[1, 0, tfm[0]], [0, 1, tfm[1]], [0, 0, 1]])
                M = T @ M
                modified = True
        if modified:
            img_np = cv2.warpAffine(img_np, M[0:2], (cols, rows))
        return img_np


    def init_tform_image(self, img):
        tforms = (self._image_offset, self._image_rotation)
        return Zeiss_OLE._tform_image_cv(img, tforms)


    def close(self):
        self.ole.close()


    def open(self):
        self.ole.open(filename=self.file_dir, write_mode=False)
        self.clear_cache()


    def clear_cache(self):
        self._stream = None
        self._storage = None
        self._image_dict = None
        self._image_block = None
        self._image_loaded = False
        self._image_downsample = None
        self._image_rotation = None
        self._image_offset = None
        for attr in self.__dict__:
            if attr.startswith('_attrcache_'):
                setattr(self, attr, None)
        gc.collect()


    def read_stream(self, stream_name, **kwarg):
        fmt = kwarg.get('fmt', None)
        prefix = kwarg.get('prefix', '')
        attrfunc = kwarg.get('attrfunc', None)
        if prefix:
            stream_name = prefix + '/' + stream_name
        # assuming little-endian
        if not self.ole.exists(stream_name):
            return None
        with self.ole.openstream(stream_name) as fin:
            s = fin.read()
        if fmt == 's1':
            s = s.split(b'\x00')[0]
            s = s.decode('utf-8', 'ignore')
        elif fmt == 'ascii':
            def isprintable(bts):
                if not bts:
                    return False
                else:
                    return all(b > 31 and b < 127 for b in bts)
            s = s.split(b'\x00')
            s = [m.decode('ascii', 'ignore') for m in filter(isprintable, s)]
        elif fmt is not None:
            fmtsz = struct.calcsize(fmt)
            datalen = len(s) // fmtsz
            s = struct.unpack(self._endian + str(datalen) + fmt, s)
        if attrfunc:
            s = attrfunc(s)
        return s


    def set_rotation(self, angle=0):
        if self._image_rotation is not None:
            to_rotate = angle - self._image_rotation
        else:
            to_rotate = angle
        self._image_rotation = angle
        if self._image_block is not None and to_rotate != 0:
            self._image_block = None
            self._image_loaded = False
            self._image_downsample = None


    def set_offset(self, offsets=(0,0)):
        if self._image_offset is not None:
            to_translate = [t1 - t2 for t1, t2 in zip(offsets, self._image_offset)]
        else:
            to_translate = offsets
        self._image_offset = offsets
        if self._image_block is not None and any(t for t in to_translate):
            self._image_block = None
            self._image_loaded = False
            self._image_downsample = None


    @staticmethod
    def _attr_first(s):
        if s is not None:
            return s[0]

    @staticmethod
    def _attr_npmean(s):
        if s is not None:
            return np.mean(s)


    def read_stream_cache(self, stream_name, attrname, **kwarg):
        prefix = kwarg.get('prefix', '')
        attrname = '_attrcache_' + attrname
        if prefix:
            attrname = attrname + '_' + prefix.replace('/', '_')
        if not hasattr(self, attrname) or getattr(self, attrname) is None:
            s = self.read_stream(stream_name, **kwarg)
            setattr(self, attrname, s)
        return getattr(self, attrname)


    def metadata(self, printfmt=True):
        yield 'file_path', self.file_dir
        metanames = Metadata_Registry.metadata_set[self.__class__.__name__]
        for mname in sorted(metanames):
            mval = getattr(self, mname)
            if callable(mval):
                mval = mval(printfmt=printfmt)
            yield mname, str(mval)


    def imagedata(self, indx):
        image_dict = self.image_dict
        imght = self.ImageHeight()
        imgwd = self.ImageWidth()
        dtype = self.DataType(npfmt=True)
        if indx not in image_dict:
            return None
        s = self.read_stream(image_dict[indx])
        return Zeiss_OLE._unpack_image(s, dtype, imght, imgwd)


    def preload_image(self, downsample=None):
        if self._image_block is None:
            t0 = time.time()
            image_dict = self.image_dict
            image_indx = list(image_dict.keys())
            image_indx.sort()
            if downsample:
                image_indx = image_indx[::downsample]
            if not image_indx:
                return
            imgdp = len(image_indx)
            for z, indx in enumerate(image_indx):
                img = self.imagedata(indx)
                if downsample:
                    img = img[::downsample,::downsample]
                img = self.init_tform_image(img)
                if z == 0:
                    image_block = np.empty_like(img, shape=(imgdp, *img.shape))
                image_block[z] = img
            self._image_block = image_block
            self._image_loaded = True
            self._image_downsample = downsample
            print("Image Loaded: " + str(round(time.time()-t0, 3)) + 'sec')

    @staticmethod
    def _unpack_image(s, dtype, imght, imgwd):
        img = np.frombuffer(s, dtype=dtype)
        return img.reshape((imght, imgwd))


    def AbsorptionScaleFactor(self, **kwarg):
        return self.read_stream('ImageInfo/AbsorptionScaleFactor', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)


    def AbsorptionScaleOffset(self, **kwarg):
        return self.read_stream('ImageInfo/AbsorptionScaleOffset', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)

    @Metadata_Registry
    def CameraBinning(self, **kwarg):
         return self.read_stream('ImageInfo/CameraBinning', fmt='I',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)

    @Metadata_Registry
    def ConeAngle(self, **kwarg):
        s = self.read_stream('ImageInfo/ConeAngle', fmt='f',
            attrfunc=Zeiss_OLE._attr_npmean, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(round(s, 1)) + ' deg'
        return s

    @Metadata_Registry
    def Current(self, **kwarg):
        s = self.read_stream('ImageInfo/Current', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(s) + ' uA'
        return s

    @Metadata_Registry
    def DataType(self, **kwarg):
        dtypes_indx = {5: ('H', 'ushort'), 10: ('f', 'float32')}
        s = self.read_stream_cache('ImageInfo/DataType', '_dtype', fmt='I',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        dt = dtypes_indx.get(s, ('B', 'uchar'))
        if kwarg.get('npfmt', False):
            return np.dtype(self._endian + dt[0])
        else:
            return dt[1]


    def Date(self, **kwarg):
        s0 = self.Dates(**kwarg)
        if s0 is None:
            return None
        s = min(s0)
        if kwarg.get('printfmt', False):
            s = s.strftime(r'%Y/%m/%d %H:%M')
        return s


    def DateMean(self, **kwarg):
        s0 = self.Dates(**kwarg)
        if s0 is None:
            return None
        t0 = min(s0)
        dt = [t - t0 for t in s0]
        s = t0 + sum(dt, timedelta()) / len(dt)
        if kwarg.get('printfmt', False):
            s = s.strftime(r'%Y/%m/%d %H:%M')
        return s


    def Dates(self, **kwarg):
        def datetimelist(ts):
            if not ts:
                return None
            timefmt = r'%m/%d/%Y %H:%M:%S.%f'
            tdlist = []
            for t in ts:
                try:
                    t0 = datetime.strptime(t, timefmt)
                    tdlist.append(t0)
                except ValueError:
                    pass
            return tdlist
        s = self.read_stream_cache('ImageInfo/Date', '_date', fmt='ascii',
            attrfunc=datetimelist, **kwarg)
        return s

    @Metadata_Registry
    def DetectorToRADistance(self, **kwarg):
        s = self.read_stream('ImageInfo/DtoRADistance', fmt='f',
            attrfunc=Zeiss_OLE._attr_npmean, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(round(s,3)) + ' mm'
        return s


    def Duration(self, **kwarg):
        s0 = self.Dates(**kwarg)
        if s0 is None:
            return None
        s = max(s0) - min(s0)
        if kwarg.get('printfmt', False):
            dt = s.total_seconds()
            s = ''
            if dt >= 3600:
                dH = dt // 3600
                s += str(dH) + 'hr '
            if dt >= 60:
                dM = int((dt // 60) % 60)
                s += str(dM) + 'm '
            dS = round(dt % 60, 3)
            s += str(dS) + 's'
        return s

    @Metadata_Registry
    def ExpTimes(self, **kwarg):
        s = self.read_stream('ImageInfo/ExpTimes', fmt='f',
            attrfunc=Zeiss_OLE._attr_npmean, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(round(s, 3)) + ' sec'
        return s


    def FileType(self, **kwarg):
        ft = {0: 'TXRM', 1: 'TXRM', 3: 'TXM'}
        indx = self.read_stream('ImageInfo/FileType', fmt='I')[0]
        return ft[indx]

    @Metadata_Registry
    def GeometricalMagnification(self, **kwarg):
        s2RA = self.SourceToRADistance(printfmt=False)
        d2RA = self.DetectorToRADistance(printfmt=False)
        if s2RA is None or d2RA is None:
            return None
        s = (abs(d2RA) + abs(s2RA)) / abs(s2RA)
        if kwarg.get('printfmt', False):
            s = round(s, 3)
        return s


    def ImageHeight(self, **kwarg):
        s = self.read_stream_cache('ImageInfo/ImageHeight', '_imght', fmt='I',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        return s


    def ImageWidth(self, **kwarg):
        s = self.read_stream_cache('ImageInfo/ImageWidth', '_imgwd', fmt='I',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        return s


    def NoOfImages(self, **kwarg):
        return self.read_stream('ImageInfo/NoOfImages', fmt='I')[0]

    @Metadata_Registry
    def ObjectiveName(self, **kwarg):
        return self.read_stream('ImageInfo/ObjectiveName', fmt='s1', **kwarg)

    @Metadata_Registry
    def OpticalMagnification(self, **kwarg):
        s = self.read_stream('ImageInfo/OpticalMagnification', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = round(s, 3)
        return s

    @Metadata_Registry
    def PixelSize(self, **kwarg):
        s = self.read_stream('ImageInfo/PixelSize', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(round(s,3)) + ' um'
        return s

    @Metadata_Registry
    def SourceFilterName(self, **kwarg):
        return self.read_stream('ImageInfo/SourceFilterName', fmt='s1', **kwarg)

    @Metadata_Registry
    def SourceToRADistance(self, **kwarg):
        s = self.read_stream('ImageInfo/StoRADistance', fmt='f',
            attrfunc=Zeiss_OLE._attr_npmean, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(round(s, 3)) + ' mm'
        return s


    def TransmissionScaleFactor(self, **kwarg):
        return self.read_stream('ImageInfo/TransmissionScaleFactor', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)


    def XPosition(self, **kwarg):
        return self.read_stream('ImageInfo/XPosition', fmt='f', **kwarg)

    def YPosition(self, **kwarg):
        return self.read_stream('ImageInfo/YPosition', fmt='f', **kwarg)

    def ZPosition(self, **kwarg):
        return self.read_stream('ImageInfo/ZPosition', fmt='f', **kwarg)

    @Metadata_Registry
    def Voltage(self, **kwarg):
        s = self.read_stream('ImageInfo/Voltage', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(s) + ' kV'
        return s

    @property
    def loaded_block_shape(self):
        if self._image_block is None:
            return None
        else:
            return self._image_block.shape

    @property
    def image_dict(self):
        if not self._image_dict:
            self._image_dict = {int(ImageDataPatternSub.sub('', m)): m
                for m in self.stream if ImageDataPattern.search(m)}
        return self._image_dict

    @property
    def stream(self):
        if not self._stream:
            stream_list = self.ole.listdir(streams=True, storages=False)
            self._stream = ['/'.join(s) for s in stream_list]
        return self._stream

    @property
    def storage(self):
        if not self._storage:
            storage_list = self.ole.listdir(streams=False, storages=True)
            self._storage = ['/'.join(s) for s in storage_list]
        return self._storage


    def change_endian(self, endian):
        assert endian in ('<', '>', '=', '|')
        self._endian = endian


    def debug_dump_all_stream(self, outfile):
        streamnames = self.stream
        with open(outfile, 'w') as f:
            f.writelines(s + '\n' for s in streamnames)



class Zeiss_TXM(Zeiss_OLE):
    registered = Metadata_Registry()
    def __init__(self, filedir):
        super().__init__(filedir)
        assert self.FileType() == 'TXM', 'Input file should be a .TXM file'

    @Metadata_Registry
    def DetectorToRADistance(self, **kwarg):
        s = super().DetectorToRADistance(printfmt=False)
        if s is None:
            return None
        s = s / 1000
        if kwarg.get('printfmt', False):
            s = str(round(s, 3)) + ' mm'
        return s

    @Metadata_Registry
    def GlobalMax(self, **kwarg):
        return self.read_stream('GlobalMinMax/GlobalMax', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)

    @Metadata_Registry
    def GlobalMin(self, **kwarg):
        return self.read_stream('GlobalMinMax/GlobalMin', fmt='f',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)

    @Metadata_Registry
    def ReconBinning(self, **kwarg):
        return self.read_stream('ReconSettings/ReconBinning', fmt='I',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)

    @Metadata_Registry
    def ReconDate(self, **kwarg):
        return self.Date(**kwarg)

    @Metadata_Registry
    def ReconImageHeight(self, **kwarg):
        return super().ImageHeight(**kwarg)

    @Metadata_Registry
    def ReconImageWidth(self, **kwarg):
        return super().ImageWidth(**kwarg)

    @Metadata_Registry
    def ReconImageDepth(self, **kwarg):
        return self.NoOfImages(**kwarg)


    def ReconTime(self, **kwarg):
        return self.Duration(**kwarg)

    @Metadata_Registry
    def NumOfProjects(self, **kwarg):
        return self.read_stream('ReconInputTomoParams/NumOfProjects', fmt='I',
            attrfunc=Zeiss_OLE._attr_first, **kwarg)

    @Metadata_Registry
    def SourceToRADistance(self, **kwarg):
        s = super().SourceToRADistance(printfmt=False)
        if s is None:
            return None
        s = -s / 1000
        if kwarg.get('printfmt', False):
            s = str(round(s, 3)) + ' mm'
        return s



class Zeiss_TXRM(Zeiss_OLE):
    registered = Metadata_Registry()
    def __init__(self, filedir):
        super().__init__(filedir)
        assert self.FileType() == 'TXRM', 'Input file should be a .TXRM file'

    def clear_cache(self):
        super().clear_cache()
        self._ref_image_dict = None
        self._amc_image_dict = None

    def ref_image(self, indx):
        image_dict = self.ref_image_dict
        imght = self.ImageHeight(prefix='MultiReferenceData')
        imgwd = self.ImageWidth(prefix='MultiReferenceData')
        dtype = self.DataType(npfmt=True, prefix='MultiReferenceData')
        s = self.read_stream(image_dict[indx])
        if indx not in image_dict:
            return None
        return Zeiss_OLE._unpack_image(s, dtype, imght, imgwd)


    def amc_image(self, indx):
        image_dict = self.amc_image_dict
        imght = self.ImageHeight(prefix='AMC')
        imgwd = self.ImageWidth(prefix='AMC')
        dtype = self.DataType(npfmt=True, prefix='AMC')
        s = self.read_stream(image_dict[indx])
        if indx not in image_dict:
            return None
        return Zeiss_OLE._unpack_image(s, dtype, imght, imgwd)

    @Metadata_Registry
    def AcquisitionDate(self, **kwarg):
        return self.Date(**kwarg)

    @Metadata_Registry
    def AcquisitionTime(self, **kwarg):
        return self.Duration(**kwarg)


    def Angles(self, **kwarg):
        s =  self.read_stream_cache('ImageInfo/Angles', '_angles', fmt='f',
            attrfunc=None, **kwarg)
        return s

    @Metadata_Registry
    def DynamicRingRemoval(self, **kwarg):
        ex = self.EncoderXShifts()
        ey = self.EncoderYShifts()
        if not ex or not ey:
            return False
        return np.ptp(ex) > 0 or np.ptp(ey) > 0


    def EncoderXShifts(self, **kwarg):
        s = self.read_stream_cache('Alignment/EncoderXShifts', '_encoder_x',
            fmt='f', **kwarg)
        return s


    def EncoderYShifts(self, **kwarg):
        s = self.read_stream_cache('Alignment/EncoderYShifts', '_encoder_y',
            fmt='f', **kwarg)
        return s

    @Metadata_Registry
    def NumOfProjects(self, **kwarg):
        return len(self.Angles())

    @Metadata_Registry
    def ImageHeight(self, **kwarg):
        return super().ImageHeight(**kwarg)

    @Metadata_Registry
    def ImageWidth(self, **kwarg):
        return super().ImageWidth(**kwarg)

    @Metadata_Registry
    def RequestedPower(self, **kwarg):
        s = self.read_stream('ImageInfo/RequestedPower', fmt='f',
            attrfunc=Zeiss_OLE._attr_npmean, **kwarg)
        if kwarg.get('printfmt', False) and s is not None:
            s = str(s) + ' W'
        return s

    @property
    def ref_image_dict(self):
        if not self._ref_image_dict:
            self._ref_image_dict = {int(RefImagePatternSub.sub('', m)): m
                for m in self.stream if RefImagePattern.search(m)}
        return self._ref_image_dict

    @property
    def amc_image_dict(self):
        if not self._amc_image_dict:
            self._amc_image_dict = {int(AMCImagePatternSub.sub('', m)): m
                for m in self.stream if AMCImagePattern.search(m)}
        return self._amc_image_dict

