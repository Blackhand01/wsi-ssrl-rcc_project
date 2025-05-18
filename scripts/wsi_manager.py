import os
import math
import random
import openslide
import numpy as np


class CropList:
    """
    The class CropList implements a type which behaves like a python list, but reads data from disk only when required.
    This allows the usage of large WSI without being constrained by the amount of available memory.
    """
    def __init__(self, indexes, size=None):
        self.indexes = indexes
        self.size = size

    def __add__(self, b):
        return CropList(self.indexes + b.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        section = self.indexes[idx]
        region = openslide.OpenSlide(
            section['filepath_slide']).read_region([section['left'],
                                                    section['top']],
                                                   section['level'],
                                                   [section['size'],
                                                    section['size']])
        region = region.convert('RGB')
        region = region.resize(size=(self.size, self.size))
        return np.array(region)

    def shuffle(self):
        random.shuffle(self.indexes)


class SectionManager:
    def __init__(self, crop_size, overlap=1):
        """
        # SectionManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        @param crop_size: crop_size
        @param overlap: overlap (%)
        """
        self.defaultSide = crop_size
        self.level = 0
        self.overlap = int(1/overlap)

    def __generateSections__(self,
                             x_start,
                             y_start,
                             width,
                             height,
                             downsample,
                             side=None):
        if side is None:
            side = self.defaultSide

        step = int(side / self.overlap)
        self.__sections__ = []

        n_crops = 0
        print('_'*15)
        print("STEP: {}".format(step))
        print("Y: {}".format(y_start))
        print("X: {}".format(x_start))
        print("W {}".format(width))
        print("H {}".format(height))
        print("DOWNSAMPLE: {}".format(downsample))

        # N.B. Crops are considered in the 0 level
        for y in range(int(math.floor(height / step))):
            for x in range(int(math.floor(width / step))):
                # x * step + side is right margin of the given crop
                if x * step + side > width or y * step + side > height:
                    continue
                n_crops += 1
                self.__sections__.append(
                    {'top': y_start + step * y, 'left': x_start + step * x, 'size': math.floor(side / downsample),
                     'augmented': False})
        print("{} CROPS".format(n_crops))
        print('_'*15)

    def crop(self, filepath_slide, slide_label=None, size=None, save_dir=None):
        self.level = 0
        slide = openslide.OpenSlide(filepath_slide)
        downsample = slide.level_downsamples[self.level]
        if 'openslide.bounds-width' in slide.properties.keys():
            bounds_width = int(slide.properties['openslide.bounds-width'])
            bounds_height = int(slide.properties['openslide.bounds-height'])
            bounds_x = int(slide.properties['openslide.bounds-x'])
            bounds_y = int(slide.properties['openslide.bounds-y'])
        else:
            print("{} - no bounds".format(filepath_slide))
            bounds_width = slide.dimensions[0]
            bounds_height = slide.dimensions[1]
            bounds_x = 0
            bounds_y = 0
        if save_dir:
            pim = slide.read_region((int(bounds_x),
                                     int(bounds_y)),
                                    slide.level_count - 1,
                                    (int(bounds_width/slide.level_downsamples[-1]),
                                     int(bounds_height/slide.level_downsamples[-1])))
            basename = os.path.basename(filepath_slide)
            pim.save(os.path.join(save_dir, basename + '.png'))

        self.__generateSections__(bounds_x,
                                  bounds_y,
                                  bounds_width,
                                  bounds_height,
                                  downsample)
        indexes = self.__sections__
        for index in indexes:
            index['filepath_slide'] = filepath_slide
            index['level'] = self.level
            index['slide_label'] = slide_label
        return CropList(indexes, size=size)
