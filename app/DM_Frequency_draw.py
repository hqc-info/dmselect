#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2022/5/30
# @Author  : HH
# @File    : DM_Frequency_draw.py
# @Project : imagecheck

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import exposure
import sys
import struct

"""
    Cand data read unit        
"""
global mark


class candUnit:
    def __init__(self, cand_path):
        self.line = None
        self.file = open(cand_path, "rb")
        raw_data = self.file.read(9)
        raw_data = self.file.read(4 * 3)
        (head_len, self.dm_num, dm_channel) = struct.unpack('3i', raw_data)
        raw_data = self.file.read(4 * 3)
        (self.dm_min, self.dm_max, self.fre) = struct.unpack('3f', raw_data)
        self.dm_step = (self.dm_max - self.dm_min) / self.dm_num
        [self.low_channel, self.high_channel] = np.frombuffer(self.file.read(16), dtype=np.float64)
        self.telescope = self.file.read(40).decode()
        self.source_file = self.file.read(256).decode()
        self.source_name = self.file.read(100).decode()
        self.stt = np.frombuffer(self.file.read(16), dtype=np.float128)
        [self.raj, self.decj] = np.frombuffer(self.file.read(16), dtype=np.float64)
        self.file.seek(head_len, 0)

    def readline(self):
        self.line = np.frombuffer(self.file.read(4 * self.dm_num), dtype=np.float32)
        return self.line

    def close(self):
        self.file.close()


def DMLargeFrequencyDraw(path, path_t):
    path_i = os.path.join(path, path_t)
    head_name = path_t.split("td")[0]
    files = os.listdir(path_i)
    fre_min = 5
    fre_max = 1200
    image_data = np.zeros((1200, 500))
    source_file = ""
    for i in files:
        tempCand = candUnit(os.path.join(path_i, i))
        if fre_min <= tempCand.fre < fre_max:
            temp_line = tempCand.readline()[:5000:10]
            # print(temp_line.shape)
            source_file = tempCand.source_file
            fre_index = int(np.round(tempCand.fre))
            if fre_index == fre_max:
                fre_index -= 1
            image_data[fre_index] += temp_line
            del temp_line
        tempCand.close()
        del tempCand
    data = image_data.T
    del image_data
    data = exposure.equalize_hist(data) * 255
    data = np.maximum(data, 254.5)  # M71
    return data, source_file


path = sys.argv[1]
long_range_data, source_file = DMLargeFrequencyDraw("./", path)

fig = plt.figure(figsize=[24, 12])
plt.rcParams['font.size'] = 24
plt.xlabel("Frequency/Hz")
plt.ylabel("DM/$pc\cdot cm^{-3}$")
plt.imshow(long_range_data, aspect="auto", cmap=plt.get_cmap('PuBuGn_r'))

plt.show()
