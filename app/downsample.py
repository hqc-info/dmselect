#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2022/2/14
# @Author  : HH
# @File    : downsample.py
# @Project : AutoProcess
import numpy as np
import struct
from tqdm import tqdm
import sys

"""
    do down sample for the preprocess data
    2->1, 4->1, 8->1
    
"""


def main():
    path = sys.argv[1]
    sumsample = int(sys.argv[2])
    save_path = path + "_downsample_{}.dat".format(sumsample)
    file = open(path, "rb")
    head_mark = file.read(9)
    head_len = np.frombuffer(file.read(4), dtype=np.int32)[0]
    data_type = np.frombuffer(file.read(4), dtype=np.int32)
    if data_type != 0:
        print("Can not deal  with such data!, the data is not the original data")
        exit()
    sample_time = np.frombuffer(file.read(4), dtype=np.float32)
    [channel_num, channel_len] = np.frombuffer(file.read(8), dtype=np.int32)
    file.seek(0)
    head_data = file.read(head_len)
    new_file = open(save_path, "wb")
    new_file.write(head_data)
    new_file.seek(17)
    new_file.write(struct.pack("f", sample_time*sumsample))
    new_file.seek(25)
    maxlen = int(np.ceil(channel_len / sumsample) * sumsample)
    output_len = int(channel_len/sumsample)
    new_file.write(struct.pack("i", output_len))
    new_file.seek(head_len)
    for channel_i in tqdm(range(channel_num)):
        tempdata = np.frombuffer(file.read(4*channel_len), dtype=np.float32)
        tempdata = tempdata[:maxlen].reshape(output_len, sumsample)
        newdata = np.sum(tempdata, axis=1, dtype=np.float32)
        new_file.write(newdata.tobytes())
        del newdata
        del tempdata
    new_file.close()
    file.close()


if __name__ == "__main__":
    main()
