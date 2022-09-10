#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# @Date    : 2021/6/15
# @Author  : HH
# @File    : readfile.py
# @Project : tool
"""
用于阅读DMSELECT中间文件头信息
"""
import numpy as np
import sys
import datetime


# mjd转datetime类
def mjd2time(mjd):
    t0 = datetime.datetime(1858, 11, 17, 0, 0, 0, 0)  # 简化儒略日起始日
    return t0 + datetime.timedelta(days=mjd)


def select_data(file):
    """
    读取中间处理文件信息
    """
    datatype = {0:"original", 1:"DM", 2:"FFT"}
    head_len = np.frombuffer(file.read(4), dtype=np.int32)
    data_type = np.frombuffer(file.read(4), dtype=np.int32)
    sample_time = np.frombuffer(file.read(4), dtype=np.float32)
    [channel_num, channel_len, dm_num]= np.frombuffer(file.read(12), dtype=np.int32)
    [dm_min, dm_max, dm_step] = np.frombuffer(file.read(12), dtype=np.float32)
    print("{} data".format(datatype.get(data_type[0])))
    print("Sampletime(us): {}".format(sample_time[0]))
    print("Data channel: {}".format(channel_num))
    print("Data channel len: {}".format(channel_len))
    print("dm_num, dm_min, dm_max, dm_step", dm_num, dm_min, dm_max, dm_step)
    if head_len > 45:
        [low_channel, high_channel] = np.frombuffer(file.read(16), dtype=np.float64)
        telescope = file.read(40).decode()
        source_file = file.read(256).decode()
        source_name = file.read(100).decode()
        stt = np.frombuffer(file.read(16), dtype=np.float128)
        file.read(16)
        frontend = file.read(100).decode()
        backend = file.read(100).decode()
        info = [telescope.replace("\x00", ""), low_channel, high_channel, source_file.replace("\x00", ""), source_name.replace("\x00", ""), stt, frontend.replace("\x00", ""), backend.replace("\x00", "")]
        print("channel: {:.3f} - {:.3f} MHz".format(info[1], info[2]))
        print("Telescope: {}".format(info[0]))
        print("Source name: {}".format(info[4]))
        print("Obs start time: {}".format( mjd2time(float(info[5][0]))))
        print("Frontend: {}".format(info[6]))
        print("Backend: {}".format(info[7]))
        # print(info)
    

def cand_data(file):
    """
    读取最终生成的候选项文件信息
    """
    head_len = np.frombuffer(file.read(4), dtype=np.int32)
    data_type = np.frombuffer(file.read(4), dtype=np.int32)
    file.read(16)
    fre = np.frombuffer(file.read(4), dtype=np.float32)
    print("Cand data")
    print("Frequency(Hz): {:.4f}".format(fre[0]))
    if head_len > 33:
        file.seek(33, 0)
        [low_channel, high_channel] = np.frombuffer(file.read(16), dtype=np.float64)
        telescope = file.read(40).decode()
        source_file = file.read(256).decode()
        source_name = file.read(100).decode()
        stt = np.frombuffer(file.read(16), dtype=np.float128)
        file.read(16)
        frontend = file.read(100).decode()
        backend = file.read(100).decode()
        info = [telescope.replace("\x00", ""), low_channel, high_channel, source_file.replace("\x00", ""), source_name.replace("\x00", ""), stt, frontend.replace("\x00", ""), backend.replace("\x00", "")]
        print("Telescope: {}".format(info[0]))
        print("Source name: {}".format(info[4]))
        print("Obs start time: {}".format( mjd2time(float(info[5][0]))))
        print("Frontend: {}".format(info[6]))
        print("Backend: {}".format(info[7]))


if __name__ == "__main__":
    file = open(sys.argv[1], "rb")
    file_mark = file.read(9).decode()
    if file_mark == "DM Select":
        select_data(file)

    elif file_mark == "Cand data":
        cand_data(file)

    else:
        print("wrong file!")
    file.close()