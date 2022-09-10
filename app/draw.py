#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# @Date    : 2021/6/1
# @Author  : HH
# @File    : draw_6_1.py
# @Project : funcCheck
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from tqdm import tqdm
import time
import sys
import datetime


# mjd转datetime类
def mjd2time(mjd):
    t0 = datetime.datetime(1858, 11, 17, 0, 0, 0, 0)  # 简化儒略日起始日
    return t0 + datetime.timedelta(days=mjd)


def draw_cand(dm_point_data, pic_data_t, dm_num, fre, save_path, info, dm_min, dm_step):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_data = np.asarray(pic_data_t)
    del pic_data_t
    pic_row = pic_data.shape[0]
    if info is None:
        fre_low = 1000
        fre_hight = 1500
    else:
        fre_low = info[1]
        fre_hight = info[2]
    max_dm_index = np.argmax(dm_point_data)

    plt.rcParams['font.size'] = 12
    fig = plt.figure(0, figsize=(15, 10))
    color_map = plt.cm.rainbow
    # 调整子图数量
    # spec = fig.add_gridspec(nrows=4, ncols=6, width_ratios=[2, 1], height_ratios=[1, 1, 1, 1])
    spec = fig.add_gridspec(nrows=4, ncols=10, height_ratios=[1, 1, 1, 1])
    # 设置画布 子图间距
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.96, hspace=0.352, wspace=0.6)

    # pic1
    ax1 = fig.add_subplot(spec[0, :8])
    plt.xlim([0, dm_num])
    plt.xlabel("DM($\mathit{pc}\\bullet{cm}^{-3}$)")
    plt.ylabel("Amplitude")
    x_list = np.arange(dm_point_data.shape[0])
    y_max = np.max(dm_point_data)
    ax1.plot(x_list, dm_point_data, "-")
    x_line = [max_dm_index, max_dm_index]
    y_layer = [0, y_max]
    ax1.plot(x_line, y_layer, "r-")
    ax1.text(max_dm_index, y_max, "DM {:.2f}".format(max_dm_index*dm_step), color="r", ha='left', va='center')
    plt.xticks(np.linspace(0, dm_num, num=5), np.around(np.linspace(0, dm_num*dm_step, num=5), 1))

    # pic2
    ax2 = fig.add_subplot(spec[1, :8], sharex=ax1)
    im = ax2.imshow(pic_data, origin='lower', interpolation='none', aspect="auto",
                    cmap=color_map)  # cmap=plt.cm.rainbow
    ax2.set(adjustable="datalim")
    plt.yticks(np.linspace(0, pic_row, num=5), np.around(np.linspace(fre_low, fre_hight, num=5), 2))
    plt.ylabel("Frequency/MHz")

    plt.xlim([0, dm_num])
    plt.xticks(np.linspace(0, dm_num, num=5), np.around(np.linspace(0, dm_num*dm_step, num=5), 1))
    plt.xlabel("DM($\mathit{pc}\\bullet{cm}^{-3}$)")

    """max line"""
    y_max = len(pic_data)
    x_line = [max_dm_index, max_dm_index]
    y_layer = [0, y_max]

    ax2.plot(x_line, y_layer, "r-")
    ax2.text(max_dm_index, -1, "DM {:.2f}".format(max_dm_index*dm_step), color="r", ha='left', va='top')

    # pic4
    ax4 = fig.add_subplot(spec[2, :5])
    plt.xlim([0, dm_num])
    plt.xlabel("DM($\mathit{pc}\\bullet{cm}^{-3}$)")
    plt.ylabel("Amplitude")
    x_list_ax4 = np.arange(int(dm_num/ 10))
    ax4.plot(x_list_ax4, dm_point_data[:int(dm_num/ 10)], "-")
    plt.xlim([0, dm_num/10])
    plt.xticks(np.linspace(0, dm_num/10, num=5), np.around(np.linspace(0, dm_num*dm_step/10, num=5), 1))

    # pic5
    ax5 = fig.add_subplot(spec[3, :5], sharex=ax4)

    ax5.imshow(pic_data[:, :int(dm_num/10)], origin='lower', interpolation='none', aspect="auto",
               cmap=color_map)  # cmap=plt.cm.rainbow
    ax5.set(adjustable="datalim")
    plt.yticks(np.linspace(0, pic_row, num=5), np.around(np.linspace(fre_low, fre_hight, num=5), 2))
    plt.ylabel("Frequency/MHz")

    plt.xlim([0, dm_num/10])
    plt.xticks(np.linspace(0, dm_num/10, num=5), np.around(np.linspace(0, dm_num*dm_step/10, num=5), 1))
    plt.xlabel("DM($\mathit{pc}\\bullet{cm}^{-3}$)")

    # pic6
    ax6 = fig.add_subplot(spec[2, 5:])
    pic6_x_min = 0
    pic6_x_max = dm_num
    if max_dm_index >= dm_num/20:
        pic6_x_min = max_dm_index - int(dm_num/20)

    if max_dm_index <= (dm_num - int(dm_num/20)):
        pic6_x_max = max_dm_index + int(dm_num/20)
    pic6_range = pic6_x_max - pic6_x_min
    plt.xlabel("DM($\mathit{pc}\\bullet{cm}^{-3}$)")
    # plt.ylabel("Amplitude")
    ax6.set_yticks([])
    pic6_x_list = np.arange(pic6_range)
    ax6.plot(pic6_x_list, dm_point_data[pic6_x_min: pic6_x_max], "-")
    plt.xlim([0, pic6_range])
    plt.xticks(np.linspace(0, pic6_range, num=5), np.around(np.linspace(pic6_x_min*dm_step, pic6_x_max*dm_step, num=5), 1))

    # pic7
    ax7 = fig.add_subplot(spec[3, 5:], sharex=ax6)
    im = ax7.imshow(pic_data[:, pic6_x_min:pic6_x_max], origin='lower', interpolation='none', aspect="auto",
                    cmap=color_map)
    ax7.set(adjustable="datalim")
    ax7.set_yticks([])
    # plt.yticks(np.linspace(0, pic_row, num=5), np.linspace(fre_low, fre_hight, num=5))
    # plt.ylabel("Frequency/MHz")

    plt.xlim([0, pic6_range])
    plt.xticks(np.linspace(0, pic6_range, num=5), np.around(np.linspace(pic6_x_min*dm_step, pic6_x_max*dm_step, num=5), 1))
    plt.xlabel("DM($\mathit{pc}\\bullet{cm}^{-3}$)")

    position = fig.add_axes([0.12, 0.04, 0.8, 0.01])  # 位置[左,下,右,上]
    cb = plt.colorbar(im, cax=position, orientation='horizontal')  # 方向

    # parameter
    ax3 = fig.add_subplot(spec[:2, 8:])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    if info is None:
        show_msg = "DM($\mathit{pc}\\bullet{cm}^{-3}$)" + """={:.1f}\n\nFrequency(Hz)={:.4f}\n\nPeriod(ms)={:.4f}\n\nDM_Max/DM_Mean={:.3f}""".format(
            max_dm_index*dm_step, fre, 1e3 / fre, dm_point_data[max_dm_index] / np.mean(dm_point_data))
        ax3.text(0, 0.6, show_msg, transform=ax3.transAxes, fontsize=12)
    else:
        show_msg = "Telescope:  {}\n\nObservation file:\n {}*\n\nObs start time:\n {}\n\nMJD start time:\n {:.14f}\n\nSource Name:  {}\n\n".format(info[0], info[3], mjd2time(float(info[5][0])), info[5][0], info[4])
        show_msg += "DM($\mathit{pc}\\bullet{cm}^{-3}$)" + """={:.1f}\n\nFrequency(Hz)={:.4f}\n\nPeriod(ms)={:.4f}\n\nDM_Max/DM_Mean={:.3f}""".format(
            max_dm_index*dm_step, fre, 1e3 / fre, dm_point_data[max_dm_index] / np.mean(dm_point_data))
        ax3.text(0, 0, show_msg, transform=ax3.transAxes, fontsize=12)
    # plt.title("Parameter")


    # save as svg and dpi 600
    # plt.show()
    pic_name = os.path.join(save_path, "pic_{:.5f}_{:.2f}.png".format(fre, max_dm_index*dm_step))
    plt.savefig(fname=pic_name, dpi=150)
    del dm_point_data
    del pic_data
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    plt.close("all")


def files_draw(file_list, save_path):
    fre_list = []
    for file_i in tqdm(file_list):
        pad_data = []
        if "pic" in file_i:
            file_path = os.path.join(path, file_i)
            with open(file_path, "rb") as raw:
                info = None
                raw_data = raw.read(9)
                raw_data = raw.read(4 * 3)
                (head_len, dm_num, dm_channel) = struct.unpack('3i', raw_data)
                raw_data = raw.read(4 * 3)
                (dm_min, dm_max, fre) = struct.unpack('3f', raw_data)
                dm_step = (dm_max-dm_min)/dm_num
                if head_len > 33:
                    [low_channel, high_channel] = np.frombuffer(raw.read(16), dtype=np.float64)
                    telescope = raw.read(40).decode()
                    source_file = raw.read(256).decode()
                    source_name = raw.read(100).decode()
                    stt = np.frombuffer(raw.read(16), dtype=np.float128)
                    raw.read(16)
                    frontend = raw.read(100).decode()
                    backend = raw.read(100).decode()
                    info = [telescope.replace("\x00", ""), low_channel, high_channel, source_file.replace("\x00", ""), source_name.replace("\x00", ""), stt, frontend.replace("\x00", ""), backend.replace("\x00", "")]
                raw.seek(head_len, 0)
                pad_num = 140  # int(dm_num * 0.4 / dm_channel)
                dm_point_data = np.frombuffer(raw.read(4 * dm_num), dtype=np.float32)
                dm_max = np.argmax(dm_point_data) / 10
                if dm_max > 1:
                    fre_list.append([fre, dm_max])
                pic_data = np.frombuffer(raw.read(4 * dm_num * dm_channel), dtype=np.float32).reshape(dm_channel,
                                                                                                      dm_num)
                for d in pic_data:
                    d = d * 255 / np.max(d)
                    for i in range(pad_num):
                        pad_data.append(d)
                draw_cand(dm_point_data, pad_data, dm_num, fre, save_path, info, dm_min, dm_step)
                del dm_point_data
                del pad_data
                del pic_data

    # fre_list = np.asarray(fre_list)
    # fre_list = fre_list[np.argsort(fre_list[:, 0])].tolist()
    # fre_info = open("file/f_info_ls.txt", "w+")
    # cand_list = []
    # 
    # for i in range(len(fre_list) - 1):
    #     mark = 0
    #     for cand_temp_i in cand_list:
    #         if fre_list[i] in cand_temp_i:
    #             mark = 1
    #     if mark == 1:
    #         continue
    #     temp_cand = [fre_list[i]]
    #     max_rate = 0
    #     for cand_i in range(i + 1, len(fre_list)):
    #         rate = fre_list[cand_i][0] / fre_list[i][0]
    #         if int(rate) > 1 and abs(round(rate) - rate) <= 0.002 and abs(fre_list[cand_i][1] - fre_list[i][1]) < \
    #                 fre_list[i][1] * 0.1 and fre_list[i][1] > 0.3:
    #             temp_cand.append(fre_list[cand_i])
    #             if max_rate <= int(rate):
    #                 max_rate = int(rate)
    #     if len(temp_cand) > 1:
    #         cand_list.append(temp_cand)
    #     # if len(temp_cand) > 1:
    #     fre_info.write("\n\n________fre({})________\n".format(max_rate))
    #     for cand in temp_cand:
    #         fre_info.write("fre:{}  DM:{}\n".format(cand[0], cand[1]))
    # fre_info.close()


def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def path_check(path1, path2):
    if os.path.exists(path1):
        if not os.path.exists(path2):
            os.makedirs(path2)
        else:
            pass
    else:
        print("put the right cand file path and image save path!")
        exit(0)


if __name__ == "__main__":
    path = sys.argv[1]
    save_path = sys.argv[2]
    path_check(path, save_path)
    # path = "/home/hh/astrodata/UsingFile/parkes_data/file/data_td525312_256_cand_data"
    # save_path = "./file_1/parkes/"
    file_list = os.listdir(path)
    files_draw(file_list, save_path)

