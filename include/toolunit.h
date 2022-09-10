#ifndef _TOOLUNIT_H
#define _TOOLUNIT_H

#include<stdio.h>

struct systemSource
{
    unsigned long availableHostMem = 0;      // kB
    unsigned long availableDeviceMem = 0;    // KB
    int cpuCore = 1;
};

struct indexSort{
    float value=0;
    unsigned long index=0;
};

struct selectInfo{
    int isSelect = 0;
    float value = 0;
    unsigned long fftIndex = 0;
    float maxTomean = 0;
    int dmValue = 0;
    float fre = 0;
    float maxpeakToabs = 0;
    float mTon = 0;
    int absindex_i =0;
};
// 检查文件夹是否存在
void is_file_dir_exist(const char*dir_path);


// func for complex data sort
void quicksort(struct indexSort* sortData, size_t maxlen, size_t begin, size_t end);
void swap(struct indexSort *a, struct indexSort *b);
unsigned int* absIndexSort(float *indata, size_t dataLen);

// filter
struct selectInfo chooseSelectData(float *selectData, int dmNum, float selectRate);
struct selectInfo chooseSelectDataWithPeak(float *selectData, int dmNum, float abs);
struct selectInfo dmPowerFilter(float* powerData, int lowFreNum, int FFTLen, struct selectInfo tempInfo);

void quicksortInfo(struct selectInfo* sortData, int maxlen, int begin, int end);
void swapInfo(struct selectInfo *a, struct selectInfo *b);

struct systemSource systemSourceCheck(int prMark);
void processBar(int progress, int total);

int fftReadChannelNum(struct systemSource source, size_t channelLen, int channel);
int selectReadChannelNum(struct systemSource source, size_t fftLen, int channel, int selectNum, int dmNum, size_t picDataSize);

char* getFileName(char *filePath);
char * addPath(char *path, char *fileName);
int isProcessPartly(struct systemSource source, int channelLen, int channel);


float fftIndexToFre(unsigned int fftIndex, struct fileinfo fftInfo);

void splitMinMax(char* msg, int *min, int *max, int num);
int rangeCount(char *msg);
void getZapChanRange(int *min, int *max, int num, char *msg);
int inZapChan(int *min, int *max, int num, int index);
int mjd2utc8(double stt); // 修正儒略历转换为UTC +8(东八区时间)
#endif
