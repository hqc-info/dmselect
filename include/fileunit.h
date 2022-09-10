#ifndef _FILEUNIT_H
#define _FILEUNIT_H

#include<stdio.h>
#include<cuda.h>
#include<cufft.h>
struct fileinfo
{
    /*
        struct for file read/write
    */
    FILE *file;
    char *path;             // the file path
    int DataType = 0;       // 0-float32 original data/ 1-int32 DM data/ 2-complex64 fft data/ 4-float32 DM data
    int HeadLen = 45;       // file head len
    int channel = 0;        // channel num
    size_t channelLen = 0;     // channel data len
    size_t FFTLen = 0;         // fft data len
    float SampleTime = 0.0; // data sampletime  us
    int DmNum = 0;          // dm times
    float DmMin = 0.0;      // the minimum DM value
    float DmMax = 0.0;      // the maximum DM value
    float DmStep = 0.0;     // DM step
    unsigned long AllDataSize = 0;   // the num of data(all data)
    unsigned long NextDataIndex = 0; // file index(for data read)
    double loFre = 0;       // fl
    double hiFre = 0;       // fh
    char telescope[40];     // 望远镜名
    char sourceFile[256];   // 观测源文件(起始文件)
    char sourceName[100];   // 源名
    long double stMJD=0;    // 观测起始时间
    double raj = 0;         // RA J2000
    double decj = 0;        // Dec J2000 
    char frontend[100];     // 望远镜使用的观测前端
    char backend[100];      // 望远镜使用的观测后端
};

/* basic data func */
struct fileinfo readfile(char *path,struct fileinfo datainfo);
struct fileinfo writefile(char *path,struct fileinfo datainfo);
void closefile(struct fileinfo datainfo);

/*  write data func */
void writeIntData(struct fileinfo datainfo, int* data, size_t writenum);
void writeFloatData(struct fileinfo datainfo, float* data, size_t writenum);
void writeComplexData(struct fileinfo datainfo, cufftComplex* data, size_t writenum);


/* read data func */
float* readChannelFloatData(struct fileinfo datainfo, int channelIndex);
float* readAllFloatData(struct fileinfo datainfo);
float* readMultiChannelFloatData(struct fileinfo datainfo, int channelIndex, int readChannelNum);
int* readIntData(struct fileinfo datainfo);
cufftComplex* readChannelComplexData(struct fileinfo datainfo, int channelIndex, struct cmds cmdData);
cufftComplex* readMultiChannelComplexData(struct fileinfo datainfo, int channelIndex, int readChannelNum, struct cmds cmdData);
cufftComplex* readAllComplexData(struct fileinfo datainfo);
cufftComplex* readIndexComplexData(struct fileinfo datainfo, int channelIndex, int startIndex, int indexRange);
/* cand data func*/
void writeCandDataHead(FILE *tempfile, int dmNum, int dmchannel, float dmmin, float dmmax, float fre, struct fileinfo datainfo);
#endif