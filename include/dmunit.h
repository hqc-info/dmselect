#ifndef _DMUNIT_H
#define _DMUNIT_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
__global__ void channelComplexSumAbs(cufftComplex *indata, float *outdata, unsigned int datalen, int channel);
__global__ void selectFunc(cufftComplex *fftData, unsigned int dataIndex, int *dmdata, cufftComplex *outdata, int channelNum, unsigned int channelLen, unsigned int dataLen, int downsample);
__global__ void partSelectFunc(cufftComplex *fftData, unsigned int dataIndex, unsigned int fftIndex, int *dmdata, cufftComplex *outdata,int channelNum, unsigned int channelLen, unsigned int dataLen, int downsample);
__global__ void selectSumDM(cufftComplex *indata, cufftComplex *outdata, int channelNum);
__global__ void selectAbsDm(cufftComplex *indata, float *outdata);
__global__ void calculatePicRowData(cufftComplex* complexDmData, float* picRowData, int sumChannelNum, int channelStep, int dmNum,  int channel);
__global__ void channelComplexSum(cufftComplex *indata, cufftComplex *outdata, unsigned int datalen, int channel);
__global__ void printinfotestf(float *indata);
__global__ void printinfotest(cufftComplex *indata);
#endif