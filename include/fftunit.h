#ifndef _FFTUNIT_H
#define _FFTUNIT_H

#include<stdio.h>
#include<cuda.h>
#include<cufft.h>


/* attention: this func got device data  */
cufftComplex* fftAllData(float *indata, int channel, int channelLen, int outPutLen);
/* attention: this func got host data  */
cufftComplex* fftPartOfData(float *indata, int channel, int channelLen, int outPutLen, int endMark);
void real_to_complex(float *r, cufftComplex **complx, int N);
struct fileinfo fftProcess(struct fileinfo datainfo, struct systemSource source, struct cmds cmdData);
__global__ void frePower(cufftComplex* fftData, float* power);
float* dmFrePower(float *dmFreData, int dataLen, int FFTLen);
float* fftAndAbsSumProcess(struct fileinfo datainfo, struct systemSource source, struct cmds cmdData, struct fileinfo *outputfile);
cufftComplex* fftPartOfDataAndAbsSum(float *indata, int channel, int channelLen, int outPutLen, int endMark, float *absSum);
#endif