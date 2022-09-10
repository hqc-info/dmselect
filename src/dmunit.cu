#include<dmunit.h>
#include<cuda.h>
#include <stdio.h>
#include <common.h>
#include <cuda_runtime.h>

__global__ void channelComplexSumAbs(cufftComplex *indata, float *outdata, unsigned int datalen, int channel){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 0;
    cufftComplex temp;
    float outValue=0;
    for(i=0;i<channel;i++){
        temp = indata[ix+i*datalen];
        outValue += sqrtf(temp.x*temp.x+temp.y*temp.y);
    }
    outdata[ix] += outValue;
}

__global__ void printinfotest(cufftComplex *indata){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    printf("complex:%f\n", indata[ix].x);
}


__global__ void printinfotestf(float *indata){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    printf("float:%f\n", indata[ix]);
}


__global__ void channelComplexSum(cufftComplex *indata, cufftComplex *outdata, unsigned int datalen, int channel){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 0;
    cufftComplex temp;
    cufftComplex outValue;
    for(i=0;i<channel;i++){
        outValue.x += indata[ix+i*datalen].x;
        outValue.y += indata[ix+i*datalen].y;
    }
    outdata[ix].x += outValue.x;
    outdata[ix].y += outValue.y;
}


__global__ void  calculatePicRowData(cufftComplex* complexDmData, float* picRowData, int sumChannelNum, int channelStep, int dmNum, int channel){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int x = (ix%dmNum);
    unsigned int y = ix/dmNum;
    cufftComplex temp;
    temp.x=0;
    temp.y=0;
    int index = 0;
    for(int loop=0; loop < sumChannelNum; loop++)
    {   
        index = (y*channelStep+loop)+x*channel;
        temp.x += complexDmData[index].x;
        temp.y += complexDmData[index].y;
    }
    picRowData[ix] = sqrtf(temp.x*temp.x+temp.y*temp.y);

}



// 筛选dm算法
__global__ void partSelectFunc(cufftComplex *fftData, unsigned int dataIndex, unsigned int fftIndex, int *dmdata, cufftComplex *outdata,int channelNum, unsigned int channelLen, unsigned int dataLen, int downsample){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * channelNum + ix;
    float pi2 = 6.28318534;
    cufftComplex temp_complex;
    cufftComplex temp_complex2;
    float s, c;
    temp_complex2 = fftData[dataIndex*channelNum+ ix];
    sincosf(pi2*dmdata[idx]*(fftIndex)/channelLen/downsample, &s, &c);
    temp_complex.x = temp_complex2.x*c + (-1)*temp_complex2.y*s;
    temp_complex.y = temp_complex2.x*s + temp_complex2.y*c;
    outdata[idx] = temp_complex;
}



// 筛选dm算法
__global__ void selectFunc(cufftComplex *fftData, unsigned int dataIndex, int *dmdata, cufftComplex *outdata, int channelNum, unsigned int channelLen, unsigned int dataLen, int downsample){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * channelNum + ix;
    float pi2 = 6.28318534;
    cufftComplex temp_complex;
    cufftComplex temp_complex2;
    float s, c;
    temp_complex2 = fftData[dataIndex+ ix*dataLen];
    sincosf(pi2*dmdata[idx]*(dataIndex)/channelLen/downsample, &s, &c);
    temp_complex.x = temp_complex2.x*c + (-1)*temp_complex2.y*s;
    temp_complex.y = temp_complex2.x*s + temp_complex2.y*c;
    outdata[idx] = temp_complex;
}


// 筛选dm值后每个通道加和
__global__ void selectSumDM(cufftComplex *indata, cufftComplex *outdata, int channelNum){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 0;
    cufftComplex temp_complex;
    temp_complex.x = 0;
    temp_complex.y = 0;
  
    for(i=0;i<channelNum;i++){
        temp_complex.x += indata[ix*channelNum+i].x;
        temp_complex.y += indata[ix*channelNum+i].y;
    }
    outdata[ix] = temp_complex;
  
  }

  // 筛选 abs
__global__ void selectAbsDm(cufftComplex *indata, float *outdata){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    cufftComplex temp = indata[ix];
    outdata[ix] = sqrtf(temp.x*temp.x+temp.y*temp.y);
  }
  