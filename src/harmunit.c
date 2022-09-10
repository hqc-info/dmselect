#include<harmunit.h>

/*
    add the harm data
*/
float* harmsum(float* absData, int harmNum, float minFreq, struct fileinfo dataInfo)
{
    int harm_i;
    size_t fftIndex_i;
    size_t fftLen = dataInfo.FFTLen;
    size_t maxHarmSumIndex = (size_t)(fftLen/harmNum);
    size_t minHarmSumIndex = (size_t)(minFreq*dataInfo.SampleTime*dataInfo.channelLen/1e6);
    float * newAbsData;
    newAbsData = (float *)malloc(sizeof(float)*maxHarmSumIndex);
    memset(newAbsData, 0, maxHarmSumIndex);
    for(harm_i=1; harm_i<harmNum; harm_i++)
    {   
        for(fftIndex_i=minHarmSumIndex; fftIndex_i<maxHarmSumIndex; fftIndex_i++)
        {
            newAbsData[fftIndex_i] += absData[(size_t)fftIndex_i*harm_i];
        }
    }
    return newAbsData;
    
}
