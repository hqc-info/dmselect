#ifndef _CMDUNIT_H
#define _CMDUNIT_H

#include<stdio.h>

struct cmds
{
    char *filePath = NULL;               // the data file path
    char *dmPath = NULL;                 // the DM data file path
    char *outputPath = NULL;             // the output path
    int saveMark = 0;                    
    int streamNum = 1;                   // Gpu stream
    int funcNum = -1;                    // func mark fft or pic select, 0 for fft/ 1 for pic draw/ 2 for singal pic draw/ 3 for DM-frequency count 
    float selectNumRate = 0.005;         // the rate of DM0(filter cand)
    float selectThresholdRate = 0.25;    // select func threshold
    int selectFrequencyNum = 20;         // power filter low frequency num
    int candIndex = 0;                   // draw singal candindex's pic 
    int picChannelStep = 128;            // pic's channel step  128
    int picSumChannelNum = 1024;         // pic's sum channel num   1024
    int picPramChange = 0;               // picSumChannelNum or picChannelStep has been changed
    int startIndex =0 ;                  // Dealing with special index 
    int indexRangeNum = 100;             // the num of index, started at startIndex
    int downsample = 1;                  // down sample for data compute
    int gpuId = 0;                       // set GPU device ID

    int *zapChanMin;                     // for zapchan
    int *zapChanMax;
    int zapChanNum = 0;
};

void argvTips();
struct cmds getCmdsData(int argc, char  **argv);
#endif