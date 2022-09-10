#include<stdio.h>
#include<string.h>
#include<fileunit.h>
#include<cmdunit.h>
#include<toolunit.h>

void writeCandDataHead(FILE *tempfile, int dmNum, int dmchannel, float dmmin, float dmmax, float fre, struct fileinfo datainfo)
{
    char filehead[] = "Cand data";
    int headlen = 0;
    if(datainfo.HeadLen > 45)
    {
        headlen = 677; 
    }
    else
    {
        headlen = 33;
    }
    fwrite(filehead, sizeof(char) , sizeof(filehead)-1, tempfile);
    fwrite(&headlen, sizeof(int) , 1, tempfile);
    fwrite(&dmNum, sizeof(int) , 1, tempfile);
    fwrite(&dmchannel, sizeof(int) , 1, tempfile);
    fwrite(&dmmin, sizeof(float) , 1, tempfile);
    fwrite(&dmmax, sizeof(float) , 1, tempfile);
    fwrite(&fre, sizeof(float) , 1, tempfile);
    if(datainfo.HeadLen > 45)
    {   
        fwrite(&datainfo.loFre, sizeof(double), 1, tempfile);
        fwrite(&datainfo.hiFre, sizeof(double), 1, tempfile);
        fwrite(datainfo.telescope, sizeof(char), 40, tempfile);
        fwrite(datainfo.sourceFile, sizeof(char), 256, tempfile);
        fwrite(datainfo.sourceName, sizeof(char), 100, tempfile);
        fwrite(&datainfo.stMJD, sizeof(long double), 1, tempfile);
        fwrite(&datainfo.raj, sizeof(double), 1, tempfile);
        fwrite(&datainfo.decj, sizeof(double), 1, tempfile);
        fwrite(datainfo.frontend, sizeof(char), 100, tempfile);
        fwrite(datainfo.backend, sizeof(char), 100, tempfile);
    }
}



struct fileinfo writefile(char *path, struct fileinfo datainfo)
{
    char filehead[] = "DM Select";
    datainfo.path = path;
    datainfo.file = fopen(path, "wb");
    fwrite(filehead, sizeof(char) , sizeof(filehead)-1, datainfo.file);
    fwrite(&datainfo.HeadLen, sizeof(int) , 1, datainfo.file);
    fwrite(&datainfo.DataType, sizeof(int) , 1, datainfo.file);
    fwrite(&datainfo.SampleTime, sizeof(float) , 1, datainfo.file);
    fwrite(&datainfo.channel, sizeof(int) , 1, datainfo.file);

    if(datainfo.DataType == 2){
        fwrite(&datainfo.FFTLen, sizeof(int) , 1, datainfo.file);
    }else{
        fwrite(&datainfo.channelLen, sizeof(int) , 1, datainfo.file);
    }


    fwrite(&datainfo.DmNum, sizeof(int) , 1, datainfo.file);
    fwrite(&datainfo.DmMin, sizeof(float) , 1, datainfo.file);
    fwrite(&datainfo.DmMax, sizeof(float) , 1, datainfo.file);
    fwrite(&datainfo.DmStep, sizeof(float) , 1, datainfo.file);
    if(datainfo.HeadLen > 45)
    {
        fwrite(&datainfo.loFre, sizeof(double), 1, datainfo.file);
        fwrite(&datainfo.hiFre, sizeof(double), 1, datainfo.file);
        fwrite(datainfo.telescope, sizeof(char), 40, datainfo.file);
        fwrite(datainfo.sourceFile, sizeof(char), 256, datainfo.file);
        fwrite(datainfo.sourceName, sizeof(char), 100, datainfo.file);
        fwrite(&datainfo.stMJD, sizeof(long double), 1, datainfo.file);
        fwrite(&datainfo.raj, sizeof(double), 1, datainfo.file);
        fwrite(&datainfo.decj, sizeof(double), 1, datainfo.file);
        fwrite(datainfo.frontend, sizeof(char), 100, datainfo.file);
        fwrite(datainfo.backend, sizeof(char), 100, datainfo.file);
    }

    return datainfo;
}


struct fileinfo readfile(char *path,struct fileinfo datainfo){
    char filehead[] = "DM Select";
    int mark;
    datainfo.file = fopen(path, "rb");
    if(datainfo.file == NULL){
        printf("\nError! file:   %s   not exists!\n", path);
        exit(0);
    }
    char *headmark;
    headmark = (char *)malloc(9*sizeof(char));
    mark = fread(headmark, sizeof(char), 9, datainfo.file);
    if (strcmp(headmark, filehead) != 0){  // check filemark
        printf("The file is not right!");
        exit(0);
    }

    datainfo.path = path;
    mark += fread(&datainfo.HeadLen, sizeof(int), 1, datainfo.file);
    mark += fread(&datainfo.DataType, sizeof(int), 1, datainfo.file);
    mark += fread(&datainfo.SampleTime, sizeof(float), 1, datainfo.file);
    mark += fread(&datainfo.channel, sizeof(int), 1, datainfo.file);
    if(datainfo.DataType == 2)
    {
        mark += fread(&datainfo.FFTLen, sizeof(int), 1, datainfo.file);
    }else{
        mark += fread(&datainfo.channelLen, sizeof(int), 1, datainfo.file);
    }
    mark += fread(&datainfo.DmNum, sizeof(int), 1, datainfo.file);
    mark += fread(&datainfo.DmMin, sizeof(float), 1, datainfo.file);
    mark += fread(&datainfo.DmMax, sizeof(float), 1, datainfo.file);
    mark += fread(&datainfo.DmStep, sizeof(float), 1, datainfo.file);
    if(datainfo.HeadLen > 45)
    {
        mark += fread(&datainfo.loFre, sizeof(double), 1, datainfo.file);
        mark += fread(&datainfo.hiFre, sizeof(double), 1, datainfo.file);
        mark += fread(datainfo.telescope, sizeof(char), 40, datainfo.file);
        mark += fread(datainfo.sourceFile, sizeof(char), 256, datainfo.file);
        mark += fread(datainfo.sourceName, sizeof(char), 100, datainfo.file);
        mark += fread(&datainfo.stMJD, sizeof(long double), 1, datainfo.file);
        mark += fread(&datainfo.raj, sizeof(double), 1, datainfo.file);
        mark += fread(&datainfo.decj, sizeof(double), 1, datainfo.file);
        mark += fread(datainfo.frontend, sizeof(char), 100, datainfo.file);
        mark += fread(datainfo.backend, sizeof(char), 100, datainfo.file);
    }
    if(mark <= 0)
    {
        printf("File read error!\n");
        exit(1);
    }

    fseek(datainfo.file, 0, SEEK_END);
    datainfo.AllDataSize = ftell(datainfo.file);
    fseek(datainfo.file, datainfo.HeadLen, 0);

    return datainfo;
}


void closefile(struct fileinfo datainfo)
{

    fclose(datainfo.file);
}


float* readChannelFloatData(struct fileinfo datainfo, int channelIndex)
/*  read channel data oringial data)   */
{
    float* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType == 2 || datainfo.DataType == 1){
        printf("Please use other func(readChannelComplexData or readAllFloatData) to read this data(FFT/DM)\n");
        exit(3);
    }


    offset = (size_t)((size_t)datainfo.HeadLen + (size_t)datainfo.channelLen* channelIndex * sizeof(float));
    // printf("channel:%d, offset:%lld\n", channelIndex, offset);
    fseek(datainfo.file, offset, 0);
    tempData = (float*)malloc(datainfo.channelLen*sizeof(float));
    mark = fread(tempData, sizeof(float), datainfo.channelLen, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;

}

int* readIntData(struct fileinfo datainfo)
/*  read channel data oringial data)   */
{
    int* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType == 2 || datainfo.DataType == 0){
        printf("Please use other func(readChannelComplexData or readAllFloatData) to read this data(FFT/DM)\n");
        exit(3);
    }


    offset = datainfo.HeadLen;
    fseek(datainfo.file, offset, 0);
    tempData = (int*)malloc(datainfo.DmNum*sizeof(int)*datainfo.channel);
    mark = fread(tempData, sizeof(int), datainfo.DmNum*datainfo.channel, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;

}

float* readFloatData(struct fileinfo datainfo)
/*  read channel data oringial data)   */
{
    float* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType == 2 || datainfo.DataType == 0){
        printf("Please use other func(readChannelComplexData or readAllFloatData) to read this data(FFT/DM)\n");
        exit(3);
    }


    offset = datainfo.HeadLen;
    fseek(datainfo.file, offset, 0);
    tempData = (float*)malloc(datainfo.DmNum*sizeof(float)*datainfo.channel);
    mark = fread(tempData, sizeof(float), datainfo.DmNum*datainfo.channel, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;

}


float* readAllFloatData(struct fileinfo datainfo)
/*  read all data (float oringial or dm data)   */
{
    float* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType == 2 || datainfo.DataType == 1){
        printf("Please use other func(readAllComplexData or readDmData) to read this data\n");
        exit(3);
    }

    offset = datainfo.HeadLen;
    fseek(datainfo.file, offset, 0);
    tempData = (float*)malloc(datainfo.channelLen*sizeof(float)*datainfo.channel);
    mark = fread(tempData, sizeof(float), datainfo.channelLen*datainfo.channel, datainfo.file);

    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;


}

float* readMultiChannelFloatData(struct fileinfo datainfo, int channelIndex, int readChannelNum)
/*  read channel data oringial data)   */
{
    float* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType == 2 || datainfo.DataType == 1){
        printf("Please use other func(readChannelComplexData or readAllFloatData) to read this data(FFT/DM)\n");
        exit(3);
    }


    offset = (size_t)((size_t)datainfo.HeadLen + (size_t)datainfo.channelLen* channelIndex * sizeof(float));
    // printf("channel:%d, offset:%lld\n", channelIndex, offset);
    fseek(datainfo.file, offset, 0);
    tempData = (float*)malloc((size_t)datainfo.channelLen*sizeof(float)*readChannelNum);
    mark = fread(tempData, sizeof(float), (size_t)datainfo.channelLen*readChannelNum, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;

}


cufftComplex* readMultiChannelComplexData(struct fileinfo datainfo, int channelIndex, int readChannelNum, struct cmds cmdData)
/*  read channel data (cufftComplex fft data) */
{
    cufftComplex* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType != 2){
        printf("Please use other func(readChannelFloatData or readDmData) to read this data\n");
        exit(4);
    }

    // channelIndex = inZapChan(cmdData.zapChanMin, cmdData.zapChanMax, cmdData.zapChanNum, channelIndex);
    

    offset = (size_t)datainfo.HeadLen + (size_t)datainfo.FFTLen*channelIndex*sizeof(cufftComplex);
    fseek(datainfo.file, offset, 0);
    tempData = (cufftComplex*)malloc((size_t)datainfo.FFTLen*readChannelNum*sizeof(cufftComplex));

    mark = fread(tempData, sizeof(cufftComplex), (size_t)datainfo.FFTLen*readChannelNum, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    // printf("read channel:%d\n", channelIndex);
  
    return tempData;
}

cufftComplex* readIndexComplexData(struct fileinfo datainfo, int channelIndex, int startIndex, int indexRange)
/*  read index data (cufftComplex fft data) */
{
    cufftComplex* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType != 2){
        printf("Please use other func(readAllFloatData or readDmData) to read this data\n");
        exit(4);
    }

    offset = (size_t)datainfo.HeadLen+(size_t)((size_t)channelIndex*datainfo.FFTLen+startIndex)*sizeof(cufftComplex);
    fseek(datainfo.file, offset, 0);
    tempData = (cufftComplex*)malloc(sizeof(cufftComplex)*indexRange);
    mark = fread(tempData, sizeof(cufftComplex), indexRange, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;
}


cufftComplex* readChannelComplexData(struct fileinfo datainfo, int channelIndex, struct cmds cmdData)
/*  read channel data (cufftComplex fft data) */
{
    cufftComplex* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType != 2){
        printf("Please use other func(readChannelFloatData or readDmData) to read this data\n");
        exit(4);
    }

    // channelIndex = inZapChan(cmdData.zapChanMin, cmdData.zapChanMax, cmdData.zapChanNum, channelIndex);
    

    offset = (size_t)((size_t)datainfo.HeadLen + (size_t)datainfo.FFTLen*channelIndex*sizeof(cufftComplex));
    fseek(datainfo.file, offset, 0);
    tempData = (cufftComplex*)malloc(datainfo.FFTLen*sizeof(cufftComplex));

    mark = fread(tempData, sizeof(cufftComplex), datainfo.FFTLen, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    // printf("read channel:%d\n", channelIndex);
  
    return tempData;
}


cufftComplex* readAllComplexData(struct fileinfo datainfo)
/*  read all data (cufftComplex fft data) */
{
    cufftComplex* tempData;
    int mark;
    size_t offset;

    if(datainfo.DataType != 2){
        printf("Please use other func(readAllFloatData or readDmData) to read this data\n");
        exit(4);
    }

    offset = datainfo.HeadLen;
    fseek(datainfo.file, offset, 0);
    tempData = (cufftComplex*)malloc(datainfo.FFTLen*sizeof(cufftComplex)*datainfo.channel);
    mark = fread(tempData, sizeof(cufftComplex), datainfo.FFTLen*datainfo.channel, datainfo.file);
    if(mark == 0){
        printf("Read Null\n");
    }
    return tempData;
}

void writeIntData(struct fileinfo datainfo, int* data, size_t writenum)
{
    fwrite(data, sizeof(int), writenum, datainfo.file);
}


void writeFloatData(struct fileinfo datainfo, float* data, size_t writenum)
{
    fwrite(data, sizeof(float), writenum, datainfo.file);
}

void writeComplexData(struct fileinfo datainfo, cufftComplex* data, size_t writenum)
{
    fwrite(data, sizeof(cufftComplex), writenum, datainfo.file);
}