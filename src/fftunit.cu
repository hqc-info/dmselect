#include <common.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <omp.h>
#include <fftunit.h>
#include <toolunit.h>
#include <fileunit.h>
#include <cmdunit.h>
#include <math.h>
#include <dmunit.h>

void real_to_complex(float *r, cufftComplex **complx, int N)
{
    int i;
    (*complx) = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

    #pragma omp parallel for num_threads(omp_get_num_procs()/2)
    for (i = 0; i < N; i++)
    {
        (*complx)[i].x = r[i];
        (*complx)[i].y = 0;
    }
}

float* dmFrePower(float *dmFreData, int dataLen, int FFTLen){
    static int firstTime = 0;
    static cufftHandle plan = 1;
    static cufftComplex *complexData, *dComplexSamples;
    static float *tempData = NULL;
    static float *tempData2 = NULL;
    if(firstTime == 0)
    {   

        CHECK_CUFFT(cufftPlan1d(&plan, dataLen, CUFFT_C2C, 1));
        CHECK(cudaMalloc((void **)&dComplexSamples, sizeof(cufftComplex *)*dataLen));
        CHECK(cudaMalloc((void **)&tempData, sizeof(float )*FFTLen));
        tempData2 = (float*)malloc(sizeof(float)*FFTLen);
        firstTime = 1;
    }
    real_to_complex(dmFreData, &complexData, dataLen);
    CHECK(cudaMemcpy((void **)dComplexSamples, complexData, sizeof(cufftComplex)*dataLen, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples, dComplexSamples,CUFFT_FORWARD));

    frePower<<<FFTLen/32, 32>>>(dComplexSamples, tempData);
    cudaDeviceSynchronize();
    free(complexData);
    CHECK(cudaMemcpy(tempData2, tempData, sizeof(float)*FFTLen, cudaMemcpyDeviceToHost));
    return tempData2;
}

__global__ void frePower(cufftComplex* fftData, float* power){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    cufftComplex tempComplex=fftData[ix];
    power[ix] = tempComplex.x*tempComplex.x + tempComplex.y+tempComplex.y;

}



cufftComplex* fftAllData(float *indata, int channel, int channelLen, int outPutLen)
{   // when GPU Mem is 3 times big than data
    // fft for all data(all data load in mem)
    // got the output data on the device
    int i = 0;
    int dataSize = channelLen*channel;
    cufftComplex *complexData, *dComplexSamples;
    cufftHandle plan = 0;
    
    unsigned long long dataOffset = 0;
    cufftComplex *tempData = NULL;
    
    printf("\nStart FFT!\n");
    CHECK_CUFFT(cufftPlan1d(&plan, channelLen, CUFFT_C2C, 1));

    // 实数转换为复数，便于计算
    real_to_complex(indata, &complexData, dataSize);
    
    // 复制数据进入显卡
    printf("FFT:start copy data to device!\n");
    CHECK(cudaMalloc((void **)&dComplexSamples, sizeof(cufftComplex *)*dataSize));
    CHECK(cudaMemcpy((void **)dComplexSamples, complexData, sizeof(cufftComplex)*dataSize, cudaMemcpyHostToDevice));

    double fftStart = seconds();
    for(i=0; i<channel; i++)
    {// fft计算，一个通道一个通道的进行计算
        CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples+dataOffset, dComplexSamples+dataOffset,CUFFT_FORWARD));
        dataOffset += channelLen;
    }

    //FFT 数据取半
    CHECK(cudaMalloc((void **)&tempData, sizeof(cufftComplex *)*outPutLen*channel));
    dataOffset = 0;
    for(i=0;i<channel;i++){
        CHECK(cudaMemcpy((tempData+dataOffset), (dComplexSamples+i*channelLen), sizeof(cufftComplex)*outPutLen, cudaMemcpyDeviceToDevice));
        dataOffset += outPutLen;
    }
    printf("FFT time cost:%.4fs\n", seconds()-fftStart);

    free(complexData);
    CHECK(cudaFree(dComplexSamples));
    CHECK_CUFFT(cufftDestroy(plan));
    return tempData;
}


cufftComplex* fftPartOfData(float *indata, int channel, int channelLen, int outPutLen, int endMark)
{   // When system resources are limited
    // Processing part of the data each time
    // got the output data on the Host
    int i = 0;
    size_t dataSize = (size_t)channelLen*channel;
    cufftComplex *dComplexSamples;//, *complexData;
    static cufftHandle plan = 0;
    static int first = 0;

    cufftReal *devicein;

    size_t dataOffset = 0;
    size_t fftOffset = 0;
    cufftComplex *tempData = NULL;
    if(first == 0 ){
        CHECK_CUFFT(cufftPlan1d(&plan, channelLen, CUFFT_R2C, 1));
        first = 1;
    }
        
    CHECK(cudaMalloc((void **)&devicein, sizeof(cufftReal)*dataSize));
    CHECK(cudaMalloc((void **)&dComplexSamples, sizeof(cufftComplex *)*(outPutLen+1)*channel));
    CHECK(cudaMemcpy((void **)devicein, indata, sizeof(cufftReal)*dataSize, cudaMemcpyHostToDevice)); 


    for(i=0; i<channel; i++)
    {// fft计算，一个通道一个通道的进行计算
        CHECK_CUFFT(cufftExecR2C(plan, devicein+dataOffset, dComplexSamples+fftOffset));
        dataOffset += channelLen;
        fftOffset +=(outPutLen+1);
    }

    free(indata);

    tempData = (cufftComplex*)malloc(sizeof(cufftComplex *)*outPutLen*channel);
    dataOffset = 0;
    fftOffset = 0;
    for(i=0;i<channel;i++){
        CHECK(cudaMemcpy((tempData+dataOffset), (dComplexSamples+fftOffset), sizeof(cufftComplex)*outPutLen, cudaMemcpyDeviceToHost));
        dataOffset += outPutLen;
        fftOffset +=(outPutLen+1);
    }

    CHECK(cudaFree(devicein));
    CHECK(cudaFree(dComplexSamples));
    if(endMark == 1){
        CHECK_CUFFT(cufftDestroy(plan));
    }
    return tempData;
}


cufftComplex* fftPartOfDataAndAbsSum(float *indata, int channel, int channelLen, int outPutLen, int endMark, float *absSum)
{   
    // abs data when the fft is processing
    int i = 0;
    size_t dataSize = (size_t)channelLen*channel;
    cufftComplex *dComplexSamples;//, *complexData;
    static cufftHandle plan = 0;
    static int first = 0;
    static float* cuAbsSum;
    // static cufftComplex* cuAbsSum;
    static double absAddTime=0;  // abs 
    cufftReal *devicein;

    size_t dataOffset = 0;
    size_t fftOffset = 0;
    cufftComplex *tempData = NULL;
    if(first == 0 ){
        CHECK_CUFFT(cufftPlan1d(&plan, channelLen, CUFFT_R2C, 1));
        first = 1;
        CHECK(cudaMalloc((void **)&cuAbsSum, sizeof(float)*(outPutLen+1)));
        CHECK(cudaMemset(cuAbsSum, 0, sizeof(float)*(outPutLen+1)));
        // CHECK(cudaMalloc((void **)&cuAbsSum, sizeof(cufftComplex)*(outPutLen+1)));
        // CHECK(cudaMemset(cuAbsSum, 0, sizeof(cufftComplex)*(outPutLen+1)));
    }
        
    CHECK(cudaMalloc((void **)&devicein, sizeof(cufftReal)*dataSize));
    CHECK(cudaMalloc((void **)&dComplexSamples, sizeof(cufftComplex *)*(outPutLen+1)*channel));
    CHECK(cudaMemcpy((void **)devicein, indata, sizeof(cufftReal)*dataSize, cudaMemcpyHostToDevice)); 


    for(i=0; i<channel; i++)
    {// fft计算，一个通道一个通道的进行计算
        CHECK_CUFFT(cufftExecR2C(plan, devicein+dataOffset, dComplexSamples+fftOffset));
        dataOffset += channelLen;
        fftOffset +=(outPutLen+1);
    }

    free(indata);
    cudaDeviceSynchronize();
    CHECK(cudaFree(devicein));
    double templtime1 = seconds();
    // channelComplexSum<<<outPutLen/32, 32>>>(dComplexSamples, cuAbsSum, outPutLen+1, channel);  
    channelComplexSumAbs<<<outPutLen/32, 32>>>(dComplexSamples, cuAbsSum, outPutLen+1, channel); 
    cudaDeviceSynchronize();
    absAddTime += seconds() - templtime1;

    tempData = (cufftComplex*)malloc(sizeof(cufftComplex *)*outPutLen*channel);
    dataOffset = 0;
    fftOffset = 0;
    for(i=0;i<channel;i++){
        CHECK(cudaMemcpy((tempData+dataOffset), (dComplexSamples+fftOffset), sizeof(cufftComplex)*outPutLen, cudaMemcpyDeviceToHost));
        dataOffset += outPutLen;
        fftOffset +=(outPutLen+1);
    }

    
    CHECK(cudaFree(dComplexSamples));
    if(endMark == 1){
        printf("\n\n\nsum and abs time cost: %f   \n\n\n", absAddTime);
        CHECK_CUFFT(cufftDestroy(plan));
        CHECK(cudaMemcpy((void **)absSum, cuAbsSum, sizeof(float)*outPutLen, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        CHECK(cudaFree(cuAbsSum));
        // exit(0);
    }
    return tempData;
}


struct fileinfo fftProcess(struct fileinfo datainfo, struct systemSource source, struct cmds cmdData){
    /*
        this func is use for fft data save or data need to be deal with part
    */
    struct fileinfo outputinfo;
    outputinfo = datainfo;
    outputinfo.DataType = 2;
    outputinfo.FFTLen = datainfo.channelLen/2;
    char *filename = getFileName(datainfo.path);
    strcat(filename, ".fft");
    outputinfo.path = addPath(cmdData.outputPath, filename);
    

    int partMark = isProcessPartly(source,  datainfo.channelLen, datainfo.channel);
    
    outputinfo = writefile(outputinfo.path, outputinfo); 

    if(partMark == 0){ // Judge whether to process files separately
        float *tempData;
        tempData = readAllFloatData(datainfo);
        cufftComplex *complexHostData = NULL;
        cufftComplex *complexDeviceData = NULL;

        complexDeviceData = fftPartOfData(tempData, outputinfo.channel, datainfo.channelLen, outputinfo.FFTLen, 1);
        complexHostData = (cufftComplex*)malloc(sizeof(cufftComplex)*outputinfo.channel*outputinfo.FFTLen);
        CHECK(cudaMemcpy(complexHostData, complexDeviceData,sizeof(cufftComplex)*outputinfo.channel*outputinfo.FFTLen, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(complexDeviceData));
        writeComplexData(outputinfo, complexHostData, (size_t)outputinfo.channel*outputinfo.FFTLen);
        free(tempData);
        free(complexHostData);
    }else{

        // func for part of data 
        cufftComplex *complexHostData = NULL;
        int readChannel =fftReadChannelNum(source, datainfo.channelLen, datainfo.channel);
        int allChannel = datainfo.channel;
        int channelCount = 0;
        int loopTimes = ceil((float)allChannel/readChannel);
        float *tempData;
        
        // tempData = (float*)malloc(sizeof(float)*datainfo.channelLen*readChannel);
        // float *tempData2;
        // size_t offset = 0;
        double Start = seconds();
        int firstmark = 0;
        for(int i=0; i<loopTimes;i++)
        {   
            if((allChannel-i*readChannel) <readChannel){
                readChannel = allChannel-i*readChannel;
            }

            if(firstmark == 0)
            {
                firstmark = 1;
            }else{
                free(tempData);
            }
            tempData = readMultiChannelFloatData(datainfo, channelCount,readChannel); 
        

            channelCount = channelCount + readChannel;

            processBar(i, loopTimes);

            if(channelCount == allChannel){
                complexHostData = fftPartOfData(tempData, readChannel, datainfo.channelLen, datainfo.channelLen/2, 1);

            }else{
                complexHostData = fftPartOfData(tempData, readChannel, datainfo.channelLen, datainfo.channelLen/2, 0);

            }
            writeComplexData(outputinfo, complexHostData, (size_t)(datainfo.channelLen/2)*readChannel);
            fflush(outputinfo.file);

        }
        printf("FFT time cost:%.3fs\n", seconds()-Start);
        free(complexHostData);
        

    }
    closefile(outputinfo);
    closefile(datainfo);

    printf("FFT file save at: %s\n\n", outputinfo.path );
    return outputinfo;
    
}

float* fftAndAbsSumProcess(struct fileinfo datainfo, struct systemSource source, struct cmds cmdData, struct fileinfo *outputfile){
    /*
        this func is use for fft data save or data need to be deal with part
    */
    struct fileinfo outputinfo;
    outputinfo = datainfo;
    outputinfo.DataType = 2;
    outputinfo.FFTLen = datainfo.channelLen/2;
    char *filename = getFileName(datainfo.path);
    strcat(filename, ".fft");
    outputinfo.path = addPath(cmdData.outputPath, filename);
    
    float* absSum;
    absSum = (float*)malloc((size_t)sizeof(float)*outputinfo.FFTLen);
    memset(absSum, 0, sizeof(float)*outputinfo.FFTLen);
    
    
    outputinfo = writefile(outputinfo.path, outputinfo); 

    // func for part of data 
    cufftComplex *complexHostData = NULL;
    int readChannel =fftReadChannelNum(source, datainfo.channelLen, datainfo.channel);
    int allChannel = datainfo.channel;
    int channelCount = 0;
    int loopTimes = ceil((float)allChannel/readChannel);
    float *tempData;
    
    // tempData = (float*)malloc(sizeof(float)*datainfo.channelLen*readChannel);
    // float *tempData2;
    // size_t offset = 0;
    double Start = seconds();
    int firstmark = 0;
    printf("\nFFT and ABS sum processing... could cost a little bit time\n");
    for(int i=0; i<loopTimes;i++)
    {   
        if((allChannel-i*readChannel) <readChannel){
            readChannel = allChannel-i*readChannel;
        }

        if(firstmark == 0)
        {
            firstmark = 1;
        }else{
            free(tempData);
        }
        tempData = readMultiChannelFloatData(datainfo, channelCount,readChannel); 
    

        channelCount = channelCount + readChannel;

        processBar(i, loopTimes);

        if(channelCount == allChannel){
            complexHostData = fftPartOfDataAndAbsSum(tempData, readChannel, datainfo.channelLen, datainfo.channelLen/2, 1, absSum);

        }else{
            complexHostData = fftPartOfDataAndAbsSum(tempData, readChannel, datainfo.channelLen, datainfo.channelLen/2, 0, absSum);

        }
        writeComplexData(outputinfo, complexHostData, (size_t)(datainfo.channelLen/2)*readChannel);
        fflush(outputinfo.file);

    }
    printf("FFT time cost:%.3fs\n", seconds()-Start);
    free(complexHostData);
    

    closefile(outputinfo);
    closefile(datainfo);

    printf("FFT file save at: %s\n\n", outputinfo.path );
    // copy info data to output file info
    memcpy(outputfile, &outputinfo, sizeof(struct fileinfo));
    return absSum;
    
}