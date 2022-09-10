#include <selectunit.h>
#include <fileunit.h>
#include <stdio.h>
#include <cuda.h>
#include <common.h>
#include <cuda_runtime.h>
#include <fftunit.h>
#include <cmdunit.h>
#include <toolunit.h>
#include <dmunit.h>
#include <stdlib.h>
#include <string.h>


void allInMemSelect(struct fileinfo datainfo, struct fileinfo dminfo, struct systemSource source, struct cmds cmdData){
    if(datainfo.DataType == 2){  // input data is fft data
        int channel = datainfo.channel;
        int channelLen = datainfo.FFTLen*2;
        int fftLen = datainfo.FFTLen;
        int dmNum = dminfo.DmNum;
        float selectThreshold = cmdData.selectThresholdRate;
        int lowFreNum = cmdData.selectFrequencyNum;
        int sumChannelNum = cmdData.picSumChannelNum;
        int channelStep = cmdData.picChannelStep;
        char *outputPath = cmdData.outputPath;
        int downsample = cmdData.downsample;

        if(cmdData.picPramChange == 0)
        {
            channelStep = ((float)channel/48);
            sumChannelNum = ((float)channel*15/96);
        }
        
        cufftComplex *hostData;     // the fft data on Host
        cufftComplex *deviceData;   // the fft data on GPU
        cufftComplex *complexDmData;  //the data after dm with fft data on all channel
        cufftComplex *complexDmDataSingle; // the data sum after dm with fft data

        float *absDeviceData;
        float *absHostData;
        // float *absDmData;
        float *dmDeviceSelectData;
        float *dmHostSelectData;
        int *hostDmData;
        int *deviceDmData;

        float candidateSelectRate = cmdData.selectNumRate;
        struct selectInfo* selectList; 
        int selectListNum = 0;
        int selectNum = fftLen * candidateSelectRate;
        unsigned int* selectIndex = NULL;  

        int blockX = 2;   // gpu calculation struct            
        int blockY = 64; 
        dim3 block (blockX, blockY);
        dim3 grid ((channel+block.x-1)/block.x, (dmNum+block.y-1)/block.y);

        hostData = readAllComplexData(datainfo);    // get fft data 
        hostDmData = readIntData(dminfo);           // get DM data
       


        CHECK(cudaMalloc((void **)&deviceData, sizeof(cufftComplex)*channel*fftLen));  // Transfer fft data to GPU
        CHECK(cudaMemcpy(deviceData, hostData, sizeof(cufftComplex)*channel*fftLen, cudaMemcpyHostToDevice));

        CHECK(cudaMalloc((void **)&deviceDmData, sizeof(int)*channel*dmNum));       // Transfer dm data to GPU
        CHECK(cudaMemcpy(deviceDmData, hostDmData, sizeof(int)*channel*dmNum, cudaMemcpyHostToDevice));

        CHECK(cudaMalloc((void **)&absDeviceData, sizeof(float)*fftLen));
        CHECK(cudaMemset(absDeviceData, 0, sizeof(float)*fftLen));
        absHostData = (float*)malloc(sizeof(float)*fftLen);
        
        channelComplexSumAbs<<<fftLen/32, 32>>>(deviceData, absDeviceData, fftLen, channel);  // get the abs data /same as DM0 data
        cudaDeviceSynchronize();

        CHECK(cudaMemcpy(absHostData, absDeviceData, sizeof(float)*fftLen, cudaMemcpyDeviceToHost));  // Transfer data to Host mem
        CHECK(cudaFree(absDeviceData));  // after transfer, del GPU abs data 

        /* sort the abs data for candidate index select */
        selectList = (selectInfo*)malloc(sizeof(selectInfo)*selectNum);
        selectIndex =  absIndexSort(absHostData, fftLen);  // get abs data index(after sort)
        free(absHostData);
        
        // absDmData = (float *)malloc(sizeof(float)*dmNum);
        CHECK(cudaMalloc((void **)&complexDmData, sizeof(cufftComplex)*dmNum*channel));
        CHECK(cudaMalloc((void **)&complexDmDataSingle, sizeof(cufftComplex)*dmNum));
        CHECK(cudaMalloc((void **)&dmDeviceSelectData, sizeof(float)*dmNum));
        dmHostSelectData = (float *)malloc(sizeof(float)*dmNum);

        /* loop parameter */
        size_t absindex_i = 0;
        
        /* temp select parameter*/
        struct selectInfo tempInfo;
        float* dmPower;
        dmPower = (float*)malloc(sizeof(float)*(dmNum/2));

        /* pic parameter*/
        int blockx2 = dmNum;
        int blocky2 = (channel-sumChannelNum)/channelStep;
        int picDataSize = blockx2*blocky2;

        float* picHostData;
        float* picDeviceData;
        picHostData = (float*)malloc(sizeof(float)*(picDataSize));
        CHECK(cudaMalloc((void **)&picDeviceData, sizeof(float)*(picDataSize)));


        /* data save part*/
        int firstTime = 0;
        char foldersuffix[] = "_cand_data";
        char dataFolder[255];
        char tempFileName[512] = "";
        char tempFilePath[512] = "";
        float tempFre = 0; 
        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, foldersuffix);
        printf("\nSave cand_data at:%s\n\n", dataFolder);
        FILE *tempFile;
        /* search loop */
        double selectStartT=seconds();
        int waringInfo;
        for(absindex_i=0; absindex_i<selectNum; absindex_i++)
        {   
            // Frequency domain de-dispersion for each channel
            selectFunc<<<grid, block>>>(deviceData, selectIndex[absindex_i], deviceDmData, complexDmData, channel, channelLen, fftLen, downsample);  
            cudaDeviceSynchronize();
            // add each channel
            selectSumDM<<<dmNum/32, 32>>>(complexDmData, complexDmDataSingle, channel);  // , didn't calculation the last 32 value
            cudaDeviceSynchronize();
            // abs data 
            selectAbsDm<<<dmNum/32, 32>>>(complexDmDataSingle, dmDeviceSelectData);
            cudaDeviceSynchronize();
            // copy data to host, get ready for filter

            CHECK(cudaMemcpy(dmHostSelectData, dmDeviceSelectData, sizeof(float)*dmNum, cudaMemcpyDeviceToHost));

            // Threshold filter
            tempInfo = chooseSelectData(dmHostSelectData, dmNum, selectThreshold);  

            // Power filter
            if(tempInfo.isSelect > 0 && tempInfo.dmValue !=0){
                dmPower =  dmFrePower(dmHostSelectData, dmNum, dmNum/2);
                tempInfo = dmPowerFilter(dmPower, lowFreNum, dmNum/2, tempInfo);
            }else{
                tempInfo.isSelect = 0;
            }


            if(tempInfo.isSelect > 0 )
            {   
 
                calculatePicRowData<<<blockx2, blocky2>>>(complexDmData, picDeviceData, sumChannelNum, channelStep, dmNum, channel);
                cudaDeviceSynchronize();
                CHECK(cudaMemcpy(picHostData, picDeviceData, sizeof(float)*picDataSize, cudaMemcpyDeviceToHost));
                
                if(firstTime == 0)
                {
                    firstTime =1;
                    char floderCommand[512] = "mkdir -p ";
                    strcat(floderCommand, dataFolder);
                    waringInfo = system(floderCommand);
                    if(waringInfo==-1)
                    {
                        printf("mkdir error\n");
                    }
                }

                tempFre = fftIndexToFre(selectIndex[absindex_i], datainfo);
                sprintf(tempFileName, "/dm%.1f_%f.pic", (float)tempInfo.dmValue*dminfo.DmStep, tempFre);
                strcat(tempFilePath, dataFolder);
                strcat(tempFilePath, tempFileName);

                tempFile = fopen(tempFilePath, "wb");
                writeCandDataHead(tempFile, dmNum, blocky2, dminfo.DmMin, dminfo.DmMax, tempFre, datainfo);
                
                fwrite(dmHostSelectData, sizeof(float), dmNum, tempFile);
                fwrite(picHostData, sizeof(float), picDataSize, tempFile);
                
                
                fclose(tempFile);
                strcpy(tempFilePath, "");


                selectList[selectListNum].dmValue = tempInfo.dmValue;
                selectList[selectListNum].isSelect = tempInfo.isSelect;
                selectList[selectListNum].fftIndex = selectIndex[absindex_i];
                selectList[selectListNum].fre = tempFre;
                selectList[selectListNum].maxTomean = tempInfo.maxTomean;
                selectListNum++;
            }

            processBar(absindex_i, selectNum);
        }

        quicksortInfo(selectList, selectListNum, 0, selectListNum-1);
        printf("Processing with time:%6.4fs\n\n", seconds()-selectStartT);

        printf("Candidate info:\n");
        for(int i=0;i<selectListNum;i++){
            printf("index:%3d ,DM:%4d,  fftIndex:%6ld, fre:%10.6fHz\n", i, selectList[i].dmValue, selectList[i].fftIndex, selectList[i].fre);
        }
        free(hostData);
        free(hostDmData);
        free(picHostData);
        free(dmPower);
        free(dmHostSelectData);
        free(selectList);
        CHECK(cudaFree(deviceData));
        CHECK(cudaFree(deviceDmData));
        CHECK(cudaFree(complexDmData));
        CHECK(cudaFree(complexDmDataSingle));
        CHECK(cudaFree(dmDeviceSelectData));
        CHECK(cudaFree(picDeviceData));

    

    }else if(datainfo.DataType == 0){  // input data is original data
        struct fileinfo datainfo1;
        struct fileinfo datainfo2;
        char *path = cmdData.filePath;
        datainfo1 = readfile(path, datainfo1);
        datainfo2 = fftProcess(datainfo1, source, cmdData);
        datainfo2 = readfile(datainfo2.path, datainfo2);
        allInMemSelect(datainfo2, dminfo, source, cmdData);
    }else{
        printf("Input data is wrong!\n");
        exit(0);
    }


}

void partInMemSelect(struct fileinfo datainfo, struct fileinfo dminfo, struct systemSource source, struct cmds cmdData){
    if(datainfo.DataType == 2){  // input data is fft data
        int channel = datainfo.channel;
        size_t channelLen = datainfo.FFTLen*2;
        size_t fftLen = datainfo.FFTLen;
        int dmNum = dminfo.DmNum;
        float selectThreshold = cmdData.selectThresholdRate;
        // int lowFreNum = cmdData.selectFrequencyNum;
        int sumChannelNum = cmdData.picSumChannelNum;
        int channelStep = cmdData.picChannelStep;
        char *outputPath = cmdData.outputPath;
        int downsample = cmdData.downsample;

        struct systemSource sourceUpdate = systemSourceCheck(1);
        
        // int *zapChanMin = cmdData.zapChanMin; // zapchan info
        // int *zapChanMax = cmdData.zapChanMax;
        // int zapChanNum = cmdData.zapChanNum;
        // int zapchan_i=0;

        // if(cmdData.picPramChange == 0)
        // {
        //     channelStep = 128;//((float)channel/48);
        //     sumChannelNum = 1024;//((float)channel*15/96);
        // }
        
        cufftComplex *hostData;     // the fft data on Host
        cufftComplex *deviceData;   // the fft data on GPU
        cufftComplex *complexDmData;  //the data after dm with fft data on all channel
        cufftComplex *complexDmDataSingle; // the data sum after dm with fft data

        float *absDeviceData;
        float *absHostData;
        // float *absDmData;
        float *dmDeviceSelectData;
        float *dmHostSelectData;
        int *hostDmData;
        int *deviceDmData;

        float candidateSelectRate = cmdData.selectNumRate;
        struct selectInfo* selectList; 
        int selectListNum = 0;
        size_t selectNum = fftLen * candidateSelectRate;
        unsigned int* selectIndex = NULL;  

        int blockX = 2;   // gpu calculation struct            
        int blockY = 64; 
        dim3 block (blockX, blockY);
        dim3 grid ((channel+block.x-1)/block.x, (dmNum+block.y-1)/block.y);\

        int blockx2 = dmNum;
        int blocky2 = (channel-sumChannelNum)/channelStep;
        int picDataSize = blockx2*blocky2;

        int readChannel = selectReadChannelNum(sourceUpdate, fftLen, channel, selectNum, dmNum, picDataSize);
        // int readChannel = 40;
        int loopTimes = ceil((float)channel/readChannel);
        int channelCount = 0;
        // size_t offset = 0;
        size_t read_i=0;
        size_t readloop=0;
        size_t select_i=0;
        int firstmark=0;
        int oldReadChannel = readChannel;
        // hostData = (cufftComplex*)malloc(sizeof(cufftComplex*)*readChannel*fftLen);
        // cufftComplex *tempData;
       
        CHECK(cudaMalloc((void **)&absDeviceData, sizeof(float)*fftLen));
        CHECK(cudaMalloc((void **)&deviceData, (size_t)sizeof(cufftComplex)*readChannel*fftLen));
        CHECK(cudaMemset(absDeviceData, 0, (size_t)sizeof(float)*fftLen));
        printf("ABS Sum:\n");
        double absSumTimeS = seconds();

        
        
        for(readloop=0; readloop<loopTimes; readloop++){

            if((channel-readloop*readChannel) <readChannel){
                readChannel = channel-readloop*readChannel;
            }
            if(firstmark == 0)
            {
                firstmark = 1;
            }else{
                free(hostData);
            }
            hostData = readMultiChannelComplexData(datainfo, channelCount, readChannel, cmdData);
            channelCount = channelCount + readChannel; 


            CHECK(cudaMemcpy(deviceData, hostData, (size_t)sizeof(cufftComplex)*readChannel*fftLen, cudaMemcpyHostToDevice));
            
            channelComplexSumAbs<<<fftLen/32, 32>>>(deviceData, absDeviceData, fftLen, readChannel);  
            cudaDeviceSynchronize();
            processBar(readloop, loopTimes);
        }
        int devicedChoosed;
        cudaGetDevice(&devicedChoosed);

        printf("\n\n device id: %d\n", devicedChoosed);

        hostDmData = readIntData(dminfo);           // get DM data
        absHostData = (float*)malloc(sizeof(float)*fftLen);
        CHECK(cudaMalloc((void **)&deviceDmData, sizeof(int)*channel*dmNum));       // Transfer dm data to GPU
        CHECK(cudaMemcpy(deviceDmData, hostDmData, sizeof(int)*channel*dmNum, cudaMemcpyHostToDevice));
        // printinfotest<<<32, 32>>>(deviceData);
        // printinfotestf<<<32, 32>>>(absDeviceData);

        
        CHECK(cudaMemcpy(absHostData, absDeviceData, sizeof(float)*fftLen, cudaMemcpyDeviceToHost));  // Transfer data to Host mem
        CHECK(cudaFree(absDeviceData));  // after transfer, del GPU abs data 
        CHECK(cudaFree(deviceData));
        printf("ABS Sum Cost Time:%fs\n",seconds() - absSumTimeS);
        printf("\n\n info %f %f %f %f %f %f %f\n",absHostData[1], absHostData[2], absHostData[3], absHostData[4], absHostData[5], absHostData[6], absHostData[7]);
        /* sort the abs data for candidate index select */
        selectList = (selectInfo*)malloc(sizeof(selectInfo)*selectNum);
        selectIndex =  absIndexSort(absHostData, fftLen);  // get abs data index(after sort)
        free(absHostData);
        
        // save select data(fft abs) for data check 
        // FILE *dataCheck;
        // printf("Save select index data\n");
        // dataCheck = fopen("./file/checkdata.txt", "w+");
        // for(int datacheck_i=0; datacheck_i< selectNum; datacheck_i++){
        //     fprintf(dataCheck, "%d  fft index:%d\n", datacheck_i,selectIndex[datacheck_i]);
        // }
        // fclose(dataCheck);
        // exit(0);


        cufftComplex *candFFTData;
        candFFTData = (cufftComplex *)malloc(sizeof(cufftComplex)*selectNum*channel);

        /* Got the fft data from the file*/
        unsigned int temp_index;
        readChannel = oldReadChannel;
        channelCount = 0;
        printf("Read index data:\n");
        double readIndexTimeS = seconds();
        for(readloop=0; readloop<loopTimes; readloop++){

            if((channel-readloop*readChannel) <readChannel){
                readChannel = channel-readloop*readChannel;
            }

            free(hostData);
            hostData = readMultiChannelComplexData(datainfo, channelCount, readChannel, cmdData);

            channelCount = channelCount + readChannel; 
            for(select_i=0; select_i<selectNum; select_i++)
            {   
                temp_index = selectIndex[select_i];
                for(read_i=0; read_i<readChannel; read_i++)
                {
                    candFFTData[(size_t)select_i*channel+readloop*oldReadChannel+read_i] = hostData[(size_t)read_i*fftLen+temp_index];
                }
            }
            processBar(readloop, loopTimes);
        }

        free(hostData);

        CHECK(cudaMalloc((void **)&deviceData, sizeof(cufftComplex)*selectNum*channel));
        CHECK(cudaMemcpy(deviceData, candFFTData, sizeof(cufftComplex)*selectNum*channel, cudaMemcpyHostToDevice));
        free(candFFTData);

        printf("Read index data Cost Time:%fs\n",seconds() - readIndexTimeS);

        // absDmData = (float *)malloc(sizeof(float)*dmNum);
        CHECK(cudaMalloc((void **)&complexDmData, sizeof(cufftComplex)*dmNum*channel));
        CHECK(cudaMalloc((void **)&complexDmDataSingle, sizeof(cufftComplex)*dmNum));
        CHECK(cudaMalloc((void **)&dmDeviceSelectData, sizeof(float)*dmNum));
        dmHostSelectData = (float *)malloc(sizeof(float)*dmNum);

        /* loop parameter */
        size_t absindex_i = 0;
        
        /* temp select parameter*/
        struct selectInfo tempInfo;
        float* dmPower;
        dmPower = (float*)malloc(sizeof(float)*(dmNum/2));

        /* pic parameter*/


        float* picHostData;
        float* picDeviceData;
        picHostData = (float*)malloc(sizeof(float)*(picDataSize));
        CHECK(cudaMalloc((void **)&picDeviceData, sizeof(float)*(picDataSize)));


        /* data save part*/
        int firstTime = 0;
        char foldersuffix[] = "_cand_data";
        char dataFolder[255];
        char tempFileName[512] = "";
        char tempFilePath[512] = "";
        float tempFre = 0; 
        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, foldersuffix);
        
        FILE *tempFile;

        /* search loop */
        int waringInfo;
        printf("Searching cand...\n");
        printf("\nSave cand_data at:%s\n", dataFolder);
        double selectStartT=seconds();
        for(absindex_i=0; absindex_i<selectNum; absindex_i++)
        {   
            // Frequency domain de-dispersion for each channel
            partSelectFunc<<<grid, block>>>(deviceData, absindex_i, selectIndex[absindex_i], deviceDmData, complexDmData, channel, channelLen, fftLen, downsample);  
            cudaDeviceSynchronize();
            // add each channel
            selectSumDM<<<dmNum/32, 32>>>(complexDmData, complexDmDataSingle, channel);  // , didn't calculation the last 32 value
            cudaDeviceSynchronize();
            // abs data 
            selectAbsDm<<<dmNum/32, 32>>>(complexDmDataSingle, dmDeviceSelectData);
            cudaDeviceSynchronize();
            // copy data to host, get ready for filter

            CHECK(cudaMemcpy(dmHostSelectData, dmDeviceSelectData, sizeof(float)*dmNum, cudaMemcpyDeviceToHost));

            // Threshold filter
            tempInfo = chooseSelectData(dmHostSelectData, dmNum, selectThreshold);  
            
            // Power filter
            // if(tempInfo.isSelect > 0 && tempInfo.dmValue !=0){
            //     dmPower =  dmFrePower(dmHostSelectData, dmNum, dmNum/2);
            //     tempInfo = dmPowerFilter(dmPower, lowFreNum, dmNum/2, tempInfo);
            // }else{
            //     tempInfo.isSelect = 0;
            // }

                
            if(tempInfo.isSelect > 0 && tempInfo.dmValue != 0 )
            {   
                
                calculatePicRowData<<<blockx2, blocky2>>>(complexDmData, picDeviceData, sumChannelNum, channelStep, dmNum, channel);
                cudaDeviceSynchronize();
                CHECK(cudaMemcpy(picHostData, picDeviceData, sizeof(float)*picDataSize, cudaMemcpyDeviceToHost));
                
                if(firstTime == 0)
                {
                    firstTime =1;
                    char floderCommand[512] = "mkdir -p ";
                    strcat(floderCommand, dataFolder);
                    waringInfo = system(floderCommand);
                    if(waringInfo==-1)
                    {
                        printf("mkdir error\n");
                    }
                }

                tempFre = fftIndexToFre(selectIndex[absindex_i], datainfo);
                sprintf(tempFileName, "/dm%.1f_%f.pic", (float)tempInfo.dmValue*dminfo.DmStep, tempFre);
                strcat(tempFilePath, dataFolder);
                strcat(tempFilePath, tempFileName);

                tempFile = fopen(tempFilePath, "wb");
                writeCandDataHead(tempFile, dmNum, blocky2, dminfo.DmMin, dminfo.DmMax, tempFre, datainfo);
                
                fwrite(dmHostSelectData, sizeof(float), dmNum, tempFile);
                fwrite(picHostData, sizeof(float), picDataSize, tempFile);
                
                
                fclose(tempFile);
                strcpy(tempFilePath, "");


                selectList[selectListNum].dmValue = tempInfo.dmValue;
                selectList[selectListNum].isSelect = tempInfo.isSelect;
                selectList[selectListNum].fftIndex = selectIndex[absindex_i];
                selectList[selectListNum].fre = tempFre;
                selectList[selectListNum].maxTomean = tempInfo.maxTomean;
                selectListNum++;
            }

            processBar(absindex_i, selectNum);
        }

        quicksortInfo(selectList, selectListNum, 0, selectListNum-1);
        printf("Processing with time:%6.4fs\n", seconds()-selectStartT);

        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, "cand_info.txt");

        printf("Save cand_info at:%s\n", dataFolder);
        // printf("Candidate info:\n");
        FILE *candInfo;
        candInfo = fopen(dataFolder, "wb+");
        for(int i=0;i<selectListNum;i++){
            fprintf(candInfo, "index:%3d ,DM:%4.1f,  fftIndex:%6ld, fre:%10.6fHz, Max:Mean:%4.3f\n", i, (float)selectList[i].dmValue*dminfo.DmStep, selectList[i].fftIndex, selectList[i].fre, selectList[i].maxTomean);
        }
        fclose(candInfo);
        free(hostDmData);
        free(picHostData);
        free(dmPower);
        free(dmHostSelectData);
        free(selectList);
        CHECK(cudaFree(deviceData));
        CHECK(cudaFree(deviceDmData));
        CHECK(cudaFree(complexDmData));
        CHECK(cudaFree(complexDmDataSingle));
        CHECK(cudaFree(dmDeviceSelectData));
        CHECK(cudaFree(picDeviceData));

    

    }else if(datainfo.DataType == 0){  // input data is original data
        struct fileinfo datainfo1;
        struct fileinfo datainfo2;
        char *path = cmdData.filePath;
        datainfo1 = readfile(path, datainfo1);

        float *absHostData;

        absHostData = fftAndAbsSumProcess(datainfo1, source, cmdData, &datainfo2);
        datainfo2 = readfile(datainfo2.path, datainfo2);
        datainfo = datainfo2;

        // partInMemSelect(datainfo2, dminfo, source, cmdData);
        
        int channel = datainfo.channel;
        size_t channelLen = datainfo.FFTLen*2;
        size_t fftLen = datainfo.FFTLen;
        int dmNum = dminfo.DmNum;
        float selectThreshold = cmdData.selectThresholdRate;
        // int lowFreNum = cmdData.selectFrequencyNum;
        int sumChannelNum = cmdData.picSumChannelNum;
        int channelStep = cmdData.picChannelStep;
        char *outputPath = cmdData.outputPath;
        int downsample = cmdData.downsample;

        struct systemSource sourceUpdate = systemSourceCheck(1);
        
        // if(cmdData.picPramChange == 0)
        // {
        //     channelStep = 128;//((float)channel/48);
        //     sumChannelNum = 1024;//((float)channel*15/96);
        // }
        
        cufftComplex *hostData;             // the fft data on Host
        cufftComplex *deviceData;           // the fft data on GPU
        cufftComplex *complexDmData;        //the data after dm with fft data on all channel
        cufftComplex *complexDmDataSingle;  // the data sum after dm with fft data

        // float *absDeviceData;
        // float *absHostData;
        // float *absDmData;
        float *dmDeviceSelectData;
        float *dmHostSelectData;
        int *hostDmData;
        int *deviceDmData;

        float candidateSelectRate = cmdData.selectNumRate;
        struct selectInfo* selectList; 
        int selectListNum = 0;
        size_t selectNum = fftLen * candidateSelectRate;
        unsigned int* selectIndex = NULL;  

        int blockX = 2;   // gpu calculation struct            
        int blockY = 64; 
        dim3 block (blockX, blockY);
        dim3 grid ((channel+block.x-1)/block.x, (dmNum+block.y-1)/block.y);\

        int blockx2 = dmNum;
        int blocky2 = (channel-sumChannelNum)/channelStep;
        int picDataSize = blockx2*blocky2;

        int readChannel = selectReadChannelNum(sourceUpdate, fftLen, channel, selectNum, dmNum, picDataSize);
        // int readChannel = 40;
        int loopTimes = ceil((float)channel/readChannel);
        int channelCount = 0;
        // size_t offset = 0;
        size_t read_i=0;
        size_t readloop=0;
        size_t select_i=0;
        int firstmark=0;
        int oldReadChannel = readChannel;
        // hostData = (cufftComplex*)malloc(sizeof(cufftComplex*)*readChannel*fftLen);
        // cufftComplex *tempData;
       
        // CHECK(cudaMalloc((void **)&absDeviceData, sizeof(float)*fftLen));
        // CHECK(cudaMalloc((void **)&deviceData, (size_t)sizeof(cufftComplex)*readChannel*fftLen));
        // CHECK(cudaMemset(absDeviceData, 0, (size_t)sizeof(float)*fftLen));
        // printf("ABS Sum:\n");
        // double absSumTimeS = seconds();

        
        
        // for(readloop=0; readloop<loopTimes; readloop++){

        //     if((channel-readloop*readChannel) <readChannel){
        //         readChannel = channel-readloop*readChannel;
        //     }
        //     if(firstmark == 0)
        //     {
        //         firstmark = 1;
        //     }else{
        //         free(hostData);
        //     }
        //     hostData = readMultiChannelComplexData(datainfo, channelCount, readChannel, cmdData);

        //     channelCount = channelCount + readChannel; 


        //     CHECK(cudaMemcpy(deviceData, hostData, (size_t)sizeof(cufftComplex)*readChannel*fftLen, cudaMemcpyHostToDevice));
        //     channelComplexSumAbs<<<fftLen/32, 32>>>(deviceData, absDeviceData, fftLen, readChannel);  
        //     cudaDeviceSynchronize();
        //     processBar(readloop, loopTimes);
        // }

        hostDmData = readIntData(dminfo);           // get DM data
        // absHostData = (float*)malloc(sizeof(float)*fftLen);
        CHECK(cudaMalloc((void **)&deviceDmData, sizeof(int)*channel*dmNum));       // Transfer dm data to GPU
        CHECK(cudaMemcpy(deviceDmData, hostDmData, sizeof(int)*channel*dmNum, cudaMemcpyHostToDevice));

        // CHECK(cudaMemcpy(absHostData, absDeviceData, sizeof(float)*fftLen, cudaMemcpyDeviceToHost));  // Transfer data to Host mem
        // CHECK(cudaFree(absDeviceData));  // after transfer, del GPU abs data 
        // CHECK(cudaFree(deviceData));
        // printf("ABS Sum Cost Time:%fs",seconds() - absSumTimeS);

        /* sort the abs data for candidate index select */
        selectList = (selectInfo*)malloc(sizeof(selectInfo)*selectNum);
        selectIndex =  absIndexSort(absHostData, fftLen);  // get abs data index(after sort)
        
        // save select data(fft abs) for data check 
        // FILE *dataCheck;
        // dataCheck = fopen("./checkdata.txt", "w+");
        // for(int datacheck_i=0; datacheck_i< selectNum; datacheck_i++){
        //     fprintf(dataCheck, "index:%d\n", selectIndex[datacheck_i]);
        // }
        // fclose(dataCheck);
        // exit(0);


        cufftComplex *candFFTData;
        candFFTData = (cufftComplex *)malloc(sizeof(cufftComplex)*selectNum*channel);

        /* Got the fft data from the file*/
        unsigned int temp_index;
        readChannel = oldReadChannel;
        channelCount = 0;
        printf("Read index data:\n");
        double readIndexTimeS = seconds();
        for(readloop=0; readloop<loopTimes; readloop++){

            if((channel-readloop*readChannel) <readChannel){
                readChannel = channel-readloop*readChannel;
            }

            if(firstmark == 0)
            {
                firstmark = 1;
            }else{
                free(hostData);
            }
            hostData = readMultiChannelComplexData(datainfo, channelCount, readChannel, cmdData);

            channelCount = channelCount + readChannel; 
            for(select_i=0; select_i<selectNum; select_i++)
            {   
                temp_index = selectIndex[select_i];
                for(read_i=0; read_i<readChannel; read_i++)
                {
                    candFFTData[(size_t)select_i*channel+readloop*oldReadChannel+read_i] = hostData[(size_t)read_i*fftLen+temp_index];
                }
            }
            processBar(readloop, loopTimes);
        }

        free(hostData);

        CHECK(cudaMalloc((void **)&deviceData, sizeof(cufftComplex)*selectNum*channel));
        CHECK(cudaMemcpy(deviceData, candFFTData, sizeof(cufftComplex)*selectNum*channel, cudaMemcpyHostToDevice));
        free(candFFTData);

        printf("Read index data Cost Time:%fs\n",seconds() - readIndexTimeS);

        // absDmData = (float *)malloc(sizeof(float)*dmNum);
        CHECK(cudaMalloc((void **)&complexDmData, sizeof(cufftComplex)*dmNum*channel));
        CHECK(cudaMalloc((void **)&complexDmDataSingle, sizeof(cufftComplex)*dmNum));
        CHECK(cudaMalloc((void **)&dmDeviceSelectData, sizeof(float)*dmNum));
        dmHostSelectData = (float *)malloc(sizeof(float)*dmNum);

        /* loop parameter */
        size_t absindex_i = 0;
        
        /* temp select parameter*/
        struct selectInfo tempInfo;
        float* dmPower;
        dmPower = (float*)malloc(sizeof(float)*(dmNum/2));

        /* pic parameter*/


        float* picHostData;
        float* picDeviceData;
        picHostData = (float*)malloc(sizeof(float)*(picDataSize));
        CHECK(cudaMalloc((void **)&picDeviceData, sizeof(float)*(picDataSize)));


        /* data save part*/
        int firstTime = 0;
        char foldersuffix[] = "_cand_data";
        char dataFolder[255];
        char dataFolder2[255];
        char tempFileName[512] = "";
        char tempFilePath[512] = "";
        float tempFre = 0; 
        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, foldersuffix);

        strcpy(dataFolder2, outputPath);
        strcat(dataFolder2, getFileName(datainfo.path));
        strcat(dataFolder2, foldersuffix);
        
        FILE *tempFile;

        /* search loop */
        int waringInfo;
        struct selectInfo *selectcands;
        selectcands = (selectInfo*)malloc(sizeof(selectInfo)*selectNum);
        int selectcandscount = 0;

        printf("Got necessary info\n");
        
        for(absindex_i=0; absindex_i<selectNum; absindex_i++)
        {      
            // Frequency domain de-dispersion for each channel
            partSelectFunc<<<grid, block>>>(deviceData, absindex_i, selectIndex[absindex_i], deviceDmData, complexDmData, channel, channelLen, fftLen, downsample);  
            cudaDeviceSynchronize();
            // add each channel
            selectSumDM<<<dmNum/32, 32>>>(complexDmData, complexDmDataSingle, channel);  // , didn't calculation the last 32 value
            cudaDeviceSynchronize();
            // abs data 
            selectAbsDm<<<dmNum/32, 32>>>(complexDmDataSingle, dmDeviceSelectData);
            cudaDeviceSynchronize();
            // copy data to host, get ready for filter

            CHECK(cudaMemcpy(dmHostSelectData, dmDeviceSelectData, sizeof(float)*dmNum, cudaMemcpyDeviceToHost));

            // Threshold filter
            tempInfo = chooseSelectDataWithPeak(dmHostSelectData, dmNum, absHostData[selectIndex[absindex_i]]);
            tempFre = fftIndexToFre(selectIndex[absindex_i], datainfo);
            if(tempInfo.dmValue > 50 && tempInfo.dmValue < dmNum-50)
            {
                selectcands[selectcandscount].isSelect = tempInfo.isSelect;
                selectcands[selectcandscount].value = tempInfo.value;
                selectcands[selectcandscount].fftIndex = selectIndex[absindex_i];
                selectcands[selectcandscount].maxTomean = tempInfo.maxTomean;
                selectcands[selectcandscount].dmValue = tempInfo.dmValue;
                selectcands[selectcandscount].fre = tempFre;
                selectcands[selectcandscount].maxpeakToabs = tempInfo.maxpeakToabs;
                selectcands[selectcandscount].mTon = tempInfo.mTon;
                selectcands[selectcandscount].absindex_i = absindex_i;
                selectcandscount += 1;
            }
            processBar(absindex_i, selectNum);
        }

        float meanmn = 0;
        float meanpeakabs = 0;
        float stdmn=0;
        float stdpeakabs=0;
        
        for(int selectcands_i=0; selectcands_i<selectcandscount; selectcands_i++)
        {
            meanmn += selectcands[selectcands_i].mTon/selectcandscount;
            meanpeakabs += selectcands[selectcands_i].maxpeakToabs/selectcandscount;
        }
        
        for(int selectcands_i=0; selectcands_i<selectcandscount; selectcands_i++)
        {
            stdmn += pow(selectcands[selectcands_i].mTon-meanmn, 2);
            stdpeakabs += pow(selectcands[selectcands_i].maxpeakToabs-meanpeakabs, 2);
        }
        
        stdmn = sqrtf( stdmn/selectcandscount);
        stdpeakabs = sqrtf( stdpeakabs/selectcandscount );
        printf("mean mn:%f, peak abs:%f,  std mn: %f,  std peak:%f\n", meanmn, meanpeakabs, stdmn, stdpeakabs);

        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, "cand_info.txt");

        printf("Save cand_info at:%s\n", dataFolder);
        // printf("Candidate info:\n");
        FILE *candInfo;
        candInfo = fopen(dataFolder, "wb+");
        for(int selectcands_i=0; selectcands_i<selectcandscount; selectcands_i++)
        {
            // fprintf(candInfo,"%3d, %4.1f, %6ld, %10.6f, %7f, %7f\n", selectcands_i, (float)selectcands[selectcands_i].dmValue*dminfo.DmStep, selectcands[selectcands_i].fftIndex, selectcands[selectcands_i].fre, selectcands[selectcands_i].mTon, selectcands[selectcands_i].maxpeakToabs);
            // if(selectcands[selectcands_i].mTon < meanmn && selectcands[selectcands_i].maxpeakToabs >(meanpeakabs+2*stdpeakabs))
            if(selectcands[selectcands_i].mTon < meanmn && selectcands[selectcands_i].maxpeakToabs >(meanpeakabs+2*stdpeakabs))
                fprintf(candInfo, "%3d, %4.1f, %6ld, %10.6f, %7f, %7f\n", selectcands_i, (float)selectcands[selectcands_i].dmValue*dminfo.DmStep, selectcands[selectcands_i].fftIndex, selectcands[selectcands_i].fre, selectcands[selectcands_i].mTon, selectcands[selectcands_i].maxpeakToabs);
        }
       
        fclose(candInfo);
        free(absHostData);


        printf("Searching cand...\n");
        printf("\nSave cand_data at:%s\n", dataFolder2);
        double selectStartT=seconds();
        // for(absindex_i=0; absindex_i<selectNum; absindex_i++)
        for(int selectcands_i=0; selectcands_i<selectcandscount; selectcands_i++)
        {   
            processBar(selectcands_i, selectcandscount);
            if(selectcands[selectcands_i].maxpeakToabs <(meanpeakabs+2*stdpeakabs))
                continue;
            absindex_i = selectcands[selectcands_i].absindex_i;
            // Frequency domain de-dispersion for each channel
            partSelectFunc<<<grid, block>>>(deviceData, absindex_i, selectIndex[absindex_i], deviceDmData, complexDmData, channel, channelLen, fftLen, downsample);  
            cudaDeviceSynchronize();
            
            // add each channel
            selectSumDM<<<dmNum/32, 32>>>(complexDmData, complexDmDataSingle, channel);  // , didn't calculation the last 32 value
            cudaDeviceSynchronize();
            
            // abs data 
            selectAbsDm<<<dmNum/32, 32>>>(complexDmDataSingle, dmDeviceSelectData);
            cudaDeviceSynchronize();
            
            // copy data to host, get ready for filter
            CHECK(cudaMemcpy(dmHostSelectData, dmDeviceSelectData, sizeof(float)*dmNum, cudaMemcpyDeviceToHost));

            // Threshold filter
            tempInfo = selectcands[selectcands_i];  
            
            // Power filter
            // if(tempInfo.isSelect > 0 && tempInfo.dmValue !=0){
            //     dmPower =  dmFrePower(dmHostSelectData, dmNum, dmNum/2);
            //     tempInfo = dmPowerFilter(dmPower, lowFreNum, dmNum/2, tempInfo);
            // }else{
            //     tempInfo.isSelect = 0;
            // }

            calculatePicRowData<<<blockx2, blocky2>>>(complexDmData, picDeviceData, sumChannelNum, channelStep, dmNum, channel);
            cudaDeviceSynchronize();
            CHECK(cudaMemcpy(picHostData, picDeviceData, sizeof(float)*picDataSize, cudaMemcpyDeviceToHost));
            
            if(firstTime == 0)
            {
                firstTime =1;
                char floderCommand[512] = "mkdir -p ";
                strcat(floderCommand, dataFolder2);
                waringInfo = system(floderCommand);
                if(waringInfo==-1)
                {
                    printf("mkdir error\n");
                }
            }

            // tempFre = fftIndexToFre(selectIndex[absindex_i], datainfo);
            sprintf(tempFileName, "/dm%.1f_%f.pic", (float)tempInfo.dmValue*dminfo.DmStep, selectcands[selectcands_i].fre);
            strcat(tempFilePath, dataFolder2);
            strcat(tempFilePath, tempFileName);

            tempFile = fopen(tempFilePath, "wb");
            writeCandDataHead(tempFile, dmNum, blocky2, dminfo.DmMin, dminfo.DmMax, selectcands[selectcands_i].fre, datainfo);
            
            fwrite(dmHostSelectData, sizeof(float), dmNum, tempFile);
            fwrite(picHostData, sizeof(float), picDataSize, tempFile);
            
            
            fclose(tempFile);
            strcpy(tempFilePath, "");


            // selectList[selectListNum].dmValue = tempInfo.dmValue;
            // selectList[selectListNum].isSelect = tempInfo.isSelect;
            // selectList[selectListNum].fftIndex = selectIndex[absindex_i];
            // selectList[selectListNum].fre = tempFre;
            // selectList[selectListNum].maxTomean = tempInfo.maxTomean;
            // selectListNum++;
        }

        // quicksortInfo(selectList, selectListNum, 0, selectListNum-1);
        printf("Processing with time:%6.4fs\n", seconds()-selectStartT);

        // strcpy(dataFolder, outputPath);
        // strcat(dataFolder, getFileName(datainfo.path));
        // strcat(dataFolder, "cand_info.txt");

        // printf("Save cand_info at:%s\n", dataFolder);
        // // printf("Candidate info:\n");
        // // FILE *candInfo;
        // candInfo = fopen(dataFolder, "wb+");
        // for(int i=0;i<selectListNum;i++){
        //     fprintf(candInfo, "index:%3d ,DM:%4.1f,  fftIndex:%6ld, fre:%10.6fHz, Max:Mean:%4.3f\n", i, (float)selectList[i].dmValue*dminfo.DmStep, selectList[i].fftIndex, selectList[i].fre, selectList[i].maxTomean);
        // }
        // fclose(candInfo);
        free(selectcands);
        free(hostDmData);
        free(picHostData);
        free(dmPower);
        free(dmHostSelectData);
        free(selectList);
        CHECK(cudaFree(deviceData));
        CHECK(cudaFree(deviceDmData));
        CHECK(cudaFree(complexDmData));
        CHECK(cudaFree(complexDmDataSingle));
        CHECK(cudaFree(dmDeviceSelectData));
        CHECK(cudaFree(picDeviceData));


    }else{
        printf("Input data is wrong!\n");
        exit(0);
    }


}


void partInMemSelectWithindex(struct fileinfo datainfo, struct fileinfo dminfo, struct systemSource source, struct cmds cmdData){
    if(datainfo.DataType == 2){  // input data is fft data
        int channel = datainfo.channel;
        size_t channelLen = (size_t)datainfo.FFTLen*2;
        size_t fftLen = datainfo.FFTLen;
        int dmNum = dminfo.DmNum;
        // float selectThreshold = cmdData.selectThresholdRate;
        // int lowFreNum = cmdData.selectFrequencyNum;
        int sumChannelNum = cmdData.picSumChannelNum;
        int channelStep = cmdData.picChannelStep;
        char *outputPath = cmdData.outputPath;
        int downsample = cmdData.downsample;
        struct systemSource sourceUpdate = systemSourceCheck(1);

        // if(cmdData.picPramChange == 0)
        // {
        //     channelStep = ((float)channel/48);
        //     sumChannelNum = ((float)channel*15/96);
        // }
        
        cufftComplex *hostData;     // the fft data on Host
        cufftComplex *deviceData;   // the fft data on GPU
        cufftComplex *complexDmData;  //the data after dm with fft data on all channel
        cufftComplex *complexDmDataSingle; // the data sum after dm with fft data

        float *absDeviceData;
        float *absHostData;
        // float *absDmData;
        float *dmDeviceSelectData;
        float *dmHostSelectData;
        int *hostDmData;
        int *deviceDmData;

        // float candidateSelectRate = cmdData.selectNumRate;
        struct selectInfo* selectList; 
        int selectListNum = 0;
        // int selectNum = fftLen * candidateSelectRate;
        int selectNum = cmdData.indexRangeNum;
        int* selectIndex = NULL;  

        int blockX = 2;   // gpu calculation struct            
        int blockY = 64; 
        dim3 block (blockX, blockY);
        dim3 grid ((channel+block.x-1)/block.x, (dmNum+block.y-1)/block.y);\

        int blockx2 = dmNum;
        int blocky2 = (channel-sumChannelNum)/channelStep;
        size_t picDataSize = blockx2*blocky2;

        int readChannel = selectReadChannelNum(sourceUpdate, fftLen, channel, selectNum, dmNum, picDataSize);
        // int readChannel = 40;
        int loopTimes = ceil((float)channel/readChannel);
        int channelCount = 0;
        // unsigned long long offset = 0;
        size_t read_i=0;
        size_t readloop=0;
        size_t select_i=0;
        int oldReadChannel = readChannel;
        // hostData = (cufftComplex*)malloc(sizeof(cufftComplex*)*readChannel*fftLen);
        // cufftComplex *tempData;
        // hostData = readAllComplexData(datainfo);    // get fft data 
       
       
        CHECK(cudaMalloc((void **)&absDeviceData, sizeof(float)*fftLen));
        CHECK(cudaMalloc((void **)&deviceData, sizeof(cufftComplex)*readChannel*fftLen));
        CHECK(cudaMemset(absDeviceData, 0, sizeof(float)*fftLen));

        hostDmData = readIntData(dminfo);           // get DM data
        absHostData = (float*)malloc(sizeof(float)*fftLen);
        CHECK(cudaMalloc((void **)&deviceDmData, sizeof(int)*channel*dmNum));       // Transfer dm data to GPU
        CHECK(cudaMemcpy(deviceDmData, hostDmData, sizeof(int)*channel*dmNum, cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(absHostData, absDeviceData, sizeof(float)*fftLen, cudaMemcpyDeviceToHost));  // Transfer data to Host mem
        CHECK(cudaFree(absDeviceData));  // after transfer, del GPU abs data 
        CHECK(cudaFree(deviceData));
        /* sort the abs data for candidate index select */
        selectList = (selectInfo*)malloc(sizeof(selectInfo)*selectNum);
        // selectIndex =  absIndexSort(absHostData, fftLen);  // get abs data index(after sort)
        
        selectIndex = (int *)malloc(sizeof(int)*selectNum);
        int startIndex = cmdData.startIndex; 
        for(int i=0; i<selectNum; i++)
        {
            selectIndex[i] = startIndex+i;
        }

        free(absHostData);

        cufftComplex *candFFTData;
        candFFTData = (cufftComplex *)malloc(sizeof(cufftComplex)*selectNum*channel);

        /* Got the fft data from the file*/
        int temp_index;
        readChannel = oldReadChannel;
        channelCount = 0;
        int firstmark = 0; 
        printf("Read index data:\n");
        for(readloop=0; readloop<loopTimes; readloop++){

            if((channel-readloop*readChannel) <readChannel){
                readChannel = channel-readloop*readChannel;
            }
            // offset = 0;
            // for(read_i=0; read_i<readChannel; read_i++)
            // {
            //     tempData = readChannelComplexData(datainfo, channelCount+read_i, cmdData);
                
            //     memcpy(hostData+offset, tempData, sizeof(cufftComplex)*fftLen);
            //     offset = offset + fftLen; 
            //     free(tempData);
            // }
            if(firstmark == 0)
            {
                firstmark = 1;
            }else{
                free(hostData);
            }
            hostData = readMultiChannelComplexData(datainfo, channelCount, readChannel, cmdData);

            channelCount = channelCount + readChannel; 
            for(select_i=0; select_i<selectNum; select_i++)
            {   
                temp_index = selectIndex[select_i];
                for(read_i=0; read_i<readChannel; read_i++)
                {
                    candFFTData[(size_t)select_i*channel+readloop*oldReadChannel+read_i] = hostData[(size_t)read_i*fftLen+temp_index];
                }
            }
            if(loopTimes !=1){
                processBar(readloop, loopTimes);
            }
        }

        free(hostData);
        // FILE *savefftindex = fopen("./fftindex_data.dat", "wb");
        // fwrite(candFFTData, sizeof(cufftComplex), (size_t)selectNum*channel, savefftindex);
        // fclose(savefftindex);

        // exit(0);
        CHECK(cudaMalloc((void **)&deviceData, sizeof(cufftComplex)*selectNum*channel));
        CHECK(cudaMemcpy(deviceData, candFFTData, sizeof(cufftComplex)*selectNum*channel, cudaMemcpyHostToDevice));
        free(candFFTData);

        // absDmData = (float *)malloc(sizeof(float)*dmNum);
        CHECK(cudaMalloc((void **)&complexDmData, sizeof(cufftComplex)*dmNum*channel));
        CHECK(cudaMalloc((void **)&complexDmDataSingle, sizeof(cufftComplex)*dmNum));
        CHECK(cudaMalloc((void **)&dmDeviceSelectData, sizeof(float)*dmNum));
        dmHostSelectData = (float *)malloc(sizeof(float)*dmNum);

        /* loop parameter */
        size_t absindex_i = 0;
        
        /* temp select parameter*/
        struct selectInfo tempInfo;
        float* dmPower;
        dmPower = (float*)malloc(sizeof(float)*(dmNum/2));

        /* pic parameter*/


        float* picHostData;
        float* picDeviceData;
        picHostData = (float*)malloc(sizeof(float)*(picDataSize));
        CHECK(cudaMalloc((void **)&picDeviceData, sizeof(float)*(picDataSize)));


        /* data save part*/
        int firstTime = 0;
        char foldersuffix[] = "_cand_data";
        char dataFolder[255];
        char tempFileName[512] = "";
        char tempFilePath[512] = "";
        float tempFre = 0; 
        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, foldersuffix);
        printf("\nSave cand_data at:%s\n\n", dataFolder);
        FILE *tempFile;
        /* search loop */
        int waringInfo;
        double selectStartT=seconds();
        for(absindex_i=0; absindex_i<selectNum; absindex_i++)
        {   
            // Frequency domain de-dispersion for each channel
            partSelectFunc<<<grid, block>>>(deviceData, absindex_i, selectIndex[absindex_i], deviceDmData, complexDmData, channel, channelLen, fftLen, downsample);  
            cudaDeviceSynchronize();
            // add each channel
            selectSumDM<<<dmNum/32, 32>>>(complexDmData, complexDmDataSingle, channel);  // , didn't calculation the last 32 value
            cudaDeviceSynchronize();
            // abs data 
            selectAbsDm<<<dmNum/32, 32>>>(complexDmDataSingle, dmDeviceSelectData);
            cudaDeviceSynchronize();
            // copy data to host, get ready for filter

            CHECK(cudaMemcpy(dmHostSelectData, dmDeviceSelectData, sizeof(float)*dmNum, cudaMemcpyDeviceToHost));

            // Threshold filter
            tempInfo = chooseSelectData(dmHostSelectData, dmNum, 1);  
            tempInfo.isSelect = 1;
            // Power filter
            // if(tempInfo.isSelect > 0 && tempInfo.dmValue !=0){
            //     dmPower =  dmFrePower(dmHostSelectData, dmNum, dmNum/2);
            //     tempInfo = dmPowerFilter(dmPower, lowFreNum, dmNum/2, tempInfo);
            // }else{
            //     tempInfo.isSelect = 0;
            // }

            
            if(tempInfo.isSelect > 0 )
            {   
 
                calculatePicRowData<<<blockx2, blocky2>>>(complexDmData, picDeviceData, sumChannelNum, channelStep, dmNum, channel);
                cudaDeviceSynchronize();
                CHECK(cudaMemcpy(picHostData, picDeviceData, sizeof(float)*picDataSize, cudaMemcpyDeviceToHost));
                
                if(firstTime == 0)
                {
                    firstTime =1;
                    char floderCommand[512] = "mkdir -p ";
                    strcat(floderCommand, dataFolder);
                    waringInfo = system(floderCommand);
                    if(waringInfo==-1)
                    {
                        printf("mkdir error\n");
                    }
                }

                tempFre = fftIndexToFre(selectIndex[absindex_i], datainfo);
                sprintf(tempFileName, "/dm%.1f_%f.pic", (float)tempInfo.dmValue*dminfo.DmStep, tempFre);
                strcat(tempFilePath, dataFolder);
                strcat(tempFilePath, tempFileName);

                tempFile = fopen(tempFilePath, "wb");
                writeCandDataHead(tempFile, dmNum, blocky2, dminfo.DmMin, dminfo.DmMax, tempFre, datainfo);
                
                fwrite(dmHostSelectData, sizeof(float), dmNum, tempFile);
                fwrite(picHostData, sizeof(float), picDataSize, tempFile);
                
                
                fclose(tempFile);
                strcpy(tempFilePath, "");


                selectList[selectListNum].dmValue = tempInfo.dmValue;
                selectList[selectListNum].isSelect = tempInfo.isSelect;
                selectList[selectListNum].fftIndex = selectIndex[absindex_i];
                selectList[selectListNum].fre = tempFre;
                selectList[selectListNum].maxTomean = tempInfo.maxTomean;
                selectListNum++;
                
            }

            processBar(absindex_i, selectNum);
        }

        quicksortInfo(selectList, selectListNum, 0, selectListNum-1);
        printf("Processing with time:%6.4fs\n\n", seconds()-selectStartT);

        strcpy(dataFolder, outputPath);
        strcat(dataFolder, getFileName(datainfo.path));
        strcat(dataFolder, "cand_info.txt");

        printf("\nSave cand_info at:%s\n", dataFolder);
        // printf("Candidate info:\n");
        FILE *candInfo;
        candInfo = fopen(dataFolder, "wb+");
        for(int i=0;i<selectListNum;i++){
            fprintf(candInfo, "index:%3d ,DM:%4.1f,  fftIndex:%6ld, fre:%10.6fHz, Max:Mean:%4.3f\n", i, (float)selectList[i].dmValue*dminfo.DmStep, selectList[i].fftIndex, selectList[i].fre, selectList[i].maxTomean);
        }
        fclose(candInfo);
        free(hostDmData);
        free(picHostData);
        free(dmPower);
        free(dmHostSelectData);
        free(selectList);
        CHECK(cudaFree(deviceData));
        CHECK(cudaFree(deviceDmData));
        CHECK(cudaFree(complexDmData));
        CHECK(cudaFree(complexDmDataSingle));
        CHECK(cudaFree(dmDeviceSelectData));
        CHECK(cudaFree(picDeviceData));

    

    }else if(datainfo.DataType == 0){  // input data is original data
        struct fileinfo datainfo1;
        struct fileinfo datainfo2;
        char *path = cmdData.filePath;
        datainfo1 = readfile(path, datainfo1);
        datainfo2 = fftProcess(datainfo1, source, cmdData);
        datainfo2 = readfile(datainfo2.path, datainfo2);
        partInMemSelect(datainfo2, dminfo, source, cmdData);
    }else{
        printf("Input data is wrong!\n");
        exit(0);
    }

}