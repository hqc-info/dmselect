#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <toolunit.h>
#include <cuda.h>
#include <math.h>
#include <string.h>
#include <common.h>
#include <fileunit.h>
#include <cmdunit.h>
#include <unistd.h>
#include <dirent.h>


void is_file_dir_exist(const char*dir_path){
    // 检查文件夹是否存在
    int waringInfo;
    if(dir_path==NULL){
        waringInfo = system("mkdir ./file");
        if(waringInfo==-1)
        {
            printf("mkdir error\n");
        }
    }
    if(opendir(dir_path)==NULL){
        waringInfo = system("mkdir ./file");
    }
}


int mjd2utc8(double stt)
{   
    int days = stt;
    int hour = (double)(stt - days)*24;
    int min = (double)((stt - days)*24 - hour)*60;
    int sec = (double)(((stt - days)*24 - hour)*60 - min)*60;
    int microsecond = (double)((((stt - days)*24 - hour)*60 - min)*60-sec)*1000000;  //微秒
    int nyear[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int lyeary[] = {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int monthlack = 0;
    int startyear = 1858;
    int startmonth = 11;
    int startday = 17;
    while(1){
        if((startyear%4==0 && startyear%100!=0)|| startyear%400==0){
            monthlack = lyeary[startmonth] - startday + 1;
            if(monthlack > days){
                startday += days;
                break;
            }else{
                startday = 1;
                startmonth += 1;
                days -= monthlack;
            }
        }else{
            monthlack = nyear[startmonth] - startday + 1;
            if(monthlack > days){
                startday += days;
                break;
            }else{
                startday = 1;
                startmonth += 1;
                days -= monthlack;
            }
        }
        if(startmonth>12){
            startmonth = 1;
            startyear += 1;
        }


    }

    printf("time:%.4d-%.2d-%.2d %.2d:%.2d:%.2d.%d\n", startyear, startmonth, startday, hour, min, sec, microsecond); 
    return 0;
}


int inZapChan(int *min, int *max, int num, int index){
    int cutnum = 0;
    for(int zapchan_i=0;zapchan_i<num;zapchan_i++){
        if(min[zapchan_i]<= index && max[zapchan_i]>index)
        {

            cutnum = index-min[zapchan_i];
            index = max[zapchan_i]+cutnum;
            if(index >= 4095)
            {
                index = min[zapchan_i]-cutnum;
            }
            index = inZapChan(min, max, num, index);
            return index;
        }

    }
    return index;
}



void splitMinMax(char* msg, int *min, int *max, int num){
    int msgLen = strlen(msg);
    // int minIndex = 0;
    // int maxIndex = 0;
    char tempStr[20];
    for(int i=0;i<msgLen;i++)
    {    
        if(msg[i]==58) // : ascii 58
        {
            strncpy(tempStr, msg, i);
            min[num] = (int)atoi(tempStr);
            strcpy(tempStr, msg+i+1);
            max[num] = (int)atoi(tempStr);
            break;
        }
       
    }
}

int rangeCount(char *msg)
{
    int msgLen = strlen(msg);
    int countNum = 0;
    for(int i=0; i<msgLen; i++)
    {
        if(msg[i]==44)   // , ascii 44
            countNum++;
    }
    return countNum+1;
}


void getZapChanRange(int *min, int *max, int num, char *msg)
{

    const char s[2] = ",";
    // const char s1[2] = ":";
    char *token;

    int countNum = 0;
    token = strtok(msg, s);

    
    while( token != NULL ) {

            // printf( "%s\n", token);

            splitMinMax(token, min, max, countNum);
            countNum ++;
            token = strtok(NULL, s);
    }
}



float fftIndexToFre(unsigned int fftIndex, struct fileinfo fftInfo){
    if(fftInfo.channelLen == 0)
    {
        fftInfo.channelLen = (size_t)fftInfo.FFTLen*2;
    }
    float fre;
    fre = (float)fftIndex *(float)((float)1e6/fftInfo.SampleTime)/fftInfo.channelLen;
    return fre;
}


// 功率筛选
struct selectInfo dmPowerFilter(float* powerData, int lowFreNum, int FFTLen, struct selectInfo tempInfo){
    double lowFrePower = 0;
    double highFrePower = 0;
    int i = 0;
    for(i=1; i<lowFreNum; i++)
    {
        lowFrePower = lowFrePower + powerData[i];
    }

    for(i=lowFreNum; i<FFTLen; i++)
    {
        highFrePower = highFrePower + powerData[i];
    }

    if(lowFrePower >= highFrePower)
    {
        tempInfo.isSelect=1;
    }else{
        tempInfo.isSelect=0;
    }

    return tempInfo;
    
}


// 交换参数
void swapInfo(struct selectInfo *a, struct selectInfo *b) 
{
    struct selectInfo temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

// 快排
void quicksortInfo(struct selectInfo* sortData, int maxlen, int begin, int end)
{
  int i, j;
  
  if(begin < end)
  {
    i = begin + 1; // 将array[begin]作为基准数，因此从array[begin+1]开始与基准数比较！
    j = end;    // array[end]是数组的最后一位
      
    while(i < j)
    {
      if(sortData[i].dmValue > sortData[begin].dmValue ) // 如果比较的数组元素大于基准数，则交换位置。
      {
        swapInfo(&sortData[i], &sortData[j]); // 交换两个数
        j--;
      }
      else
      {
        i++; // 将数组向后移一位，继续与基准数比较。
      }
    }
  
    if(sortData[i].dmValue >= sortData[begin].dmValue) 
    {
      i--;
    }
  
    swapInfo(&sortData[begin], &sortData[i]); // 交换array[i]与array[begin]
     
    quicksortInfo(sortData, maxlen, begin, i);
    quicksortInfo(sortData, maxlen, j, end);
  }
}

// 交换参数
void swap(struct indexSort *a, struct indexSort *b) 
{
    struct indexSort temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

void quicksort(struct indexSort* sortData, size_t maxlen, size_t begin, size_t end)
{
    size_t i, j;
  
  if(begin < end)
  {
    i = begin + 1; // 将array[begin]作为基准数，因此从array[begin+1]开始与基准数比较！
    j = end;    // array[end]是数组的最后一位

    while(i < j)
    {
      if(sortData[i].value < sortData[begin].value ) // 如果比较的数组元素大于基准数，则交换位置。
      {
        swap(&sortData[i], &sortData[j]); // 交换两个数
        j--;
      }
      else
      {
        i++; // 将数组向后移一位，继续与基准数比较。
      }
    }
  
    if(sortData[i].value <= sortData[begin].value) 
    {
      i--;
    }
  
    swap(&sortData[begin], &sortData[i]); // 交换array[i]与array[begin]
     
    quicksort(sortData, maxlen, begin, i);
    quicksort(sortData, maxlen, j, end);
  }
}

// 选择小于selectRate的数据
struct selectInfo chooseSelectData(float *selectData, int dmNum, float selectRate){
    int maxIndex=0;
    float maxValue=0;
    double mean=0;
    int i=0;
    float temp;
    struct selectInfo tempInfo;
    for(i=0;i<dmNum;i++){
        temp = selectData[i];
        if(temp>maxValue){
            maxIndex = i;
            maxValue = temp;
        }
        mean +=(temp/dmNum);
    }
    
    if(mean <= selectRate*maxValue)
    {  
        tempInfo.isSelect=1;
    }else{
        tempInfo.isSelect=0;
    }
    tempInfo.maxTomean = (float)maxValue/mean;
    tempInfo.value = maxValue;
    tempInfo.dmValue=maxIndex;
    return tempInfo;
}


// 选择小于selectRate的数据
struct selectInfo chooseSelectDataWithPeak(float *selectData, int dmNum, float abs){
    int maxIndex=0;
    float maxValue=0;
    double mean=0;
    int i=0;
    float temp;
    struct selectInfo tempInfo;
    for(i=0;i<dmNum;i++){
        temp = selectData[i];
        if(temp>maxValue){
            maxIndex = i;
            maxValue = temp;
        }
        mean +=(temp/dmNum);
    }

    int N = 0;
    int M = 0;
    for(i=0;i<dmNum-1;i++)
    {
        if(i == 0 && maxValue==selectData[0])
        {
            M+=1;
            N+=1;
        }else if(selectData[i-1]< selectData[i] && selectData[i] > selectData[i+1])
        {
            if(selectData[i]>0.7*maxValue)
            {
                M+=1;
                if(selectData[i]>0.95*maxValue)
                {
                    N+=1;
                }
            }
        }
    }
    tempInfo.mTon = (float)M/N; 
    tempInfo.maxpeakToabs = (float)maxValue/abs;
    tempInfo.value = maxValue;
    tempInfo.dmValue = maxIndex;
    return tempInfo;
}



//  abs后的DM排序Index
unsigned int* absIndexSort(float *indata, size_t dataLen){
    struct indexSort* sortData;
    size_t low = 0;
    size_t high = dataLen-1;
    double sortSatrt = seconds();
    printf("\nABS data sort\n");
    sortData = (indexSort*)malloc(sizeof(indexSort)*dataLen);
    size_t i=0;


    for(i=0; i<dataLen; i++){
        sortData[i].value = indata[i];
        sortData[i].index = i;
    }
    // static FILE *testdata;
    // testdata = fopen("./dat.txt","w");
    
    quicksort(sortData, dataLen, low, high);

    unsigned int* indexData;
    indexData = (unsigned int*)malloc((size_t)sizeof(unsigned int)*dataLen);
    for(i=0; i<dataLen; i++){
        indexData[i] = sortData[i].index;
        // fprintf(testdata, "%d\n", indexData[i]);

    }
    // fclose(testdata);
    printf("Sort cost time:%6.4fs\n\n", seconds()-sortSatrt);
    return indexData;
}


int isProcessPartly(struct systemSource source, int channelLen, int channel)
{
    size_t singalChannelMem = ceil((double)channelLen/1000)*4;
    int hostChannel = (source.availableHostMem-(300*1000))/(singalChannelMem*3);   // Reserve 100m memory for system
    int gpuChannel = (source.availableDeviceMem-(300*1000))/(singalChannelMem*3)-1;
    if(hostChannel>=channel && gpuChannel>=channel)
        return 0;
    else
        return 1;
}


int fftReadChannelNum(struct systemSource source, size_t channelLen, int channel)
{   

    size_t singalChannelMem = (size_t)(channelLen/1000)*4;   // singal channel mem/KB
    int hostChannel = (size_t)(source.availableHostMem-(400*1000))/(size_t)(singalChannelMem*3)-2;   // Reserve 200m memory for system
    int gpuChannel = (size_t)(source.availableDeviceMem-(500*1000))/(size_t)(singalChannelMem*3)-2;  // ??????? problem
    printf("device mem:%ld, singalChannelMem:%ld\n", source.availableDeviceMem, singalChannelMem);

    printf("Host can deal with %d channel, Gpu can deal with %d channel\n", hostChannel, gpuChannel);
    
    if(hostChannel>=channel && gpuChannel>=channel)
    {
        return channel;
    }


    if(hostChannel<= gpuChannel)
        return hostChannel;
    else
        return gpuChannel;
}


int selectReadChannelNum(struct systemSource source, size_t fftLen, int channel, int selectNum, int dmNum, size_t picDataSize)
{   

    size_t singalChannelMem = ceil((double)fftLen/1000)*8;   // singal channel mem/KB
    int saveSpace = 400*1000;
    int hostChannel = (double)((size_t)source.availableHostMem-saveSpace*2- (size_t)((size_t)singalChannelMem*2 +(size_t)selectNum*20+ (size_t)8*selectNum*channel+ (size_t)2*dmNum+ (size_t)dmNum*channel*4 + (size_t)picDataSize*4)/1000)/(singalChannelMem);   // Reserve 200m memory for system
    int gpuChannel = (double)(source.availableDeviceMem-saveSpace- (size_t)((size_t)4*fftLen+(size_t)4*channel*dmNum+(size_t)8*selectNum*channel+(size_t)8*dmNum*channel+ (size_t)8*dmNum-(size_t)4*dmNum- (size_t)4*picDataSize)/1000)/(singalChannelMem);
    
    printf("Select func:Host can deal with %d channel, Gpu can deal with %d channel\n", hostChannel, gpuChannel);
    
    if(hostChannel>=channel && gpuChannel>=channel)
    {
        return channel;
    }

    if(hostChannel<= gpuChannel)
        return hostChannel;
    else
        return gpuChannel;
}





char* getFileName(char *filePath){
    if(filePath==NULL){
        printf("File path is NULL, can't get file name\n");
        exit(0);
    }
    char path[4096];
    char name[255];
    char* fileName;
    fileName = (char*)malloc(sizeof(char)*255);
    strcpy(path, filePath);
    char *token;

    char s[] = "/";
    char s1[] = ".";
    token = strtok(path, s);
    
    while( token != NULL )
    {
        strcpy(name, token);
        token = strtok(NULL, s);
    }
    token = strtok(name, s1);
    strcpy( fileName,name);
    return fileName;
}

char * addPath(char *path, char *fileName)
{
    char *outpath;
    char *p;
    outpath = (char*)malloc(sizeof(char)*4096);
    strcat(outpath, path);
    strcat(outpath, "/");
    strcat(outpath, fileName);
    
    p = strstr(outpath, "//");
    if(p != NULL)
    {
        memset(outpath,0,sizeof(char)*4096);
        strcat(outpath, path);
        strcat(outpath, fileName);
    }

    return outpath;
}


void processBar(int progress, int total)

{
    static int first = 0;
    static double startTime = 0;
    progress = progress +1;
    if(first==0)
    {   startTime = seconds();
        if(progress >= total)
        {
            printf("Process at:100%%\n");
            first = 0;
            startTime = 0;
            return;
        }else{
            printf("Process at:%3d%%", 0);
            fflush(stdout);
            first =1;
        }
    }else{
        if(total-progress>0){
            printf("\rProcess at:%3d%%   Remaining time for this part:%8.2fs",(progress*100)/total, ((double)(seconds()-startTime)/progress) *((float)total-progress)); 
            fflush(stdout);
        }else{
            printf("\rProcess at:%3d%%                                            ",(progress*100)/total); 
            fflush(stdout);
        }
    }
    if(progress >= total)
    {
        printf("\n");
        first = 0;
    }

}


struct systemSource systemSourceCheck(int prMark)
{   
    struct systemSource source;
    if(prMark == 0){
        printf("System source check!\n");

        // Get cpu core number
        printf("System cpu num is:%15d\n", get_nprocs_conf());
        printf("System enable cpu num is:%8d\n", get_nprocs());

    }
    source.cpuCore = get_nprocs();

    // Get the system available mem size 
    
    FILE *file;
    file =fopen("/proc/meminfo","r");
    if(file == NULL){
        fprintf(stderr,"cannot open /proc/meminfo\n");
        exit(0);
    }
    char keyword[20];
    char valuech[20];
    long mem        =0;
    long free_mem   =0;
    int mark = 0;
    mark = fscanf(file,"MemTotal: %s kB\n",keyword);
    mem = atol(keyword);
    mark = fscanf(file,"MemFree: %s kB\n",valuech);
    mark = fscanf(file,"MemAvailable: %s kB\n",valuech);
    free_mem=atol(valuech);
    fclose(file);

    if(mark == 0){  // file read check
        printf("Read Host meminfo error!\n");
        exit(0);
    }
    
    source.availableHostMem = free_mem;


    // Get device Mem size
    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total);

    if(prMark == 0)
    {   printf("Mem size:              kB            MB\n");
        printf("Host MemTotal: %14ld kB / %5.0f MB \nHost MemAvailable: %10ld kB / %5.0f MB\n", mem, (float)mem/1e3,free_mem, (float)free_mem/1e3);
        printf("GPU MemTotal: %15.0f kB / %5.0f MB \nGPU Memavalible: %12.0f kB / %5.0f MB \n\n", (float)total/1000, (float)total/1e6, (float)avail/1000, (float)avail/1e6);
    }
    
    source.availableDeviceMem = (long)avail/1000;  // KB not KiB 


    return source;

}
