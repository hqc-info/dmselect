#include<stdio.h>
#include<fileunit.h>
#include<cuda.h>
#include<cufft.h>
#include<cmdunit.h>
#include<toolunit.h>
#include<fftunit.h>
#include<common.h>
#include<math.h>
#include<selectunit.h>
#include<cuda_runtime.h>


int main(int argc, char  **argv)
{   
    char const *defaultPath = "./file";   // default output folder
    is_file_dir_exist(defaultPath);       // check output folder
    double startTime = seconds();
    struct cmds cmdData;
    cmdData = getCmdsData(argc, argv);    // got input parameter 
    struct systemSource source;
    CHECK(cudaSetDevice(cmdData.gpuId));  // choose the GPU device with gpuId 
    source = systemSourceCheck(0);
    
    if(cmdData.funcNum == 0)              // fft function, turn the original data to fft data
    {
        struct fileinfo datainfo1;        // original data things
        struct fileinfo datainfo2;        // fft data things
        char *path = cmdData.filePath;
        datainfo1 = readfile(path, datainfo1);  // init the data of original, fopen, read file head such thing  
        datainfo2 = fftProcess(datainfo1, source, cmdData);  //  fft process
    }else if(cmdData.funcNum == 1)
    {   
        struct fileinfo datainfo;
        struct fileinfo dminfo;
        char *path = cmdData.filePath;
        char *dmpath = cmdData.dmPath;
        if(path == NULL || dmpath == NULL){
            printf("Lack of file!\n");
            exit(0);
        }
        datainfo = readfile(path, datainfo);
        dminfo = readfile(dmpath, dminfo);
        // allInMemSelect(datainfo, dminfo, source, cmdData);
        partInMemSelect(datainfo, dminfo, source, cmdData);
    }else if(cmdData.funcNum == 2)
    {
        struct fileinfo datainfo;
        struct fileinfo dminfo;
        char *path = cmdData.filePath;
        char *dmpath = cmdData.dmPath;
        if(path == NULL || dmpath == NULL){
            printf("Lack of file!\n");
            exit(0);
        }
        datainfo = readfile(path, datainfo);
        dminfo = readfile(dmpath, dminfo);
        partInMemSelectWithindex(datainfo, dminfo, source, cmdData);
    }else if(cmdData.funcNum == 3)
    {
        
    }

    printf("\nAll process time:%fs\n", seconds()- startTime);
    return 0;
}