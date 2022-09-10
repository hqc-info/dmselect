#include<cmdunit.h>
#include<string.h>
#include<stdlib.h>
#include<toolunit.h>
void argvTips(){
    // shell 错误提示
    printf("*********Tips*********\n");
    printf( "   -h           help\n"
            "   -f           input file original data or fft data\n"
            "   -d           dm data\n"
            "   -s           save mark 0/1 not save/save\n"
            "   -sp          save path\n"
            "   -cr          the rate of abs cand num\n"
            "   -cf          choose the program func\n"
            "           0 for FFT\n"
            "           1 for DM select\n"
            "           2 for special index\n"
            "   -ct          threshold_filter's threshold\n"
            "   -cfn         power_filter's low frequency num\n"
            "   -ci          cand fft index for draw single pic\n"
            "   -cin         he num of index, started at (ci)startIndex\n"
            "   -ds          the downsample num, default is 1\n"
            "   -cs          draw single pic channel step\n"
            "   -scn         draw single pic sum channel num\n"
            "   -zapchan     min:max remove channel data\n"
            "   -gpu         set gpu device, default is 0\n");
    printf("*******Tips End*******\n");
    exit(0);
}


struct cmds getCmdsData(int argc, char  **argv){
    // got the shell msg
    struct cmds shellData;
    char cmd1[] = "-h";
    char cmd2[] = "-f";
    char cmd3[] = "-d";
    char cmd4[] = "-s";
    char cmd5[] = "-cr";
    char cmd6[] = "-cf";
    char cmd7[] = "-ct";
    char cmd8[] = "-cfn";
    char cmd9[] = "-ci";
    char cmd10[] = "-cs";
    char cmd11[] = "-scn";
    char cmd12[] = "-sp";
    char cmd13[] = "-cin";
    char cmd14[] = "-zapchan";
    char cmd15[] = "-ds";
    char cmd16[] = "-gpu";

    for(int i=0; i<argc; i++)
    {
        if(strcmp(argv[i], cmd1) == 0)
        { // -h help
            argvTips();
        
        }else if(strcmp(argv[i], cmd2) == 0)
        { // -f 
            shellData.filePath = argv[i+1];
            printf("Data file path:%s\n", shellData.filePath);
            i = i+1;
        }else if(strcmp(argv[i], cmd3) == 0)
        { // -d
            shellData.dmPath = argv[i+1];
            printf("DM file path:%s\n", shellData.dmPath);
            i = i+1; 
        }else if(strcmp(argv[i], cmd4) == 0)
        { // -s
            shellData.saveMark = (int)atoi(argv[i+1]);
            i = i+1; 
        }else if(strcmp(argv[i], cmd12) == 0)
        { // -sp
            shellData.outputPath = argv[i+1];
            printf("Output path:%s\n", shellData.outputPath);
            i = i+1; 
        }else if(strcmp(argv[i], cmd5) == 0)
        { // -cr
            shellData.selectNumRate = (float)atof(argv[i+1]);
            i = i+1; 
        }else if(strcmp(argv[i], cmd6) == 0)
        { // -cf
            shellData.funcNum = (int)atoi(argv[i+1]);
            i = i+1; 
        }else if(strcmp(argv[i], cmd7) == 0)
        { // -ct
            shellData.selectThresholdRate = (float)atof(argv[i+1]);
            i = i+1; 
        }else if(strcmp(argv[i], cmd8) == 0)
        { // -cfn
            shellData.selectFrequencyNum = (int)atoi(argv[i+1]);
            i = i+1; 
        }else if(strcmp(argv[i], cmd9) == 0)
        { // -ci
            shellData.candIndex = (int)atoi(argv[i+1]);
            shellData.startIndex = shellData.candIndex;
            i = i+1; 
        }else if(strcmp(argv[i], cmd10) == 0)
        { // -cs
            shellData.picChannelStep = (int)atoi(argv[i+1]);
            shellData.picPramChange = 1;
            i = i+1; 
        }else if(strcmp(argv[i], cmd11) == 0)
        { // -scn
            shellData.picSumChannelNum = (int)atoi(argv[i+1]);
            shellData.picPramChange = 1;
            i = i+1; 
        }else if(strcmp(argv[i], cmd13) == 0)
        {//cin cmd
            shellData.indexRangeNum = (int)atoi(argv[i+1]);
            i = i+1; 
        }else if(strcmp(argv[i], cmd14) == 0)
        {// zapchan cmd
            shellData.zapChanNum=rangeCount(argv[i+1]);
            shellData.zapChanMin = (int*)malloc(sizeof(int)*shellData.zapChanNum);
            shellData.zapChanMax = (int*)malloc(sizeof(int)*shellData.zapChanNum);
            getZapChanRange(shellData.zapChanMin, shellData.zapChanMax, shellData.zapChanNum, argv[i+1]);
            i = i+1;
        }else if(strcmp(argv[i], cmd15) == 0)
        {//cin cmd
            shellData.downsample = (int)atoi(argv[i+1]);
            if(shellData.downsample<1)
            {
                shellData.downsample = 1;
            }
            i = i+1; 
        }else if(strcmp(argv[i], cmd16) == 0)
        {//GPU device id cmd
            shellData.gpuId = (int)atoi(argv[i+1]);
            if(shellData.gpuId<0)
            {
                shellData.gpuId = 0;
            }
            i = i+1; 
        }
        else{
            void argvTips();
        }
    }
    if(shellData.outputPath == NULL)
    {
        shellData.outputPath = (char*)malloc(sizeof(char)*20);
        strcpy(shellData.outputPath, "./file/");
    }

    return shellData;
}