#include "prepdmdata_unit.h"
#include "misc_utils.h"

#include <sys/types.h>
#include <pwd.h>
#include <ctype.h>
#include "fitsio.h"
#include "mask.h"
#include "makeinf.h"
#include <string.h>

#define HEADLEN 689

static long long currentspectra = 0;
#define SWAP(a,b) tmpswap=(a);(a)=(b);(b)=tmpswap;
extern int clip_times(float *rawdata, int ptsperblk, int numchan,
                      float clip_sigma, float *good_chan_levels);

void writeDmProcessFileHead(FILE *file, int dataType, float sampleTime, int channelNum, int channelLen, int dmNum, float dmMin, float dmMax, float dmStep, struct spectra_info *s){
    int headLen = HEADLEN;
    char *file_mark = "DM Select";
    chkfwrite(file_mark, sizeof(char), 9, file);
    chkfwrite(&headLen, sizeof(int), 1, file);
    chkfwrite(&dataType, sizeof(int), 1, file);
    chkfwrite(&sampleTime, sizeof(float), 1, file);
    chkfwrite(&channelNum, sizeof(int), 1, file);
    chkfwrite(&channelLen, sizeof(int), 1, file);
    chkfwrite(&dmNum, sizeof(int), 1, file);
    chkfwrite(&dmMin, sizeof(float), 1, file);
    chkfwrite(&dmMax, sizeof(float), 1, file);
    chkfwrite(&dmStep, sizeof(float), 1, file);
    chkfwrite(&s->lo_freq, sizeof(double), 1, file);
    double hi_freq = s->lo_freq+s->BW;
    chkfwrite(&hi_freq, sizeof(double), 1, file);
    char *telescope;
    char *filename;
    char *source;
    char *frontend;
    char *backend;
    telescope = (char*)malloc(sizeof(char)*40);
    source = (char*)malloc(sizeof(char)*100);
    frontend = (char*)malloc(sizeof(char)*100);
    backend = (char*)malloc(sizeof(char)*100);
    filename = (char*)malloc(sizeof(char)*256);
    memset(filename, 0, sizeof(char)*256);
    memset(frontend, 0, sizeof(char)*100);
    memset(backend, 0, sizeof(char)*100);
    memset(source, 0, sizeof(char)*100);
    memset(telescope, 0, sizeof(char)*40);

    strcat(telescope, s->telescope);
    strcat(filename, s->filenames[0]);
    strcat(source, s->source);
    strcat(frontend, s->frontend);
    strcat(backend, s->backend);
    chkfwrite(telescope, sizeof(char), 40, file);
    chkfwrite(filename, sizeof(char), 256, file);
    chkfwrite(source, sizeof(char), 100, file);
    chkfwrite(&s->start_MJD[0], sizeof(long double), 1, file);
    chkfwrite(&s->ra2000, sizeof(double), 1, file);
    chkfwrite(&s->dec2000, sizeof(double), 1, file);
    chkfwrite(frontend, sizeof(char), 100, file);
    chkfwrite(backend, sizeof(char), 100, file);
}


int read_psrdata_change(float *fdata, int numspect, struct spectra_info *s,
                 int *delays, int *padding,
                 int *maskchans, int *nummasked, mask * obsmask, struct dmSelectHead fileinfo)
// This routine reads numspect from the raw pulsar data defined in
// "s". Time delays and a mask are applied to each channel.  It
// returns the # of points read if successful, 0 otherwise.  If
// padding is returned as 1, then padding was added and statistics
// should not be calculated.  maskchans is an array of length numchans
// contains a list of the number of channels that were masked.  The #
// of channels masked is returned in nummasked.  obsmask is the mask
// structure to use for masking.
{
    int ii, jj, numread = 0, offset;
    double starttime = 0.0;
    static float *tmpswap, *rawdata1, *rawdata2;
    static float *currentdata, *lastdata;
    static int firsttime = 1, numsubints = 1, allocd = 0, mask = 0;
    static double duration = 0.0;

    *nummasked = 0;
    if (firsttime) {
        if (numspect % s->spectra_per_subint) {
            fprintf(stderr,
                    "Error!:  numspect %d must be a multiple of %d in read_psrdata()!\n",
                    numspect, s->spectra_per_subint);
            exit(-1);
        } else
            numsubints = numspect / s->spectra_per_subint;
        if (obsmask->numchan)
            mask = 1;
        rawdata1 = gen_fvect(numsubints * s->spectra_per_subint * s->num_channels);
        rawdata2 = gen_fvect(numsubints * s->spectra_per_subint * s->num_channels);
        allocd = 1;
        duration = numsubints * s->time_per_subint;
        currentdata = rawdata1;
        lastdata = rawdata2;
    }
    /* Read, convert and de-disperse */
    if (allocd) {
        while (1) {
            starttime = currentspectra * s->dt;
            numread = read_rawblocks(currentdata, numsubints, s, padding);
            // 读取 返回读取的长度
            if (mask)
                *nummasked = check_mask(starttime, duration, obsmask, maskchans);
                // 检查需要进行掩模操作的通道，其中nummask为-1时，即所有的通道都需要进行掩模操作
            currentspectra += numread * s->spectra_per_subint;//spectra_per_subint光谱数目

            /* Clip nasty RFI if requested and we're not masking all the channels */
            if ((s->clip_sigma > 0.0) && !(mask && (*nummasked == -1)))
                clip_times(currentdata, numspect, s->num_channels, s->clip_sigma,
                           s->padvals);   // 数据去噪声

            if (mask) {
                if (*nummasked == -1) { /* If all channels are masked */
                    for (ii = 0; ii < numspect; ii++)
                        memcpy(currentdata + ii * s->num_channels,
                               s->padvals, s->num_channels * sizeof(float));
                } else if (*nummasked > 0) {    /* Only some of the channels are masked */
                    int channum;
                    for (ii = 0; ii < numspect; ii++) {
                        offset = ii * s->num_channels;
                        for (jj = 0; jj < *nummasked; jj++) {
                            channum = maskchans[jj];
                            currentdata[offset + channum] = s->padvals[channum];
                        }
                    }
                }
            }

            if (s->num_ignorechans) { // These are channels we explicitly zero
                int channum;
                for (ii = 0; ii < numspect; ii++) {
                    offset = ii * s->num_channels;
                    for (jj = 0; jj < s->num_ignorechans; jj++) {
                        channum = s->ignorechans[jj];
                        currentdata[offset + channum] = 0.0;
                    }
                }
            }

            if (!firsttime)
            {
                /*
                
                    进行消色散的延时处理
                    在处理之前进行数据的提取
                    //

                */
               
                // 数据转换，通道的每个时间点的数据存储在一起
                static int total_data = 0;
                static int save_type = 0;
                static int output_len = 0;
                if(save_type == 0){
                    if((s->N % numspect) == 0){
                        save_type = 1;
                    }else
                    {
                       save_type = 2;
                       output_len = s->N - (s->N % numspect);
                    }
                    
                }
                // approx_mean 2331-start 2332-running 2333-stop
                // delays      delays[0]-total_num
                if(save_type == 1)
                {   delays[0] = s->N;
                    write_original_data(currentdata, lastdata, numspect, s->num_channels,
                             delays, 0.0, fdata, fileinfo, s);
                    
                }else{
                    delays[0] = output_len;
                    if ((s->N - total_data) > numspect){
                        write_original_data(currentdata, lastdata, numspect, s->num_channels,
                                    delays, 0.0, fdata, fileinfo, s);
                    }
                }
                total_data = total_data + numspect;
                // float_dedisp(currentdata, lastdata, numspect, s->num_channels,
                //              delays, 0.0, fdata);


            }
            SWAP(currentdata, lastdata);
            if (numread != numsubints) {
                vect_free(rawdata1);
                vect_free(rawdata2);
                allocd = 0;
            }
            if (firsttime)
                firsttime = 0;
            else
                break;
        }
        return numsubints * s->spectra_per_subint;
    } else {
        return 0;
    }
}


void write_original_data(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, struct dmSelectHead fileinfo, struct spectra_info *s)
// De-disperse a stretch of data with numpts * numchan points. The
// delays (in bins) are in delays for each channel.  The result is
// returned in result.  The input data and delays are always in
// ascending frequency order.  Input data are ordered in time, with
// the channels stored together at each time point.
{
    int ii, jj, kk, zz;
    static int start_mark = 0, total_num = 0, data_num_count = 0;
    static FILE *total_output_file;
    static long long *file_offset;
    // 初始化输出数据
    for (ii = 0; ii < numpts; ii++)
        result[ii] = -approx_mean;
    
    if(start_mark == 0)
    {   // 初始化文件
        start_mark = 1;
        total_num = delays[0];
        char *tfilenm1;
        tfilenm1 = (char *) calloc(100, 1);
        sprintf(tfilenm1, "%s_td%d_%d.dat", fileinfo.outfilename, total_num, numchan);
        total_output_file = chkfopen(tfilenm1, "wb");
        writeDmProcessFileHead(total_output_file, 0, fileinfo.sampleTime, numchan, total_num, fileinfo.dmNum, fileinfo.dmMin, fileinfo.dmMax, fileinfo.dmStep, s);
        
        file_offset = (long long*)calloc(numchan, sizeof(long long));
        for(int i=0;i<numchan;i++){  // 初始写入偏移量
            file_offset[i] = (long long)i*total_num*sizeof(float)+HEADLEN;
        }
    }

    // 写出数据
    for (ii = 0; ii < numchan; ii++) { 
        jj = ii;

        for (kk = 0; kk < numpts; kk++, jj += numchan)
            result[kk] = lastdata[jj];
        fseek(total_output_file, file_offset[ii], SEEK_SET);
        chkfwrite(result, sizeof(float), numpts, total_output_file);                     
    }
    
    // 更新偏移量
    for(int i=0;i<numchan;i++)
    {
        file_offset[i] = (long long)file_offset[i] + (long long)numpts*sizeof(float);
    }

    // 关闭文件
    if(total_num == data_num_count)
    {
        for(int i=0; i<numchan; i++)
        {
            fclose(total_output_file);
        }
    }
     
}