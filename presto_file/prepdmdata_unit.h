#include <stdio.h>
#include "backend_common.h"
struct dmSelectHead {
    float sampleTime;
    int channelNum;
    int channelLen;
    int dmNum;
    float dmMin;
    float dmMax;
    float dmStep;
    char* outfilename;
};
void writeDmProcessFileHead(FILE *file, int dataType, float sampleTime, int channelNum, int channelLen, int dmNum, float dmMin, float dmMax, float dmStep, struct spectra_info *s);
int read_psrdata_change(float *fdata, int numspect, struct spectra_info *s, int *delays, int *padding, int *maskchans, int *nummasked, mask * obsmask, struct dmSelectHead fileinfo);
void write_original_data(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, struct dmSelectHead fileinfo, struct spectra_info *s);
void writeDmProcessFileHead(FILE *file, int dataType, float sampleTime, int channelNum, int channelLen, int dmNum, float dmMin, float dmMax, float dmStep, struct spectra_info *s);