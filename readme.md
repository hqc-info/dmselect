# DmSelect Alpha Version (Searching for pulsars with phase characteristics)
## environment 
Linux computer with Nvidia GPU
> - cuda 11.0  
> - sm_60 /Cuda Compute Capability greater than 6.1    
> - [Presto](https://github.com/scottransom/presto) 4.0  
> - python 3.6  
> - matplotlib  
> - numpy
> - scikit-image  
> - tqdm  
---
## Install
For presto func support
>- cp ./presto_file/*   [Presto install path]/src
>- cd [Presto install path]/src
>- make

Compile GPU code
>- make 

## Processing flow

Use Presto test file [GBT_Lband_PSR.fil](http://www.cv.nrao.edu/~sransom/GBT_Lband_PSR.fil)  as an example
- Preprocess with Presto
> - rfifind -time 2 -o gbt ./GBT_Lband_PSR.fil
- Use mask to process files and generate intermediate files. The default generated intermediate file is 0-1000 $cm^{-3}pc$, and the step size is 0.1 $cm^{-3}pc$. Use -dmend -dmstep to change the end DM value and step. 
> - prepdmdata -nobary -mask ./gbt_rfifind.mask -o gbt ./GBT_Lband_PSR.fil
- Now two files are obtained, one is the data file and the other is the DM file:  gbt_td530400_96.dat, gbt_DM_All_96_10000.dm
- Use the compiled dmselect in the app folder
> - ./dmselect -cf 1 -f ./gbt_td530400_96.dat -d ./gbt_DM_All_96_10000.dm -cr 0.01 -cs 6 -scn 16
> - Different files, -cs -scn are different. The default parameters are for FAST 4096 channel files  
>
> |dmselect cmd||
> |---|---|  
> |-h | help|
> | -f | data file | 
> | -d  |DM file|  
> | -sp |save path, default: ./file  |
> | -cf |func 0-fft 1-select 2-index deal|  
> | -cr |Screening ratio |
> | -ds |downsample|
> | -cs |Data processing channel step size  |
> | -scn|Number of channels to add  |
> | -ci |index  |
> | -cin| the num around index  |  

- After the calculation is completed, the ./file folder contains fft data, candidate folder, candidate info
- Draw DM-Frequency image with candidate folder
> - ./DM_Frequency_draw.py ./file/gbt_td530400_96_cand_data/

![DM-Frequency](https://github.com/hqc-info/dmselect/blob/main/img/GBT.png)
- 0-500 $cm^{-3}pc$, 5-1200Hz, The contrast of different files may need to be fine tuned

- Use draw.py to draw a multi-channel phase aligned image of the candidate, the generated files are saved in the image folder.

> - ./draw.py ./gbt_td530400_96_cand_data/ ./image  

![multi-channel phase aligned](https://github.com/hqc-info/dmselect/blob/main/img/pic_216.37234_61.80.png)
