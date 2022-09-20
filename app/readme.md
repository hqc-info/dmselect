## 中间文件说明
---
### 数据文件说明
文件头由下表中的几个部分组成：  
|名称|说明|数据类型|长度/byte|补充说明|   
| --- | --- | --- | --- | --- | 
|file mark|文件标识|ASCII|9|标识头为"DM Select"|   
|head len|整个文件头的长度|int32|4|便于后期在文件头中添加其他信息|  
|data type|文件中数据的类型|int32|4|0代表原始数据、1为DM数据、2为FFT数据|  
|sample time|采样时间(单位毫秒)|float32|4|无|  
|channel num|通道数量|int32|4|无|  
|channel len|每个通道的数据长度|int32|4|当数据为dm时,该值可为0|  
|dm num|DM的次数|int32|4|当数据为原始/FFT数据时，该值可为0|  
|dm min|DM最小值|float32|4|当数据不为DM时，该值为0|  
|dm max|DM最大值|float32|4|当数据不为DM时，该值为0|  
|dm step|DM步长|float32|4|当数据不为DM时，该值为0|
|low channel(MHz)|观测起始频率|double|8|fre low|
|high channel(MHz)|观测结束频率|double|8|fre high| 
|telescope|望远镜名称|char|1*40|望远镜名为char，总长为256个字符(ASCII)|
|soruce file|观测文件名|char|1*256|观测远文件名称，总长度为256个字符(ASCII)|
|source name|源名|char|1*100|同上|
|STT_*|观测开始时间(修正儒略历)|long double/float128|16|修正儒略历，转换为UTC时间|
|RA J2000|NONE|double|8|同上|
|Dec J2000|NONE|double|8|同上| 
|Frontend|前端|char|1*100|同上|
|Backend|后端|char|1*100|同上|
|Data|数据部分|int32/float32|N|每个通道数据挨着写入，复数数据被拆分为两个浮点，前一个浮点为实部、后一个浮点为虚部|  
- fft/original data 数据储存格式为 [(channel0)]、[(channel1)]...[(channel4095)]
- dm data 数据存储格式为[(dm0.0)channel0-channel-4095]、[(dm0.1)]...[(dm999.9)]
- 当无其他改动时，文件头长度的典型值为689byte
### 筛选数据说明
|名称|说明|数据类型|长度/byte|补充说明|
| --- | --- | --- | --- | --- | 
|file mark|文件标识|ASCII|9|标识头为"Cand data"|
|head len|文件头长度|int32|4|文件头的长度|
|dm Num|DM计算次数|int32|4|文件中的DM值计算数量|
|dm channel|DM生成的数据行数|int32|4|None|
|dm min|DM最小值|float32|4|None|
|dm max|DM最大值|float32|4|None|
|fre|频点频率|float32|4|None|
|low channel(MHz)|观测起始频率|double|8|fre low|
|high channel(MHz)|观测结束频率|double|8|fre high| 
|telescope|望远镜名称|char|1*40|望远镜名为char，总长为256个字符(ASCII)|
|soruce file|观测文件名|char|1*256|观测远文件名称，总长度为256个字符(ASCII)|
|source name|源名|char|1*100|同上|
|STT_*|观测开始时间(修正儒略历)|long double/float128|16|修正儒略历，转换为UTC时间|
|RA J2000|NONE|double|8|同上|
|Dec J2000|NONE|double|8|同上| 
|Frontend|前端|char|1*100|同上|
|Backend|后端|char|1*100|同上|
|DATA|数据|float|*|*|
<!-- |Max DM Phase info|data|complex64|*|*| -->
- 数据存储方式为：[dm0-dm999.9]、[dm0-dm999.9]...[dm0-dm999.9]
<!-- - Max DM Phase Data: [channel0 - channel4095] -->
- 当无其他改动时，文件头长度的典型值为677byte
