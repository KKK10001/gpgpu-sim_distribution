1. Arise3 CModel的激励是一些列API的集合，文件是xxx.cls

2. dump目录下存放的是Arise3 CModel通过脚本将原始的激励转换成"[PC] 0xXXXX_XXXX_XXXX_XXXX-YYYY-YYYY-YYYY-YYYY instruction_string"的格式的.txt
1) 其中，第二列是128-bit的指令字，第三列是对应的汇编指令名字
2) 第二列的指令字对于在VOLTA上实现扩展指令而言不重要
3) 第一列的PC数值实际就是按照+1 step增长的。理论上，128-bit的指令集应该PC也是128-bit，但实际存放不下.
即便如此，合理的设计应该采用(PC >> n)的方式，例如32-bit ISA，有(PC >> 4)，RTL内部采用这个shift之后的PC。传给编译器的时候再(pc_in_use << 4)。那么，对应到128-bit ISA，就是(PC >> 16), (pc_in_use << 16)
但是，Arise3的设计就是这么傻逼
