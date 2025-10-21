#!/bin/bash

# 重新生成对象并用g++显式链接该库（使用之前推荐流程）
nvcc -ccbin /usr/bin/g++-11 --cudart=shared -c hello.cu -o hello.o

g++-11 hello.o -o hello_cu_gpgpu_sim_again \
  -Wl,-rpath,"$GPGPU_LIB" -Wl,--enable-new-dtags \
  "$GPGPU_LIB/libcudart.so" -ldl -lpthread

# 验证可执行的 NEEDED / RUNPATH
readelf -d ./hello_cu_gpgpu_sim_again | egrep 'NEEDED|RUNPATH|RPATH'
ldd ./hello_cu_gpgpu_sim_again | grep libcudart

# 6) 运行程序，检测步骤应该不再报错
./hello_cu_gpgpu_sim_again
