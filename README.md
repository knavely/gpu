gpu
===
libwb
Compiling and Running on Linux and Windows
by Tran Minh Quan
This is a tutorial explains how to compile and run your Machine Problems (MPs) offline without separating on building libwb.

Caution: If you don't have NVIDIA GPUs (CUDA Capable GPUs) on your local machine, you cannot run the executable binaries.

First, regardless your platform is, please install CUDA 5.5 and Cmake 2.8 (http://www.cmake.org/) , then set the path appropriate to these things (linux).

Check out the source codes (only skeleton codes) for MPs as following

https://github.com/hvcl/hetero13

git clone https://github.com/abduld/libwb
1. If you are under Linux environment, you should use gcc lower than 4.7 (mine is 4.4.7). Ortherwise, it will not be compatible with nvcc

cd libwb
ls
mkdir build
cd build/
cmake ..
make -j4
./MP0
