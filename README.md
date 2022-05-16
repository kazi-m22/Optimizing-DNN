# Optimizing-DNN

In this project, I have written a plain deep neural network in C++ (easily executable in C also) and optimized step by step by using loop optimization, intel x86 vector extension, OpenMP, MPI and Cuda. For training I have used **MNIST dataset**. You can just unzip the dataset and keep on the same folder to run.

## Loop Optimization
For loop optimization, I have implemented loop interchange, loop blocking and unrolling and access pattern manipulation.

## x86 Vecotr Extension
For leveragin SIMD feature provided by intel, I have used SSE2 which can be found here https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE2

## OpenMP and NUMA
Using OpenMP API, I have further improved the performance and usign NUMA tuning, the performance got even better. In my setup, I have **two NUMA nodes** where I have found following configuration most suitable:

OMP_PLACES = threads
OMP_PROC_BIND = false
OMP_NUM_THREADS = 20

## MPI
Used Message Passing Interface on Loop optimized code. In this part, I have used **All Gather** and **Broadcast**. But it can be done in different way. 

## CUDA
Using CUDA feature of NVIDIA GPUs, the loop optimized code will be improved even further. Right now the skull is ready.

## Performance
Below is the runtime to train the model upto 98% percent accuracy on MNIST dataset.
![alt text](https://github.com/theUnspecified/Optimizing-DNN/blob/main/git_runtime.png)
