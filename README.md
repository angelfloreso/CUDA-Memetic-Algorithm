# CUDA Accelerated Memetic Algorithm

## Requeriments
Ubuntu 16.04
Cuda compilation tools, release 7.5, V7.5.17

## Compile
nvcc --default-stream per-thread -lcurand -o test RRG.cu 

## Profiler analysis
nvprof —f -o analysis.nvprof ./test Instancias/Tominaga2SSGeneratedData.txt Salidas/salida.txt

## Check gpu proccess
watch -n 0.5 nvidia-smi

## Memcheck
cuda-memcheck --leak-check full --racecheck-report all ./test Instancias/Tominaga2SSGeneratedData.txt Salidas/salida2.txt

## References
[Notes](https://icl.utk.edu/~mgates3/docs/cuda.html)

[CUDA C Programing Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)

[CUDA By Example Code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)
