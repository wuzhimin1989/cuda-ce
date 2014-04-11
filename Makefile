OBJ=OBJ_CUDA.o CUDA_link.o PrologInterpreter.o followsunEval.o EnsembleSPSS.o instance.o Autoscaling.o AstarSearch.o constructDAG.o deadlineEval.o ensembleEval.o
 
PrologInter: $(OBJ)
	g++ -g $(OBJ) -L/usr/local/cuda-5.5/lib64 -I /user/local/cuda-5.5/include -lcuda -lcudart -lcudadevrt -fopenmp -o PrologInter
 
%.o: %.cpp
	g++ -g -c -I ./boost_1_50_0 -fopenmp $<
 
CUDA_link.o: OBJ_CUDA.o
	nvcc -g -G -gencode arch=compute_35,code=sm_35 -dlink -o CUDA_link.o OBJ_CUDA.o -lcudadevrt
 
OBJ_CUDA.o: cudaFairCounterexample_lib.cu
	nvcc -g -G cudaFairCounterexample_lib.cu -o OBJ_CUDA.o -gencode arch=compute_35,code=sm_35 -dc
 
clean:
	rm *.o
 
cleanGPU:
	rm OBJ_CUDA.o
	rm CUDA_link.o
