//#include <stdlib.h>
//#include <stdio.h>
//#include <ctype.h>
//#include <string.h>
#include <iostream>
#include <queue>
#include <iomanip>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sm_11_atomic_functions.h>
#include "device_launch_parameters.h"
using namespace std;

__constant__ int SCCSIZE = 0;
__constant__ int TOTALSIZE = 0;
__constant__ int TASKTYPE = 0;
__constant__ int WARP_T = 32;
__constant__ int BLOCK_T = 512;
__constant__ int INITIAL_T = 32;
__constant__ int EXPAND_LEVEL = 4;
__constant__ int BLOCK_SYN_THRESHOLD = 8;


/***********Template Queue*****************/
template <class Type>  
class QueueItem
{  
public:  
	QueueItem(const Type &t) : item(t), next(0) {}  
	Type item;  
	QueueItem *next;  
};  

template <class Type>  
class Queue
{  
public:  
	Queue() : head(0),tail(0) {this->size = 0}  
	Queue(const Queue &Q):head(0), tail(0)
	{  
		copy_elems(Q);  
		this->size = Q.count();
	}  
	template <class Type2> 
	Queue<Type>& operator=(const Queue<Type2>&);  
	~Queue()
	{ 
		destroy();
	}  
	Type& front() 
	{ 
		return head->item;
	}  
	const Type &front() const { return head->item; }  
	void push( const Type& );  
	void pop();  
	int count();
	bool empty() const
	{  
		return head == 0;  
	}  
private:
	int size;
	QueueItem<Type> *head;  
	QueueItem<Type> *tail;  
	void destroy();  
	void copy_elems(const Queue&);  
}; 

template <class Type>
int Queue<Type>::count()
{
	return this->size;
}

template <class Type>  
void Queue<Type>::push( const Type& val)  
{  
	QueueItem<Type> *pt = new QueueItem<Type>(val);  
	if(empty())  
		head = tail = pt;  
	else {  
		tail->next = pt;  
		tail = pt;  
	} 
	this->size++;
}  

template <class Type>  
void Queue<Type>::pop()  
{  
	QueueItem<Type>* p = head;  
	head = head->next;
	this->size--;
	delete p;  
}  

template <class Type>  
void Queue<Type>::destroy()  
{  
	while(!empty())  
		pop();  
}  

template <class Type>  
void Queue<Type>::copy_elems(const Queue& orig)  
{  
	for(QueueItem<Type>* pt = orig.head;pt;pt = pt->next)  
		push(pt->item);  
} 

template <class Type>  
template <class Type2> 
Queue<Type>& Queue<Type>::operator=( const Queue<Type2>& orig)  
{  
	if((void*)this == (void*)&orig) {  
		*this;  
	}  
	Queue<Type2> tmp(orig);  
	destroy(); // delete  

	while(!tmp.empty())  
	{  
		push(tmp.front());  
		tmp.pop();  
	}  
	this->size = orig.count();
	return *this;  
}  
/********************************************************************/

//class pathnode
class Pathnode{
public:
	int Nid;
	queue<int> ** tmppath;
	int queueindex;

	Pathnode(){}
	Pathnode(int nid)
	{
		Nid = nid;
		if(TASKTYPE == 1)
		{
			tmppath = new queue<int> * [TOTALSIZE-SCCSIZE];
		}
		else
		{
			tmppath = new queue<int>* [SCCSIZE - 1];
		}
		queueindex = 0;
	}

	Pathnode& operator =(const Pathnode &)
	{

	}
};

//class Gqueue for global memeory access
class GQueue{
public:
	Pathnode ** G_queue;
	//Pathnode ** G_Backup_queue;
	int * G_queue_size;
	//int * G_backup_queue_size; //as a backup
	int blockcount;
	//int backupblockcount;

	GQueue()
	{
		blockcount = 0;
		backupblockcount = 0;
	}
	~GQueue(){;}
};


/***************Global variant****************/
__device__ GQueue G_Queue;
__device__ bool G_ifsccReach;
__device__ Pathnode ** P_G_sequence_index; //as a sequencial array to do task partition
__device__ int * P_taskd_index;

//for child use
__device__ Pathnode ** C_G_sequence_index; //as a sequencial array to do task partition
__device__ int * C_taskd_index;

__device__ int Child_Expandedtask;
__device__ bool Child_syn_need;
__device__ bool Child_need_back2parent;

__device__ int * Child_Queue_index;


//for the syn between blocks 
__device__ int SynMutex; //for simple syn  
__device__ int * Arrayin;
__device__ int * Arrayout;


//syn between blocks
__device__ void __gpu_blocks_simple_syn(int goalval)
{
	//thread ID in a block
	int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
	
	// only thread 0 is used for synchronization
	if (tid_in_block == 0) 
	{
		atomicAdd(&SynMutex, 1);
			
		//only when all blocks add 1 to g_mutex will
		//g_mutex equal to goalVal
		while(SynMutex != goalval) {
			;
		}
	}
	cudaThreadSynchronize();
}
__device__ void __gpu_blocks_tree_syn(int goalval, int * arrayin, int * arrayout)
{
	// thread ID in a block
	int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
	int nBlockNum = gridDim.x * gridDim.y;
	int bid = blockIdx.x * gridDim.y + blockIdx.y;
	
	// only thread 0 is used for synchronization
	if (tid_in_block == 0) 
	{
		Arrayin[bid] = goalval;
	}
	
	if (bid == 1)
	{
		if (tid_in_block < nBlockNum) 
		{
			while (Arrayin[tid_in_block] != goalval) 
			{;}
		}
		cudaThreadSynchronize();
	
		if (tid_in_block < nBlockNum)
		{
			Arrayout[tid_in_block] = goalval;
		}
	}
	
	if (tid_in_block == 0)
	{
		while (Arrayout[bid] != goalval)
		{ ;}
	}
	cudaThreadSynchronize();
}

/*******************************************/

//Quick search for if scc reach
__global__ void BSearchIfreach(bool * theresult, int * searchlist, int size, int key)
{
	int Bslow = 0;
	int Bshigh = size-1;
	int Bsmid = Bslow + (Bshigh-Bslow)/2;

	while(Bslow > Bshigh)
	{
		if(key == searchlist[Bsmid])
		{
			*theresult = true;
		}
		else if(key > searchlist[Bsmid])
		{
			Bslow = Bsmid;
		}
		else
		{
			Bshigh = Bsmid;
		}
		Bsmid = Bslow + (Bshigh-Bslow)/2;
	}
}

__global__ void GPath(int startid, int * scc, int ** outgoing, int * path2scc)
{
	int threadindex = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ Queue<Pathnode> Init_S_queue[8];
	extern __shared__ int queuesize;
	extern __shared__ bool ifexpand;
	extern __shared__ bool ifSccReach;
	extern __shared__ unsigned int path2sccmutex;
	extern __shared__ bool iffinish;

	Pathnode * tmpnode;

	if(threadindex == 0)
	{
		tmpnode = new Pathnode(startid);
		tmpnode->tmppath[tmpnode->queueindex] = new queue<int>();
		tmpnode->tmppath[tmpnode->queueindex]->push(startid);

		tmpnode->queueindex ++;
		Init_S_queue[threadindex % 7].push(*tmpnode);
		queuesize = 1;
		ifexpand = false;
		ifSccReach = false;
		path2sccmutex = 0;
	}

	cudaThreadSynchronize();

	do{
		if(threadindex < queuesize)
		{
			Pathnode * peeknode = &Init_S_queue[threadindex % 7].front();
			
			if(peeknode)
			{
				int succ_num = 0;
				
				//judge if belong to scc(sorted)
				BSearchIfreach(&ifSccReach,scc,SCCSIZE,peeknode->Nid);

				if(ifSccReach == true)
				{
					while(!iffinish)  
					{  
						if(atomicExch(&path2sccmutex, 1))   //use lock to modify the path2scc
						{
							for(int i=0; i< peeknode->tmppath[0]->size();i++)
							{
								path2scc[i] = (peeknode->tmppath[0]->front());
							}
							iffinish = true;
							atomicExch(&path2sccmutex, 0);
						}
					}
					break;
				}

				while(outgoing[peeknode->Nid][succ_num] != -1)
				{					
					int pathcount = 0;
					bool ifnewjudge = true;

					tmpnode = new Pathnode(outgoing[peeknode->Nid][succ_num]);
					while(pathcount < peeknode->queueindex)
					{
						tmpnode->tmppath[tmpnode->queueindex] = peeknode->tmppath[tmpnode->queueindex];
						tmpnode->tmppath[tmpnode->queueindex]->push(tmpnode->Nid);
						tmpnode->queueindex++;
						pathcount ++;
					}
					
					Init_S_queue[threadindex % 7].push(*tmpnode);
					succ_num++;
				}
				Init_S_queue[threadindex % 7].pop();
			}			
		}
		if(threadindex == 0)
			iffinish = false;

		cudaThreadSynchronize();

		if(threadindex== 0)
		{
			for(int i = 0; i < 8; i++)
				queuesize += Init_S_queue[i].count();
			if(queuesize > INITIAL_T)
				ifexpand = true;
		}
		cudaThreadSynchronize();
	}while(ifexpand);

	int expandedtasksize = 0;
	int childbsize = 0;

	if(!ifSccReach && threadindex == 0)
	{
		/*!!!important!!!FOR THIS PART, how many task to put in each block is very important, in order to decrease the time to call child, 
		* maybe the thread in each block should be more than the task, this can be verified in experiments*/

		for(int i=0; i<8; i++)                   
		{
			expandedtasksize += Init_S_queue[i].count();
		}
		childbsize = expandedtasksize / WARP_T + 1;
		G_Queue.G_queue = new Pathnode * [childbsize];
		G_Queue.G_queue_size = new int [childbsize];
		G_Queue.blockcount = childbsize;

		for(int j=0; j<childbsize; j++)
		{
			G_Queue.G_queue[j] = new __device__ Pathnode[TOTALSIZE-SCCSIZE - WARP_T];
			G_Queue.G_queue_size[j] = 0;
		}
		
		int tmpcount = 0;
		for(int j = 0; j < 8; j++)
		{
			for(int m = 0; m < Init_S_queue[j].count() - INITIAL_T/8; m++)
			{
				G_Queue.G_queue[tmpcount][G_Queue.G_queue_size[tmpcount]] = Init_S_queue[j].front();    //not sure about if the memory copy will work,need confirm.
				G_Queue.G_queue_size[tmpcount]++;
				Init_S_queue[j].pop();
				tmpcount++;
				tmpcount=tmpcount%(childbsize-1);
			}
		}
	}

	cudaThreadSynchronize();
	
	int expandtime = 1;
	queuesize = 0;
	while(!G_ifsccReach && !ifSccReach)
	{
		//this can be expanded to two version, one is iterative, the other is recursive!
		bool ifneedsyn = false;    

		//rearrange tasks
		if(threadindex == 0)
		{
			if(expandtime > 1)  
			{
				P_taskd_index = new int[childbsize + 1];   //add 1 is for the end of the last block
				for(int i = 0; i < G_Queue.blockcount + 1; i++)
				{
					queuesize += G_Queue.G_queue_size[i];
					P_taskd_index[i] = queuesize;
				}
			}
			P_G_sequence_index = new __device__ Pathnode * [queuesize];
			expandedtasksize = queuesize;
		}

		if(threadindex < childbsize)
		{
			int beginindex = P_taskd_index[threadindex];
			/*for(int i=0; i<G_Queue.G_queue_size[threadindex]; i++)
			{
			P_G_sequence_index[i+beginindex] = &G_Queue.G_queue[threadindex][i];
			}*/
			//if array are sequential stored then:
			P_G_sequence_index[beginindex] = G_Queue.G_queue[threadindex];
		}
		if(childbsize > BLOCK_T)
		{
			int beginindex, leftsize;
			for(int j = 0; j<childbsize/BLOCK_T; j++)
			{
				beginindex = P_taskd_index[j*BLOCK_T + threadindex];
				P_G_sequence_index[beginindex] = G_Queue.G_queue[j*BLOCK_T + threadindex];
			}
			leftsize = childbsize % BLOCK_T;
			if(threadindex < leftsize)
				P_G_sequence_index[P_taskd_index[(childbsize/BLOCK_T)*BLOCK_T + threadindex]] = G_Queue.G_queue[(childbsize/BLOCK_T)*BLOCK_T + threadindex];
		}
		////////////////////////////////
	
		if(threadindex == 0)
		{
			int averagetask = expandedtasksize/childbsize + 1;
			if(averagetask > WARP_T)
			{
				childbsize = expandedtasksize/WARP_T + 1;
				averagetask = expandedtasksize/childbsize + 1;
			}

			for(int i=0; i<childbsize + 1; i++) 
			{
				P_taskd_index[i] = i*averagetask;
			}

			if(childbsize > 1)
			{
				Arrayin = new __device__ int[childbsize];
				Arrayout = new __device__ int[childbsize];
				ChildPath<<<(EXPAND_LEVEL*(averagetask)), childbsize>>>(P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing);
				cudaDeviceSynchronize();
			}
			else
			{
				Arrayin = new __device__ int[1];
				Arrayout = new __device__ int[1];
				ChildPath<<<512,1>>>(P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing);
				cudaDeviceSynchronize();
			}
			//call child path,how to combine each block to just one SM?
			
			expandtime++;
			expandedtasksize = 0;
			ifneedsyn = true;
		}

		if(ifneedsyn)
			cudaThreadSynchronize();
	}	
}

__global__ void ChildPath(Pathnode ** G_sequence_Queue, int * taskindex, int * p2scc, int * scc, int ** outgoing)   //dynamic parallel in cuda
{
	int inblocktindex = threadIdx.x;
	int globalthreadindex = blockDim.x * blockIdx.x + threadIdx.x;

	extern __shared__ Queue<Pathnode> Child_Init_S_queue[8];
	extern __shared__ int queuesize;
	extern __shared__ bool ifSccReach;
	extern __shared__ bool iffinish;
	extern __shared__ unsigned int C_path2sccmutex;


	int duration=taskindex[blockIdx.x + 1] - taskindex[blockIdx.x];
	int goalVal = 0;

	Pathnode * Childpeeknode;

	if(inblocktindex == 0)
	{
		queuesize = duration;
		ifSccReach = false;
	}
	if(globalthreadindex == 0)
	{
		Child_syn_need = false;
		Child_need_back2parent = false;
		Child_Queue_index = new __device__ int[gridDim.x];
		Child_Expandedtask = 0;
	}

	if(gridDim.x < BLOCK_SYN_THRESHOLD)
		__gpu_blocks_simple_syn(gridDim.x);
	else
		__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);
	
	while(!G_ifsccReach && !Child_need_back2parent)
	{
		//copy data from global memory to shared memory
		duration=taskindex[blockIdx.x + 1] - taskindex[blockIdx.x];

		if(inblocktindex < duration)
		{
			for(int i=0; i<duration/8; i++)
			{
				Child_Init_S_queue[inblocktindex % 7].push(*(G_sequence_Queue)[taskindex[blockIdx.x]+ i*duration + inblocktindex]);
			}
		}
		if(inblocktindex < duration - (duration/8)*8)
		{
			Child_Init_S_queue[inblocktindex % 7].push(*(G_sequence_Queue)[taskindex[blockIdx.x] + (duration/8)*8 + inblocktindex]);
		}
		cudaThreadSynchronize();

		if(globalthreadindex == 0)   //!not confirmed if needed this
			free(*G_sequence_Queue);
		//////////////////////////////////////////////////
		if(inblocktindex < queuesize)
		{
			Childpeeknode = &Child_Init_S_queue[inblocktindex % 7].front();
			if(Childpeeknode)
			{
				int succ_num = 0;
				BSearchIfreach(&ifSccReach, scc, SCCSIZE, Childpeeknode->Nid);

				if(ifSccReach == true)
				{
					while(!iffinish)  
					{  
						if(atomicExch(&C_path2sccmutex, 1))   //use lock to modify the path2scc
						{
							for(int i=0; i< Childpeeknode->tmppath[0]->size();i++)
							{
								p2scc[i] = (Childpeeknode->tmppath[0]->front());
							}
							iffinish = true;
							atomicExch(&C_path2sccmutex, 0);
						}
					}

					if(inblocktindex == 0)
						G_ifsccReach = true;
					break;
				}

				Pathnode * tmpnode;

				while(outgoing[Childpeeknode->Nid][succ_num] != -1)
				{
					int pathcount = 0;
					bool ifnewjudge = true;

					tmpnode = new Pathnode(outgoing[Childpeeknode->Nid][succ_num]);
					while(pathcount < Childpeeknode->queueindex)
					{
						tmpnode->tmppath[tmpnode->queueindex] = Childpeeknode->tmppath[tmpnode->queueindex];
						tmpnode->tmppath[tmpnode->queueindex]->push(tmpnode->Nid);
						tmpnode->queueindex++;
						pathcount ++;
					}

					Child_Init_S_queue[inblocktindex % 7].push(*tmpnode);
					succ_num++;
				}

				Child_Init_S_queue[inblocktindex % 7].pop();				
			}
		}

		if(inblocktindex == 0)
			iffinish = false;
		cudaThreadSynchronize();

		//calculate queuesize;
		int cpbackindex[8];
		if(inblocktindex == 0)
		{
			for(int i = 0; i < 8; i++)
			{
				queuesize += Child_Init_S_queue[i].count();
				cpbackindex[i]=queuesize;
			}
			if(queuesize > blockDim.x)
				Child_syn_need = true;
			Child_Queue_index[blockIdx.x] = queuesize;
		}

		if(gridDim.x < BLOCK_SYN_THRESHOLD)
			__gpu_blocks_simple_syn(gridDim.x);
		else
			__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

		if(Child_syn_need)
		{
			if(globalthreadindex == 0)
			{
				int averagetask, lefttask;
				for(int i=0; i<G_Queue.blockcount;i++)
				{
					Child_Expandedtask += G_Queue.G_queue_size[i]; 
				}
				
				if(Child_Expandedtask > (gridDim.x * blockDim.x))
					Child_need_back2parent = true;
				else
				{
					G_sequence_Queue = new Pathnode *[Child_Expandedtask];
					averagetask = Child_Expandedtask/(gridDim.x);
					lefttask = Child_Expandedtask - averagetask*(gridDim.x);
					for(int i=0;i<gridDim.x;i++)
					{
						if(i<lefttask)
							taskindex[i] = averagetask+1;
						else
							taskindex[i] = averagetask;
					}
				}
			}

			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

			if(blockIdx.x < G_Queue.blockcount)
			{
				if(inblocktindex<8)
				{
					for(int m = 0; m < Child_Init_S_queue[inblocktindex].count(); m++)
					{
						G_Queue.G_queue[blockIdx.x][cpbackindex[inblocktindex]+m] = Child_Init_S_queue[inblocktindex].front();    //not sure about if the memory copy will work,need confirm.
						Child_Init_S_queue[inblocktindex].pop();
					}
				}
				if(inblocktindex == 0)
				{
					G_sequence_Queue[blockIdx.x * queuesize] = G_Queue.G_queue[blockIdx.x];
					G_Queue.G_queue_size[blockIdx.x] = queuesize;
				}
			}
			else
			{
				if(inblocktindex<8)
				{
					for(int m = 0; m< Child_Init_S_queue[inblocktindex].count(); m++)
					{
						int tmp = blockIdx.x % G_Queue.blockcount;
						G_Queue.G_queue[m][Child_Queue_index[m]+cpbackindex[inblocktindex]+m] = Child_Init_S_queue[inblocktindex].front();
						Child_Init_S_queue[inblocktindex].pop();
					}
				}
				if(inblocktindex == 0)
				{
					G_sequence_Queue[blockIdx.x * queuesize] = G_Queue.G_queue[blockIdx.x];
					G_Queue.G_queue_size[blockIdx.x % G_Queue.blockcount] += queuesize;
				}
			}

			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_simple_syn(goalVal++, Arrayin, Arrayout);

		}

	}
}


//Cuda Quicksort
__global__ void Gquicksort(int * data, int left, int right)
{
	int nleft, nright;
	cudaStream_t s1, s2;

	partition(data, left, right, &nleft, &nright);

	if(left < nright)
	{
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		Gquicksort<<<1,1,s1>>>(data, left, nright);
	}
	if(nleft < right)
	{
		cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		Gquicksort<<<1,1,s2>>>(data, nleft, right);
	}

}
void partition(int* array, int left, int right, int * nleft, int *nright)
{
	int index = left;
	int pivot = array[index];	
	swap(array[index], array[right]);
	for (int i=left; i<right; i++)
	{
		if (array[i] > pivot)    
			swap(array[index++], array[i]);
	}
	swap(array[right], array[index]);
	*nleft = index-1;
	*nright = index+1;
}
////////////////////////

string CudaIsfair(int fairnesstype, int sccsize, int * sccnodelist, int * evetlist, int ** outgoingtrainsition)
{
	TASKTYPE = 0;
}

__global__ void CudaUnion()
{

}



//outtoing array: should add a -1 in the end of each array.
string CudaGetCounterexample(int initial_t, int sccsize, int accsize, int totalsize, int startID, int * sccnodelist, int * acceptlist, int ** outgoingtransition, int outgoingwidth) //sccnodelist and acceptlist should be sorted for quick search
{
	int deviceCount;
	int * G_path2scc, *H_path2scc;
	int * G_path2acc, *H_path2acc;
	int ** G_outgoing;
	int * G_sccnodelist;
	int * G_acceptlist;
	int i;
	size_t acturalsize;

	SCCSIZE = sccsize;
	TOTALSIZE = totalsize;
	TASKTYPE = 1;
	INITIAL_T = initial_t;

	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
		return NULL;

	//cudasetdevice();  //optional to use
	H_path2scc = new int[totalsize-sccsize];
	for(i = 0; i < totalsize-sccsize; i++)
		H_path2scc = -1;
	H_path2scc = new int[sccsize];
	for(i = 0; i < totalsize-sccsize; i++)
		H_path2scc = -1;

	cudaMalloc((void**)&G_path2scc, sizeof(int)*(totalsize-sccsize));
	cudaMalloc((void**)&G_path2acc, sizeof(int)*sccsize);
	cudaMalloc((void**)&G_sccnodelist, sizeof(int)*sccsize);
	cudaMalloc((void**)&G_acceptlist, sizeof(int)*accsize); 
	cudaMallocPitch((void**)&G_outgoing, &acturalsize, sizeof(int)*outgoingwidth, totalsize);    //outgoing from pat should be a n*m

	cudaMemcpy(G_path2scc,H_path2scc,sizeof(int)*(totalsize-sccsize),cudaMemcpyHostToDevice);
	cudaMemcpy(G_path2acc,H_path2acc,sizeof(int)*(sccsize),cudaMemcpyHostToDevice);
	cudaMemcpy(G_sccnodelist,sccnodelist,sizeof(int)*sccsize, cudaMemcpyHostToDevice);
	cudaMemcpy(G_acceptlist,acceptlist,sizeof(int)*sccsize, cudaMemcpyHostToDevice);
	cudaMemcpy2D(G_outgoing,acturalsize,outgoingtransition,sizeof(int)*outgoingwidth,outgoingwidth,totalsize,cudaMemcpyHostToDevice);
	/*cudaStream_t counterexampleStream[2];
	cudaStreamCreate(&counterexampleStream[0]);
	cudaStreamCreate(&counterexampleStream[1]);
	*/
	dim3 blockparameterp(INITIAL_T,1,1);
	dim3 gridparameterp(1,1,1);
	//int gridparameter = 1; optional
	Gquicksort<<<1,1>>>(&sccnodelist, 0, SCCSIZE - 1);
	cudaDeviceSynchronize();
	//cudaSetDevice();  optional to use
	GPath<<<blockparameterp, gridparameterp, 32>>>(startID, G_sccnodelist, G_outgoing, G_path2scc);
	cudaMemcpy(H_path2scc,G_path2scc, sizeof(int)*(totalsize-sccsize), cudaMemcpyDeviceToHost);

	GPath<<<blockparameterp,gridparameterp, 32>>>()

}

