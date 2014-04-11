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

__constant__ int SCCSIZE;
__constant__ int TOTALSIZE;
__constant__ int TASKTYPE;
__constant__ int WARP_T = 32;
__constant__ int BLOCK_T = 512;
__constant__ int INITIAL_T;
__constant__ int EXPAND_LEVEL = 4;
__constant__ int BLOCK_SYN_THRESHOLD = 8;


//class Gqueue for global memeory access
class GQueue{
public:
	int ** G_queue;
	//Pathnode ** G_Backup_queue;
	int * G_queue_size;
	//int * G_backup_queue_size; //as a backup
	int blockcount;
	//int backupblockcount;

	GQueue()
	{	//backupblockcount = 0;
	}
	~GQueue(){}
};


/***************Global variant****************/
__device__ GQueue G_Queue;
__device__ bool G_ifsccReach;
__device__ int ** P_G_sequence_index; //as a sequencial array to do task partition
__device__ int * P_taskd_index;

//for child use
__device__ int ** C_G_sequence_index; //as a sequencial array to do task partition
__device__ int * C_taskd_index;

__device__ int Child_Expandedtask;
__device__ bool Child_syn_need;
__device__ bool Child_need_back2parent;

__device__ int * Child_Queue_index;

//for the syn between blocks 
__device__ int SynMutex; //for simple syn  
__device__ int * Arrayin;
__device__ int * Arrayout;


//how to use const_restrict memory?

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
	__syncthreads();
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
		__syncthreads();
	
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
	__syncthreads();
}

/*******************************************/

//Quick search for if scc reach
__device__ void BSearchIfreach(bool * theresult, int * searchlist, int size, int key)
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

__global__ void ChildPath(int **, int *, int *, int *, int **, int *, int * ); 
__global__ void GPath(int startid, int * scc, int ** outgoing, int * path2scc, int * pathrecording, int * G_pathrecMutex)  //for optimization, if outgoing is not very big, it can be stored in the specific memory in kepler
{
	int threadindex = blockIdx.x * blockDim.x + threadIdx.x;
	int inblocktid = threadindex;
	int Squeueindex,Squeueposindex;

	__shared__ int Init_S_Queue_head[4][32];
	__shared__ int Init_S_Queue_tail[4][32];

	int i,j;
	for(i=0;i<32;i++)
	{
		for(j=0;j<4;j++)
		{
			Init_S_Queue_head[j][i] = j*(TOTALSIZE-SCCSIZE);
			Init_S_Queue_tail[j][i] = j*(TOTALSIZE-SCCSIZE);
		}
	}

	extern __shared__ int S[];
	//__shared__ int * S_Pathrecord = S;
	__shared__ int * S_Inherit_relation = S;  
	if(threadindex == 0)
	{
		for(i = 0; i< (TOTALSIZE-SCCSIZE); i++)
		{	
			//S_pathrecord[i].Nid = i;
			S_Inherit_relation[i] = -1;
		}
 	}

	__shared__ int *pathrecordmutex = &S_Inherit_relation[TOTALSIZE-SCCSIZE];
	
	if(threadindex == 0)
	{
		for(i = 0; i< TOTALSIZE-SCCSIZE; i++)
		{	
			pathrecordmutex[i] = 0;
		}
	}
	__shared__ int * Init_S_Queue[32];
    //__shared__ int Init_S_queue[32][4*(TOTALSIZE-SCCSIZE)];
	if(threadindex == 0)
	{
		Init_S_Queue[0]= &pathrecordmutex[TOTALSIZE-SCCSIZE];
		for(i = 0; i < 4*(TOTALSIZE-SCCSIZE); i++) 
		{
			Init_S_Queue[0][i] = -1;
		}
		for(i = 1; i<32; i++)
		{
			Init_S_Queue[i]= &Init_S_Queue[i-1][i*(TOTALSIZE-SCCSIZE)*4];
			for(j = 0; j < 4*(TOTALSIZE-SCCSIZE); j++) 
			{
				Init_S_Queue[i][j] = -1;
			}
		}
	}
	
	__syncthreads();

	__shared__ int queuesize;
	__shared__ bool ifexpand;
	__shared__ bool ifSccReach;
	__shared__ unsigned int path2sccmutex;
	__shared__ bool iffinish;

	int tmpnode;
	//bool ifnew;

	Squeueindex = (inblocktid/32+1)%3;
	Squeueposindex = inblocktid % 31;
	int writeindex, readindex;

	if(inblocktid == 0)
	{
		BSearchIfreach(&ifSccReach,scc,SCCSIZE, startid);
		if(!ifSccReach)
		{
			writeindex = Init_S_Queue_tail[Squeueindex][Squeueposindex];
			Init_S_Queue[Squeueposindex][writeindex]=startid;
			Init_S_Queue_tail[Squeueindex][Squeueposindex]++; //move head tail, need modification in the S_queue part.

			queuesize = 1;
			ifexpand = false;
			ifSccReach = false;
			path2sccmutex = 0;
		}

	}

	__syncthreads();

	if(!ifSccReach)
	{
		do{
			if(threadindex < queuesize)
			{
				readindex = Init_S_Queue_head[Squeueindex][Squeueposindex];
				int peeknode = Init_S_Queue[Squeueposindex][readindex];
			
				if(peeknode != -1)
				{
					int succ_num = 0;
					int relationindex;
				
					//judge if belong to scc(sorted)
					BSearchIfreach(&ifSccReach,scc,SCCSIZE, peeknode);

					if(ifSccReach == true)
					{
						while(!iffinish)  
						{  
							if(atomicExch(&path2sccmutex, 1))   //use lock to modify the path2scc
							{
								path2scc[0] = peeknode;
								relationindex = peeknode;
								for(i=1; S_Inherit_relation[relationindex] != startid; i++)
								{
									path2scc[i] = S_Inherit_relation[relationindex];
									relationindex = path2scc[i];
								}

								iffinish = true;
								atomicExch(&path2sccmutex, 0);
							}
						}
						break;
					}

					while(outgoing[peeknode][succ_num] != -1)
					{					

						tmpnode = outgoing[peeknode][succ_num];
						if(atomicExch(&pathrecordmutex[tmpnode], 1))
						{
							if(S_Inherit_relation[tmpnode] != -1)
							{
								atomicExch(&path2sccmutex, 0);
								succ_num++;
								continue;
							}
							S_Inherit_relation[tmpnode] = peeknode;
							writeindex = Init_S_Queue_tail[Squeueindex][Squeueposindex];
							atomicExch(&path2sccmutex, 0);

							Init_S_Queue[Squeueposindex][writeindex] = tmpnode;
							Init_S_Queue_tail[Squeueindex][Squeueposindex]++;
						}
						else
						{
							//ifnew = false;
							succ_num++;
							continue;
						}
												
						succ_num++;
					}
					Init_S_Queue_head[Squeueindex][Squeueposindex]++;
				}			
			}
			if(threadindex == 0)
				iffinish = false;
	
			__syncthreads();

			if(inblocktid == 0)
			{
				for(i = 0; i < 32; i++)
				{
					for(j = 0; j < 4; j++)
						queuesize += (Init_S_Queue_tail[j][i] - Init_S_Queue_head[j][i]);
				}
				if(queuesize > INITIAL_T)
					ifexpand = true;
			}
			__syncthreads();
		}while(ifexpand);

		int expandedtasksize = 0;
		int childbsize = 0;

		if(!ifSccReach && inblocktid == 0)
		{
			/*!!!important!!!FOR THIS PART, how many task to put in each block is very important, in order to decrease the time to call child, 
			* maybe the thread in each block should be more than the task, this can be verified in experiments*/

			expandedtasksize = queuesize;

			childbsize = expandedtasksize / WARP_T + 1;
			G_Queue.G_queue = new int * [childbsize];
			G_Queue.G_queue_size = new int [childbsize];
			G_Queue.blockcount = childbsize;

			for(int j=0; j<childbsize; j++)
			{
				if(TASKTYPE == 1)
					G_Queue.G_queue[j] = new int[TOTALSIZE-SCCSIZE]; //queue stored in Global Queue
				else
					G_Queue.G_queue[j] = new int[SCCSIZE];
				G_Queue.G_queue_size[j] = 0;
			}
		
			int tmpcount = 0;
			int tmp;
			int tmpstart, tmpend;
			for(int i = 0; i < 4; i++)
			{
				for(int j = 0; j < 32; j++)
				{
					readindex = Init_S_Queue_head[i][j];
					writeindex = Init_S_Queue_tail[i][j];
					for(int m = 0; m < writeindex-readindex; m++)
					{
						tmpstart = Init_S_Queue_head[i][j];
						tmpend = Init_S_Queue_tail[i][j];
						tmp = Init_S_Queue[j][tmpstart];
						if(pathrecording[tmp] != -1)
						{
							if(atomicExch(&G_pathrecMutex[tmp], 1))
							{
								pathrecording[tmp] = S_Inherit_relation[tmp];
								
								atomicExch(&path2sccmutex, 0);
								S_Inherit_relation[tmp] = -1;
							}
							else
							{
								Init_S_Queue_head[i][j]++;
								tmpcount++;
								tmpcount=tmpcount%(childbsize-1);
								continue;
							}
							G_Queue.G_queue[tmpcount][G_Queue.G_queue_size[tmpcount]] = Init_S_Queue[j][tmpstart];    //not sure about if the memory copy will work,need confirm.
							G_Queue.G_queue_size[tmpcount]++;
						}
						Init_S_Queue_head[i][j]++;
						tmpcount++;
						tmpcount=tmpcount%(childbsize-1);
						
					}
				}
			}
		}

		__syncthreads();
	
		int expandtime = 1;
		queuesize = 0;
		while(!G_ifsccReach && !ifSccReach)
		{
			//this can be expanded to two version, one is iterative, the other is recursive!
			bool ifneedsyn = false;    

			//rearrange tasks
			if(inblocktid == 0)
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
				P_G_sequence_index = new  int * [queuesize];
				expandedtasksize = queuesize;
			}

			if(inblocktid < childbsize)
			{
				int beginindex = P_taskd_index[inblocktid];
				/*for(int i=0; i<G_Queue.G_queue_size[threadindex]; i++)
				{
				P_G_sequence_index[i+beginindex] = &G_Queue.G_queue[threadindex][i];
				}*/
				//if array are sequential stored then:
				P_G_sequence_index[beginindex] = G_Queue.G_queue[inblocktid];
			}
			if(childbsize > BLOCK_T)    //adjust the map between virtual P_G_Quene to G_queue; 
			{
				int beginindex, leftsize;
				for(int j = 0; j<childbsize/BLOCK_T; j++)
				{
					beginindex = P_taskd_index[j*BLOCK_T + inblocktid];
					P_G_sequence_index[beginindex] = G_Queue.G_queue[j*BLOCK_T + inblocktid];
				}
				leftsize = childbsize % BLOCK_T;
				if(inblocktid < leftsize)
					P_G_sequence_index[P_taskd_index[(childbsize/BLOCK_T)*BLOCK_T + inblocktid]] = G_Queue.G_queue[(childbsize/BLOCK_T)*BLOCK_T + inblocktid];
			}
			////////////////////////////////
	
			if(inblocktid == 0)
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
					Arrayin = new int[childbsize];
					Arrayout = new int[childbsize];
					ChildPath<<<(EXPAND_LEVEL*(averagetask)), childbsize>>>(startid,P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing, pathrecording, G_pathrecMutex);
					cudaDeviceSynchronize();
				}
				else
				{
					Arrayin = new int[1];
					Arrayout = new int[1];
					ChildPath<<<512,1>>>(startid,P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing, pathrecording, G_pathrecMutex);
					cudaDeviceSynchronize();
				}
				//call child path,how to combine each block to just one SM?
			
				expandtime++;
				expandedtasksize = 0;
				ifneedsyn = true;
			}

			if(ifneedsyn)
				__syncthreads();
		}	
	}
}

__global__ void ChildPath(int startid, int ** G_sequence_Queue, int * taskindex, int * p2scc, int * scc, int ** outgoing, int * pathrecording, int * G_pathrecMutex)   //dynamic parallel in cuda, all static data could be stored in specific storage of Kepler
{
	int inblocktindex = threadIdx.x;
	int globalthreadindex = blockDim.x * blockIdx.x + threadIdx.x;
	int Squeueindex, Squeueposindex;
	int i,j;
	////////////////////////////
	__shared__ int C_Init_S_Queue_head[4][32];
	__shared__ int C_Init_S_Queue_tail[4][32];
	for(i=0;i<32;i++)
	{
		for(j=0;j<4;j++)
		{
			C_Init_S_Queue_head[j][i] = j*(TOTALSIZE-SCCSIZE);
			C_Init_S_Queue_tail[j][i] = j*(TOTALSIZE-SCCSIZE);
		}
	}
	extern __shared__ int C[];
	//__shared__ int * S_Pathrecord = S;
	__shared__ int * C_S_Inherit_relation = C;  
	if(inblocktindex == 0)
	{
		for(i = 0; i< (TOTALSIZE-SCCSIZE); i++)
		{	
			//S_pathrecord[i].Nid = i;
			C_S_Inherit_relation[i] = -1;
		}
	}

	__shared__ int *pathrecordmutex = &C_S_Inherit_relation[TOTALSIZE-SCCSIZE];

	if(inblocktindex == 0)
	{
		for(i = 0; i< TOTALSIZE-SCCSIZE; i++)
		{	
			pathrecordmutex[i] = 0;
		}
	}
	__shared__ int * C_Init_S_Queue[32];
	//__shared__ int Init_S_queue[32][4*(TOTALSIZE-SCCSIZE)];
	if(inblocktindex == 0)
	{
		C_Init_S_Queue[0]= &pathrecordmutex[TOTALSIZE-SCCSIZE];
		for(i = 0; i < 4*(TOTALSIZE-SCCSIZE); i++) 
		{
			C_Init_S_Queue[0][i] = -1;
		}
		for(i = 1; i<32; i++)
		{
			C_Init_S_Queue[i]= &C_Init_S_Queue[i-1][i*(TOTALSIZE-SCCSIZE)*4];
			for(j = 0; j < 4*(TOTALSIZE-SCCSIZE); j++) 
			{
				C_Init_S_Queue[i][j] = -1;
			}
		}
	}
	///////////////////////////////////
	__syncthreads();
       
	__shared__ int queuesize;
	__shared__ bool ifSccReach;
	__shared__ bool iffinish;
	__shared__ unsigned int C_path2sccmutex;


	int duration=taskindex[blockIdx.x + 1] - taskindex[blockIdx.x];
	int goalVal = 0;

	int Childpeeknode;

	if(inblocktindex == 0)
	{
		queuesize = duration;
		ifSccReach = false;
	}
	if(globalthreadindex == 0)
	{
		Child_syn_need = false;
		Child_need_back2parent = false;
		Child_Queue_index = new int[gridDim.x];
		Child_Expandedtask = 0;
	}

	if(gridDim.x < BLOCK_SYN_THRESHOLD)
		__gpu_blocks_simple_syn(gridDim.x);
	else
		__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);
	
	Squeueindex = inblocktindex/32;
	Squeueposindex = inblocktindex % 31;
	int cwriteindex, creadindex;
	while(!G_ifsccReach && !Child_need_back2parent)
	{
		//copy data from global memory to shared memory
		duration=taskindex[blockIdx.x + 1] - taskindex[blockIdx.x];

		if(inblocktindex < duration)
		{
			for(int i=0; i<duration/blockDim.x; i++)
			{
				cwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
				C_Init_S_Queue[Squeueposindex][cwriteindex]=(*(G_sequence_Queue)[taskindex[blockIdx.x]+ i * blockDim.x + inblocktindex]);
				C_Init_S_Queue_tail[Squeueindex][Squeueposindex]++;
			}
		}
		if(inblocktindex < duration - (duration/blockDim.x)*blockDim.x)
		{
			cwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
			C_Init_S_Queue[Squeueposindex][cwriteindex] = (*(G_sequence_Queue)[taskindex[blockIdx.x] + (duration/blockDim.x)*blockDim.x + inblocktindex]);
			C_Init_S_Queue_tail[Squeueindex][Squeueposindex]++;
		}
		__syncthreads();

		if(globalthreadindex == 0)   //!not confirmed if needed this
			free(*G_sequence_Queue);
		//////////////////////////////////////////////////
		if(inblocktindex < queuesize)
		{
			creadindex = C_Init_S_Queue_head[Squeueindex][Squeueposindex];
			Childpeeknode = C_Init_S_Queue[Squeueposindex][creadindex];
			if(Childpeeknode)
			{
				int succ_num = 0;
				int relationindex;
				BSearchIfreach(&ifSccReach, scc, SCCSIZE, Childpeeknode);

				if(ifSccReach == true)
				{
					while(!iffinish)  
					{  
						if(atomicExch(&C_path2sccmutex, 1))   //use lock to modify the path2scc
						{
							p2scc[0] = Childpeeknode;
							relationindex = Childpeeknode;
							for(i=1; C_S_Inherit_relation[relationindex] != -1; i++)
							{
								p2scc[i] = C_S_Inherit_relation[relationindex];
								relationindex = p2scc[i];
							}

							for(j=0; pathrecording[relationindex] != startid; j++)
							{
								p2scc[i+j] = pathrecording[relationindex];
								relationindex = p2scc[i+j];
							}
							iffinish = true;
							atomicExch(&C_path2sccmutex, 0);
						}
						else
							break;
					}

					if(inblocktindex == 0)
						G_ifsccReach = true;
					break;
				}

				int tmpnode;

				while(outgoing[Childpeeknode][succ_num] != -1)
				{
					//int pathcount = 0;
					//bool ifnewjudge = true;

					tmpnode = outgoing[Childpeeknode][succ_num];
					if(atomicExch(&pathrecordmutex[tmpnode], 1))
					{
						if(C_S_Inherit_relation[tmpnode] != -1)
						{
							atomicExch(&pathrecordmutex[tmpnode], 0);
							succ_num++;
							continue;
						}
						C_S_Inherit_relation[tmpnode] = C_S_Inherit_relation[Childpeeknode];
						cwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
						C_Init_S_Queue[Squeueposindex][cwriteindex] = tmpnode;
						C_Init_S_Queue_tail[Squeueindex][Squeueposindex]++;
					}
					else
					{
						//ifnew = false;
						succ_num++;
						continue;
					}

					succ_num++;
				}

				C_Init_S_Queue[Squeueposindex][creadindex] = -1;
				C_Init_S_Queue_head[Squeueindex][Squeueposindex]++;
			}
		}

		if(inblocktindex == 0)
			iffinish = false;
		__syncthreads();

		//calculate queuesize;
		int cpbackindex[8];
		if(inblocktindex == 0)
		{
			for(int j = 0; j < 4; j++)
			{
				for(int i = 0; i < 32; i++)
				{
					queuesize += (C_Init_S_Queue_tail[j][i] - C_Init_S_Queue_head[j][i]);;
					cpbackindex[i]=queuesize;
				}
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
					G_sequence_Queue = new int *[Child_Expandedtask];
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

			//copy back path
			if(blockIdx.x < G_Queue.blockcount)
			{
				if(inblocktindex < WARP_T*4)
				{
					int tcwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
					int tcreadindex = C_Init_S_Queue_head[Squeueindex][Squeueposindex];
					for(int m = 0; m < tcreadindex-tcwriteindex; m++)
					{
						cwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
						creadindex = C_Init_S_Queue_head[Squeueindex][Squeueposindex];
						int tmpid =  C_Init_S_Queue[Squeueposindex][creadindex];
						if(pathrecording[tmpid] == -1)
						{
							if(atomicExch(&G_pathrecMutex[tmpid], 1))
							{
								pathrecording[tmpid] = C_S_Inherit_relation[tmpid];
								atomicExch(&G_pathrecMutex[tmpid], 0);
								C_S_Inherit_relation[tmpid] = -1;
							}
							else
							{
								C_Init_S_Queue[Squeueposindex][creadindex] = -1;
								C_Init_S_Queue_head[Squeueindex][Squeueposindex]++;
								continue;
							}
						}
						G_Queue.G_queue[blockIdx.x][cpbackindex[inblocktindex]+m] = C_Init_S_Queue[Squeueposindex][creadindex];    //not sure about if the memory copy will work,need confirm.
						C_Init_S_Queue[Squeueposindex][creadindex]=-1;
						C_Init_S_Queue_head[Squeueindex][Squeueposindex]++;
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
				if(inblocktindex < WARP_T*4)
				{
					int tcwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
					int tcreadindex = C_Init_S_Queue_head[Squeueindex][Squeueposindex];
					for(int m = 0; m< tcwriteindex-tcreadindex; m++)
					{
						cwriteindex = C_Init_S_Queue_tail[Squeueindex][Squeueposindex];
						creadindex = C_Init_S_Queue_head[Squeueindex][Squeueposindex];
						int tmpid =  C_Init_S_Queue[Squeueposindex][creadindex];
						if(pathrecording[tmpid] == -1)
						{
							if(atomicExch(&G_pathrecMutex[tmpid], 1))
							{
								pathrecording[tmpid] = C_S_Inherit_relation[tmpid];
								atomicExch(&G_pathrecMutex[tmpid], 0);
								C_S_Inherit_relation[tmpid]=-1;
							}
							else
							{
								C_Init_S_Queue[Squeueposindex][creadindex] = -1;
								C_Init_S_Queue_head[Squeueindex][Squeueposindex]++;
								continue;
							}
						}
						//int tmp = blockIdx.x % G_Queue.blockcount;
						G_Queue.G_queue[m][Child_Queue_index[m]+cpbackindex[inblocktindex]+m] = C_Init_S_Queue[Squeueposindex][creadindex];
						C_Init_S_Queue[Squeueposindex][creadindex] = -1;
						C_Init_S_Queue_head[Squeueindex][Squeueposindex]++;
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
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

		}

	}
}

__device__ void partition(int* array, int left, int right, int * nleft, int *nright)
{
	int index = left;
	int pivot = array[index];
	int tmp;	
	tmp = array[index];
	array[index]=array[right];
	array[right]=tmp;
	for (int i=left; i<right; i++)
	{
		if (array[i] > pivot)
		{    
			tmp = array[index];
			array[index++]=array[i];
			array[i]=tmp;
		}
	}
	tmp = array[right];
	array[right]=array[index];
	array[index]=tmp;
	*nleft = index-1;
	*nright = index+1;
}

//Cuda Quicksort
__global__ void Gquicksort(int ** data, int left, int right)
{
	int nleft, nright;
	cudaStream_t s1, s2;

	partition(*data, left, right, &nleft, &nright);

	if(left < nright)
	{
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		Gquicksort<<<1,1,0,s1>>>(data, left, nright);
	}
	if(nleft < right)
	{
		cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		Gquicksort<<<1,1,0,s2>>>(data, nleft, right);
	}

}
////////////////////////


//outtoing array: should add a -1 in the end of each array.
int * CudaPath(int initial_t, int sccsize,  int totalsize, int startID, int * sccnodelist, int ** outgoingtransition, int outgoingwidth) //sccnodelist and acceptlist should be sorted for quick search
{
	int deviceCount;
	int * G_path2scc, *H_path2scc;
	//int * G_path2acc, *H_path2acc;
	int ** G_outgoing;
	int * G_sccnodelist;

	int * G_pathrecordingMutex, *H_pathcordingMutex;
	Pathnode * G_pathrecording, *H_pathrecording;
	//int * G_acceptlist;
	int i=1;
	size_t acturalsize;

	string returnresult;

	cudaMemcpyToSymbol(&SCCSIZE, &sccsize, sizeof(int));
	cudaMemcpyToSymbol(&TOTALSIZE,&totalsize, sizeof(int));
	cudaMemcpyToSymbol(&TASKTYPE,&i,sizeof(int));
	cudaMemcpyToSymbol(&INITIAL_T,&initial_t,sizeof(int));
	//SCCSIZE = sccsize;
	//TOTALSIZE = totalsize;
	//TASKTYPE = 1;
	//INITIAL_T = initial_t;

	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
		return NULL;

	//cudasetdevice();  //optional to use
	H_path2scc = new int[totalsize-sccsize];
	H_pathrecording = new Pathnode[totalsize - sccsize];
	H_pathcordingMutex = new int[totalsize-sccsize];
	for(i = 0; i < totalsize - sccsize; i++)
	{
		H_path2scc[i] = -1;
		H_pathrecording[i].Nid = i;
		H_pathcordingMutex[i] = 0;
	}

	cudaMalloc((void**)&G_path2scc, sizeof(int)*(totalsize-sccsize));
	
	cudaMalloc((void**)&G_sccnodelist, sizeof(int)*sccsize);

	cudaMalloc((void**)&G_pathrecording, sizeof(Pathnode)*(totalsize - sccsize));

	cudaMalloc((void**)&G_pathrecordingMutex, sizeof(int)*(totalsize - sccsize));
	
	cudaMallocPitch((void**)&G_outgoing, &acturalsize, sizeof(int)*outgoingwidth, totalsize);    //outgoing from pat should be a n*m

	cudaMemcpy(G_path2scc,H_path2scc,sizeof(int)*(totalsize-sccsize),cudaMemcpyHostToDevice);
	
	cudaMemcpy(G_sccnodelist,sccnodelist,sizeof(int)*sccsize, cudaMemcpyHostToDevice);

	cudaMemcpy(G_pathrecording, H_pathrecording, sizeof(Pathnode)*(totalsize-sccsize), cudaMemcpyHostToDevice);
	
	cudaMemcpy(G_pathrecordingMutex, H_pathcordingMutex, sizeof(int)*(totalsize-sccsize), cudaMemcpyHostToDevice);

	cudaMemcpy2D(G_outgoing,acturalsize,outgoingtransition,sizeof(int)*outgoingwidth,outgoingwidth,totalsize,cudaMemcpyHostToDevice);
	/*cudaStream_t counterexampleStream[2];
	cudaStreamCreate(&counterexampleStream[0]);
	cudaStreamCreate(&counterexampleStream[1]);
	*/
	dim3 blockparameterp(initial_t,1,1);
	dim3 gridparameterp(1,1,1);
	//int gridparameter = 1; optional
	Gquicksort<<<1,1>>>(&sccnodelist, 0, sccsize - 1);
	cudaDeviceSynchronize();
	//cudaSetDevice();  optional to use
	GPath<<<blockparameterp, gridparameterp, 32>>>(startID, G_sccnodelist, G_outgoing, G_path2scc, G_pathrecording, G_pathrecordingMutex);
	cudaMemcpy(H_path2scc,G_path2scc, sizeof(int)*(totalsize-sccsize), cudaMemcpyDeviceToHost);
		
	return H_path2scc;
}

int * CudaFindshortestpath(int initial_t, int sccsize, int accsize, int totalsize, int startID, int *acclist, int ** outgoingtransition, int outgoingwidth)
{
	int devicecount;
	int * G_path2acc, *H_path2acc;
	int ** G_outgoing;
	int * G_acceptlist;
	int i;
	size_t acturalsize;

	//SCCSIZE = sccsize;
	//TASKTYPE = 2;
	//INITIAL_T = initial_t;

	//NEED MODIFICATION : FIND SHORTEST PATH SHOULE ELIMATE NON-SCC in OUTGOING TO BUILD A SCCOUTGOING
	cudaGetDeviceCount(&devicecount);
	if(devicecount == 0)
		return NULL;

	H_path2acc = new int[sccsize];
	for(i = 0; i < sccsize; i++)
		H_path2acc[i] = -1;

	cudaMalloc((void**)&G_path2acc, sizeof(int)*sccsize);
	cudaMalloc((void**)&G_acceptlist, sizeof(int)*accsize); 
	cudaMallocPitch((void**)&G_outgoing, &acturalsize, sizeof(int)*outgoingwidth, totalsize);  

	cudaMemcpy(G_path2acc,H_path2acc,sizeof(int)*(sccsize),cudaMemcpyHostToDevice);
	cudaMemcpy(G_acceptlist,acclist,sizeof(int)*sccsize, cudaMemcpyHostToDevice);
	cudaMemcpy2D(G_outgoing,acturalsize,outgoingtransition,sizeof(int)*outgoingwidth,outgoingwidth,totalsize,cudaMemcpyHostToDevice);

	dim3 blockparameterp(initial_t,1,1);
	dim3 gridparameterp(1,1,1);
	//int gridparameter = 1; optional
	Gquicksort<<<1,1>>>(&acclist, 0, accsize- 1);
	cudaDeviceSynchronize();

	//GPath<<<blockparameterp,gridparameterp, 32>>>(startID, G_acceptlist, G_outgoing, G_path2acc);
	cudaMemcpy(H_path2acc, G_path2acc, sizeof(int)*sccsize, cudaMemcpyDeviceToHost);

	return H_path2acc;
}


//test main
int main()
{
	int ** outgoing;
	int * sccnlist;
	int * result;

	sccnlist = new int[3];
	outgoing = new int*[15];
	for(int i=0;i<15;i++)
		outgoing[i] = new int[3];

	sccnlist[0] = 10;
	sccnlist[1] = 17;
	sccnlist[2] = 18;

	for(int i=0; i<15;i++)
	{
		outgoing[i][0]=i*2+1;
		outgoing[i][1]=i*2+2;
		outgoing[i][2]=-1;
	}

	result = CudaPath(8,3,31,0,sccnlist,outgoing,3);
	return 1;
}
