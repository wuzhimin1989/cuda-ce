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
__constant__ int MAX_THREAD_SMX = 2048;
__constant__ int MAX_BLOCK_SMX = 16;
__constant__ int MAX_BLOCK_THRESHOLD = 16;
__constant__ int BLOCK_T = 1024;
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

class PathNode{
public:
	int presucc;
	int selfid;
};
/***************Global variant****************/
__device__ GQueue G_Queue;
__device__ bool G_ifsccReach;
__device__ int ** P_G_sequence_index; //as a sequencial array to do task partition
__device__ int * P_taskd_index;

//for child use
__device__ int ** C_G_sequence_index; //as a sequencial array to do task partition
__device__ int * C_taskd_index;

//__device__ int * CBackBlockindex; //use as the index to copy back to global memory.
__device__ int * CBackBlockTasksize; //use to record the task size in each block, used for duplicate eliminataion.

__device__ int Child_Expandedtask;
__device__ bool Child_syn_need;
__device__ bool Child_need_back2parent;

__device__ int * Child_Queue_index;

__device__ bool * DuplicateEli;
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
	int inblocktid = threadIdx.x;

	int i,j;
	int tmpwcount,tmpqcount;
	int MAX_QUEUE_SIZE;
	int QUEUE_AVAI_LENGTH,SUCC_SIZE;
	int WARPSIZE;
	int tmpnode,peeknode;
	int succ_num = 0;
	int relationindex;
	int writeindex, readindex;
	PathNode precord;
	int averagetask, inblockwarpnum;

	int tmpcount = 0;
	int tmp;
	int tmpstart, tmpend;

	__shared__ bool ifinblockadjustment;
	__shared__ bool ifinwarpimbalance;
	__shared__ int queuesize;
	__shared__ bool ifexpand;
	__shared__ bool ifSccReach;
	__shared__ unsigned int path2sccmutex;
	__shared__ bool iffinish;

	__shared__ int Init_S_Queue_head[32];  
	__shared__ int Init_S_Queue_tail[32];
	__shared__ int Init_S_Queue_indexbackup[32];
	__shared__ int S_pathrecord_head[32];
	__shared__ int S_pathrecord_tail[32]; //no need backup for parent
	__shared__ int inblockexceednum[32];
	__shared__ int inblockavailength[32];

	
	if(inblocktid == 0)
	{
		for(i=0;i<32;i++)
		{
			inblockexceednum[j] = 0;
			Init_S_Queue_head[i] = 0;
			Init_S_Queue_tail[i] = 0;
			Init_S_Queue_indexbackup[i] = 0;
		}

		if(TASKTYPE == 0)
			DuplicateEli = new bool[TOTALSIZE - SCCSIZE];
		else
			DuplicateEli = new bool[SCCSIZE];
	}
	
	extern __shared__ int S[];
	
	__shared__ int * Init_S_Queue[32];
    
	if(inblocktid == 0)
	{
		Init_S_Queue[0]= S;
		for(i = 0; i < WARP_T; i++) 
		{
			for(j = 0;j < WARP_T; j++)
				Init_S_Queue[i][j] = -1;
		}
		for(i = 1; i<WARP_T; i++)
		{
			Init_S_Queue[i]= &Init_S_Queue[i-1][WARP_T];
			for(j = 0; j < WARP_T; j++) 
			{
				Init_S_Queue[i][j] = -1;
			}
		}
	}

	__shared__ PathNode * S_Precord_Queue[32];
	if(inblocktid == 0)
	{
		S_Precord_Queue[0] = &S_Precord_Queue[31][2*WARP_T];
		for(i=1; i<WARP_T;i++)
		{
			S_Precord_Queue[i] = &S_Precord_Queue[i-1][WARP_T];
		}
	}
	__syncthreads();


	if(inblocktid == 0)
	{
		BSearchIfreach(&ifSccReach,scc,SCCSIZE, startid);
		if(!ifSccReach)
		{
			Init_S_Queue[0][0]=startid;
			Init_S_Queue_tail[0]++; //move head tail, need modification in the S_queue part.

			queuesize = 1;
			ifexpand = false;
			ifSccReach = false;
			ifinblockadjustment = false;
			path2sccmutex = 0;
		}

	}

	__syncthreads();

	if(!ifSccReach)
	{
		do{
			if(inblocktid < queuesize)
			{
				readindex = Init_S_Queue_head[inblocktid];
				peeknode = Init_S_Queue[inblocktid][readindex];
			
				if(peeknode != -1)
				{
					succ_num = 0;
					relationindex;
				
					//judge if belong to scc(sorted)
					BSearchIfreach(&ifSccReach,scc,SCCSIZE, peeknode);

					if(ifSccReach == true)
					{
						for(i=0;i<S_pathrecord_tail[inblocktid];i++)
						{
							precord = S_Precord_Queue[inblocktid][i];
							if(pathrecording[precord.selfid]!=-1)
								continue;
							else
							{
								if(atomicExch(&G_pathrecMutex[precord.selfid],1))
								{
									pathrecording[precord.selfid] = precord.presucc;
									atomicExch(&G_pathrecMutex[precord.selfid],0);
								}
								else
									continue;
							}
						}

						while(!iffinish)  
						{  
							if(atomicExch(&path2sccmutex, 1))   //use lock to modify the path2scc
							{
								path2scc[0] = peeknode;
								relationindex = peeknode;
								for(i=1; pathrecording[relationindex] != startid; i++)
								{
									path2scc[i] = pathrecording[relationindex];
									relationindex = path2scc[i];
								}
								path2scc[i] = startid;
								iffinish = true;
								atomicExch(&path2sccmutex, 0);
							}
						}

						break;
					}

					
					readindex = Init_S_Queue_head[inblocktid];
					writeindex = Init_S_Queue_tail[inblocktid];
					QUEUE_AVAI_LENGTH = WARP_T-(writeindex-readindex+WARP_T)%WARP_T;

					while(outgoing[peeknode][succ_num] > 0)
					{					
						SUCC_SIZE = outgoing[peeknode][0];
						if(SUCC_SIZE < QUEUE_AVAI_LENGTH)
						{
							tmpnode = outgoing[peeknode][succ_num+1];
							
							(S_Precord_Queue[inblocktid][S_pathrecord_tail[inblocktid]]).selfid = tmpnode;
							(S_Precord_Queue[inblocktid][S_pathrecord_tail[inblocktid]]).presucc = peeknode;

							S_pathrecord_tail[inblocktid]++;

							writeindex = Init_S_Queue_tail[inblocktid];

							Init_S_Queue[inblocktid][writeindex] = tmpnode;
							if(Init_S_Queue_tail[inblocktid]++ == WARP_T)
							{
								Init_S_Queue_tail[inblocktid] -= WARP_T;
							}
										
							succ_num++;
						}
						else
						{
							ifinblockadjustment = true;   //if use atomic operation?
							inblockexceednum[inblocktid] = SUCC_SIZE; 
							inblockavailength[inblocktid] = QUEUE_AVAI_LENGTH;
							break;
						}
						inblockexceednum[inblocktid]= succ_num-1-QUEUE_AVAI_LENGTH;  //HOW TO adujustment inblock
						inblockavailength[inblocktid] = QUEUE_AVAI_LENGTH-succ_num-1;

					}

					if(!ifinblockadjustment)
					{
						Init_S_Queue[inblocktid][readindex] = -1;
						if(Init_S_Queue_head[inblocktid]++ == WARP_T)
						{
							Init_S_Queue_head[inblocktid]-= WARP_T;
						}
					}
				}			
			}

			if(inblocktid == 0)
			{
				iffinish = false;
				if(!ifinblockadjustment)
				{
					for(i = 0;i < WARP_T; i++)
					{
						if((Init_S_Queue_tail[i]-Init_S_Queue_head[i]+WARP_T)%WARP_T == 0)
						{
							for(j=0; j < WARP_T; j++)
							{
								if((Init_S_Queue_tail[i]-Init_S_Queue_head[i]+WARP_T)%WARP_T > 1)
								{
									Init_S_Queue[i][Init_S_Queue_tail[i]++] = Init_S_Queue[j][Init_S_Queue_tail[j]--];
									Init_S_Queue_tail[j] = (Init_S_Queue_tail[j]+WARP_T)%WARP_T;
									break;
								}
							}
						}
					}	
				}
			}

			__syncthreads();

			if(inblocktid == 0)
			{
				queuesize = 0;
				for(i = 0; i < 32; i++)
				{
					queuesize += (Init_S_Queue_tail[i] - Init_S_Queue_head[i] + WARP_T)%WARP_T;
				}
				if(queuesize > INITIAL_T)
				{
					ifexpand = true;
				}
				else
				{					
					if(ifinblockadjustment == true)   //inblock adjustment
					{
						for(i = 0;i<32;i++)
						{	
							succ_num = 1;
							tmpqcount=0;
							if(inblockexceednum[i] > 0)
							{
								readindex = Init_S_Queue_head[i];
								peeknode = Init_S_Queue[i][readindex];
								while((tmpnode=outgoing[tmpnode][succ_num]) != -1)
								{
									while(true)
									{
										if(inblockavailength[tmpqcount] > WARP_T/2) //balance inblock while exceed;
										{
											writeindex = Init_S_Queue_tail[tmpqcount];
											Init_S_Queue[tmpqcount][writeindex] = tmpnode;
											if(Init_S_Queue_tail[tmpqcount]++ == WARP_T)
											{
												Init_S_Queue_tail[tmpqcount] -= WARP_T;
											}
											succ_num++;
											inblockavailength[tmpqcount]--;
											break;
										}
										else
										{
											tmpqcount++;
											if(tmpqcount == 31)
											{											
												tmpqcount = 0;

											}
										}
									}
								}
								
							}
						}
					}

				
					queuesize = 0;
					for(i = 0; i < 32; i++)
					{
						queuesize += (Init_S_Queue_tail[i] - Init_S_Queue_head[i] + WARP_T)%WARP_T;
					}
					if(queuesize > INITIAL_T)
						ifexpand = true;
				}
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

			if(expandedtasksize % WARP_T == 0)
				childbsize = expandedtasksize/WARP_T;
			else
				childbsize = expandedtasksize/WARP_T + 1;

			G_Queue.G_queue = new int * [childbsize];
			G_Queue.G_queue_size = new int [childbsize];
			G_Queue.blockcount = childbsize;

			for(j=0; j<childbsize; j++)
			{
				if(TASKTYPE == 1)
					G_Queue.G_queue[j] = new int[TOTALSIZE-SCCSIZE]; //queue stored in Global Queue
				else
					G_Queue.G_queue[j] = new int[SCCSIZE];
				G_Queue.G_queue_size[j] = 0;
			}
		
			for(i = 0; i < 32; i++)
			{
				while(S_pathrecord_head[i] != S_pathrecord_tail[i])
				{
					precord = S_Precord_Queue[i][S_pathrecord_head[i]++];
					if(pathrecording[precord.selfid] != -1)
						continue;
					else
					{
						pathrecording[precord.selfid] = precord.presucc;
						DuplicateEli[precord.presucc] = true;
					}
				}
				precord = S_Precord_Queue[i][S_pathrecord_tail[i]];
				if(pathrecording[precord.selfid] == -1)
				{
					pathrecording[precord.selfid] = precord.presucc;
					DuplicateEli[precord.presucc] = true;
				}
				S_pathrecord_head[i] = 0;
				S_pathrecord_tail[i] = 0;
			}

			for(j = 0; j < 32; j++)
			{
				readindex = Init_S_Queue_head[j];
				writeindex = Init_S_Queue_tail[j];
				for(int m = 0; m < ((writeindex - readindex + WARP_T) % WARP_T); m++)
				{
					tmpstart = Init_S_Queue_head[j];
					tmpend = Init_S_Queue_tail[j];
					tmp = Init_S_Queue[j][tmpstart];
					if(DuplicateEli[tmp])
					{
						if(Init_S_Queue_head[j]++ == WARP_T)
							Init_S_Queue_head[j] -= WARP_T;
						tmpcount++;
						tmpcount=tmpcount%(childbsize-1);
						continue;				
					}
					G_Queue.G_queue[tmpcount][G_Queue.G_queue_size[tmpcount]] = Init_S_Queue[j][tmpstart];    //not sure about if the memory copy will work,need confirm.
					G_Queue.G_queue_size[tmpcount]++;

					if(Init_S_Queue_head[j]++ == WARP_T)
						Init_S_Queue_head[j] -= WARP_T;
					tmpcount++;
					tmpcount=tmpcount%(childbsize-1);
						
				}
			}
			
		}

		__syncthreads();
	
		int expandtime = 1;
		int averagetask, lastblocktask;
		while(!G_ifsccReach && !ifSccReach)
		{
			//this can be expanded to two version, one is iterative, the other is recursive!
			bool ifneedsyn = false;
			queuesize = 0;

			//rearrange tasks
			if(inblocktid == 0)
			{
				if(expandtime == 1)
				{
					P_taskd_index = new int[childbsize + 1];   //add 1 is for the end of the last block
					P_taskd_index[0] = 0;
					for(int i = 0; i < G_Queue.blockcount + 1; i++)
					{
						queuesize += G_Queue.G_queue_size[i];
						P_taskd_index[i+1] = queuesize;
					}
					P_G_sequence_index = new  int * [queuesize];
					for(i=0;i<childbsize;i++)
						P_G_sequence_index[P_taskd_index[i]] = G_Queue.G_queue[i];
				}
				else
				{
					for(int i = 0; i < G_Queue.blockcount + 1; i++)
					{
						queuesize += G_Queue.G_queue_size[i];
					}
					expandedtasksize = queuesize;
					if((childbsize=expandedtasksize/WARP_T) < MAX_BLOCK_THRESHOLD)  //here max block threshold is not static, it is based on GPU architecture.
					{
						averagetask = WARP_T;
						if(expandedtasksize % WARP_T == 0)
						{
							childbsize = expandedtasksize/WARP_T;
							lastblocktask = 0;
						}
						else
						{
							childbsize = expandedtasksize/WARP_T + 1;
							lastblocktask = expandedtasksize % WARP_T;
						}
						inblockwarpnum = averagetask * EXPAND_LEVEL;
					}
					else
					{
						averagetask = 2*WARP_T;
						while((childbsize=expandedtasksize/averagetask) > MAX_BLOCK_THRESHOLD)
						{
							averagetask += WARP_T;
						}
						lastblocktask = expandedtasksize % averagetask;
						inblockwarpnum = averagetask * EXPAND_LEVEL;   //it is possible that the warp num exceed the limit.
					}

					P_taskd_index = new int[childbsize + 1];
					for(i=0; i<childbsize;i++)
					{
						P_taskd_index[i] = i*averagetask;
					}
					if(lastblocktask != 0)
						P_taskd_index[i+1] = i*averagetask + lastblocktask;
					else
						P_taskd_index[i+1] = 0;

					P_G_sequence_index = new  int * [expandedtasksize];
					P_G_sequence_index[0] = G_Queue.G_queue[0];
					for(i = 0, j = 0; i < G_Queue.blockcount - 1; i++)
					{
						j += G_Queue.G_queue_size[i];
						P_G_sequence_index[j] = G_Queue.G_queue[i+1];
					}
				}
			
			}
			__syncthreads();

			////////////////////////////////
	
			if(inblocktid == 0)  //if add warp in a single block or in mutiple blokcs is needed to eveluate.
			{
				if(childbsize > 1)
				{
					Arrayin = new int[childbsize];
					Arrayout = new int[childbsize];
					ChildPath<<<inblockwarpnum, childbsize>>>(startid,P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing, pathrecording, G_pathrecMutex);
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
			
				
				expandedtasksize = 0;
				ifneedsyn = true;
			}
			expandtime++;

			if(ifneedsyn)
				__syncthreads();
		}	
	}
}

__global__ void ChildPath(int startid, int ** G_sequence_Queue, int * taskindex, int * p2scc, int * scc, int ** outgoing, int * pathrecording, int * G_pathrecMutex)   //dynamic parallel in cuda, all static data could be stored in specific storage of Kepler
{
	int inblocktindex = threadIdx.x;
	int globalthreadindex = blockDim.x * blockIdx.x + threadIdx.x;
	int i,j;
	////////////////////////////
	int TotalBlockTask = 0;
	int MAX_QUEUE_SIZE = WARP_T;  //warp_t 
	int QUEUE_AVAI_LENGTH,SUCC_SIZE; 
	int WARPID, Inwarptid, WARPNum, LWarpTask;  
	bool IFLastHead,IFLastBlock;
	

	int relationindex;  //used for path recording
	int cwriteindex, creadindex;  //used for read/write queue  

	int duration=taskindex[blockIdx.x + 1] - taskindex[blockIdx.x];   //tasks of the whole blokc
	int goalVal = 0; //used for syn among blocks

	int Childpeeknode,tmpnode;
	int succ_num = 1;
	int GBcount;

	PathNode cprecord;

	//tmp used variables
	int tmpwcount,tmpqcount;
	//Squeueindex = inblocktindex/32;
	//Squeueposindex = inblocktindex % 31;
	WARPID = inblocktindex/WARP_T;
	Inwarptid = inblocktindex%WARP_T;
	if(blockDim.x % WARP_T == 0)
		WARPNum = blockDim.x / WARP_T;
	else
	{
		WARPNum = blockDim.x / WARP_T + 1;
	}

	if(WARPID == WARPNum - 1)
	{
		IFLastHead = true;
	}	
	if(blockIdx.x == blockDim.x - 1)
	{
		IFLastBlock = true;
	}

	__shared__ int BlockQueuesize;

	__shared__ bool ifSccReach;
	__shared__ bool iffinish;
	__shared__ unsigned int C_path2sccmutex;

	extern __shared__ int C[];
	__shared__ int * C_Init_S_WarpQueueHead[32];
	if(inblocktindex == 0)
	{
		C_Init_S_WarpQueueHead[0] = C;
		for(i = 1; i < WARPNum; i++)
			C_Init_S_WarpQueueHead[i] = &C_Init_S_WarpQueueHead[i-1][WARPNum];
	}
	__shared__ int * C_Init_S_WarpQueueTail[32];	
	if(inblocktindex == 0)
	{
		C_Init_S_WarpQueueTail[0] = &C_Init_S_WarpQueueHead[31][WARPNum];
		for(i = 1; i < WARPNum; i++)
			C_Init_S_WarpQueueTail[i] = &C_Init_S_WarpQueueTail[i-1][WARPNum];
	}
	__shared__ int * C_Warp_Pathtail[32];
	if(inblocktindex == 0)
	{
		C_Warp_Pathtail[0] = &C_Init_S_WarpQueueTail[31][WARPNum];
		for(i = 1; i < WARPNum; i++)
			C_Warp_Pathtail[i] = &C_Warp_Pathtail[i-1][WARPNum];
	}
	__shared__ int * Warptasknum = &C_Warp_Pathtail[31][WARPNum];
	if(inblocktindex == 0)
	{
		for(i = 1; i < WARPNum; i++)
			Warptasknum[i] = 0;
	}
	__shared__ int * Inwarpqueuelength[32];
	if(inblocktindex == 0)
	{
		Inwarpqueuelength[0] = &Inwarpqueuelength[31][WARPNum];
		for(i = 1; i < WARPNum; i++)
			Inwarpqueuelength[i] = &Inwarpqueuelength[i-1][WARPNum];
	}
	__shared__ PathNode * C_Warp_PathRecording[32];
	if(inblocktindex == 0)
	{
		C_Warp_PathRecording[0] = (PathNode *)&C_Init_S_WarpQueueTail[31][MAX_QUEUE_SIZE];
		for(i = 1; i < WARPNum; i++)
			C_Warp_PathRecording[i] = &C_Warp_PathRecording[i-1][MAX_QUEUE_SIZE];
	}
	__shared__ int * cpbackindex = (int *)&C_Warp_PathRecording[31][MAX_QUEUE_SIZE];

	__shared__ int * C_Init_S_WarpQueue[32];
	if(inblocktindex == 0)
	{
		C_Init_S_WarpQueue[0] = &cpbackindex[WARPNum];
		for(i = 1; i < WARPNum; i++)
			C_Init_S_WarpQueue[i] = &C_Init_S_WarpQueue[i-1][MAX_QUEUE_SIZE];
	}
	__shared__ bool * ifallwithtask = (bool*)&C_Init_S_WarpQueue[31][MAX_QUEUE_SIZE];
	if(inblocktindex == 0)
	{
		for(i = 0; i < WARPNum; i++)
			ifallwithtask[i] = false;
	}
	__shared__ bool * ifinblockadjustment = &ifallwithtask[WARPNum];
	if(inblocktindex == 0)
	{
		for(i = 0; i < WARPNum; i++)
			ifinblockadjustment[i] = false;
	}
	if(inblocktindex == 0)
	{
		BlockQueuesize = duration;
		ifSccReach = false;
		ifinblockadjustment = false;
	}
///////////////////////////////////
		
	if(globalthreadindex == 0)
	{
		Child_syn_need = false;
		Child_need_back2parent = false;
		Child_Queue_index = new int[gridDim.x];
		Child_Expandedtask = 0;
		CBackBlockTasksize = new int[gridDim.x];
	}

	if(gridDim.x < BLOCK_SYN_THRESHOLD)
		__gpu_blocks_simple_syn(gridDim.x);
	else
		__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);
	
	//add warpdecision process

	while(!G_ifsccReach && !Child_need_back2parent)
	{
		//copy data from global memory to shared memory
		duration=taskindex[blockIdx.x + 1] - taskindex[blockIdx.x];

		if(Inwarptid == 0)  //for lastblock, the tasks is not the multiple of warp_t
		{
			j = 0;
			for(i=0; i < duration/WARPNum; i++)
			{
				C_Init_S_WarpQueue[j][C_Init_S_WarpQueueTail[j][WARPID]] = *(G_sequence_Queue)[taskindex[blockIdx.x] + WARPID * duration/WARPNum + i];
				if(C_Init_S_WarpQueueTail[j][WARPID]++ == WARP_T)
					C_Init_S_WarpQueueTail[j][WARPID] -= WARP_T;
				
				if(j == 31)
					ifallwithtask[WARPID] = true;  //according to the setting of parent, this condition may not happen currently.

				j++;
				j = j % WARP_T;
			}
		}

		if(Inwarptid == 0 && IFLastBlock == true)
		{
			LWarpTask = duration - WARPNum*(duration/WARPNum);
			if(WARPID < LWarpTask)
				C_Init_S_WarpQueue[j][C_Init_S_WarpQueueTail[j][WARPID]] = *(G_sequence_Queue)[taskindex[blockIdx.x] + WARPNum * (duration/WARPNum) + WARPID];
			if(C_Init_S_WarpQueueTail[j][WARPID]++ == MAX_QUEUE_SIZE)
				C_Init_S_WarpQueueTail[j][WARPID] -= MAX_QUEUE_SIZE;
			if(j == 31)
				ifallwithtask[WARPID] = true;
		}
		__syncthreads();

		if(globalthreadindex == 0)   //!not confirmed if needed this
			free(*G_sequence_Queue);

		while(!ifallwithtask && !ifSccReach)  //Initial step, guarantee each queue has task.
		{
			if(C_Init_S_WarpQueueTail[Inwarptid][WARPID] != 0)
			{
				creadindex = C_Init_S_WarpQueueHead[Inwarptid][WARPID];
				Childpeeknode = C_Init_S_WarpQueue[Inwarptid][creadindex];
				if(Childpeeknode)
				{
					BSearchIfreach(&ifSccReach, scc, SCCSIZE, Childpeeknode);
					if(ifSccReach == true)
					{
						G_ifsccReach = true;
						for(i=0;i<C_Warp_Pathtail[Inwarptid][WARPID];i++)
						{
							cprecord = C_Warp_PathRecording[Inwarptid][i];
							if(pathrecording[cprecord.selfid]!=-1)
								continue;
							else
							{
								if(atomicExch(&G_pathrecMutex[cprecord.selfid],1))
								{
									pathrecording[cprecord.selfid] = cprecord.presucc;
									atomicExch(&G_pathrecMutex[cprecord.selfid],0);
								}
								else
									continue;
							}
						}

						while(!iffinish)  
						{  
							if(atomicExch(&C_path2sccmutex, 1))   //use lock to modify the path2scc
							{
								if(atomicExch(&C_path2sccmutex, 1))   //use lock to modify the path2scc
								{
									p2scc[0] = Childpeeknode;
									relationindex = Childpeeknode;
									for(i=1; pathrecording[relationindex] != startid; i++)
									{
										p2scc[i] = pathrecording[relationindex];
										relationindex = p2scc[i];
									}
									p2scc[i] = startid;
									iffinish = true;
									atomicExch(&C_path2sccmutex, 0);
								}
							}
						}
						break;
					}

					cwriteindex = C_Init_S_WarpQueueTail[Inwarptid][WARPID];
					while(outgoing[Childpeeknode][succ_num] != -1)
					{
						tmpnode = outgoing[Childpeeknode][succ_num];
						(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]]).selfid = tmpnode;
						(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]++]).presucc = Childpeeknode;
						C_Init_S_WarpQueue[Inwarptid][cwriteindex] = tmpnode;
						if(C_Init_S_WarpQueueTail[Inwarptid][WARPID]++ == MAX_QUEUE_SIZE)
							C_Init_S_WarpQueueTail[Inwarptid][WARPID] -= MAX_QUEUE_SIZE;

						C_Init_S_WarpQueueHead[Inwarptid][WARPID]++;
					}
				}
			}
			__syncthreads();

			if(Inwarptid == 0)
			{
				for(i = 0;i < WARP_T; i++)
				{
					if(C_Init_S_WarpQueueTail[i][WARPID] == 0)
					{
						for(j=0; j < WARP_T; j++)
						{
							if(C_Init_S_WarpQueueTail[j][WARPID] > 1)
							{
								C_Init_S_WarpQueue[i][C_Init_S_WarpQueueTail[i][WARPID]++] = C_Init_S_WarpQueue[j][C_Init_S_WarpQueueTail[j][WARPID]--];
								C_Warp_PathRecording[i][C_Init_S_WarpQueueTail[i][WARPID]++] = C_Warp_PathRecording[j][C_Warp_Pathtail[j][WARPID]--];
								break;
							}
						}
					}
				}
				if(j == 0)
					ifallwithtask[WARPID] = true;
			}
			__syncthreads();
		}

		if(gridDim.x < BLOCK_SYN_THRESHOLD)
			__gpu_blocks_simple_syn(gridDim.x);
		else
			__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

		//////////////////////////////////////////////////
		if(!G_ifsccReach)
		{
			creadindex = C_Init_S_WarpQueueHead[Inwarptid][WARPID];
			Childpeeknode = C_Init_S_WarpQueue[Inwarptid][creadindex];
			if(Childpeeknode)
			{
				succ_num = 1;
				BSearchIfreach(&ifSccReach, scc, SCCSIZE, Childpeeknode);

				if(ifSccReach == true)
				{
					G_ifsccReach = true;
					for(i=0;i<C_Warp_Pathtail[Inwarptid][WARPID];i++)
					{
						cprecord = C_Warp_PathRecording[Inwarptid][i];
						if(pathrecording[cprecord.selfid]!=-1)
							continue;
						else
						{
							if(atomicExch(&G_pathrecMutex[cprecord.selfid],1))
							{
								pathrecording[cprecord.selfid] = cprecord.presucc;
								atomicExch(&G_pathrecMutex[cprecord.selfid],0);
							}
							else
								continue;
						}
					}

					while(!iffinish)  
					{  
						if(atomicExch(&C_path2sccmutex, 1))   //use lock to modify the path2scc
						{
							p2scc[0] = Childpeeknode;
							relationindex = Childpeeknode;
							for(i=1; pathrecording[relationindex] != startid; i++)
							{
								p2scc[i] = pathrecording[relationindex];
								relationindex = p2scc[i];
							}

							iffinish = true;
							atomicExch(&C_path2sccmutex, 0);
						}
					}

					break;
				}

				cwriteindex = C_Init_S_WarpQueueTail[Inwarptid][WARPID];
				QUEUE_AVAI_LENGTH = MAX_QUEUE_SIZE - (cwriteindex-creadindex+MAX_QUEUE_SIZE)%MAX_QUEUE_SIZE;
				
				//SCCSIZE = outgoing[Childpeeknode][0];

				while(outgoing[Childpeeknode][succ_num] != -1)
				{
					//int pathcount = 0;
					//bool ifnewjudge = true;
					/*if(SCCSIZE < QUEUE_AVAI_LENGTH)
					{*/
					tmpnode = outgoing[Childpeeknode][succ_num];
						
					(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]]).selfid = tmpnode;
					(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]]).presucc = Childpeeknode;

					cwriteindex = C_Init_S_WarpQueueTail[Inwarptid][WARPID];
					C_Init_S_WarpQueue[Inwarptid][cwriteindex] = tmpnode;
					
					if(C_Init_S_WarpQueueTail[Inwarptid][WARPID]++ == MAX_QUEUE_SIZE)
					{
						C_Init_S_WarpQueueTail[Inwarptid][WARPID] -= MAX_QUEUE_SIZE;
					}

					succ_num++;
					//}
					//else
					//{
					//	ifinblockadjustment = true;   //if use atomic operation?
					//	inblockexceednum[Inwarptid][WARPID] = SUCC_SIZE; 
					//	inblockavailength[Inwarptid][WARPID] = QUEUE_AVAI_LENGTH;
					//	break;
					//}
					//inblockexceednum[Squeueindex][Squeueposindex]= succ_num-1-QUEUE_AVAI_LENGTH;  //HOW TO adujustment inblock
					//inblockavailength[Squeueindex][Squeueposindex] = QUEUE_AVAI_LENGTH-succ_num-1;
				}
				
				if(C_Init_S_WarpQueueHead[Inwarptid][WARPID]++ == MAX_QUEUE_SIZE)
				{
					C_Init_S_WarpQueueHead[Inwarptid][WARPID]-= MAX_QUEUE_SIZE;
				}
				Inwarpqueuelength[Inwarptid][WARPID] = (C_Init_S_WarpQueueTail[Inwarptid][WARPID] - C_Init_S_WarpQueueHead[Inwarptid][WARPID]);
				
			}
		}

		if(inblocktindex == 0)
			iffinish = false;
		__syncthreads();

		if(Inwarptid == 0)
		{
			for(i = 0; i < 32; i++)
				Warptasknum[WARPID] += Inwarpqueuelength[i][WARPID];
			if(Warptasknum[WARPID] > 32)
				ifinblockadjustment[WARPID] = true;
		}
		__syncthreads();

		//calculate queuesize;
		//int cpbackindex[4][32];
		int indexcount=0;
		int tmp;
		PathNode tmp2;
		if(inblocktindex == 0)
		{
			for(j = 0; j < WARPNum; j++)
			{
				TotalBlockTask += Warptasknum[j];
				cpbackindex[indexcount]=TotalBlockTask;
				indexcount++;
			}
			if(TotalBlockTask > blockDim.x)
			{
				Child_syn_need = true;
				Child_Queue_index[blockIdx.x] = TotalBlockTask;
			}
			else  //adjustment intra block, inter warps
			{
				for(i = 0; i < WARPNum; i++)
				{
					if(ifinblockadjustment[i] == false)   //inblock adjustment
					{
						int tmpreadindex1, tmpreadindex2;
						for(j = 0; j<WARPNum; j++)
						{
							if(ifinblockadjustment[j] == true && Warptasknum[j] == 32)
								continue;

							succ_num = 1;
							tmpwcount=0;  //mark the queue

							while(Warptasknum[j] > 32 && Warptasknum[i] < 32)
							{
								while(Inwarpqueuelength[tmpwcount][j] > 1)
								{
									tmpreadindex1 = C_Init_S_WarpQueueTail[tmpwcount][j]--;
									tmp = C_Init_S_WarpQueue[tmpwcount][tmpreadindex1];
									C_Init_S_WarpQueue[tmpwcount][C_Init_S_WarpQueueTail[tmpwcount][i]++] = tmpnode;

									tmpreadindex2 = C_Warp_Pathtail[tmpwcount][j]--;
									tmp2 = C_Warp_PathRecording[tmpwcount][tmpreadindex2];
									(C_Warp_PathRecording[tmpwcount][C_Warp_Pathtail[tmpwcount][i]]).selfid = tmp2.selfid;
									(C_Warp_PathRecording[tmpwcount][C_Warp_Pathtail[tmpwcount][i]]).presucc = tmp2.presucc;

									Inwarpqueuelength[tmpwcount][j]--;
									Warptasknum[j]--;
									Warptasknum[i]++;
								}
								tmpwcount++;
							}
						}
					}
				}
				continue;
			}

		}

		if(gridDim.x < BLOCK_SYN_THRESHOLD)
			__gpu_blocks_simple_syn(gridDim.x);
		else
			__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

		if(Child_syn_need)
		{
			if(globalthreadindex == 0)
			{
				int averagetask, lefttask, GBcount, tmpmark;

				for(i=0; i<gridDim.x;i++)
				{
					Child_Expandedtask += Child_Queue_index[i]; 
					CBackBlockTasksize[i] = Child_Queue_index[i];
				}
				
				GBcount = G_Queue.blockcount;
				for(i = GBcount; i<gridDim.x; i++) //calculate the start index of each block in Gqueue.
				{
					tmpmark = i / (GBcount-1);
					j = i % (GBcount - 1);
					while(tmpmark > 0)
					{
						Child_Queue_index[i] += Child_Queue_index[j + (tmpmark-1)*GBcount];
						tmpmark--;
					}
				}
				
				if(Child_Expandedtask > (gridDim.x * blockDim.x))
					Child_need_back2parent = true;
				else
				{
					
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

			//copy back path and tasks
			int tcreadindex;
			int tpcreadindex;
			
			if(inblocktindex == 0)
			{
				for(i = 0; i < WARPNum; i++) //copy back path first.
				{
					for(j = 0; j < 32; j++)
					{
						tpcreadindex = C_Warp_Pathtail[j][i];
						cprecord = C_Warp_PathRecording[j][tpcreadindex];
						if(pathrecording[cprecord.selfid] != -1)
						{
							if(atomicExch(&G_pathrecMutex[cprecord.selfid], 1))
							{
								pathrecording[cprecord.selfid]=cprecord.presucc;
								DuplicateEli[cprecord.presucc] = true;
								atomicExch(&G_pathrecMutex[cprecord.selfid], 0);
							}
						}
						else
						{
							C_Warp_Pathtail[j][i]--; 
							continue;
						}
						C_Warp_Pathtail[j][i]--;
					}
				}
				
				tmpqcount = 0;
				tmp = blockIdx.x / GBcount;

				for(i = 0; i < WARPNum; i++)
				{
					for(j = 0; j < 32; j++)
					{
						tcreadindex = C_Init_S_WarpQueueHead[j][i];
						tmpnode = C_Init_S_WarpQueue[j][tcreadindex];
						if(DuplicateEli[tmpnode] == false)
						{
							G_Queue.G_queue[tmp][Child_Queue_index[blockIdx.x] + tmpqcount] = tmpnode;
							G_Queue.G_queue_size[tmp]++;
							tmpqcount++;
						}
						else
						{
							CBackBlockTasksize[blockIdx.x]--;
						}
					}
				}
			}
			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);
			
			if(globalthreadindex == 0)
			{
				int wholetasknum = 0;
				for(i = 0; i < blockDim.x; i++)
					wholetasknum += CBackBlockTasksize[i];
				
				P_G_sequence_index= new int * [wholetasknum];
				P_G_sequence_index[0] = G_Queue.G_queue[0];
				for(i = 0; i < blockDim.x; i++)
				{
					P_G_sequence_index[CBackBlockTasksize[i]-1] = &G_Queue.G_queue[i/GBcount][CBackBlockTasksize[i]];
					P_G_sequence_index[CBackBlockTasksize[i]] = &G_Queue.G_queue[(i+1)/GBcount][Child_Queue_index[i+1]];
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
	int * G_pathrecording, *H_pathrecording;
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
	H_pathrecording = new int[totalsize - sccsize];
	H_pathcordingMutex = new int[totalsize-sccsize];
	for(i = 0; i < totalsize - sccsize; i++)
	{
		H_path2scc[i] = -1;
		H_pathrecording[i] = -1;
		H_pathcordingMutex[i] = 0;
	}

	cudaMalloc((void**)&G_path2scc, sizeof(int)*(totalsize-sccsize));
	
	cudaMalloc((void**)&G_sccnodelist, sizeof(int)*sccsize);

	cudaMalloc((void**)&G_pathrecording, sizeof(int)*(totalsize - sccsize));

	cudaMalloc((void**)&G_pathrecordingMutex, sizeof(int)*(totalsize - sccsize));
	
	cudaMallocPitch((void**)&G_outgoing, &acturalsize, sizeof(int)*outgoingwidth, totalsize);    //outgoing from pat should be a n*m

	cudaMemcpy(G_path2scc,H_path2scc,sizeof(int)*(totalsize-sccsize),cudaMemcpyHostToDevice);
	
	cudaMemcpy(G_sccnodelist,sccnodelist,sizeof(int)*sccsize, cudaMemcpyHostToDevice);

	cudaMemcpy(G_pathrecording, H_pathrecording, sizeof(int)*(totalsize-sccsize), cudaMemcpyHostToDevice);
	
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
		outgoing[i] = new int[4];

	sccnlist[0] = 10;
	sccnlist[1] = 17;
	sccnlist[2] = 18;

	for(int i=0; i<15;i++)
	{
		outgoing[i][0]=2;  // the first position record the amout of succ.
		outgoing[i][1]=i*2+1;
		outgoing[i][2]=i*2+2;
		outgoing[i][3]=-1;
	}

	result = CudaPath(8,3,31,0,sccnlist,outgoing,3);
	return 1;
}
