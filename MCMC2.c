/* Convolution example; originally written by Lucas Wilcox.
 * Minor modifications by Georg Stadler.
 * The function expects a bitmap image (*.ppm) as input, as
 * well as a number of blurring loops to be performed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timing.h"
#include "cl-helper.h"
//#include "plotInfoPrintOut.c"

int K=1024;

void dataAnalysis(float *, int, int, float);
void plotInfoPrintOut(float *,int,char *);
void printTimeSeries(float *,int);
void print_kernel_info(cl_command_queue queue, cl_kernel knl)
{
  // get device associated with the queue
  cl_device_id dev;
  CALL_CL_SAFE(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(dev), &dev, NULL));

  char kernel_name[4096];
  CALL_CL_SAFE(clGetKernelInfo(knl, CL_KERNEL_FUNCTION_NAME,
        sizeof(kernel_name), &kernel_name, NULL));
  kernel_name[4095] = '\0';
  printf("Info for kernel %s:\n", kernel_name);

  size_t kernel_work_group_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(kernel_work_group_size), &kernel_work_group_size, NULL));
  printf("  CL_KERNEL_WORK_GROUP_SIZE=%zd\n", kernel_work_group_size);

  size_t preferred_work_group_size_multiple;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(preferred_work_group_size_multiple),
        &preferred_work_group_size_multiple, NULL));
  printf("  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=%zd\n",
      preferred_work_group_size_multiple);

  cl_ulong kernel_local_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(kernel_local_mem_size), &kernel_local_mem_size, NULL));
  printf("  CL_KERNEL_LOCAL_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_local_mem_size);

  cl_ulong kernel_private_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
        sizeof(kernel_private_mem_size), &kernel_private_mem_size, NULL));
  printf("  CL_KERNEL_PRIVATE_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_private_mem_size);
}


//float parallelMCMC(int num,int walkersN){
int main(int argc, char *argv[]){
  
  const int dim=2;//data dimension
  const int num=1024*1024*256;//number of samples you want
  const int walkersN=2*512*1/2;//number of walkers, red + black
  int MaxIter=num/walkersN;
  size_t global_size[]={walkersN/2};

  int i;

  // --------------------------------------------------------------------------
  // allocate CPU buffers
  // --------------------------------------------------------------------------
  float *initRed=(float *)malloc(walkersN/2*dim*sizeof(float));
  float *initBlk=(float *)malloc(walkersN/2*dim*sizeof(float));
  float *finalSamples=(float *)malloc(num*dim*sizeof(float));
  //initialize CPU buffers
  srand48(0);
  for(i=0;i<walkersN/2*dim;i++){
    initRed[i]=(float)drand48();
    initBlk[i]=(float)drand48();
  }

  // --------------------------------------------------------------------------
  // get an OpenCL context and queue
  // --------------------------------------------------------------------------
  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);
  print_device_info_from_queue(queue);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("stretch2.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "stretch", NULL);
  free(knl_text);

  char *knl_text_rand = read_file("Kernel_Ranluxcl_Init.cl");
  cl_kernel init_rand_knl = kernel_from_string(ctx, knl_text_rand, "Kernel_Ranluxcl_Init", NULL);
  free(knl_text_rand);
  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_int status;
  size_t deviceDataSize = walkersN/2*dim*sizeof(float);
  cl_mem redwalkersD = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				     deviceDataSize, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  
  cl_mem blkwalkersD = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				     deviceDataSize, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  size_t rand_lux_state_buffer_size = walkersN/2 * 7 * sizeof(cl_float4);
  
  cl_mem ranluxcltabD = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				       rand_lux_state_buffer_size, 0, &status);

  size_t proposal_size = walkersN/2*dim*sizeof(float);
  cl_mem proposalD = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				    proposal_size, 0, &status);
  

  // --------------------------------------------------------------------------
  // transfer to device
  // --------------------------------------------------------------------------
  size_t initialSize=walkersN/2*dim*sizeof(float);
  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, redwalkersD, /*blocking*/ CL_TRUE, /*offset*/ 0,
        initialSize, initRed, 0, NULL, NULL));

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, blkwalkersD, /*blocking*/ CL_TRUE, /*offset*/ 0,
        initialSize, initBlk, 0, NULL, NULL));

  // --------------------------------------------------------------------------
  // run code on device
  // --------------------------------------------------------------------------
  cl_int ins=1;
  
  CALL_CL_SAFE(clSetKernelArg(init_rand_knl, 0, sizeof(ins), &ins));
  CALL_CL_SAFE(clSetKernelArg(init_rand_knl, 1, sizeof(ranluxcltabD), &ranluxcltabD));
 
  CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, init_rand_knl, 1, NULL,
          global_size, NULL, 0, NULL, NULL));

  CALL_CL_SAFE(clFinish(queue));

  cl_int walkersND = walkersN;
  cl_int dimD = dim;
 
  
  CALL_CL_SAFE(clSetKernelArg(knl, 2, sizeof(walkersND), &walkersND));
  CALL_CL_SAFE(clSetKernelArg(knl, 3, sizeof(ranluxcltabD), &ranluxcltabD));
  CALL_CL_SAFE(clSetKernelArg(knl, 4, sizeof(dimD), &dimD));
  CALL_CL_SAFE(clSetKernelArg(knl, 5, sizeof(proposalD), &proposalD));
  // --------------------------------------------------------------------------
  // print kernel info
  // --------------------------------------------------------------------------
  print_kernel_info(queue, knl);


  int tag=0;
  float comm_time=0;
  timestamp_type tic_out, toc_out, tic_in, toc_in;
  get_timestamp(&tic_out);
  
  for(int iter = 1; iter < MaxIter; iter++)
  {
    //copy to final chain
    get_timestamp(&tic_in);

    CALL_CL_SAFE(clEnqueueReadBuffer(
        queue, redwalkersD, /*blocking*/ CL_TRUE, /*offset*/ 0,
        deviceDataSize, finalSamples+tag,
        0, NULL, NULL));
    tag+=walkersN/2*dim;
    CALL_CL_SAFE(clEnqueueReadBuffer(
        queue, blkwalkersD, /*blocking*/ CL_TRUE, /*offset*/ 0,
        deviceDataSize, finalSamples+tag,
        0, NULL, NULL));
    tag+=walkersN/2*dim;

    get_timestamp(&toc_in);
    comm_time+=(float)timestamp_diff_in_seconds(tic_in,toc_in);
    //update redwalkers
    CALL_CL_SAFE(clSetKernelArg(knl, 0, sizeof(redwalkersD), &redwalkersD));
    CALL_CL_SAFE(clSetKernelArg(knl, 1, sizeof(blkwalkersD), &blkwalkersD));
    

    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 1, NULL,
          global_size, NULL, 0, NULL, NULL));
    CALL_CL_SAFE(clFinish(queue));

    //update blkwalkers
    CALL_CL_SAFE(clSetKernelArg(knl, 0, sizeof(blkwalkersD), &blkwalkersD));
    CALL_CL_SAFE(clSetKernelArg(knl, 1, sizeof(redwalkersD), &redwalkersD));
    

    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 1, NULL,
          global_size, NULL, 0, NULL, NULL));
    CALL_CL_SAFE(clFinish(queue));
    
    //printf("Iter:%d\n",iter);
    
  }
  get_timestamp(&tic_in);

  CALL_CL_SAFE(clEnqueueReadBuffer(
        queue, redwalkersD, /*blocking*/ CL_TRUE, /*offset*/ 0,
        deviceDataSize, finalSamples+tag,
        0, NULL, NULL));
  tag+=walkersN/2*dim;
  CALL_CL_SAFE(clEnqueueReadBuffer(
        queue, blkwalkersD, /*blocking*/ CL_TRUE, /*offset*/ 0,
        deviceDataSize, finalSamples+tag,
        0, NULL, NULL));

  get_timestamp(&toc_in);
  comm_time+=(float)timestamp_diff_in_seconds(tic_in,toc_in);
  //  CALL_CL_SAFE(clFinish(queue));
  get_timestamp(&toc_out);

  float elapsed =(float)timestamp_diff_in_seconds(tic_out,toc_out);
  printf("Time elapsed in sampling:%f s\n", elapsed-comm_time);
  printf("Sample speed:%f Million Pts/s\n",num/(elapsed-comm_time)/(1e6));
  printf("Time elapsed in communication: %f s\n", comm_time);
  printf("Communication speed: %f GB/s\n",num*dim*sizeof(float)/comm_time/(1e9));

  // --------------------------------------------------------------------------
  // transfer back & check
  // --------------------------------------------------------------------------


  //calculate autocorrelation time
  //dataAnalysis(finalSamples,num*dim,dim,elapsed);
  //plot last 10000 pts
  //plotInfoPrintOut(finalSamples+num*dim-10000,10000,"plotInfo_ensemble.py");
  
  //printTimeSeries(finalSamples,num*dim);
  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clReleaseMemObject(redwalkersD));
  CALL_CL_SAFE(clReleaseMemObject(blkwalkersD));
  CALL_CL_SAFE(clReleaseMemObject(ranluxcltabD));
  CALL_CL_SAFE(clReleaseMemObject(proposalD));
  CALL_CL_SAFE(clReleaseKernel(knl));
  CALL_CL_SAFE(clReleaseKernel(init_rand_knl));
  CALL_CL_SAFE(clReleaseCommandQueue(queue));
  CALL_CL_SAFE(clReleaseContext(ctx));
  free(initRed);
  free(initBlk);
  free(finalSamples);
  //return elapsed;
}
/*
int main(int argc, char *argv[]){
  int K=1;
  int nodes[K];
  int nums[K];
  float times[K];
  for(int i=0;i<K;i++){
    nodes[i]=pow(2,i);
    nums[i]=nodes[i]*10000;
    times[i]=parallelMCMC(nums[i],nodes[i]);
    printf("Time elapsed:%f s\n",times[i]);
  }
  
}
*/
