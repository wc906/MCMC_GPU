#include "ranluxcl.cl"
#include "distribution.h"
kernel void stretch(
       global float* redwalkers,	//walkers to be updated, add to end
       global float* blkwalkers,	//walkers fixed
       int walkerN,		//all walkers numbers(K), red + black, need to be even
       global float4 *ranluxcltab,	//random number generator state 
       int dim,			//dimension of data(N)
       global float* Y			//space to put proposal
       )
{      

	//set random number generator
	ranluxcl_state_t ranluxclstate;
	ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
	float4 randnum;
	randnum = ranluxcl(&ranluxclstate);
	
	//get workitem id , read old walker data
	int k = get_global_id(0);
//	float *xk = (float *)redwalkers+redOffset+k*dim;
//	float *y = (float *)Y+k*dim;
	
	//randomly choose from blkwalkers
	int j = (int)(randnum.s0 * (walkerN / 2));
//	float *xj = (float *)blkwalkers+blkOffset+j*dim;

	//randomly choose z from g(z) (with parameter a=2)
	//g(z)=1/sqrt(2*z) for 1/2 < z < 2
	//cdf for g(z), p(x)=sqrt(2*x)-1 for 1/2 < x < 2
	//inverse for p(x), invp(x)=(x+1)^2/2 for 0 < x < 1
	float z = (randnum.s1+1)*(randnum.s1+1)/2;

	//get proposal
	int i=0;
	for(i=0;i<dim;i++){
		*(Y+k*dim+i)=*(blkwalkers+j*dim+i)+
				z*(*(redwalkers+k*dim+i)-
				*(blkwalkers+j*dim+i));
	}
		
	//get acceptance rate
	float q= pow(z,dim-1)*probability(Y+k*dim,dim)/
				probability(redwalkers+k*dim,dim);
	
	//randomly accept proposal
//	float *xk2 = (float*)redwalkers+redOffset+(k+walkerN/2)*dim;
	if (randnum.s2<=q){
		//write y to redwalkers end
	   	for(i=0;i<dim;i++){
//			xk2[i]=y[i];
			*(redwalkers+k*dim+i)
			=*(Y+k*dim+i);
			
	   	}
	}


//test
//	*(redwalkers+redOffset+k*dim+walkerN/2*dim)=probability(Y+k*dim,dim);
//	*(redwalkers+redOffset+k*dim+1+walkerN/2*dim)=pow(z,dim-1);
//test

	//upload random number state
	ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

  	return;
}

