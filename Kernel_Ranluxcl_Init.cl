
// initialization for random number generator 

#include "ranluxcl.cl"

kernel void Kernel_Ranluxcl_Init(private int ins, global float4 *ranluxcltab){
	ranluxcl_initialization(ins, ranluxcltab);
}