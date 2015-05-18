//MCMC
//Weikun Chen

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timing.h"

void dataAnalysis(float*,int,int,float);
double genrand();
void plotInfoPrintOut(float *,int,char *);
typedef float (*myfunc)(float *,int);
void printTimeSeries(float *,int);

float probability(float * x, int dim){
  //Rosenbrock density
  float x1=*x;
  float x2=*(x+1);
  return exp(-(100*(x2-x1*x1)*(x2-x1*x1)+(1-x1)*(1-x1))/20);
}

float f(float *x, int d){
  int i;
  for (i=0;i<d;i++){
    if (*(x+i)<0 || *(x+i)>1){
      return 0.;
    }
  }
  return 1;
}

float f1(float *x,int d){
  float temp=*x;
  if ( temp < 0. ){
    return 0.;  
  }
  if ( temp > 1. ){
    return 0.;
  }
  return 6*temp*(1.-temp);
}

float fDoubleWell(float *x,int d){
  float temp=*x;
  float beta=6.;
  return exp(-beta*(temp*temp-1)*(temp*temp-1));
}
float f2D(float *x,int d){
  float epsilon=0.5;//epsilon needs to be less than 1
  float x1=x[0];
  float x2=x[1];
  if (x1-3.*x2>epsilon || x1-3*x2<-epsilon){
    return 0;
  }
  if ( x1>1 || x1<-1){
    return 0;
  }
  return 3/(4*epsilon);
}

//sample from a standard normal using box-muller
float randn(){
  const float two_pi=2.0*3.14159265358;
  //  static bool generate=0;
  //generate=!generate;
  //static float z0,z1;
  //if(!generate){
  // return z1;
  //}
  float z0;
  float u1,u2;
  u1=(float)genrand();
  u2=(float)genrand();
  z0=sqrt(-2.0*log(u1))*cos(two_pi*u2);
  //z1=sqrt(-2.0*log(u1))*sin(two_pi*u2);
  return z0;
}


//propose a new x by sampling from a isotropic multivariate Gaussian
//with mean x(t) and covariance matrix r^2*I
int propose(float *mean,float r,int d,float *proposal){
  int i=0;
  float temp;
  for(i=0;i<d;i++){
    temp=randn();
    *(proposal+i)=*(mean+i)+r*temp;
  }
  return 0;
}



void mcmc (myfunc funcPtr, int d, int num, float r, float *x){
 
  float q;//acceptance rate in each step
  int i,j;
 
  float *proposal=(float *)malloc(d*sizeof(float));
  //initialize first point
  for(i=0;i<d;i++){
    *(x+i)=(float)genrand();
  }
  //start MCMC
  for(i=1;i<num;i++){
    propose(x+(i-1)*d,r,d,proposal);
    q = (*funcPtr)(proposal,d)/(*funcPtr)(x+(i-1)*d,d);
    float r=(float)genrand();
    if(r<=q){
      //accept proposal
      for(j=0;j<d;j++){
	*(x+i*d+j)=*(proposal+j);
      }
    }
    else{
      for(j=0;j<d;j++){
	*(x+i*d+j)=*(x+(i-1)*d+j);
      }
    }
  }

  free(proposal);
}
int main(){

  timestamp_type time1,time2;
  myfunc funcPtr = &probability;
  int d=2;
  int num=1024*10000;
  float r=0.5;//sigma for proposal isotropic gaussian 
  float *x=(float *)malloc(num*d*sizeof(float));//sample points
  get_timestamp(&time1);
  mcmc(funcPtr,d,num,r,x);
  get_timestamp(&time2);
  float elapsed =(float)timestamp_diff_in_seconds(time1,time2);
  printf("Time elapsed:%f s\n", elapsed);
  printf("Sample speed:%f Million Pts/s\n",num/elapsed/(1e6));


  dataAnalysis(x,num*d,d,elapsed);
  plotInfoPrintOut(x+num*d-10000,10000,"plotInfo_metropolis.py");
  printTimeSeries(x,num*d);
}
