#include <stdlib.h>
#include <stdio.h>


void dataAnalysis(float *x, int length, int dim,float time){
  //autocorrelation time for f(x1,x2)=x1, and f(x1,x2)=x2.
  // dim>=2

  float covariance(float *,int,int);
  int num=length/dim;
  float *x1=(float *)malloc(num*sizeof(float));
  float *x2=(float *)malloc(num*sizeof(float));
  int i;
  int inf=1000;
  float mean1=0,mean2=0;
  for(i=0;i<num;i++){
    x1[i]=x[dim*i];
    x2[i]=x[dim*i+1];
    mean1+=x1[i];
    mean2+=x2[i];
  }
  mean1=mean1/num;
  mean2=mean2/num;
  for(i=0;i<num;i++){
    x1[i]=x1[i]-mean1;
    x2[i]=x2[i]-mean2;
  }
  float cov1[inf];
  float cov2[inf];
  for(i=0;i<inf;i++){
    cov1[i]=covariance(x1,i,num);
    cov2[i]=covariance(x2,i,num);
  }
  for(i=0;i<inf;i++){
    cov1[i]=cov1[i]/cov1[0];
    cov2[i]=cov2[i]/cov2[0];
  }
  float t1=1,t2=1;
  for(i=1;i<inf;i++){
    t1+=2*cov1[i];
    t2+=2*cov2[i];
  }
  printf("Time:%f s\n",time);
  printf("autocorrelation time for x1: %f\n",t1);
  printf("autocorrelation time for x2: %f\n",t2);
  //plotInfo


  free(x1);
  free(x2);  
}
float covariance(float *x,int lag,int num){
  //mean 0
  int i;
  float sum=0;
  for(i=0;i<num-lag;i++){
    sum+=x[i]*x[i+lag];
  }
  sum=sum/(num-lag);
  return sum;
}
