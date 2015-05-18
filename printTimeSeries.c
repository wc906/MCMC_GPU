#include <stdlib.h>
#include <stdio.h>

void printTimeSeries(float *result, int length){
  float *x1=(float *)malloc(length/2*sizeof(float));
  float *x2=(float *)malloc(length/2*sizeof(float));
  for(int i=0;i<length/2;i++){
    x1[i]=result[2*i];
    x2[i]=result[2*i+1];
  }
  FILE *ptr = fopen("timeseries_x1","w");
  if(ptr){
    for (int i=0;i<length/2;i++){
      fprintf(ptr,"%f\n",x1[i]);
    }
  }
  fclose(ptr);

  ptr = fopen("timeseries_x2","w");
  if(ptr){
    for (int i=0;i<length/2;i++){
      fprintf(ptr,"%f\n",x2[i]);
    }
  }
  fclose(ptr);
  free(x1);
  free(x2);
}
