#include <stdlib.h>
#include <stdio.h>
void printOutArray(FILE *ptr, float *result,int length, char *name){
  fprintf(ptr,"   %s = np.array([\\\n",name);
  fprintf(ptr,"   [");
  for (int i=0;i<length-1;i++){
    fprintf(ptr,"%f , ",result[i]);
  }
  fprintf(ptr,"%f ]])\n",result[length-1]);
  fprintf(ptr,"   data[ '%s' ] = %s\n",name,name);
}

void plotInfoPrintOut(float *result, int length, char* filename){
  float *x1=(float *)malloc(length/2*sizeof(float));
  float *x2=(float *)malloc(length/2*sizeof(float));
  for(int i=0;i<length/2;i++){
    x1[i]=result[2*i];
    x2[i]=result[2*i+1];
  }
  FILE *ptr = fopen(filename,"w");
  if(ptr){
    fprintf(ptr,"# Output file plotInfo.py. See MCMC.c for more info.\n\n");
    fprintf(ptr,"import numpy as np\n");
    fprintf(ptr,"def RunData():\n");
    fprintf(ptr,"   num = %d\n",length);
    fprintf(ptr,"   data = { 'num' : num }\n");
    printOutArray(ptr,x1,length/2,"x");
    printOutArray(ptr,x2,length/2,"y");
    fprintf(ptr,"   return data\n");
  }
  free(x1);
  free(x2);
  fclose(ptr);
}
