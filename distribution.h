
float probability(global float * x, int dim){
  //Rosenbrock density
  float x1=*x;
  float x2=*(x+1);
  return exp(-(100*(x2-x1*x1)*(x2-x1*x1)+(1-x1)*(1-x1))/20);
}
