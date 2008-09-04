#include <dolfin.h>

using namespace dolfin; 

// simple program for checking that the operators etc. work properly

int main() 
{
  Matrix A(12,12); 

  BlockMatrix AA(2,2); 
  AA(0,0) = A; 
  AA(1,0) = A; 
  AA(0,1) = A; 

  Vector x(A.size(0)); 
  BlockVector xx(2); 
  xx(0) = x; 
  xx(1) = x; 

  Vector y(A.size(1)); 
  BlockVector yy(2); 
  yy(0) = y; 
  yy(1) = y; 

  AA.mult(xx,yy); 
  
}; 


