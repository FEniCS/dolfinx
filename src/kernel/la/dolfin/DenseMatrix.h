#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H

#include <dolfin/constants.h>

namespace dolfin {
  
  class DirectSolver;
  class Matrix;
  
  class DenseMatrix{
  public:
	 
	 DenseMatrix(int m, int n);
	 DenseMatrix(Matrix& A);
	 ~DenseMatrix();
	 
	 void init(int m, int n);
	 
	 real& operator() (int i, int j);
	 real  operator() (int i, int j) const;

	 int size(int dim) const;
	 
	 friend class DirectSolver;
	 
  private:
	 
	 int m,n;
	 real **values;
	 int *permutation;
	 
  };

}
  
#endif
