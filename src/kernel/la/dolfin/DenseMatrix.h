#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H

#include <dolfin/constants.h>

namespace dolfin {
  
  class DirectSolver;
  class SparseMatrix;
  
  class DenseMatrix{
  public:
	 
	 DenseMatrix(int m, int n);
	 DenseMatrix(SparseMatrix &A);
	 ~DenseMatrix();
	 
	 void init(int m, int n);

	 real& operator() (int i, int j);
	 real  operator() (int i, int j) const;
	 
	 void set (int i, int j, real value);
	 int  size
	 (int dim);
	 real get (int i, int j);
	 
	 void DisplayAll();
	 void DisplayRow(int i);
	 
	 friend class DirectSolver;
	 
  private:
	 
	 int m,n;
	 real **values;
	 int *permutation;
	 
  };

}
  
#endif
