// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class MyMatrix : public Matrix {
public:

  MyMatrix(int m, int n) : Matrix(m,n, Matrix::generic)
  {
    dolfin::cout << "Creating a generic matrix" << dolfin::endl;
  }

  void mult(const Vector& x, Vector& Ax) const
  {
    for (int i = 0; i < size(0); i++) {
      if ( i == 0 )
	Ax(i) = 2.0*x(i) - x(i+1);
      else if ( i == (size(0) - 1) ) 
	Ax(i) = -x(i-1) + 2.0*x(i);
      else
	Ax(i) = -x(i-1) + 2.0*x(i) - x(i+1);
    }
  }

};

int main()
{
  dolfin_set("output", "plain text");
  
  // Number of unknowns
  int m = 5;

  // A generic matrix
  MyMatrix A(m,m);

  // Right-hand side
  Vector b(m);
  b = 1.0;

  // Solution vector
  Vector x;

  // Solve system
  A.solve(x,b);

  // Save solution to file
  File file("solution.m");
  file << x;

  return 0;
}
