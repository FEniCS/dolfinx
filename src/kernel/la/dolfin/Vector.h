// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Modifications by Georgios Foufas (2002)

#ifndef __VECTOR_HH
#define __VECTOR_HH

#include <iostream>

#include <dolfin/dolfin_constants.h>

namespace dolfin {
  
  class Vector{
  public:
	 
	 Vector  ();
	 Vector  (int size);
	 ~Vector ();

	 void resize(int size);
	 int size();
	 
	 void operator=(Vector &vector);
	 void operator=(real scalar);
	 
	 real& operator()(int index);
	 
	 void operator+=(Vector &vector);
	 void operator+=(real scalar);	 
	 void operator*=(real scalar);
	 
	 real operator*(Vector &vector);
	 
	 real norm ();
	 real norm (int i);
	 void add(real scalar, Vector &vector);
	 
	 friend class DirectSolver;
	 friend class SparseMatrix;
	 friend class SISolver;
	 
	 friend ostream& operator << (ostream& output, Vector& vector);
	 
  private:
	 
	 int n;
	 real *values;
	 
  };

}
  
#endif
