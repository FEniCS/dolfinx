// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Modifications by Georgios Foufas (2002)

#ifndef __VECTOR_H
#define __VECTOR_H

#include <iostream>

#include <dolfin/constants.h>

namespace dolfin {
  
  class Vector{
  public:
	 
	 Vector  ();
	 Vector  (int size);
	 ~Vector ();

	 void resize(int size);
	 int size();
	 int bytes();

	 real& operator()(int index);
	 
	 void operator=(Vector &vector);
	 void operator=(real scalar);
	 
	 void operator+=(Vector &vector);
	 void operator+=(real scalar);	 
	 void operator*=(real scalar);
	 
	 real operator*(Vector &vector);
	 
	 real norm ();
	 real norm (int i);
	 void add(real scalar, Vector &vector);
	 
	 // Output
	 void show();
	 friend ostream& operator << (ostream& output, Vector& vector);
	 
	 // Friends
	 friend class DirectSolver;
	 friend class SparseMatrix;
	 friend class SISolver;
	 
  private:
	 
	 int n;
	 real *values;
	 
  };

}
  
#endif
