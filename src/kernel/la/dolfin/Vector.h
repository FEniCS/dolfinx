// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Modifications by Georgios Foufas (2002)

#ifndef __VECTOR_H
#define __VECTOR_H

#include <iostream>
#include <dolfin/Variable.h>
#include <dolfin/constants.h>

namespace dolfin {
  
  class Vector : public Variable {
  public:
	 
	 Vector  ();
	 Vector  (int size);
	 Vector  (Vector &vector);
	 ~Vector ();

	 void init(int size);
	 int size() const;
	 int bytes() const;

	 real& operator()(int i);
	 real  operator()(int i) const;
	 
	 void operator=(Vector &vector);
	 void operator=(real scalar);
	 
	 void operator-=(Vector &vector);

	 void operator+=(Vector &vector);
	 void operator+=(real scalar);	 
	 void operator*=(real scalar);
	 
	 real operator*(Vector &vector);
	 
	 real norm ();
	 real norm (int i);
	 void add(real scalar, Vector &vector);
	 
	 // Output
	 void show() const;
	 friend std::ostream& operator << (std::ostream& output, Vector& vector);
	 
	 // Friends
	 friend class DirectSolver;
	 friend class Matrix;
	 friend class SISolver;
	 
  private:
	 
	 int n;
	 real *values;
	 
  };

}
  
#endif
