// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Contributions by: Georgios Foufas (2002)
//                   Johan Jansson (2003)

#ifndef __VECTOR_H
#define __VECTOR_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>

namespace dolfin {
  
  class Vector : public Variable {
  public:
	 
    Vector();
    Vector(int size);
    Vector(const Vector &vector);
    Vector(real x0);
    Vector(real x0, real x1);
    Vector(real x0, real x1, real x2);
    ~Vector ();
    
    void init(int size);
    int size() const;
    int bytes() const;
    
    real& operator()(int i);
    real  operator()(int i) const;
    
    void operator=(const Vector &vector);
    void operator=(real scalar);
    
    void operator+=(const Vector &vector);
    void operator+=(real scalar);	 
    void operator*=(real scalar);
    void operator-=(const Vector &vector);
    
    real operator*(const Vector &vector);
    
    real norm ();
    real norm (int i);
    void add(real scalar, Vector &vector);
    
    // Output
    void show() const;
    friend LogStream& operator<< (LogStream& stream, const Vector& vector);
    
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
