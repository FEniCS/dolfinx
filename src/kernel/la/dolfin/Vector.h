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
    Vector(const Vector& x);
    Vector(real x0);
    Vector(real x0, real x1);
    Vector(real x0, real x1, real x2);
    ~Vector ();
    
    void init(int size);
    void clear();
    int size() const;
    int bytes() const;
    
    real& operator()(int i);
    real  operator()(int i) const;
    
    void operator=(const Vector& x);
    void operator=(real scalar);
    
    void operator+=(const Vector& x);
    void operator+=(real a);	 
    void operator*=(real a);
    void operator-=(const Vector& x);
    
    real operator*(const Vector& x);
    
    real norm () const;
    real norm (int i) const;
    void add(real a, Vector& x);
    
    // Output
    void show() const;
    friend LogStream& operator<< (LogStream& stream, const Vector& vector);
    
    // Friends
    friend class DirectSolver;
    friend class Matrix;
    friend class SISolver;
    
  private:
    
    void alloc(int size);

    int n;
    real *values;
    
  };

}

#endif
