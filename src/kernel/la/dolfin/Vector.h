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
#include <dolfin/Matrix.h>

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

    real  operator()(int i) const;    
    real& operator()(int i);
    
    void operator=(const Vector& x);
    void operator=(real scalar);
    void operator=(const Matrix::Row& row);
    void operator=(const Matrix::Column& col);

    void operator+=(const Vector& x);
    void operator+=(const Matrix::Row& row);
    void operator+=(const Matrix::Column& col);
    void operator+=(real a);	 

    void operator-=(const Vector& x);
    void operator-=(const Matrix::Row& row);
    void operator-=(const Matrix::Column& col);
    void operator-=(real a);

    void operator*=(real a);
    
    real operator*(const Vector& x);
    real operator*(const Matrix::Row& row);
    real operator*(const Matrix::Column& col);
    
    real norm () const;
    real norm (int i) const;
    
    void add(real a, Vector& x);
    void add(real a, const Matrix::Row& row);
    void add(real a, const Matrix::Column& col);
    
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
