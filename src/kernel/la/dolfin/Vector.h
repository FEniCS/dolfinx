// Copyright (C) 2002 Johan Hoffman and Anders Logg.
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
    Vector(unsigned int size);
    Vector(const Vector& x);
    Vector(real x0);
    Vector(real x0, real x1);
    Vector(real x0, real x1, real x2);
    ~Vector ();
    
    void init(unsigned int size);
    void clear();
    unsigned int size() const;
    unsigned int bytes() const;

    real  operator()(unsigned int i) const;    
    real& operator()(unsigned int i);
    
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
    
    real operator*(const Vector& x) const;
    real operator*(const Matrix::Row& row) const;
    real operator*(const Matrix::Column& col) const;
    
    real norm () const;
    real norm (unsigned int i) const;
    
    void add(real a, Vector& x);
    void add(real a, const Matrix::Row& row);
    void add(real a, const Matrix::Column& col);

    void rand();
    
    // Output
    void show() const;
    friend LogStream& operator<< (LogStream& stream, const Vector& vector);
    
    // Friends
    friend class DirectSolver;
    friend class Matrix;
    friend class SISolver;
    
  private:
    
    void alloc(unsigned int size);

    unsigned int n;
    real *values;
    
  };

}

#endif
