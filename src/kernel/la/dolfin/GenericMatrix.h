// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <dolfin/Matrix.h>

namespace dolfin {

  class DenseMatrix;
  class SparseMatrix;

  class GenericMatrix {
  public:
    
    virtual void init(int m, int n) = 0;
    virtual void clear() = 0;
    
    virtual int size(int dim) const = 0;
    virtual int size() const = 0;
    virtual int rowsize(int i) const = 0;
    virtual int bytes() const = 0;
    
    virtual real operator()(int i, int& j, int pos) const = 0;
    virtual real operator()(int i, int j) const = 0;

    virtual void operator=  (real a) = 0;
    virtual void operator=  (const DenseMatrix& A) = 0;
    virtual void operator=  (const SparseMatrix& A) = 0;
    virtual void operator+= (const DenseMatrix& A) = 0;
    virtual void operator+= (const SparseMatrix& A) = 0;
    virtual void operator-= (const DenseMatrix& A) = 0;
    virtual void operator-= (const SparseMatrix& A) = 0;
    virtual void operator*= (real a) = 0;
    
    virtual real norm() const = 0;

    virtual real mult (Vector& x, int i) const = 0;
    virtual void mult (Vector& x, Vector& Ax) const = 0;

    virtual void resize() = 0;
    virtual void ident(int i) = 0;
    virtual void initrow(int i, int rowsize) = 0;
    virtual bool endrow(int i, int pos) const = 0;
    virtual int  perm(int i) const = 0;

    virtual void show() const = 0;

    friend class Matrix;
    friend class Matrix::Element;

  protected:
    
    virtual void alloc(int m, int n) = 0;

    virtual real read  (int i, int j) const = 0;
    virtual void write (int i, int j, real value) = 0;
    virtual void add   (int i, int j, real value) = 0;
    virtual void sub   (int i, int j, real value) = 0;
    virtual void mult  (int i, int j, real value) = 0;
    virtual void div   (int i, int j, real value) = 0;

    virtual real** getvalues() = 0;
    virtual int* getperm() = 0;

  };

}

#endif
