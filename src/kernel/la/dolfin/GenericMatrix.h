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

    virtual real  operator()(int i, int j) const = 0;    
    virtual real* operator[](int i) = 0;
    virtual real  operator()(int i, int& j, int pos) const = 0;

    virtual void operator=  (real a) = 0;
    virtual void operator=  (const DenseMatrix& A) = 0;
    virtual void operator=  (const SparseMatrix& A) = 0;
    virtual void operator+= (const DenseMatrix& A) = 0;
    virtual void operator+= (const SparseMatrix& A) = 0;
    virtual void operator-= (const DenseMatrix& A) = 0;
    virtual void operator-= (const SparseMatrix& A) = 0;
    virtual void operator*= (real a) = 0;
    
    virtual real norm() const = 0;

    virtual real mult    (const Vector& x, int i) const = 0;
    virtual void mult    (const Vector& x, Vector& Ax) const = 0;
    virtual void multt   (const Vector& x, Vector& Ax) const = 0;
    virtual real multrow (const Vector& x, int i) const = 0;
    virtual real multcol (const Vector& x, int j) const = 0;

    virtual void resize() = 0;
    virtual void ident(int i) = 0;
    virtual void addrow() = 0;
    virtual void addrow(const Vector& x) = 0;
    virtual void initrow(int i, int rowsize) = 0;
    virtual bool endrow(int i, int pos) const = 0;
    virtual void settransp(const DenseMatrix& A) = 0;
    virtual void settransp(const SparseMatrix& A) = 0;

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
    virtual real** const getvalues() const = 0;

    virtual void initperm() = 0;
    virtual void clearperm() = 0;

    virtual int* getperm() = 0;
    virtual int* const getperm() const = 0;

  };

}

#endif
