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
    
    GenericMatrix();
    GenericMatrix(int m, int n);
    virtual ~GenericMatrix();
    
    virtual void init(int m, int n);
    virtual void clear();
    
    virtual int size(int dim) const;
    virtual int size() const;
    virtual int rowsize(int i) const;
    virtual int bytes() const;

    virtual real  operator()(int i, int j) const;    
    virtual real* operator[](int i);
    virtual real  operator()(int i, int& j, int pos) const;

    virtual void operator=  (real a);
    virtual void operator=  (const DenseMatrix& A);
    virtual void operator=  (const SparseMatrix& A);
    virtual void operator=  (const GenericMatrix& A);
    virtual void operator+= (const DenseMatrix& A);
    virtual void operator+= (const SparseMatrix& A);
    virtual void operator+= (const GenericMatrix& A);
    virtual void operator-= (const DenseMatrix& A);
    virtual void operator-= (const SparseMatrix& A);
    virtual void operator-= (const GenericMatrix& A);
    virtual void operator*= (real a);
    
    virtual real norm() const;

    virtual real mult    (const Vector& x, int i) const;
    virtual void mult    (const Vector& x, Vector& Ax) const;
    virtual void multt   (const Vector& x, Vector& Ax) const;
    virtual real multrow (const Vector& x, int i) const;
    virtual real multcol (const Vector& x, int j) const;

    virtual void resize();
    virtual void ident(int i);
    virtual void addrow();
    virtual void addrow(const Vector& x);
    virtual void initrow(int i, int rowsize);
    virtual bool endrow(int i, int pos) const;
    virtual void settransp(const DenseMatrix& A);
    virtual void settransp(const SparseMatrix& A);
    virtual void settransp(const GenericMatrix& A);

    virtual void show() const;
    friend LogStream& operator<< (LogStream& stream, const GenericMatrix& A);

    friend class Matrix;
    friend class Matrix::Element;

  protected:
    
    virtual void alloc(int m, int n);

    virtual real read  (int i, int j) const;
    virtual void write (int i, int j, real value);
    virtual void add   (int i, int j, real value);
    virtual void sub   (int i, int j, real value);
    virtual void mult  (int i, int j, real value);
    virtual void div   (int i, int j, real value);

    virtual real** getvalues();
    virtual real** const getvalues() const;

    virtual void initperm();
    virtual void clearperm();

    virtual int* getperm();
    virtual int* const getperm() const;

    // Dimension
    int m;
    int n;
    
  };

}

#endif
