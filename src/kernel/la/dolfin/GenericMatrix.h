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
    GenericMatrix(unsigned int m, unsigned int n);
    virtual ~GenericMatrix();
    
    virtual void init(unsigned int m, unsigned int n);
    virtual void clear();
    
    virtual unsigned int size(unsigned int dim) const;
    virtual unsigned int size() const;
    virtual unsigned int rowsize(unsigned int i) const;
    virtual unsigned int bytes() const;

    virtual real  operator()(unsigned int i, unsigned int j) const;    
    virtual real* operator[](unsigned int i);
    virtual real  operator()(unsigned int i, unsigned int& j, unsigned int pos) const;

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

    virtual real mult    (const Vector& x, unsigned int i) const;
    virtual void mult    (const Vector& x, Vector& Ax) const;
    virtual void multt   (const Vector& x, Vector& Ax) const;
    virtual real multrow (const Vector& x, unsigned int i) const;
    virtual real multcol (const Vector& x, unsigned int j) const;

    virtual void resize();
    virtual void ident(unsigned int i);
    virtual void lump(Vector& a) const;
    virtual void addrow();
    virtual void addrow(const Vector& x);
    virtual void initrow(unsigned int i, unsigned int rowsize);
    virtual bool endrow(unsigned int i, unsigned int pos) const;
    virtual void settransp(const DenseMatrix& A);
    virtual void settransp(const SparseMatrix& A);
    virtual void settransp(const GenericMatrix& A);

    virtual void show() const;
    friend LogStream& operator<< (LogStream& stream, const GenericMatrix& A);

    friend class Matrix;
    friend class Matrix::Element;

  protected:
    
    virtual void alloc(unsigned int m, unsigned int n);

    virtual real read  (unsigned int i, unsigned int j) const;
    virtual void write (unsigned int i, unsigned int j, real value);
    virtual void add   (unsigned int i, unsigned int j, real value);
    virtual void sub   (unsigned int i, unsigned int j, real value);
    virtual void mult  (unsigned int i, unsigned int j, real value);
    virtual void div   (unsigned int i, unsigned int j, real value);

    virtual real** getvalues();
    virtual real** const getvalues() const;

    virtual void initperm();
    virtual void clearperm();

    virtual unsigned int* getperm();
    virtual unsigned int* const getperm() const;

    // Dimension
    unsigned int m;
    unsigned int n;
    
  };

}

#endif
