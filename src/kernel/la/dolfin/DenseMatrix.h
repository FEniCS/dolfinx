// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003

#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/GenericMatrix.h>

namespace dolfin {
  
  class SparseMatrix;

  class DenseMatrix : public GenericMatrix {
  public:

    DenseMatrix ();
    DenseMatrix (unsigned int m, unsigned int n);
    DenseMatrix (const DenseMatrix& A);
    DenseMatrix (const SparseMatrix& A);
    ~DenseMatrix ();
    
    void init(unsigned int m, unsigned int n);
    void clear();
    
    unsigned int size(unsigned int dim) const;
    unsigned int size() const;    
    unsigned int rowsize(unsigned int i) const;
    unsigned int bytes() const;
    
    real  operator()(unsigned int i, unsigned int j) const;
    real* operator[](unsigned int i);
    real  operator()(unsigned int i, unsigned int& j, unsigned int pos) const;

    void operator=  (real a);
    void operator=  (const DenseMatrix& A);
    void operator=  (const SparseMatrix& A);
    void operator+= (const DenseMatrix& A);
    void operator+= (const SparseMatrix& A);
    void operator-= (const DenseMatrix& A);
    void operator-= (const SparseMatrix& A);
    void operator*= (real a);
    
    real norm() const;

    real mult    (const Vector& x, unsigned int i) const;
    void mult    (const Vector& x, Vector& Ax) const;
    void multt   (const Vector& x, Vector& Ax) const;
    void mult    (const DenseMatrix& B, DenseMatrix& AB) const;
    real multrow (const Vector& x, unsigned int i) const;
    real multcol (const Vector& x, unsigned int j) const;
    
    void resize();
    void ident(unsigned int i);
    void lump(Vector& a) const;
    void addrow();
    void addrow(const Vector& x);
    void initrow(unsigned int i, unsigned int rowsize);
    bool endrow(unsigned int i, unsigned int pos) const;
    void settransp(const DenseMatrix& A);
    void settransp(const SparseMatrix& A);
    real rowmax(unsigned int i) const;
    real colmax(unsigned int i) const;
    real rowmin(unsigned int i) const;
    real colmin(unsigned int i) const;
    real rowsum(unsigned int i) const;
    real colsum(unsigned int i) const;
    real rownorm(unsigned int i, unsigned int type) const;
    real colnorm(unsigned int i, unsigned int type) const;

    void show() const;
    friend LogStream& operator<< (LogStream& stream, const DenseMatrix& A);

    friend class SparseMatrix;

  protected:

    void alloc(unsigned int m, unsigned int n);
    
    real read  (unsigned int i, unsigned int j) const;
    void write (unsigned int i, unsigned int j, real value);
    void add   (unsigned int i, unsigned int j, real value);
    void sub   (unsigned int i, unsigned int j, real value);
    void mult  (unsigned int i, unsigned int j, real value);
    void div   (unsigned int i, unsigned int j, real value);
   
    real** getvalues();
    real** const getvalues() const;

    void initperm();
    void clearperm();

    unsigned int* getperm();
    unsigned int* const getperm() const;

  private:
    
    real** values;

    unsigned int* permutation;
    
  };
  
}

#endif
