// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by: Georgios Foufas 2002, 2003
//              Erik Svensson, 2003

#ifndef __SPARSE_MATRIX_H
#define __SPARSE_MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/GenericMatrix.h>

namespace dolfin {

  class DenseMatrix;
  
  class SparseMatrix : public GenericMatrix {
  public:
  
    SparseMatrix ();
    SparseMatrix (unsigned int m, unsigned int n);
    SparseMatrix (const SparseMatrix& A);
    SparseMatrix (const DenseMatrix& A);
    ~SparseMatrix ();
    
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
    real multrow (const Vector& x, unsigned int i) const;
    real multcol (const Vector& x, unsigned int j) const;

    void resize();
    void ident(unsigned int i);
    void addrow();
    void addrow(const Vector& x);
    void initrow(unsigned int i, unsigned int rowsize);
    bool endrow(unsigned int i, unsigned int pos) const;
    void settransp(const DenseMatrix& A);
    void settransp(const SparseMatrix& A);

    void show() const;
    friend LogStream& operator<< (LogStream& stream, const SparseMatrix& A);

    friend class DenseMatrix;

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

    void resizeRow(unsigned int i,unsigned  int rowsize);
    
    // Data
    unsigned int* rowsizes;
    int** columns;
    real** values;
    
    // Additional size to allocate when needed
    unsigned int allocsize;
    
  };
  
}

#endif
