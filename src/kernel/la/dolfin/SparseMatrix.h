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
    SparseMatrix (int m, int n);
    SparseMatrix (const SparseMatrix& A);
    SparseMatrix (const DenseMatrix& A);
    ~SparseMatrix ();
    
    void init(int m, int n);
    void clear();
    
    int size(int dim) const;
    int size() const;
    int rowsize(int i) const;
    int bytes() const;
    
    real  operator()(int i, int j) const;
    real* operator[](int i);
    real  operator()(int i, int& j, int pos) const;
    
    void operator=  (real a);
    void operator=  (const DenseMatrix& A);
    void operator=  (const SparseMatrix& A);
    void operator+= (const DenseMatrix& A);
    void operator+= (const SparseMatrix& A);
    void operator-= (const DenseMatrix& A);
    void operator-= (const SparseMatrix& A);
    void operator*= (real a);
    
    real norm() const;

    real mult    (const Vector& x, int i) const;
    void mult    (const Vector& x, Vector& Ax) const;
    void multt   (const Vector& x, Vector& Ax) const;
    real multrow (const Vector& x, int i) const;
    real multcol (const Vector& x, int j) const;

    void resize();
    void ident(int i);
    void addrow();
    void addrow(const Vector& x);
    void initrow(int i, int rowsize);
    bool endrow(int i, int pos) const;
    void settransp(const DenseMatrix& A);
    void settransp(const SparseMatrix& A);

    void show() const;
    friend LogStream& operator<< (LogStream& stream, const SparseMatrix& A);

    friend class DenseMatrix;

  protected:
    
    void alloc(int m, int n);

    real read  (int i, int j) const;
    void write (int i, int j, real value);
    void add   (int i, int j, real value);
    void sub   (int i, int j, real value);
    void mult  (int i, int j, real value);
    void div   (int i, int j, real value);

    real** getvalues();
    real** const getvalues() const;

    void initperm();
    void clearperm();

    int* getperm();
    int* const getperm() const;
    
  private:

    void resizeRow(int i, int rowsize);
    
    // Dimension
    int m, n;
    
    // Data
    int*   rowsizes;
    int**  columns;
    real** values;
    
    // Additional size to allocate when needed
    int allocsize;
    
  };
  
}

#endif
