// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MATRIX_SPARSITY_H
#define __MATRIX_SPARSITY_H

#include <dolfin/GenericSparsity.h>

namespace dolfin {

  class Matrix;

  class MatrixSparsity : public GenericSparsity {
  public:

    MatrixSparsity(int N, const Matrix& A_);
    ~MatrixSparsity();

    Type type() const;

    class Iterator : public GenericSparsity::Iterator {
    public:

      Iterator(int i, const MatrixSparsity& sparsity);
      ~Iterator();

      Iterator& operator++();
      int operator*() const;
      bool end() const;

    private:

      const MatrixSparsity& s;
      int pos;

    };

    Iterator* createIterator(int i) const;

    friend class Iterator;

  private:

    const Matrix& A;

  };

}

#endif
