// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EMPTY_SPARSITY_H
#define __EMPTY_SPARSITY_H

#include <dolfin/GenericSparsity.h>

namespace dolfin {

  class EmptySparsity : public GenericSparsity {
  public:

    EmptySparsity(int N);
    ~EmptySparsity();

    Type type() const;

    class Iterator : public GenericSparsity::Iterator {
    public:

      Iterator(int i, const EmptySparsity& sparsity);
      ~Iterator();

      Iterator& operator++();
      int operator*() const;
      bool end() const;

    private:

      const EmptySparsity& s;

    };

    Iterator* createIterator(int i) const;

    friend class Iterator;

  };

}

#endif
