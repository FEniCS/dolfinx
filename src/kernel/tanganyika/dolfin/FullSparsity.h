// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FULL_SPARSITY_H
#define __FULL_SPARSITY_H

#include <dolfin/GenericSparsity.h>

namespace dolfin {

  class FullSparsity : public GenericSparsity {
  public:

    FullSparsity(int N);
    ~FullSparsity();

    Type type() const;

    class Iterator : public GenericSparsity::Iterator {
    public:

      Iterator(int i, const FullSparsity& sparsity);
      ~Iterator();

      Iterator& operator++();
      int operator*() const;
      bool end() const;

    private:

      const FullSparsity& s;
      int pos;
      bool at_end;

    };

    Iterator* createIterator(int i) const;

    friend class Iterator;

  };

}

#endif
