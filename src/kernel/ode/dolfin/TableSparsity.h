// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TABLE_SPARSITY_H
#define __TABLE_SPARSITY_H

#include <dolfin/Array.h>
#include <dolfin/GenericSparsity.h>

namespace dolfin {

  class TableSparsity : public GenericSparsity {
  public:

    TableSparsity(unsigned int N);
    ~TableSparsity();

    void setsize(unsigned int i, unsigned int size);
    void set(unsigned int i, unsigned int j);

    Type type() const;

    class Iterator : public GenericSparsity::Iterator {
    public:

      Iterator(unsigned int i, const TableSparsity& sparsity);
      ~Iterator();

      Iterator& operator++();
      unsigned int operator*() const;
      bool end() const;

    private:

      const TableSparsity& s;
      unsigned int pos;
      bool at_end;

    };

    Iterator* createIterator(unsigned int i) const;

    friend class Iterator;

  private:

    Array<int>* list;

  };

}

#endif
