// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_SPARSITY_H
#define __GENERIC_SPARSITY_H

namespace dolfin {

  class GenericSparsity {
  public:

    enum Type { empty, full, automatic, table, matrix };

    GenericSparsity(int N);
    virtual ~GenericSparsity();

    virtual void setsize(int i, int size);
    virtual void set(int i, int j);

    virtual Type type() const = 0;

    class Iterator {
    public:

      Iterator(int i);
      virtual ~Iterator();

      virtual Iterator& operator++() = 0;
      virtual int operator*() const = 0;
      virtual bool end() const = 0;

    protected:

      int i;

    };

    virtual Iterator* createIterator(int i) const = 0;

  protected:

    int N;
    
  };
  
}

#endif
