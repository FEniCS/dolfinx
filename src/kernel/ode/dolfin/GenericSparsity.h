// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_SPARSITY_H
#define __GENERIC_SPARSITY_H

namespace dolfin {

  class GenericSparsity {
  public:

    enum Type { automatic, full, empty, matrix, table };

    GenericSparsity(unsigned int N);
    virtual ~GenericSparsity();

    virtual void setsize(unsigned int i, unsigned int size);
    virtual void set(unsigned int i, unsigned int j);

    virtual Type type() const = 0;

    class Iterator {
    public:

      Iterator(unsigned int i);
      virtual ~Iterator();

      virtual Iterator& operator++() = 0;
      virtual unsigned int operator*() const = 0;
      virtual bool end() const = 0;

    protected:

      unsigned int i;

    };

    virtual Iterator* createIterator(unsigned int i) const = 0;

  protected:

    unsigned int N;
    
  };
  
}

#endif
