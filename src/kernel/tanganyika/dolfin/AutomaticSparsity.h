// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __AUTOMATIC_SPARSITY_H
#define __AUTOMATIC_SPARSITY_H

#include <dolfin/GenericSparsity.h>

namespace dolfin {

  class ODE;
  class Vector;

  class AutomaticSparsity : public GenericSparsity {
  public:

    AutomaticSparsity(int N, ODE& ode);
    ~AutomaticSparsity();

    Type type() const;

    class Iterator : public GenericSparsity::Iterator {
    public:

      Iterator(int i, const AutomaticSparsity& sparsity);
      ~Iterator();

      Iterator& operator++();
      int operator*() const;
      bool end() const;

    private:

      const AutomaticSparsity& s;
      int pos;
      bool at_end;

    };

    Iterator* createIterator(int i) const;

    friend class Iterator;

  private:

    void computeSparsity(ODE& ode);
    bool checkdep(ODE& ode, Vector& u, real f0, int i, int j);

    ShortList<int>* list;
    real increment;

  };

}

#endif
