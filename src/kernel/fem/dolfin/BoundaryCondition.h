// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/constants.h>

namespace dolfin {

  class BoundaryCondition {
  public:
    
    enum Type { DIRICHLET, NEUMANN };
    
    BoundaryCondition(int components = 1) {
      this->components = components;
      _type = NEUMANN;
      np = 0;
      _val = new real[components];

      for(int i = 0; i < components; i++)
      {
	_val[i]  = 0.0;
      }
    }

    ~BoundaryCondition() {
      delete [] _val;
    }
    
    Point coord() const {
      if ( np == 0 )
	dolfin_error("Node is not specified.");
      return np->coord();
    }
    
    const Node& node() const {
      if ( np == 0 )
	dolfin_error("Node is not specified.");
      return *np;
    }

    Type type() const {
      return _type;
    }
    
    real val(int component = 0) const {
      return _val[component];
    }
    
    void set(Type type, real val, int component = 0) {
      _type = type;
      _val[component] = val;
    }

    friend class Galerkin;
    
  private:
    
    void update(Node* np) {
      this->np = np;
      _type = NEUMANN;

      for(int i = 0; i < components; i++)
      {
	_val[i]  = 0.0;
      }
    }
    
    Node* np;
    
    Type _type;
    real* _val;
    int components;
  };

}

#endif
