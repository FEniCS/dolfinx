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
    
    BoundaryCondition() {
      _type = NEUMANN;
      _val  = 0.0;
      np = 0;
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
    
    real val() const {
      return _val;
    }
    
    void set(Type type, real val) {
      _type = type;
      _val = val;
    }
    
    friend class Galerkin;
    
  private:
    
    void update(Node* np) {
      this->np = np;
      _type = NEUMANN;
      _val = 0.0;
    }
    
    Node* np;
    
    Type _type;
    real _val;
    
  };

}

#endif
