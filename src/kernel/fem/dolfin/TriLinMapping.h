// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Mapping from reference element (0,0) - (1,0) - (0,1)
// to a given triangle in R^2 (i.e. z = 0).

#ifndef __TRILIN_MAPPING_H
#define __TRILIN_MAPPING_H

#include <dolfin/Mapping.h>

namespace dolfin {

  class TriLinMapping : public Mapping {
  public:

	 TriLinMapping();

	 FunctionSpace::ElementFunction dx(const FunctionSpace::ShapeFunction &v) const;
	 FunctionSpace::ElementFunction dy(const FunctionSpace::ShapeFunction &v) const;
	 FunctionSpace::ElementFunction dz(const FunctionSpace::ShapeFunction &v) const;
	 FunctionSpace::ElementFunction dt(const FunctionSpace::ShapeFunction &v) const;
	 
	 void update(const Cell &cell);

  };

}

#endif
