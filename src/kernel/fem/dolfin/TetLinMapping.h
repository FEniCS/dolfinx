// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Mapping from reference element (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1)
// to a given tetrahedron in in R^3.

#ifndef __TETLIN_MAPPING_H
#define __TETLIN_MAPPING_H

#include <dolfin/Mapping.h>

namespace dolfin {

  class TetLinMapping : public Mapping {
  public:

	 TetLinMapping();

	 const FunctionSpace::ElementFunction dx(const FunctionSpace::ShapeFunction &v) const;
	 const FunctionSpace::ElementFunction dy(const FunctionSpace::ShapeFunction &v) const;
	 const FunctionSpace::ElementFunction dz(const FunctionSpace::ShapeFunction &v) const;
	 const FunctionSpace::ElementFunction dt(const FunctionSpace::ShapeFunction &v) const;

	 void update(const Cell &cell);

  };

}

#endif
