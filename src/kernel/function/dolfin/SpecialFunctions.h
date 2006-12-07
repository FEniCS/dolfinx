// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2006-12-07

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

namespace dolfin
{

  /// This is the zero function.
  class Zero : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 0.0;
    }
  };

  /// This is the unity function.
  class Unity : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 1.0;
    }
  };

  /// This function represents the local mesh size on a given mesh.
  class MeshSize : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return cell().diameter();
    }
  };

  /// This function represents the outward unit normal on mesh facets.
  class FacetNormal : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return cell().normal(facet(), i);
    }
  };

}

#endif
