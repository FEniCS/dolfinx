// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-04-05

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include <dolfin/Function.h>

namespace dolfin
{

  /// This is the zero function.
  class Zero : public Function
  {
    void eval(real* values, const real* coordinates)
    {
      values[0] = 0.0;
    }
  };

  /// This is the unity function.
  class Unity : public Function
  {
    void eval(real* values, const real* coordinates)
    {
      values[0] = 1.0;
    }
  };

  /// This function represents the local mesh size on a given mesh.
  class MeshSize : public Function
  {
    void eval(real* values, const real* coordinates)
    {
      // FIXME: Not implemented
      dolfin_error("Not implemented");
      //return cell().diameter();
    }
  };

  /// This function represents the inverse of the local mesh size on a given mesh.
  class InvMeshSize : public Function
  {
    void eval(real* values, const real* coordinates)
    {
      // FIXME: Not implemented
      dolfin_error("Not implemented");
      //return 1.0/cell().diameter();
    }
  };

  /// This function represents the outward unit normal on mesh facets.
  class FacetNormal : public Function
  {
    void eval(real* values, const real* coordinates)
    {
      // FIXME: Not implemented
      dolfin_error("Not implemented");
      //return cell().normal(facet(), i);
    }
  };

}

#endif
