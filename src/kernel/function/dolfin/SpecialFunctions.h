// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-04-30

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include <dolfin/Cell.h>
#include <dolfin/Function.h>

namespace dolfin
{

  /// This function represents the local mesh size on a given mesh.
  class MeshSize : public Function
  {
  public:

    MeshSize(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x)
    {
      return cell().diameter();
    }
    
  };

  /// This function represents the inverse of the local mesh size on a given mesh.
  class InvMeshSize : public Function
  {
  public:

    InvMeshSize(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x)
    {
      return 1.0 / cell().diameter();
    }

  };

  /// This function represents the outward unit normal on mesh facets.
  class FacetNormal : public Function
  {
  public:

    FacetNormal(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x)
    {
      // FIXME: Not implemented
      dolfin_error("Not implemented");
      //return cell().normal(facet(), i);
    }
  };

}

#endif
