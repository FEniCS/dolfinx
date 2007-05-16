// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-05-09

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

    real eval(const real* x) const
    {
      return cell().diameter();
    }
    
  };

  /// This function represents the inverse of the local mesh size on a given mesh.
  class InvMeshSize : public Function
  {
  public:

    InvMeshSize(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x) const
    {
      return 1.0 / cell().diameter();
    }

  };

  /// This function represents the outward unit normal on mesh facets.
  /// Note that it is only nonzero on cell facets (not on cells).
  class FacetNormal : public Function
  {
  public:

    FacetNormal(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      if (facet() >= 0)
      {
        for (uint i = 0; i < cell().dim(); i++)
          values[i] = cell().normal(facet(), i);
      }
      else
      {
        for (uint i = 0; i < cell().dim(); i++)
          values[i] = 0.0;
      }
    }

  };

}

#endif
