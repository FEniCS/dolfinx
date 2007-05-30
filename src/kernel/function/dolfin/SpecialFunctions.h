// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007.
//
// First added:  2006-02-09
// Last changed: 2007-05-29

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
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

  /// This function represents the average of the local mesh size on a given mesh.
  class AvgMeshSize : public Function
  {
  public:

    AvgMeshSize(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x) const
    {
      // If there is no facet (assembling on interior), return cell diameter
      if (facet() <= 0)
        return cell().diameter();
      else
      {
        // Create facet from the global facet number
        Facet facet0(mesh(), cell().entities(cell().mesh().topology().dim() - 1)[facet()]);

        // If there are two cells connected to the facet
        if (facet0.numEntities(cell().mesh().topology().dim()) == 2)
        {
          // Create the two connected cells and return the average of their diameter
          Cell cell0(mesh(), facet0.entities(cell().mesh().topology().dim())[0]);
          Cell cell1(mesh(), facet0.entities(cell().mesh().topology().dim())[1]);

          return (cell0.diameter() + cell1.diameter())/2.0;
        }
        // Else there is only one cell connected to the facet and the average is the cell diameter
        else
          return cell().diameter();
      }
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
