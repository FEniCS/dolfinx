// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-07-01

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/fem/UFC.h>
#include "Function.h"

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
    
    /// Compute minimal cell diameter
    real min() const
    {
      CellIterator c(mesh());
      real hmin = c->diameter();
      for (; !c.end(); ++c)
        hmin = std::min(hmin, c->diameter());
      return hmin;
    }

    /// Compute maximal cell diameter
    real max() const
    {
      CellIterator c(mesh());
      real hmax = c->diameter();
      for (; !c.end(); ++c)
        hmax = std::max(hmax, c->diameter());
      return hmax;
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
      if (facet() < 0)
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

    uint rank() const
    {
      return 1;
    }
    
    uint dim(uint i) const
    {
      if(i > 0)
        error("Invalid dimension %d in FacetNormal::dim.", i);
      return mesh().geometry().dim();
    }

  };

  /// This function represents the area/length of a mesh facet.
  class FacetArea : public Function
  {
  public:

    FacetArea(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      if (facet() >= 0)
        values[0] = cell().facetArea(facet());
      else
        values[0] = 0.0;
    }

  };

  /// This function represents the inverse area/length of a mesh facet.
  class InvFacetArea : public Function
  {
  public:

    InvFacetArea(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      if (facet() >= 0)
        values[0] = 1.0 / cell().facetArea(facet());
      else
        values[0] = 0.0;
    }

  };

  /// This function determines if the current facet is an outflow facet with
  /// respect to the current cell. It accepts as argument the mesh and a form
  /// M = dot(n, v)*ds, a functional, defined on the normal vector to the
  /// facet and velocity vector integrated over the exterior of the cell.
  /// The function returns 1.0 if the dot product > 0, 0.0 otherwise.
  class OutflowFacet : public Function
  {
  public:

    OutflowFacet(Mesh& mesh, Form& form) : Function(mesh), form(form)
    {
      // Some simple sanity checks on form
      if (!(form.form().rank() == 0 && form.form().num_coefficients() == 2))
        error("Invalid form: rank = %d, number of coefficients = %d. Must be rank 0 form with 2 coefficients.", 
                  form.form().rank(), form.form().num_coefficients());
      if (!(form.form().num_cell_integrals() == 0 && form.form().num_exterior_facet_integrals() == 1 
            && form.form().num_interior_facet_integrals() == 0))
        error("Invalid form: Must have exactly 1 exterior facet integral");

      form.updateDofMaps(mesh);
      ufc = new UFC(form.form(), mesh, form.dofMaps());
    }

    ~OutflowFacet()
    {
      delete ufc;
    }

    real eval(const real* x) const
    {
      // If there is no facet (assembling on interior), return 0.0
      if (facet() < 0)
        return 0.0;
      else
      {
        // Copy cell, cannot call interpolate with const cell()
        Cell cell0(cell());
        ufc->update(cell0);

        // Interpolate coefficients on cell and current facet
        for (dolfin::uint i = 0; i < form.coefficients().size(); i++)
          form.coefficients()[i]->interpolate(ufc->w[i], ufc->cell, *ufc->coefficient_elements[i], cell0, facet());

        // Get exterior facet integral (we need to be able to tabulate ALL facets of a given cell)
        ufc::exterior_facet_integral* integral = ufc->exterior_facet_integrals[0];

        // Call tabulate_tensor on exterior facet integral, dot(velocity, facet_normal)
        integral->tabulate_tensor(ufc->A, ufc->w, ufc->cell, facet());
      }

      // If dot product is positive, the current facet is an outflow facet
      if (ufc->A[0] > DOLFIN_EPS)
      {
        return 1.0;
      }
      else
        return 0.0;
    }


  private:

    UFC* ufc;
    Form& form;

  };

}

#endif
