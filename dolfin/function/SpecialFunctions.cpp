// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007, 2008.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2008-07-17
// Last changed: 2008-11-03

#include <dolfin/common/constants.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/fem/Form.h>
#include "FunctionSpace.h"
#include "SpecialFunctions.h"
#include "Data.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshSize::MeshSize() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshSize::MeshSize(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshSize::eval(double* values, const Data& data) const
{
  values[0] = data.cell().diameter();
}
//-----------------------------------------------------------------------------
double MeshSize::min() const
{
  CellIterator c(function_space().mesh());
  double hmin = c->diameter();
  for (; !c.end(); ++c)
    hmin = std::min(hmin, c->diameter());
  return hmin;
}
//-----------------------------------------------------------------------------
double MeshSize::max() const
{
  CellIterator c(function_space().mesh());
  double hmax = c->diameter();
  for (; !c.end(); ++c)
    hmax = std::max(hmax, c->diameter());
  return hmax;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
InvMeshSize::InvMeshSize() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
InvMeshSize::InvMeshSize(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void InvMeshSize::eval(double* values, const Data& data) const
{
  values[0] = 1.0 / data.cell().diameter();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
AvgMeshSize::AvgMeshSize() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
AvgMeshSize::AvgMeshSize(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AvgMeshSize::eval(double* values, const Data& data) const
{
  // Get the cell
  const Cell& cell = data.cell();

  // If there is no facet (assembling on interior), return cell diameter
  if (!data.on_facet())
  {
    values[0] = cell.diameter();
    return;
  }
  else
  {
    // Get the facet and the mesh
    const uint facet = data.facet();
    const Mesh& mesh = cell.mesh();

    // Create facet from the global facet number
    Facet facet0(mesh, cell.entities(mesh.topology().dim() - 1)[facet]);

    // If there are two cells connected to the facet
    if (facet0.numEntities(mesh.topology().dim()) == 2)
    {
      // Create the two connected cells and return the average of their diameter
      Cell cell0(mesh, facet0.entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet0.entities(mesh.topology().dim())[1]);

      values[0] = (cell0.diameter() + cell1.diameter())/2.0;
      return;
    }
    // Else there is only one cell connected to the facet and the average is 
    // the cell diameter
    else
    {
      values[0] = cell.diameter();
      return;
    }
  }
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FacetNormal::FacetNormal() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FacetNormal::FacetNormal(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetNormal::eval(double* values, const Data& data) const
{
  const Cell& cell = data.cell();

  if (data.on_facet())
  {
    const uint facet = data.facet();
    for (uint i = 0; i < cell.dim(); i++)
      values[i] = cell.normal(facet, i);
  }
  else
  {
    for (uint i = 0; i < cell.dim(); i++)
      values[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
dolfin::uint FacetNormal::rank() const
{
  return 1;
}
//-----------------------------------------------------------------------------    
dolfin::uint FacetNormal::dim(uint i) const
{
  if(i > 0)
    error("Invalid dimension %d in FacetNormal::dim.", i);
  return function_space().mesh().geometry().dim();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FacetArea::FacetArea() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FacetArea::FacetArea(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(double* values, const Data& data) const
{
  if (data.on_facet())
    values[0] = data.cell().facetArea(data.facet());
  else
    values[0] = 0.0;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
InvFacetArea::InvFacetArea() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
InvFacetArea::InvFacetArea(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void InvFacetArea::eval(double* values, const Data& data) const
{
  if (data.on_facet() >= 0)
    values[0] = 1.0 / data.cell().facetArea(data.facet());
  else
    values[0] = 0.0;
}
//-----------------------------------------------------------------------------
OutflowFacet::OutflowFacet(const Form& form) : form(form), 
                                               V(form.function_spaces()), ufc(form)                            
{
  // Some simple sanity checks on form
  if (!(form.rank() == 0 && form.ufc_form().num_coefficients() == 2))
    error("Invalid form: rank = %d, number of coefficients = %d. Must be rank 0 form with 2 coefficients.", 
              form.rank(), form.ufc_form().num_coefficients());

  if (!(form.ufc_form().num_cell_integrals() == 0 && form.ufc_form().num_exterior_facet_integrals() == 1 
        && form.ufc_form().num_interior_facet_integrals() == 0))
    error("Invalid form: Must have exactly 1 exterior facet integral");
}
//-----------------------------------------------------------------------------
OutflowFacet::~OutflowFacet()
{
  //delete ufc;
}
//-----------------------------------------------------------------------------
void OutflowFacet::eval(double* values, const Data& data) const
{
  // If there is no facet (assembling on interior), return 0.0
  if (!data.on_facet())
  {
    values[0] = 0.0;
    return;
  }
  else
  {
    ufc.update( data.cell() );

    // Interpolate coefficients on cell and current facet
    for (uint i = 0; i < form.coefficients().size(); i++)
      form.coefficient(i).interpolate(ufc.w[i], ufc.cell, data.facet());

    // Get exterior facet integral (we need to be able to tabulate ALL facets 
    // of a given cell)
    ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];

    // Call tabulate_tensor on exterior facet integral, 
    // dot(velocity, facet_normal)
    integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell, data.facet());
  }

   // If dot product is positive, the current facet is an outflow facet
  if (ufc.A[0] > DOLFIN_EPS)
     values[0] = 1.0;
  else
     values[0] = 0.0;
}
//-----------------------------------------------------------------------------
