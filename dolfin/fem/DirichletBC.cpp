// Copyright (C) 2007-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian Oelgaard, 2008
// Modified by Martin Sandve Alnes, 2008
// Modified by Johan Hake, 2009
//
// First added:  2007-04-10
// Last changed: 2010-09-16

#include <boost/scoped_ptr.hpp>
#include <boost/assign/list_of.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/LinearAlgebraFactory.h>
#include "GenericDofMap.h"
#include "FiniteElement.h"
#include "UFCMesh.h"
#include "UFCCell.h"
#include "DirichletBC.h"

using namespace dolfin;

const std::set<std::string> DirichletBC::methods = boost::assign::list_of("topological")("geometric")("pointwise");

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V,
                         const GenericFunction& g,
                         const SubDomain& sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    g(reference_to_no_delete_pointer(g)),
    method(method), user_sub_domain(reference_to_no_delete_pointer(sub_domain))
{
  check();
  parameters = default_parameters();
  init_from_sub_domain(user_sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         boost::shared_ptr<const SubDomain> sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    g(g),
    method(method), user_sub_domain(sub_domain)
{
  check();
  parameters = default_parameters();
  init_from_sub_domain(user_sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V,
                         const GenericFunction& g,
                         const MeshFunction<uint>& sub_domains,
                         uint sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    g(reference_to_no_delete_pointer(g)),
    method(method), user_sub_domain(static_cast<SubDomain*>(0))
{
  check();
  parameters = default_parameters();
  init_from_mesh_function(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         const MeshFunction<uint>& sub_domains,
                         uint sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    g(g),
    method(method), user_sub_domain(static_cast<SubDomain*>(0))
{
  check();
  parameters = default_parameters();
  init_from_mesh_function(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V,
                         const GenericFunction& g,
                         uint sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    g(reference_to_no_delete_pointer(g)),
    method(method), user_sub_domain(static_cast<SubDomain*>(0))
{
  check();
  parameters = default_parameters();
  init_from_mesh(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         uint sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    g(g),
    method(method), user_sub_domain(static_cast<SubDomain*>(0))
{
  check();
  parameters = default_parameters();
  init_from_mesh(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V,
                         const GenericFunction& g,
                         const std::vector<std::pair<uint, uint> >& markers,
                         std::string method)
  : BoundaryCondition(V),
    g(reference_to_no_delete_pointer(g)),
    method(method), user_sub_domain(static_cast<SubDomain*>(0)),
    facets(markers)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         const std::vector<std::pair<uint, uint> >& markers,
                         std::string method)
  : BoundaryCondition(V),
    g(g),
    method(method), user_sub_domain(static_cast<SubDomain*>(0)),
    facets(markers)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const DirichletBC& bc)
  : BoundaryCondition(bc._function_space)
{
  // Set default parameters
  parameters = default_parameters();

  *this = bc;
}
//-----------------------------------------------------------------------------
DirichletBC::~DirichletBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const DirichletBC& DirichletBC::operator= (const DirichletBC& bc)
{
  _function_space = bc._function_space;
  g = bc.g;
  method = bc.method;
  user_sub_domain = bc.user_sub_domain;
  facets = bc.facets;

  return *this;
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A) const
{
  apply(&A, 0, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericVector& b) const
{
  apply(0, &b, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A, GenericVector& b) const
{
  apply(&A, &b, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericVector& b, const GenericVector& x) const
{
  apply(0, &b, &x);
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A,
                        GenericVector& b,
                        const GenericVector& x) const
{
  apply(&A, &b, &x);
}
//-----------------------------------------------------------------------------
void DirichletBC::zero(GenericMatrix& A) const
{
  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, double> boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data);

  // Copy boundary value data to arrays
  uint* dofs = new uint[boundary_values.size()];
  std::map<uint, double>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
    dofs[i++] = boundary_value->first;

  // Modify linear system (A_ii = 1)
  A.zero(boundary_values.size(), dofs);

  // Finalise changes to A
  A.apply("insert");

  // Clear temporary arrays
  delete [] dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::pair<dolfin::uint, dolfin::uint> >& DirichletBC::markers()
{
  return facets;
}
//-----------------------------------------------------------------------------
const GenericFunction& DirichletBC::value()
{
  return *g;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericFunction> DirichletBC::value_ptr()
{
  return g;
}
//-----------------------------------------------------------------------------
bool DirichletBC::is_compatible(GenericFunction& v) const
{
  // This function only checks the values at vertices when it should
  // really check that the dof functionals agree. The check here is
  // neither necessary nor sufficient to guarantee compatible boundary
  // boundary conditions but a more robust test requires access to the
  // function space.

  error("is_compatible() has not been updated for the new Function class interface.");

  /*
  // Compute value size
  uint size = 1;
  const uint rank = g->function_space().element().value_rank();
  for (uint i = 0; i < rank ; i++)
    size *= g->function_space().element().value_dimension(i);
  double* g_values = new double[size];
  double* v_values = new double[size];

  // Get mesh
  const Mesh& mesh = _function_space->mesh();

  // Iterate over facets
  for (uint f = 0; f < facets.size(); f++)
  {
    // Create cell and facet
    uint cell_number  = facets[f].first;
    uint facet_number = facets[f].second;
    Cell cell(mesh, cell_number);
    Facet facet(mesh, facet_number);

    // Make cell and facet available to user-defined function
    error("Does the new GenericFunction class need an 'update' function?");
    //g->update(cell, facet_number);
    //v.update(cell, facet_number);

    // Iterate over facet vertices
    for (VertexIterator vertex(facet); !vertex.end(); ++vertex)
    {
      // Evaluate g and v at vertex
      g->eval(g_values, vertex->x());
      v.eval(v_values, vertex->x());

      // Check values
      for (uint i = 0; i < size; i++)
      {
        if (std::abs(g_values[i] - v_values[i]) > DOLFIN_EPS)
        {
          Point p(mesh.geometry().dim(), vertex->x());
          cout << "Incompatible function value " << v_values[i] << " at x = " << p << ", should be " << g_values[i] << "." << endl;
          delete [] g_values;
          delete [] v_values;
          return false;
        }
      }
    }
  }

  delete [] g_values;
  delete [] v_values;
  */

  return true;
}
//-----------------------------------------------------------------------------
void DirichletBC::set_value(const GenericFunction& g)
{
  this->g = reference_to_no_delete_pointer(g);
}
//-----------------------------------------------------------------------------
void DirichletBC::set_value(boost::shared_ptr<const GenericFunction> g)
{
  this->g = g;
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix* A,
                        GenericVector* b,
                        const GenericVector* x) const
{
  // Check arguments
  check_arguments(A, b, x);

  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, double> boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data);

  // Copy boundary value data to arrays
  const uint size = boundary_values.size();
  std::vector<uint> dofs(size);
  std::vector<double> values(size);
  std::map<uint, double>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
  {
    dofs[i]     = boundary_value->first;
    values[i++] = boundary_value->second;
  }

  // Modify boundary values for nonlinear problems
  if (x)
  {
    // Gather values
    std::vector<double> x_values(size);
    x->get(&x_values[0], dofs.size(), &dofs[0]);

    // Modify RHS entries
    for (uint i = 0; i < size; i++)
      values[i] = x_values[i] - values[i];
  }

  info(PROGRESS, "Applying boundary conditions to linear system.");

  // Modify RHS vector (b[i] = value) and apply changes
  if (b)
  {
    b->set(&values[0], size, &dofs[0]);
    b->apply("insert");
  }

  // Modify linear system (A_ii = 1) and apply changes
  if (A)
  {
    const bool use_ident = parameters["use_ident"];
    if (use_ident)
    {
      A->ident(size, &dofs[0]);
    }
    else
    {
      for (uint i = 0; i < size; i++)
      {
        std::pair<uint, uint> ij(dofs[i], dofs[i]);
        A->setitem(ij, 1.0);
      }
    }
    A->apply("insert");
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::check() const
{
  // Check for common errors, message below might be cryptic
  if (g->value_rank() == 0 && _function_space->element().value_rank() == 1)
    error("Unable to create boundary condition. Reason: Expecting a vector-valued boundary value but given function is scalar.");
  if (g->value_rank() == 1 && _function_space->element().value_rank() == 0)
    error("Unable to create boundary condition. Reason: Expecting a scalar boundary value but given function is vector-valued.");

  // Check that value shape of boundary value
  check_equal(g->value_rank(), _function_space->element().value_rank(),
              "create boundary condition", "value rank");
  for (uint i = 0; i < g->value_rank(); i++)
    check_equal(g->value_dimension(i), _function_space->element().value_dimension(i),
                "create boundary condition", "value dimension");

  // Check that boundary condition method is known
  if (methods.count(method) == 0)
    error("Unable to create boundary condition, unknown method specified.");

  // Check that the mesh is ordered
  if (!_function_space->mesh().ordered())
    error("Unable to create boundary condition, mesh is not correctly ordered (consider calling mesh.order()).");
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_sub_domain(boost::shared_ptr<const SubDomain> sub_domain)
{
  assert(facets.size() == 0);

  // FIXME: This can be made more efficient, we should be able to
  // FIXME: extract the facets without first creating a MeshFunction on
  // FIXME: the entire mesh and then extracting the subset. This is done
  // FIXME: mainly for convenience (we may reuse mark() in SubDomain).

  // Create mesh function for sub domain markers on facets
  const uint dim = _function_space->mesh().topology().dim();
  _function_space->mesh().init(dim - 1);
  MeshFunction<uint> sub_domains(_function_space->mesh(), dim - 1);

  // Set geometric dimension (needed for SWIG interface)
  sub_domain->_geometric_dimension = _function_space->mesh().geometry().dim();

  // Mark everything as sub domain 1
  sub_domains = 1;

  // Mark the sub domain as sub domain 0
  sub_domain->mark(sub_domains, 0);

  // Initialize from mesh function
  init_from_mesh_function(sub_domains, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_mesh_function(const MeshFunction<uint>& sub_domains,
                                          uint sub_domain)
{
  assert(facets.size() == 0);

  // Make sure we have the facet - cell connectivity
  const uint dim = _function_space->mesh().topology().dim();
  _function_space->mesh().init(dim - 1, dim);

  // Build set of boundary facets
  for (FacetIterator facet(_function_space->mesh()); !facet.end(); ++facet)
  {
    // Skip facets not on this boundary
    if (sub_domains[*facet] != sub_domain)
      continue;

    // Get cell to which facet belongs (there may be two, but pick first)
    const Cell cell(_function_space->mesh(), facet->entities(dim)[0]);

    // Get local index of facet with respect to the cell
    const uint facet_number = cell.index(*facet);

    // Copy data
    facets.push_back(std::pair<uint, uint>(cell.index(), facet_number));
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_mesh(uint sub_domain)
{
  assert(facets.size() == 0);

  // For this to work, the mesh *needs* to be ordered according to
  // the UFC ordering before it gets here. So reordering the mesh
  // here will either have no effect (if the mesh is already ordered
  // or it won't do anything good (since the markers are wrong anyway).
  // In conclusion: we don't need to order the mesh here.

  cout << "Creating sub domain markers for boundary condition." << endl;

  // Get data
  const std::vector<uint>* facet_cells   = _function_space->mesh().data().array("boundary facet cells");
  const std::vector<uint>* facet_numbers = _function_space->mesh().data().array("boundary facet numbers");
  const std::vector<uint>* indicators    = _function_space->mesh().data().array("boundary indicators");

  // Check data
  if (!facet_cells)
  {
    info(_function_space->mesh().data());
    error("Mesh data \"boundary facet cells\" not available.");
  }
  if (!facet_numbers)
  {
    info(_function_space->mesh().data());
    error("Mesh data \"boundary facet numbers\" not available.");
  }
  if (!indicators)
  {
    info(_function_space->mesh().data());
    error("Mesh data \"boundary indicators\" not available.");
  }

  // Get size
  const uint size = facet_cells->size();
  assert(size == facet_numbers->size());
  assert(size == indicators->size());

  // Build set of boundary facets
  for (uint i = 0; i < size; i++)
  {
    // Skip facets not on this boundary
    if ((*indicators)[i] != sub_domain)
      continue;

    // Copy data
    facets.push_back(std::pair<uint, uint>((*facet_cells)[i], (*facet_numbers)[i]));
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc(std::map<uint, double>& boundary_values,
                             BoundaryCondition::LocalData& data) const
{
  // Choose strategy
  if (method == "topological")
    compute_bc_topological(boundary_values, data);
  else if (method == "geometric")
    compute_bc_geometric(boundary_values, data);
  else if (method == "pointwise")
    compute_bc_pointwise(boundary_values, data);
  else
    error("Unknown method for application of boundary conditions.");
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_topological(std::map<uint, double>& boundary_values,
                                         BoundaryCondition::LocalData& data) const
{
  assert(_function_space);
  assert(g);

  // Special case
  if (facets.size() == 0)
  {
    warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get mesh and dofmap
  const Mesh& mesh = _function_space->mesh();
  const GenericDofMap& dofmap = _function_space->dofmap();

  // Create UFC cell object
  UFCCell ufc_cell(mesh);

  // Iterate over facets
  Progress p("Computing Dirichlet boundary values, topological search", facets.size());
  for (uint f = 0; f < facets.size(); f++)
  {
    // Get cell number and local facet number
    const uint cell_number = facets[f].first;
    const uint facet_number = facets[f].second;

    // Create cell
    Cell cell(mesh, cell_number);
    ufc_cell.update(cell, facet_number);

    // Restrict coefficient to cell
    g->restrict(data.w, _function_space->element(), cell, ufc_cell);

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(data.cell_dofs, ufc_cell, cell_number);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(data.facet_dofs, facet_number);

    // Debugging print:
    /*
       cout << endl << "Handling BC's for:" << endl;
       cout << "Cell:  " << facet.entities(facet.dim() + 1)[0] << endl;
       cout << "Facet: " << local_facet << endl;
    */

    // Pick values for facet
    for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
    {
      const uint dof = data.cell_dofs[data.facet_dofs[i]];
      const double value = data.w[data.facet_dofs[i]];
      boundary_values[dof] = value;
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_geometric(std::map<uint, double>& boundary_values,
                                       BoundaryCondition::LocalData& data) const
{
  assert(_function_space);
  assert(g);

  // Special case
  if (facets.size() == 0)
  {
    warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get mesh and dofmap
  const Mesh& mesh = _function_space->mesh();
  const GenericDofMap& dofmap = _function_space->dofmap();

  // Initialize facets, needed for geometric search
  info(TRACE, "Computing facets, needed for geometric application of boundary conditions.");
  mesh.init(mesh.topology().dim() - 1);

  // Iterate over facets
  Progress p("Computing Dirichlet boundary values, geometric search", facets.size());
  for (uint f = 0; f < facets.size(); f++)
  {
    // Get cell number and local facet number
    const uint cell_number = facets[f].first;
    const uint facet_number = facets[f].second;

    // Create facet
    Cell cell(mesh, cell_number);
    Facet facet(mesh, cell.entities(mesh.topology().dim() - 1)[facet_number]);

    // Create UFC cell object
    UFCCell ufc_cell(mesh);

    // Loop the vertices associated with the facet
    for (VertexIterator vertex(facet); !vertex.end(); ++vertex)
    {
      // Loop the cells associated with the vertex
      for (CellIterator c(*vertex); !c.end(); ++c)
      {
        ufc_cell.update(*c, facet_number);

        bool interpolated = false;

        // Tabulate coordinates of dofs on cell
        dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

        // Loop over all dofs on cell
        for (uint i = 0; i < dofmap.local_dimension(ufc_cell); ++i)
        {
          // Check if the coordinates are on current facet and thus on boundary
          if (!on_facet(data.coordinates[i], facet))
            continue;

          if (!interpolated)
          {
            // Tabulate dofs on cell
            dofmap.tabulate_dofs(data.cell_dofs, ufc_cell, c->index());

            /// Restrict coefficient to cell
            g->restrict(data.w, _function_space->element(), cell, ufc_cell);
          }

          // Set boundary value
          const uint dof = data.cell_dofs[i];
          const double value = data.w[i];
          boundary_values[dof] = value;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_pointwise(std::map<uint, double>& boundary_values,
                                       BoundaryCondition::LocalData& data) const
{
  assert(_function_space);
  assert(g);
  assert(user_sub_domain);

  // Get mesh and dofmap
  const Mesh& mesh = _function_space->mesh();
  const GenericDofMap& dofmap = _function_space->dofmap();

  // Create UFC cell object
  UFCCell ufc_cell(mesh);

  // Iterate over cells
  Progress p("Computing Dirichlet boundary values, pointwise search", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    ufc_cell.update(*cell);

    // Tabulate coordinates of dofs on cell
    dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

    // Interpolate function only once and only on cells where necessary
    bool interpolated = false;

    // Loop all dofs on cell
    for (uint i = 0; i < dofmap.local_dimension(ufc_cell); ++i)
    {
      // Check if the coordinates are part of the sub domain
      if (!user_sub_domain->inside(data.array_coordinates[i], false))
        continue;

      if (!interpolated)
      {
        interpolated = true;

        // Tabulate dofs on cell
        dofmap.tabulate_dofs(data.cell_dofs, ufc_cell, cell->index());

        // Restrict coefficient to cell
        g->restrict(data.w, _function_space->element(), *cell, ufc_cell);
      }

      // Set boundary value
      const uint dof = data.cell_dofs[i];
      const double value = data.w[i];
      boundary_values[dof] = value;
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
bool DirichletBC::on_facet(double* coordinates, Facet& facet) const
{
  // Check if the coordinates are on the same line as the line segment
  if (facet.dim() == 1)
  {
    // Create points
    Point p(coordinates[0], coordinates[1]);
    const Point v0 = Vertex(facet.mesh(), facet.entities(0)[0]).point();
    const Point v1 = Vertex(facet.mesh(), facet.entities(0)[1]).point();

    // Create vectors
    const Point v01 = v1 - v0;
    const Point vp0 = v0 - p;
    const Point vp1 = v1 - p;

    // Check if the length of the sum of the two line segments vp0 and vp1 is
    // equal to the total length of the facet
    if ( std::abs(v01.norm() - vp0.norm() - vp1.norm()) < DOLFIN_EPS )
      return true;
    else
      return false;
  }
  // Check if the coordinates are in the same plane as the triangular facet
  else if (facet.dim() == 2)
  {
    // Create points
    Point p(coordinates[0], coordinates[1], coordinates[2]);
    const Point v0 = Vertex(facet.mesh(), facet.entities(0)[0]).point();
    const Point v1 = Vertex(facet.mesh(), facet.entities(0)[1]).point();
    const Point v2 = Vertex(facet.mesh(), facet.entities(0)[2]).point();

    // Create vectors
    const Point v01 = v1 - v0;
    const Point v02 = v2 - v0;
    const Point vp0 = v0 - p;
    const Point vp1 = v1 - p;
    const Point vp2 = v2 - p;

    // Check if the sum of the area of the sub triangles is equal to the total
    // area of the facet
    if (std::abs(v01.cross(v02).norm() - vp0.cross(vp1).norm() - vp1.cross(vp2).norm()
        - vp2.cross(vp0).norm()) < DOLFIN_EPS)
      return true;
    else
      return false;
  }

  error("Unable to determine if given point is on facet (not implemented for given facet dimension).");

  return false;
}
//-----------------------------------------------------------------------------
void DirichletBC::get_bc(uint* indicators, double* values) const
{
  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, double> boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data);

  std::map<uint, double>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
  {
    i = boundary_value->first;
    indicators[i] = 1;
    values[i] = boundary_value->second;
  }
}
//-----------------------------------------------------------------------------
