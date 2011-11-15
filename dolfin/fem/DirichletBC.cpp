// Copyright (C) 2007-2011 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Kristian Oelgaard, 2008
// Modified by Martin Sandve Alnes, 2008
// Modified by Johan Hake, 2009
//
// First added:  2007-04-10
// Last changed: 2011-09-19

#include <map>
#include <utility>
#include <boost/assign/list_of.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Constant.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshDomains.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
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

const std::set<std::string> DirichletBC::methods
            = boost::assign::list_of("topological")("geometric")("pointwise");

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V, const GenericFunction& g,
                         const SubDomain& sub_domain, std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(reference_to_no_delete_pointer(g)),
    _method(method), _user_sub_domain(reference_to_no_delete_pointer(sub_domain))
{
  check();
  parameters = default_parameters();
  init_from_sub_domain(_user_sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         boost::shared_ptr<const SubDomain> sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(g), _method(method), _user_sub_domain(sub_domain)
{
  check();
  parameters = default_parameters();
  init_from_sub_domain(_user_sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V, const GenericFunction& g,
                         const MeshFunction<uint>& sub_domains,
                         uint sub_domain, std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(reference_to_no_delete_pointer(g)),
    _method(method)
{
  check();
  parameters = default_parameters();
  init_from_mesh_function(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         boost::shared_ptr<const MeshFunction<uint> > sub_domains,
                         uint sub_domain,
                         std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(g), _method(method)
{
  check();
  parameters = default_parameters();
  init_from_mesh_function(*sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V, const GenericFunction& g,
                         uint sub_domain, std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(reference_to_no_delete_pointer(g)), _method(method)
{
  check();
  parameters = default_parameters();
  init_from_mesh(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         uint sub_domain, std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(g), _method(method)
{
  check();
  parameters = default_parameters();
  init_from_mesh(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         const std::vector<std::pair<uint, uint> >& markers,
                         std::string method)
  : BoundaryCondition(V),
    Hierarchical<DirichletBC>(*this),
    g(g), _method(method), facets(markers)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const DirichletBC& bc)
  : BoundaryCondition(bc._function_space),
    Hierarchical<DirichletBC>(*this)
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
  _method = bc._method;
  _user_sub_domain = bc._user_sub_domain;
  facets = bc.facets;

  // Call assignment operator for base class
  Hierarchical<DirichletBC>::operator=(bc);

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
void DirichletBC::get_boundary_values(Map& boundary_values,
                         std::string method) const
{
  // Create local data
  BoundaryCondition::LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data, method);
}
//-----------------------------------------------------------------------------
void DirichletBC::zero(GenericMatrix& A) const
{
  // A map to hold the mapping from boundary dofs to boundary values
  Map boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data, _method);

  // Copy boundary value data to arrays
  std::vector<uint> dofs(boundary_values.size());
  Map::const_iterator bv;
  uint i = 0;
  for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
    dofs[i++] = bv->first;

  // Modify linear system (A_ii = 1)
  A.zero(boundary_values.size(), &dofs[0]);

  // Finalise changes to A
  A.apply("insert");
}
//-----------------------------------------------------------------------------
void DirichletBC::zero_columns(GenericMatrix& A,
                               GenericVector& b,
                               double diag_val) const
{
  Map bv_map;
  get_boundary_values(bv_map, _method);

  // Create lookup table of dofs
  //const uint nrows = A.size(0); // should be equal to b.size()
  const uint ncols = A.size(1); // should be equal to max possible dof+1

  std::pair<uint,uint> rows = A.local_range(0);
  //std::pair<uint,uint> cols = A.local_range(1);

  std::vector<char> is_bc_dof(ncols);
  std::vector<double> bc_dof_val(ncols);
  for (Map::const_iterator bv = bv_map.begin();  bv != bv_map.end();  ++bv)
  {
    is_bc_dof[bv->first] = 1;
    bc_dof_val[bv->first] = bv->second;
  }

  // Scan through all columns of all rows, setting to zero if is_bc_dof[column]
  // At the same time, we collect corrections to the RHS

  std::vector<uint>   cols;
  std::vector<double> vals;
  std::vector<double> b_vals;
  std::vector<uint>   b_rows;

  for (uint row=rows.first; row<rows.second; row++)
  {
    // If diag_val is nonzero, the matrix is a diagonal block (nrows==ncols),
    // and we can set the whole BC row
    if (diag_val != 0.0 && is_bc_dof[row])
    {
      A.getrow(row, cols, vals);
      for (uint j=0; j<cols.size(); j++)
        vals[j] = (cols[j]==row) * diag_val;
      A.setrow(row, cols, vals);
      A.apply("insert");
      b.setitem(row, bc_dof_val[row]*diag_val);
    }
    else // Otherwise, we scan the row for BC columns
    {
      A.getrow(row, cols, vals);
      bool row_changed=false;
      for (uint j=0; j<cols.size(); j++)
      {
        const uint col = cols[j];

        // Skip columns that aren't BC, and entries that are zero
        if (!is_bc_dof[col] || vals[j] == 0.0)
          continue;

        // We're going to change the row, so make room for it
        if (!row_changed)
        {
          row_changed = true;
          b_rows.push_back(row);
          b_vals.push_back(0.0);
        }

        b_vals.back() -= bc_dof_val[col]*vals[j];
        vals[j] = 0.0;
      }
      if (row_changed)
      {
        A.setrow(row, cols, vals);
        A.apply("insert");
      }
    }
  }

  b.add(&b_vals.front(), b_rows.size(), &b_rows.front());
  b.apply("add");
}
//-----------------------------------------------------------------------------
const std::vector<std::pair<dolfin::uint, dolfin::uint> >& DirichletBC::markers() const
{
  return facets;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericFunction> DirichletBC::value() const
{
  return g;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const SubDomain> DirichletBC::user_sub_domain() const
{
  return _user_sub_domain;
}
//-----------------------------------------------------------------------------
bool DirichletBC::is_compatible(GenericFunction& v) const
{
  // This function only checks the values at vertices when it should
  // really check that the dof functionals agree. The check here is
  // neither necessary nor sufficient to guarantee compatible boundary
  // boundary conditions but a more robust test requires access to the
  // function space.

  dolfin_error("DirichletBC.cpp",
               "call is_compatible",
               "This function has not been updated for the new Function class interface");

  /*
  // Compute value size
  uint size = 1;
  const uint rank = g->function_space().element()->value_rank();
  for (uint i = 0; i < rank ; i++)
    size *= g->function_space().element()->value_dimension(i);
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
    dolfin_error("DirichletBC.cpp",
                 "add proper message here",
                 "Does the new GenericFunction class need an 'update' function?");
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
void DirichletBC::homogenize()
{
  const uint value_rank = g->value_rank();
  if (!value_rank)
  {
    boost::shared_ptr<Constant> zero(new Constant(0.0));
    set_value(zero);
  } else if (value_rank == 1)
  {
    const uint value_dim = g->value_dimension(0);
    std::vector<double> values(value_dim, 0.0);
    boost::shared_ptr<Constant> zero(new Constant(values));
    set_value(zero);
  } else
  {
    std::vector<uint> value_shape;
    for (uint i = 0; i < value_rank; i++)
      value_shape.push_back(g->value_dimension(i));
    std::vector<double> values(g->value_size(), 0.0);
    boost::shared_ptr<Constant> zero(new Constant(value_shape, values));
    set_value(zero);
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::set_value(boost::shared_ptr<const GenericFunction> g)
{
  this->g = g;
}
//-----------------------------------------------------------------------------
std::string DirichletBC::method() const
{
  return _method;
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix* A,
                        GenericVector* b,
                        const GenericVector* x) const
{
  // Check arguments
  check_arguments(A, b, x);

  // A map to hold the mapping from boundary dofs to boundary values
  Map boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data, _method);

  // Copy boundary value data to arrays
  const uint size = boundary_values.size();
  std::vector<uint> dofs(size);
  std::vector<double> values(size);
  Map::const_iterator bv;
  uint i = 0;
  for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
  {
    dofs[i]     = bv->first;
    values[i++] = bv->second;
  }

  // Modify boundary values for nonlinear problems
  if (x)
  {
    // Get values (these must reside in local portion (including ghost
    // values) of the vector
    std::vector<double> x_values(size);
    x->get_local(&x_values[0], dofs.size(), &dofs[0]);

    // Modify RHS entries
    for (uint i = 0; i < size; i++)
      values[i] = x_values[i] - values[i];
  }

  log(PROGRESS, "Applying boundary conditions to linear system.");

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
      A->ident(size, &dofs[0]);
    else
    {
      for (uint i = 0; i < size; i++)
      {
        std::pair<uint, uint> ij(dofs[i], dofs[i]);
        A->setitem(ij, 1.0);
      }
    }

    // Apply changes
    A->apply("add");
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::check() const
{
  assert(g);
  assert(_function_space->element());

  // Check for common errors, message below might be cryptic
  if (g->value_rank() == 0 && _function_space->element()->value_rank() == 1)
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "Expecting a vector-valued boundary value but given function is scalar");
  }
  if (g->value_rank() == 1 && _function_space->element()->value_rank() == 0)
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "Expecting a scalar boundary value but given function is vector-valued");
  }

  // Check that value shape of boundary value
  dolfin_error("DirichletBC.cpp",
               "create Dirichlet boundary condition",
               "Illegal value rank (%d), expecting (%d)",
               g->value_rank(), _function_space->element()->value_rank());
  for (uint i = 0; i < g->value_rank(); i++)
  {
    dolfin_error("DirichletBC.cpp",
                 "Illcreate Dirichlet boundary condition",
                 "Illegal value dimension (%d), expecting (%d)",
                 g->value_dimension(i), _function_space->element()->value_dimension(i));
  }

  // Check that boundary condition method is known
  if (methods.count(_method) == 0)
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "unknown method (\"%s\")", _method.c_str());
  }

  // Check that the mesh is ordered
  assert(_function_space->mesh());
  if (!_function_space->mesh()->ordered())
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                  "Mesh is not ordered according to the UFC numbering convention. Consider calling mesh.order()");
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_sub_domain(boost::shared_ptr<const SubDomain> sub_domain)
{
  assert(facets.size() == 0);

  // FIXME: This can be made more efficient, we should be able to
  // FIXME: extract the facets without first creating a MeshFunction on
  // FIXME: the entire mesh and then extracting the subset. This is done
  // FIXME: mainly for convenience (we may reuse mark() in SubDomain).

  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Create mesh function for sub domain markers on facets
  const uint dim = mesh.topology().dim();
  _function_space->mesh()->init(dim - 1);
  MeshFunction<uint> sub_domains(mesh, dim - 1);

  // Set geometric dimension (needed for SWIG interface)
  sub_domain->_geometric_dimension = mesh.geometry().dim();

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

  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Make sure we have the facet - cell connectivity
  const uint dim = mesh.topology().dim();
  mesh.init(dim - 1, dim);

  // Build set of boundary facets
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Skip facets not on this boundary
    if (sub_domains[*facet] != sub_domain)
      continue;

    // Get cell to which facet belongs (there may be two, but pick first)
    const Cell cell(mesh, facet->entities(dim)[0]);

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

  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Assign domain numbers for each facet
  const uint D = mesh.topology().dim();
  const std::map<std::pair<uint, uint>, uint>&
    markers = mesh.domains().markers(D - 1).values();
  std::map<std::pair<uint, uint>, uint>::const_iterator mark;
  for (mark = markers.begin(); mark != markers.end(); ++mark)
  {
    if (mark->second == sub_domain)
      facets.push_back(mark->first);
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc(Map& boundary_values,
                             BoundaryCondition::LocalData& data,
                             std::string method) const
{
  // Set method if dafault
  if (method == "default")
    method = _method;

  // Choose strategy
  if (method == "topological")
    compute_bc_topological(boundary_values, data);
  else if (method == "geometric")
    compute_bc_geometric(boundary_values, data);
  else if (method == "pointwise")
    compute_bc_pointwise(boundary_values, data);
  else
  {
    dolfin_error("DirichletBC.cpp",
                 "compute boundary conditions",
                 "Unknown method for application of boundary conditions");
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_topological(Map& boundary_values,
                                         BoundaryCondition::LocalData& data) const
{
  assert(_function_space);
  assert(g);

  // Special case
  if (facets.size() == 0)
  {
    if (MPI::num_processes() == 1)
      warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get mesh and dofmap
  assert(_function_space->mesh());
  assert(_function_space->dofmap());
  const Mesh& mesh = *_function_space->mesh();
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Create UFC cell object
  UFCCell ufc_cell(mesh);

  // Iterate over facets
  assert(_function_space->element());
  Progress p("Computing Dirichlet boundary values, topological search", facets.size());
  for (uint f = 0; f < facets.size(); ++f)
  {
    // Get cell number and local facet number
    const uint cell_number  = facets[f].first;
    const uint facet_number = facets[f].second;

    // Create cell
    Cell cell(mesh, cell_number);
    ufc_cell.update(cell, facet_number);

    // Restrict coefficient to cell
    g->restrict(&data.w[0], *_function_space->element(), cell, ufc_cell);

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(&data.cell_dofs[0], cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(&data.facet_dofs[0], facet_number);

    // Pick values for facet
    for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
    {
      const uint global_dof = data.cell_dofs[data.facet_dofs[i]];
      const double value = data.w[data.facet_dofs[i]];
      boundary_values[global_dof] = value;
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_geometric(Map& boundary_values,
                                      BoundaryCondition::LocalData& data) const
{
  assert(_function_space);
  assert(_function_space->element());
  assert(g);

  // Special case
  if (facets.size() == 0)
  {
    if (MPI::num_processes() == 1)
      warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get mesh and dofmap
  assert(_function_space->mesh());
  assert(_function_space->dofmap());
  const Mesh& mesh = *_function_space->mesh();
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Initialize facets, needed for geometric search
  log(TRACE, "Computing facets, needed for geometric application of boundary conditions.");
  mesh.init(mesh.topology().dim() - 1);

  // Iterate over facets
  Progress p("Computing Dirichlet boundary values, geometric search", facets.size());
  for (uint f = 0; f < facets.size(); ++f)
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
        for (uint i = 0; i < dofmap.cell_dimension(c->index()); ++i)
        {
          // Check if the coordinates are on current facet and thus on boundary
          if (!on_facet(&(data.coordinates[i][0]), facet))
            continue;

          if (!interpolated)
          {
            // Tabulate dofs on cell
            dofmap.tabulate_dofs(&data.cell_dofs[0], *c);

            // Restrict coefficient to cell
            g->restrict(&data.w[0], *_function_space->element(), cell, ufc_cell);
          }

          // Set boundary value
          const uint global_dof = data.cell_dofs[i];
          const double value = data.w[i];
          boundary_values[global_dof] = value;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_pointwise(Map& boundary_values,
                                      BoundaryCondition::LocalData& data) const
{
  assert(_function_space);
  assert(_function_space->element());
  assert(g);
  assert(_user_sub_domain);

  // Get mesh and dofmap
  assert(_function_space->mesh());
  assert(_function_space->dofmap());
  const Mesh& mesh = *_function_space->mesh();
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Geometric dim
  const uint gdim = mesh.geometry().dim();

  // Create UFC cell object
  UFCCell ufc_cell(mesh);

  // Iterate over cells
  Progress p("Computing Dirichlet boundary values, pointwise search", mesh.num_cells());
  Array<double> x(gdim);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    ufc_cell.update(*cell);

    // Tabulate coordinates of dofs on cell
    dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

    // Interpolate function only once and only on cells where necessary
    bool already_interpolated = false;

    // Loop all dofs on cell
    for (uint i = 0; i < dofmap.cell_dimension(cell->index()); ++i)
    {
      // Check if the coordinates are part of the sub domain (calls user-defined 'indside' function)
      for (uint j = 0; j < gdim; ++j)
        x[j] = data.coordinates[i][j];
      if (!_user_sub_domain->inside(x, false))
        continue;

      if (!already_interpolated)
      {
        already_interpolated = true;

        // Tabulate dofs on cell
        dofmap.tabulate_dofs(&data.cell_dofs[0], *cell);

        // Restrict coefficient to cell
        g->restrict(&data.w[0], *_function_space->element(), *cell, ufc_cell);
      }

      // Set boundary value
      const uint global_dof = data.cell_dofs[i];
      const double value = data.w[i];
      boundary_values[global_dof] = value;
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
    const Point p(coordinates[0], coordinates[1], coordinates[2]);
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

  dolfin_error("DirichletBC.cpp",
               "determine if given point is on facet",
               "Not implemented for given facet dimension");

  return false;
}
//-----------------------------------------------------------------------------
