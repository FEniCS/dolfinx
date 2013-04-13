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
// Modified by Joachim B. Haga, 2012
//
// First added:  2007-04-10
// Last changed: 2012-02-29

#include <map>
#include <utility>
#include <boost/assign/list_of.hpp>
#include <boost/serialization/utility.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/RangedIndexSet.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Constant.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshDomains.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Restriction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include "GenericDofMap.h"
#include "FiniteElement.h"
#include "UFCCell.h"
#include "DirichletBC.h"

using namespace dolfin;

const std::set<std::string> DirichletBC::methods
            = boost::assign::list_of("topological")("geometric")("pointwise");

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V,
                         const GenericFunction& g,
                         const SubDomain& sub_domain,
                         std::string method,
                         bool check_midpoint)
  : Hierarchical<DirichletBC>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    _g(reference_to_no_delete_pointer(g)),
    _method(method),
    _user_sub_domain(reference_to_no_delete_pointer(sub_domain)),
    _check_midpoint(check_midpoint)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         boost::shared_ptr<const SubDomain> sub_domain,
                         std::string method,
                         bool check_midpoint)
  : Hierarchical<DirichletBC>(*this),
    _function_space(V),
    _g(g), _method(method),
    _user_sub_domain(sub_domain),
    _check_midpoint(check_midpoint)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V, const GenericFunction& g,
                         const MeshFunction<std::size_t>& sub_domains,
                         std::size_t sub_domain, std::string method)
  : Hierarchical<DirichletBC>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    _g(reference_to_no_delete_pointer(g)),
    _method(method),
    _user_mesh_function(reference_to_no_delete_pointer(sub_domains)),
    _user_sub_domain_marker(sub_domain),
    _check_midpoint(true)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         boost::shared_ptr<const MeshFunction<std::size_t> > sub_domains,
                         std::size_t sub_domain,
                         std::string method)
  : Hierarchical<DirichletBC>(*this), _function_space(V), _g(g),
    _method(method), _user_mesh_function(sub_domains),
    _user_sub_domain_marker(sub_domain),
    _check_midpoint(true)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const FunctionSpace& V, const GenericFunction& g,
                         std::size_t sub_domain, std::string method)
  : Hierarchical<DirichletBC>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    _g(reference_to_no_delete_pointer(g)), _method(method),
    _user_sub_domain_marker(sub_domain),
    _check_midpoint(true)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         std::size_t sub_domain, std::string method)
  : Hierarchical<DirichletBC>(*this),
    _function_space(V),
    _g(g),
    _method(method),
    _user_sub_domain_marker(sub_domain),
    _check_midpoint(true)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace> V,
                         boost::shared_ptr<const GenericFunction> g,
                         const std::vector<std::pair<std::size_t, std::size_t> >& markers,
                         std::string method)
  : Hierarchical<DirichletBC>(*this),
    _function_space(V),
    _g(g),
    _method(method),
    facets(markers),
    _check_midpoint(true)
{
  check();
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(const DirichletBC& bc)
  : Hierarchical<DirichletBC>(*this)
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
  _g = bc._g;
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
void DirichletBC::gather(Map& boundary_values) const
{
  Timer timer("DirichletBC gather");

  typedef std::vector<std::pair<std::size_t, double> > bv_vec_type;
  typedef std::map<std::size_t, bv_vec_type> map_type;

  typedef boost::unordered_map<std::size_t, std::vector<unsigned int> > shared_dof_type;
  typedef shared_dof_type::const_iterator shared_dof_iterator;
  typedef std::vector<unsigned int>::const_iterator proc_iterator;

  dolfin_assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *_function_space->dofmap();
  const shared_dof_type& shared_dofs = dofmap.shared_dofs();

  // Create list of boundary values to send to each processor

  map_type proc_map;
  for (Map::const_iterator bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
  {
    // If the boundary value is attached to a shared dof, add it to the list of
    // boundary values for each of the processors that share it
    shared_dof_iterator shared_dof = shared_dofs.find(bv->first);
    if (shared_dof != shared_dofs.end())
      for (proc_iterator proc = shared_dof->second.begin(); proc != shared_dof->second.end(); ++proc)
        proc_map[*proc].push_back(*bv);
  }

  // Distribute the lists between neighbours

  map_type received_bvs;
  MPI::distribute(dofmap.neighbours(), proc_map, received_bvs);

  // Add the received boundary values to the local boundary values

  for (map_type::const_iterator it = received_bvs.begin(); it != received_bvs.end(); ++it)
    boundary_values.insert(it->second.begin(), it->second.end());
}
//-----------------------------------------------------------------------------
void DirichletBC::get_boundary_values(Map& boundary_values,
                         std::string method) const
{
  // Create local data
  LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data, method);
}
//-----------------------------------------------------------------------------
void DirichletBC::zero(GenericMatrix& A) const
{
  // A map to hold the mapping from boundary dofs to boundary values
  Map boundary_values;

  // Create local data for application of boundary conditions
  LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data, _method);

  // Copy boundary value data to arrays
  std::vector<dolfin::la_index> dofs(boundary_values.size());
  Map::const_iterator bv;
  std::size_t i = 0;
  for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
    dofs[i++] = bv->first;

  // Modify linear system (A_ii = 1)
  A.zero(boundary_values.size(), dofs.data());

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
  //const std::size_t nrows = A.size(0); // should be equal to b.size()
  const std::size_t ncols = A.size(1); // should be equal to max possible dof+1

  std::pair<std::size_t, std::size_t> rows = A.local_range(0);

  std::vector<char> is_bc_dof(ncols);
  std::vector<double> bc_dof_val(ncols);
  for (Map::const_iterator bv = bv_map.begin();  bv != bv_map.end();  ++bv)
  {
    is_bc_dof[bv->first] = 1;
    bc_dof_val[bv->first] = bv->second;
  }

  // Scan through all columns of all rows, setting to zero if is_bc_dof[column]
  // At the same time, we collect corrections to the RHS

  std::vector<std::size_t> cols;
  std::vector<double> vals;
  std::vector<double> b_vals;
  std::vector<dolfin::la_index> b_rows;

  for (std::size_t row = rows.first; row < rows.second; row++)
  {
    // If diag_val is nonzero, the matrix is a diagonal block (nrows==ncols),
    // and we can set the whole BC row
    if (diag_val != 0.0 && is_bc_dof[row])
    {
      A.getrow(row, cols, vals);
      for (std::size_t j = 0; j < cols.size(); j++)
        vals[j] = (cols[j] == row)*diag_val;
      A.setrow(row, cols, vals);
      A.apply("insert");
      b.setitem(row, bc_dof_val[row]*diag_val);
    }
    else // Otherwise, we scan the row for BC columns
    {
      A.getrow(row, cols, vals);
      bool row_changed = false;
      for (std::size_t j = 0; j < cols.size(); j++)
      {
        const std::size_t col = cols[j];

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
const std::vector<std::pair<std::size_t, std::size_t> >& DirichletBC::markers() const
{
  return facets;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericFunction> DirichletBC::value() const
{
  return _g;
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
  std::size_t size = 1;
  const std::size_t rank = g->function_space().element()->value_rank();
  for (std::size_t i = 0; i < rank ; i++)
    size *= g->function_space().element()->value_dimension(i);
  double* g_values = new double[size];
  double* v_values = new double[size];

  // Get mesh
  const Mesh& mesh = _function_space->mesh();

  // Iterate over facets
  for (std::size_t f = 0; f < facets.size(); f++)
  {
    // Create cell and facet
    std::size_t cell_number  = facets[f].first;
    std::size_t facet_number = facets[f].second;
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
      for (std::size_t i = 0; i < size; i++)
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
  _g = reference_to_no_delete_pointer(g);
}
//-----------------------------------------------------------------------------
void DirichletBC::homogenize()
{
  const std::size_t value_rank = _g->value_rank();
  if (!value_rank)
  {
    boost::shared_ptr<Constant> zero(new Constant(0.0));
    set_value(zero);
  } else if (value_rank == 1)
  {
    const std::size_t value_dim = _g->value_dimension(0);
    std::vector<double> values(value_dim, 0.0);
    boost::shared_ptr<Constant> zero(new Constant(values));
    set_value(zero);
  } else
  {
    std::vector<std::size_t> value_shape;
    for (std::size_t i = 0; i < value_rank; i++)
      value_shape.push_back(_g->value_dimension(i));
    std::vector<double> values(_g->value_size(), 0.0);
    boost::shared_ptr<Constant> zero(new Constant(value_shape, values));
    set_value(zero);
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::set_value(boost::shared_ptr<const GenericFunction> g)
{
  _g = g;
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
  Timer timer("DirichletBC apply");

  // Check arguments
  check_arguments(A, b, x);

  // A map to hold the mapping from boundary dofs to boundary values
  Map boundary_values;

  // Create local data for application of boundary conditions
  LocalData data(*_function_space);

  // Compute dofs and values
  compute_bc(boundary_values, data, _method);

  // Copy boundary value data to arrays
  const std::size_t size = boundary_values.size();
  std::vector<dolfin::la_index> dofs(size);
  std::vector<double> values(size);
  Map::const_iterator bv;
  std::size_t counter = 0;
  for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
  {
    dofs[counter]     = bv->first;
    values[counter++] = bv->second;
  }

  // Modify boundary values for nonlinear problems
  if (x)
  {
    // Get values (these must reside in local portion (including ghost
    // values) of the vector
    std::vector<double> x_values(size);
    x->get_local(&x_values[0], dofs.size(), &dofs[0]);

    // Modify RHS entries
    for (std::size_t i = 0; i < size; i++)
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
      for (std::size_t i = 0; i < size; i++)
      {
        std::pair<std::size_t, std::size_t> ij(dofs[i], dofs[i]);
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
  dolfin_assert(_g);
  dolfin_assert(_function_space->element());

  // Check for common errors, message below might be cryptic
  if (_g->value_rank() == 0 && _function_space->element()->value_rank() == 1)
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "Expecting a vector-valued boundary value but given function is scalar");
  }
  if (_g->value_rank() == 1 && _function_space->element()->value_rank() == 0)
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "Expecting a scalar boundary value but given function is vector-valued");
  }

  // Check that value shape of boundary value
  if (_g->value_rank() != _function_space->element()->value_rank())
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "Illegal value rank (%d), expecting (%d)",
                 _g->value_rank(), _function_space->element()->value_rank());
  }
  for (std::size_t i = 0; i < _g->value_rank(); i++)
  {
    if (_g->value_dimension(i) != _function_space->element()->value_dimension(i))
    {
      dolfin_error("DirichletBC.cpp",
                   "create Dirichlet boundary condition",
                   "Illegal value dimension (%d), expecting (%d)",
                   _g->value_dimension(i),
                   _function_space->element()->value_dimension(i));
    }
  }

  // Check that boundary condition method is known
  if (methods.count(_method) == 0)
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "unknown method (\"%s\")", _method.c_str());
  }

  // Check that the mesh is ordered
  dolfin_assert(_function_space->mesh());
  if (!_function_space->mesh()->ordered())
  {
    dolfin_error("DirichletBC.cpp",
                 "create Dirichlet boundary condition",
                 "Mesh is not ordered according to the UFC numbering convention. Consider calling mesh.order()");
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::init_facets() const
{
  Timer timer("DirichletBC init facets");

  if (facets.size() > 0)
    return;

  if (_user_sub_domain)
    init_from_sub_domain(_user_sub_domain);
  else if (_user_mesh_function)
    init_from_mesh_function(*_user_mesh_function, _user_sub_domain_marker);
  else
    init_from_mesh(_user_sub_domain_marker);
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_sub_domain(boost::shared_ptr<const SubDomain> sub_domain) const
{
  dolfin_assert(facets.size() == 0);

  // FIXME: This can be made more efficient, we should be able to
  // FIXME: extract the facets without first creating a MeshFunction on
  // FIXME: the entire mesh and then extracting the subset. This is done
  // FIXME: mainly for convenience (we may reuse mark() in SubDomain).

  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Create mesh function for sub domain markers on facets
  const std::size_t dim = mesh.topology().dim();
  _function_space->mesh()->init(dim - 1);
  MeshFunction<std::size_t> sub_domains(mesh, dim - 1);

  // Set geometric dimension (needed for SWIG interface)
  sub_domain->_geometric_dimension = mesh.geometry().dim();

  // Mark everything as sub domain 1
  sub_domains = 1;

  // Mark the sub domain as sub domain 0
  sub_domain->mark(sub_domains, 0, _check_midpoint);

  // Initialize from mesh function
  init_from_mesh_function(sub_domains, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_mesh_function(const MeshFunction<std::size_t>& sub_domains,
                                          std::size_t sub_domain) const
{
  dolfin_assert(facets.size() == 0);
  dolfin_assert(_function_space->mesh());

  // Get mesh
  const Mesh& mesh = *_function_space->mesh();

  // Get restriction if any
  boost::shared_ptr<const Restriction> restriction
    = _function_space->dofmap()->restriction();

  // Make sure we have the facet - cell connectivity
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1, D);

  // Build set of boundary facets
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Skip facets not on this boundary
    if (sub_domains[*facet] != sub_domain)
      continue;

    // Get cell to which facet belongs. If mesh is restricted, make
    // sure we pick the right cell in case there are two.
    dolfin_assert(facet->num_entities(D) > 0);
    const unsigned int* cell_indices = facet->entities(D);
    std::size_t cell_index = 0;
    if (restriction && facet->num_entities(D) > 1)
    {
      if (restriction->contains(D, cell_indices[0]))
        cell_index = cell_indices[0];
      else if (restriction->contains(D, cell_indices[1]))
        cell_index = cell_indices[1];
      else
      {
        dolfin_error("DirichletBC.cpp",
                     "create Dirichlet boundary condition",
                     "Boundary facet is not adjacent to a cell inside the restriction.");
      }
    }
    else
    {
      cell_index = facet->entities(D)[0];
    }
    const Cell cell(mesh, cell_index);

    // Get local index of facet with respect to the cell
    const size_t facet_number = cell.index(*facet);

    // Copy data
    facets.push_back(std::pair<std::size_t, std::size_t>(cell.index(), facet_number));
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::init_from_mesh(std::size_t sub_domain) const
{
  dolfin_assert(facets.size() == 0);

  // For this to work, the mesh *needs* to be ordered according to
  // the UFC ordering before it gets here. So reordering the mesh
  // here will either have no effect (if the mesh is already ordered
  // or it won't do anything good (since the markers are wrong anyway).
  // In conclusion: we don't need to order the mesh here.

  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Assign domain numbers for each facet
  const std::size_t D = mesh.topology().dim();
  dolfin_assert(mesh.domains().markers(D - 1));
  const std::map<std::pair<std::size_t, std::size_t>, std::size_t>&
    markers = mesh.domains().markers(D - 1)->values();
  std::map<std::pair<std::size_t, std::size_t>, std::size_t>::const_iterator mark;
  for (mark = markers.begin(); mark != markers.end(); ++mark)
  {
    if (mark->second == sub_domain)
      facets.push_back(mark->first);
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc(Map& boundary_values, LocalData& data,
                             std::string method) const
{
  Timer timer("DirichletBC compute bc");

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
                                         LocalData& data) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_g);

  // Extract the list of facets where the BC should be applied
  init_facets();

  // Special case
  if (facets.size() == 0)
  {
    if (MPI::num_processes() == 1)
      warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get mesh and dofmap
  dolfin_assert(_function_space->mesh());
  dolfin_assert(_function_space->dofmap());
  const Mesh& mesh = *_function_space->mesh();
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Create UFC cell object
  UFCCell ufc_cell(mesh);

  // Iterate over facets
  dolfin_assert(_function_space->element());
  Progress p("Computing Dirichlet boundary values, topological search", facets.size());
  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    // Get cell number and local facet number
    const std::size_t cell_number  = facets[f].first;
    const std::size_t facet_number = facets[f].second;

    // Create cell
    Cell cell(mesh, cell_number);
    ufc_cell.update(cell, facet_number);

    // Restrict coefficient to cell
    _g->restrict(&data.w[0], *_function_space->element(), cell, ufc_cell);

    // Tabulate dofs on cell
    const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(cell.index());

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(data.facet_dofs, facet_number);

    // Pick values for facet
    for (std::size_t i = 0; i < dofmap.num_facet_dofs(); i++)
    {
      const std::size_t global_dof = cell_dofs[data.facet_dofs[i]];
      const double value = data.w[data.facet_dofs[i]];
      boundary_values[global_dof] = value;
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_geometric(Map& boundary_values,
                                       LocalData& data) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  dolfin_assert(_g);

  // Extract the list of facets where the BC *might* be applied
  init_facets();

  // Special case
  if (facets.size() == 0)
  {
    if (MPI::num_processes() == 1)
      warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Get mesh and dofmap
  dolfin_assert(_function_space->mesh());
  dolfin_assert(_function_space->dofmap());
  const Mesh& mesh = *_function_space->mesh();
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Initialize facets, needed for geometric search
  log(TRACE, "Computing facets, needed for geometric application of boundary conditions.");
  mesh.init(mesh.topology().dim() - 1);

  // Speed up the computations by only visiting (most) dofs once
  RangedIndexSet already_visited(dofmap.is_view()
                                 ? std::pair<std::size_t, std::size_t>(0, 0)
                                 : dofmap.ownership_range());

  // Iterate over facets
  Progress p("Computing Dirichlet boundary values, geometric search", facets.size());
  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    // Get cell number and local facet number
    const std::size_t cell_number = facets[f].first;
    const std::size_t facet_number = facets[f].second;

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

        bool tabulated = false;
        bool interpolated = false;

        // Tabulate dofs on cell
        const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(c->index());

        // Loop over all dofs on cell
        for (std::size_t i = 0; i < cell_dofs.size(); ++i)
        {
          const std::size_t global_dof = cell_dofs[i];

          // Tabulate coordinates if not already done
          if (!tabulated)
          {
            dofmap.tabulate_coordinates(data.coordinates, ufc_cell);
            tabulated = true;
          }

          // Check if the coordinates are on current facet and thus on boundary
          if (!on_facet(&(data.coordinates[i][0]), facet))
            continue;

          // Skip already checked dofs
          if (already_visited.in_range(global_dof) && !already_visited.insert(global_dof))
            continue;

          // Restrict if not already done
          if (!interpolated)
          {
            _g->restrict(&data.w[0], *_function_space->element(), cell, ufc_cell);
            interpolated = true;
          }

          // Set boundary value
          const double value = data.w[i];
          boundary_values[global_dof] = value;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::compute_bc_pointwise(Map& boundary_values,
                                       LocalData& data) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  dolfin_assert(_g);

  if (!_user_sub_domain)
    dolfin_error("DirichletBC.cpp",
                 "computing Dirichlet boundary values, pointwise search",
                 "A SubDomain is required for pointwise search");

  // Get mesh and dofmap
  dolfin_assert(_function_space->mesh());
  dolfin_assert(_function_space->dofmap());
  const Mesh& mesh = *_function_space->mesh();
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Geometric dim
  const std::size_t gdim = mesh.geometry().dim();

  // Create UFC cell object
  UFCCell ufc_cell(mesh);

  // Speed up the computations by only visiting (most) dofs once
  RangedIndexSet already_visited(dofmap.is_view()
                                 ? std::pair<std::size_t, std::size_t>(0,0)
                                 : dofmap.ownership_range());

  // Iterate over cells
  Progress p("Computing Dirichlet boundary values, pointwise search", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    ufc_cell.update(*cell);

    // Tabulate coordinates of dofs on cell
    dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

    // Tabulate dofs on cell
    const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(cell->index());

    // Interpolate function only once and only on cells where necessary
    bool already_interpolated = false;

    // Loop all dofs on cell
    for (std::size_t i = 0; i < dofmap.cell_dimension(cell->index()); ++i)
    {
      const std::size_t global_dof = cell_dofs[i];

      // Skip already checked dofs
      if (already_visited.in_range(global_dof) && !already_visited.insert(global_dof))
        continue;

      // Check if the coordinates are part of the sub domain (calls user-defined 'inside' function)
      Array<double> x(gdim, &data.coordinates[i][0]);
      if (!_user_sub_domain->inside(x, false))
        continue;

      if (!already_interpolated)
      {
        already_interpolated = true;

        // Restrict coefficient to cell
        _g->restrict(&data.w[0], *_function_space->element(), *cell, ufc_cell);
      }

      // Set boundary value
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
void DirichletBC::check_arguments(GenericMatrix* A, GenericVector* b,
                                  const GenericVector* x) const
{
  dolfin_assert(_function_space);

  // Check matrix and vector dimensions
  if (A && x && A->size(0) != x->size())
  {
    dolfin_error("BoundaryCondition.cpp",
                 "apply boundary condition",
                 "Matrix dimension (%d rows) does not match vector dimension (%d) for application of boundary conditions",
                 A->size(0), x->size());
  }

  if (A && b && A->size(0) != b->size())
  {
    dolfin_error("BoundaryCondition.cpp",
                 "apply boundary condition",
                 "Matrix dimension (%d rows) does not match vector dimension (%d) for application of boundary conditions",
                 A->size(0), b->size());
  }

  if (x && b && x->size() != b->size())
  {
    dolfin_error("BoundaryCondition.cpp",
                 "apply boundary condition",
                 "Vector dimension (%d rows) does not match vector dimension (%d) for application of boundary conditions",
                 x->size(), b->size());
  }

  // Check dimension of function space
  if (A && A->size(0) < _function_space->dim())
  {
    dolfin_error("BoundaryCondition.cpp",
                 "apply boundary condition",
                 "Dimension of function space (%d) too large for application of boundary conditions to linear system (%d rows)",
                 _function_space->dim(), A->size(0));
  }

  if (x && x->size() < _function_space->dim())
  {
    dolfin_error("BoundaryCondition.cpp",
                 "apply boundary condition",
                 "Dimension of function space (%d) too large for application to boundary conditions linear system (%d rows)",
                 _function_space->dim(), x->size());
  }

  if (b && b->size() < _function_space->dim())
  {
    dolfin_error("BoundaryCondition.cpp",
                 "apply boundary condition",
                 "Dimension of function space (%d) too large for application to boundary conditions linear system (%d rows)",
                 _function_space->dim(), b->size());
  }

  // FIXME: Check case A.size() > _function_space->dim() for subspaces
}
//-----------------------------------------------------------------------------
DirichletBC::LocalData::LocalData(const FunctionSpace& V)
  : w(V.dofmap()->max_cell_dimension(), 0.0),
    facet_dofs(V.dofmap()->num_facet_dofs(), 0),
    coordinates(boost::extents[V.dofmap()->max_cell_dimension()][V.mesh()->geometry().dim()])
{
  // Do nothing
}
//-----------------------------------------------------------------------------
