// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells 2007
// Modified by Johan Hake 2009
//
// First added:  2007-07-08
// Last changed: 2011-11-14

#include <map>
#include <utility>
#include <vector>
#include <boost/unordered_map.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/log.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Vertex.h>
#include "BoundaryCondition.h"
#include "FiniteElement.h"
#include "GenericDofMap.h"
#include "UFCMesh.h"
#include "PeriodicBC.h"

using namespace dolfin;

// Comparison operator for hashing coordinates. Note that two
// coordinates are considered equal if equal to within round-off.
struct lt_coordinate
{
  bool operator() (const std::vector<double>& x, const std::vector<double>& y) const
  {
    unsigned int n = std::max(x.size(), y.size());
    for (unsigned int i = 0; i < n; ++i)
    {
      double xx = 0.0;
      double yy = 0.0;

      if (i < x.size())
        xx = x[i];
      if (i < y.size())
        yy = y[i];

      if (xx < (yy - DOLFIN_EPS))
        return true;
      else if (xx > (yy + DOLFIN_EPS))
        return false;
    }

    return false;
  }
};


// FIXME: Change this to boost::unordered_map

// Mapping from coordinates to dof pairs
// coordinate_map maps a coordinate to a pair of (master dof, slave dof)
// and each master dof/slave dof is a pair of the degree of freedom
// index AND its parallel owner.
typedef std::pair<int, int> dof_data;
typedef std::pair<dof_data, dof_data> dof_pair;
typedef std::map<std::vector<double>, dof_pair, lt_coordinate> coordinate_map;
typedef coordinate_map::iterator coordinate_iterator;

// To apply periodic BCs in parallel, we'll need to reduce the coordinate
// map onto each processor, and have that apply the BCs. This function
// below provides the reduction operator used in the MPI reduce.
struct merge_coordinate_map
{
  coordinate_map operator() (coordinate_map x, coordinate_map y)
  {
    coordinate_map z;
    for (coordinate_iterator it = x.begin(); it != x.end(); ++it)
      z[it->first] = it->second;

    for (coordinate_iterator it = y.begin(); it != y.end(); ++it)
    {
      coordinate_iterator match = z.find(it->first);
      if (match != z.end())
      {
        // Copy the degree of freedom indices and their parallel owners
        match->second.first.first
          = std::max(it->second.first.first, match->second.first.first);
        match->second.first.second
          = std::max(it->second.first.second, match->second.first.second);
        match->second.second.first
          = std::max(it->second.second.first, match->second.second.first);
        match->second.second.second
          = std::max(it->second.second.second, match->second.second.second);
      }
      else
        z[it->first] = it->second;
    }

    return z;
  }
};
//-----------------------------------------------------------------------------
PeriodicBC::PeriodicBC(const FunctionSpace& V, const SubDomain& sub_domain)
  : BoundaryCondition(V),
    _sub_domain(reference_to_no_delete_pointer(sub_domain))
{
  // Build mapping between dofs
  rebuild();
}
//-----------------------------------------------------------------------------
PeriodicBC::PeriodicBC(boost::shared_ptr<const FunctionSpace> V,
                       boost::shared_ptr<const SubDomain> sub_domain)
  : BoundaryCondition(V), _sub_domain(sub_domain)
{
  // Build mapping between dofs
  rebuild();
}
//-----------------------------------------------------------------------------
PeriodicBC::~PeriodicBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix& A) const
{
  apply(&A, 0, 0);
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericVector& b) const
{
  apply(0, &b, 0);
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix& A, GenericVector& b) const
{
  apply(&A, &b, 0);
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericVector& b, const GenericVector& x) const
{
  apply(0, &b, &x);
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix& A, GenericVector& b,
                       const GenericVector& x) const
{
  apply(&A, &b, &x);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const SubDomain> PeriodicBC::sub_domain() const
{
  return _sub_domain;
}
//-----------------------------------------------------------------------------
void PeriodicBC::rebuild()
{
  cout << "Building mapping between periodic degrees of freedom." << endl;

  // Build list of dof pairs
  std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > > dof_pairs;
  compute_dof_pairs(dof_pairs);

  // Resize arrays
  const std::size_t num_dof_pairs = dof_pairs.size();
  master_dofs.resize(num_dof_pairs);
  slave_dofs.resize(num_dof_pairs);
  if (MPI::num_processes() > 1)
  {
    master_owners.resize(num_dof_pairs);
    slave_owners.resize(num_dof_pairs);
  }

  // Store master and slave dofs
  for (std::size_t i = 0; i < dof_pairs.size(); ++i)
  {
    master_dofs[i] = dof_pairs[i].first.first;
    slave_dofs[i]  = dof_pairs[i].second.first;
  }

  if (MPI::num_processes() > 1)
  {
    // Store master and slave dofs
    master_owners.resize(num_dof_pairs);
    slave_owners.resize(num_dof_pairs);
    for (std::size_t i = 0; i < dof_pairs.size(); ++i)
    {
      master_owners[i] = dof_pairs[i].first.second;
      slave_owners[i]  = dof_pairs[i].second.second;
      dolfin_assert(master_owners[i] < MPI::num_processes());
      dolfin_assert(slave_owners[i] < MPI::num_processes());
    }
  }
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix* A, GenericVector* b,
                       const GenericVector* x) const
{

  // The current handling of periodic boundary conditions adds entries in the matrix that
  // were not accounted for in the original sparsity pattern. This causes PETSc to crash,
  // since by default it will not let you do this (as it requires extra memory allocation
  // and is very inefficient). However, until that's fixed, this hack should stay in, so
  // that we can run these problems at all.
  #ifdef HAS_PETSC
  if (A && has_type<PETScMatrix>(*A))
  {
    PETScMatrix& petsc_A = A->down_cast<PETScMatrix>();
    MatSetOption(*petsc_A.mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  }
  #endif

  if (MPI::num_processes() > 1)
  {
    parallel_apply(A, b, x);
    return;
  }

  const std::size_t num_dof_pairs = master_dofs.size();
  dolfin_assert(num_dof_pairs > 0);

  log(PROGRESS, "Applying periodic boundary conditions to linear system.");

  // Check arguments
  check_arguments(A, b, x);

  // Add slave rows to master rows
  for (std::size_t i = 0; i < num_dof_pairs; ++i)
  {
    // Add slave row to master row in A
    if (A)
    {
      std::vector<std::size_t> columns;
      std::vector<double> values;
      A->getrow(slave_dofs[i], columns, values);
      std::vector<dolfin::la_index> _columns(columns.begin(), columns.end());
      A->add(values.data(), 1, &master_dofs[i], _columns.size(), _columns.data());
      A->apply("add");
    }

    // Add slave row to master row in b
    if (b)
    {
      double value;
      b->get_local(&value, 1, &slave_dofs[i]);
      b->add(&value, 1, &master_dofs[i]);
      b->apply("add");
    }
  }

  // Insert 1 in master columns and -1 in slave columns in slave rows
  if (A)
  {
    // Zero out slave rows
    A->zero(num_dof_pairs, &slave_dofs[0]);
    A->apply("insert");

    // Insert 1 and -1
    dolfin::la_index cols[2];
    const double vals[2] = {1.0, -1.0};
    for (std::size_t i = 0; i < num_dof_pairs; ++i)
    {
      const dolfin::la_index row = slave_dofs[i];
      cols[0] = master_dofs[i];
      cols[1] = slave_dofs[i];
      A->set(vals, 1, &row, 2, cols);
    }
    A->apply("insert");
  }

  // Modify boundary values for nonlinear problems
  std::vector<double> rhs_values_slave(num_dof_pairs, 0.0);
  if (x)
  {
    std::vector<double> rhs_values_master(num_dof_pairs);
    x->get_local(&rhs_values_master[0], num_dof_pairs, &master_dofs[0]);
    x->get_local(&rhs_values_slave[0],  num_dof_pairs, &slave_dofs[0]);
    for (std::size_t i = 0; i < num_dof_pairs; i++)
      rhs_values_slave[i] = rhs_values_master[i] - rhs_values_slave[i];
  }

  // Zero slave rows in right-hand side
  if (b)
  {
    b->set(&rhs_values_slave[0], num_dof_pairs, &slave_dofs[0]);
    b->apply("insert");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::pair<std::size_t, std::size_t>,
    std::pair<std::size_t, std::size_t> > > PeriodicBC::compute_dof_pairs() const
{
  std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > > dof_pairs;
  compute_dof_pairs(dof_pairs);
  return dof_pairs;
}
//-----------------------------------------------------------------------------
void PeriodicBC::compute_dof_pairs(
  std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > >& dof_pairs) const
{
  dof_pairs.clear();

  // Get function space
  dolfin_assert(function_space());
  const FunctionSpace& V = *function_space();

  // Compute pairs
  extract_dof_pairs(V, dof_pairs);
}
//-----------------------------------------------------------------------------
void PeriodicBC::extract_dof_pairs(const FunctionSpace& V,
   std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > >& dof_pairs) const
{
  // Call recursively for subspaces, should work for arbitrary nesting
  dolfin_assert(V.element());
  const std::size_t num_sub_spaces = V.element()->num_sub_elements();
  if (num_sub_spaces > 0)
  {
    for (std::size_t i = 0; i < num_sub_spaces; ++i)
    {
      cout << "Extracting matching degrees of freedom for sub space " << i << "." << endl;
      extract_dof_pairs((*V[i]), dof_pairs);
    }
    return;
  }

  // We should have a non-mixed element by now
  dolfin_assert(V.element()->num_sub_elements() == 0);

  dolfin_assert(_function_space);
  dolfin_assert(_function_space->dofmap());
  const std::pair<std::size_t, std::size_t> ownership_range
      = _function_space->dofmap()->ownership_range();

  // Get mesh and dofmap
  dolfin_assert(V.mesh());
  dolfin_assert(V.dofmap());
  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = *V.dofmap();

  // Get dimensions
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Set geometric dimension (needed for SWIG interface)
  dolfin_assert(_sub_domain);
  _sub_domain->_geometric_dimension = gdim;

  // Make sure we have the facet - cell connectivity
  mesh.init(tdim - 1, tdim);

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*_function_space);

  // Arrays used for mapping coordinates
  std::vector<double> x(gdim);
  std::vector<double> y(gdim);

  // Wrap x and y (Array view of x and y)
  Array<double> _x(gdim, &x[0]);
  Array<double> _y(gdim, &y[0]);

  // Mapping from coordinates to dof pairs
  coordinate_map coordinate_dof_pairs;

  // MPI process number
  const std::size_t process_number = MPI::process_number();

  // Iterate over all facets of the mesh (not only the boundary)
  Progress p("Applying periodic boundary conditions", mesh.size(tdim - 1));
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Get cell (there may be two, but pick first) and local facet index
    const Cell cell(mesh, facet->entities(tdim)[0]);
    const std::size_t local_facet = cell.index(*facet);

    // Tabulate dofs and coordinates on cell
    const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(cell.index());
    dofmap.tabulate_coordinates(data.coordinates, cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(&data.facet_dofs[0], local_facet);

    // Iterate over facet dofs
    for (std::size_t i = 0; i < dofmap.num_facet_dofs(); ++i)
    {
      // Get dof and coordinate of dof
      const std::size_t local_dof = data.facet_dofs[i];
      const int global_dof = cell_dofs[local_dof];

      // Only handle dofs that are owned by this process
      if (global_dof >= (int) ownership_range.first && global_dof < (int) ownership_range.second)
      {
        std::copy(data.coordinates[local_dof].begin(),
                  data.coordinates[local_dof].end(), x.begin());

        // Set y = x
        y.assign(x.begin(), x.end());

        // Map coordinate from H to G
        _sub_domain->map(_x, _y);

        // Check if coordinate is inside the domain G or in H
        const bool on_boundary = (facet->num_entities(tdim) == 1);
        if (_sub_domain->inside(_x, on_boundary))
        {
          // Check if coordinate exists from before
          coordinate_iterator it = coordinate_dof_pairs.find(x);
          if (it != coordinate_dof_pairs.end())
          {
            // Exists from before, so set dof associated with x
            it->second.first = dof_data(global_dof, process_number);
          }
          else
          {
            // Doesn't exist, so create new pair (with illegal second value)
            dof_data g_dofs(global_dof, process_number);
            dof_data l_dofs(-1, -1);
            dof_pair pair(g_dofs, l_dofs);
            coordinate_dof_pairs[x] = pair;
          }
        }
        else if (_sub_domain->inside(_y, on_boundary))
        {
          // Copy coordinate to std::vector
          x.assign(y.begin(), y.end());

          // Check if coordinate exists from before
          coordinate_iterator it = coordinate_dof_pairs.find(x);
          if (it != coordinate_dof_pairs.end())
          {
            // Exists from before, so set dof associated with y
            it->second.second = dof_data(global_dof, process_number);
          }
          else
          {
            // Doesn't exist, so create new pair (with illegal first value)
            dof_data g_dofs(-1, -1);
            dof_data l_dofs(global_dof, process_number);
            dof_pair pair(g_dofs, l_dofs);
            coordinate_dof_pairs[x] = pair;
          }
        }
      }
    }

    p++;
  }

  #ifdef HAS_MPI
  coordinate_map final_coordinate_dof_pairs
      = MPI::all_reduce(coordinate_dof_pairs, merge_coordinate_map());
  #else
  coordinate_map final_coordinate_dof_pairs = coordinate_dof_pairs;
  #endif

  // Fill up list of dof pairs
  for (coordinate_iterator it = final_coordinate_dof_pairs.begin();
                                it != final_coordinate_dof_pairs.end(); ++it)
  {
    // Check dofs
    if (it->second.first.first == -1 || it->second.second.first == -1)
    {
      cout << "At coordinate: x =";
      for (std::size_t j = 0; j < gdim; ++j)
        cout << " " << it->first[j];
      cout << endl;
      dolfin_error("PeriodicBC.cpp",
                   "apply periodic boundary condition",
                   "Could not find a pair of matching degrees of freedom");
    }

    if (it->second.first.second >= (int) MPI::num_processes() || it->second.first.second < 0)
    {
      cout << "At coordinate: x =";
      for (std::size_t j = 0; j < gdim; ++j)
        cout << " " << it->first[j];
      cout << endl;
      cout << "degree of freedom: " << it->second.first.first << endl;
      cout << "master owner: " << it->second.first.second << endl;
      dolfin_error("PeriodicBC.cpp",
                   "apply periodic boundary condition",
                   "Invalid master owner");
    }

    if (it->second.second.second >= (int) MPI::num_processes() || it->second.second.second < 0)
    {
      cout << "At coordinate: x =";
      for (std::size_t j = 0; j < gdim; ++j)
        cout << " " << it->first[j];
      cout << endl;
      cout << "degree of freedom: " << it->second.second.first << endl;
      cout << "slave owner: " << it->second.second.second << endl;
      dolfin_error("PeriodicBC.cpp",
                   "apply periodic boundary condition",
                   "Invalid slave owner");
    }

    // Store dofs
    dof_pairs.push_back(it->second);
  }
}
//-----------------------------------------------------------------------------
void PeriodicBC::parallel_apply(GenericMatrix* A, GenericVector* b,
                                const GenericVector* x) const
{
  const std::size_t num_dof_pairs = master_dofs.size();
  dolfin_assert(num_dof_pairs > 0);

  log(PROGRESS, "Applying periodic boundary conditions to linear system.");

  // Check arguments
  check_arguments(A, b, x);

  // Add the slave equations to its corresponding master equation
  if (A)
  {
    // Data for a row -- master dof, cols, vals
    typedef boost::tuples::tuple<dolfin::la_index, std::vector<std::size_t>, std::vector<double> > row_type;

    // All rows to be sent to a particular process
    typedef std::vector<row_type> rows_type;

    // Which processor should the row data be sent to?
    typedef std::map<std::size_t, rows_type> row_map_type;

    row_map_type row_map;
    //const std::pair<std::size_t, std::size_t> local_range = A->local_range(0);
    const std::pair<dolfin::la_index, dolfin::la_index> local_range(A->local_range(0).first, A->local_range(0).second);
    std::set<std::size_t> communicating_processors;

    for (std::size_t i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= local_range.first && slave_dofs[i] < local_range.second)
      {
        std::vector<std::size_t> columns;
        std::vector<double> values;
        A->getrow(slave_dofs[i], columns, values);
        row_type row = row_type(master_dofs[i], columns, values);

        std::size_t master_owner = master_owners[i];

        if (row_map.find(master_owner) == row_map.end())
        {
          rows_type rows_list; rows_list.push_back(row);
          row_map[master_owner] = rows_list;
        }
        else
          row_map[master_owner].push_back(row);

        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= local_range.first && master_dofs[i] < local_range.second)
      {
        std::size_t slave_owner = slave_owners[i];
        communicating_processors.insert(slave_owner);
      }
    }

    row_map_type received_rows;
    MPI::distribute(communicating_processors, row_map, received_rows);

    for (row_map_type::const_iterator proc_it = received_rows.begin();
            proc_it != received_rows.end(); ++proc_it)
    {
      rows_type rows = proc_it->second;
      for (rows_type::const_iterator row_it = rows.begin(); row_it != rows.end(); ++row_it)
      {
        row_type row = *row_it;
        const dolfin::la_index& master_dof = row.get<0>();
        const std::vector<dolfin::la_index> columns(row.get<1>().begin(), row.get<1>().end());
        const std::vector<double>& values = row.get<2>();
        A->add(&values[0], 1, &master_dof, columns.size(), &columns[0]);
      }
    }

    A->apply("add"); // effect the adding on of the slave rows
  }

  if (A) // now set the slave rows
  {
    A->zero(num_dof_pairs, &slave_dofs[0]);
    A->apply("insert");

    // Insert 1 and -1 in the master and slave column of each slave row
    dolfin::la_index cols[2];
    const double vals[2] = {1.0, -1.0};
    const std::pair<dolfin::la_index, dolfin::la_index> local_range(A->local_range(0).first, A->local_range(0).second);
    for (std::size_t i = 0; i < num_dof_pairs; ++i)
    {
      if (slave_dofs[i] >= local_range.first && slave_dofs[i] < local_range.second)
      {
        const dolfin::la_index row = slave_dofs[i];
        cols[0] = master_dofs[i];
        cols[1] = slave_dofs[i];
        A->set(vals, 1, &row, 2, cols);
      }
    }

    A->apply("insert");
  }

  // Modify boundary values for nonlinear problems.
  // This doesn't change x, it only changes what the slave
  // entries of b will be set to.
  std::vector<double> rhs_values_slave(num_dof_pairs, 0.0);
  if (x)
  {
    typedef boost::tuples::tuple<dolfin::la_index, double> x_type; // dof index, value
    typedef std::vector<x_type> xs_type; // all x-information to be sent to a particular process
    typedef std::map<std::size_t, xs_type> x_map_type; // processor to send it to

    std::set<std::size_t> communicating_processors;
    x_map_type x_map;
    const std::pair<dolfin::la_index, dolfin::la_index> local_range(x->local_range().first, x->local_range().second);
    for (std::size_t i = 0; i < num_dof_pairs; ++i)
    {
      if (slave_dofs[i] >= local_range.first && slave_dofs[i] < local_range.second)
      {
        double value;
        x->get_local(&value, 1, &slave_dofs[i]);
        x_type data = x_type(i, value);
        std::size_t master_owner = master_owners[i];
        if (x_map.find(master_owner) == x_map.end())
        {
          xs_type x_list; x_list.push_back(data);
          x_map[master_owner] = x_list;
        }
        else
          x_map[master_owner].push_back(data);

        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= local_range.first && master_dofs[i] < local_range.second)
      {
        std::size_t slave_owner = slave_owners[i];
        communicating_processors.insert(slave_owner);
      }
    }

    x_map_type received_x;
    MPI::distribute(communicating_processors, x_map, received_x);
    for (x_map_type::const_iterator proc_it = received_x.begin(); proc_it != received_x.end(); ++proc_it)
    {
      xs_type xs = proc_it->second;
      for (xs_type::const_iterator x_it = xs.begin(); x_it != xs.end(); ++x_it)
      {
        const dolfin::la_index dof_idx = x_it->get<0>();
        const double slave_value = x_it->get<1>();
        double master_value;
        x->get_local(&master_value, 1, &master_dofs[dof_idx]);
        rhs_values_slave[dof_idx] = master_value - slave_value;
      }
    }
  }

  // Add the slave equations to the master equations, then set the slave
  // rows to the appropriate values.
  if (b)
  {
    typedef boost::tuples::tuple<std::size_t, double> vec_type; // master dof, value that should be added
    typedef std::vector<vec_type> vecs_type; // all vec_types that should be sent to a particular process
    typedef std::map<std::size_t, vecs_type> vec_map_type; // which processor should the vec_data be sent to?

    vec_map_type vec_map;
    const std::pair<dolfin::la_index, dolfin::la_index> local_range(b->local_range().first, b->local_range().second);
    std::set<std::size_t> communicating_processors;
    for (std::size_t i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= local_range.first && slave_dofs[i] < local_range.second)
      {
        double value;
        b->get_local(&value, 1, &slave_dofs[i]);
        vec_type data = vec_type(master_dofs[i], value);

        std::size_t master_owner = master_owners[i];
        if (vec_map.find(master_owner) == vec_map.end())
        {
          vecs_type list; list.push_back(data);
          vec_map[master_owner] = list;
        }
        else
          vec_map[master_owner].push_back(data);

        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= local_range.first && master_dofs[i] < local_range.second)
      {
        const std::size_t slave_owner = slave_owners[i];
        communicating_processors.insert(slave_owner);
      }
    }

    vec_map_type received_vecs;
    MPI::distribute(communicating_processors, vec_map, received_vecs);

    for (vec_map_type::const_iterator proc_it = received_vecs.begin();
            proc_it != received_vecs.end(); ++proc_it)
    {
      vecs_type vecs = proc_it->second;
      for (vecs_type::const_iterator vec_it = vecs.begin(); vec_it != vecs.end(); ++vec_it)
      {
        const dolfin::la_index master_dof   = vec_it->get<0>();
        const double value      = vec_it->get<1>();
        b->add(&value, 1, &master_dof);
      }
    }

    b->apply("add"); // effect the adding on of the slave rows
  }

  if (b) // now zero the slave rows
  {
    const std::pair<dolfin::la_index, dolfin::la_index> local_range(b->local_range().first, b->local_range().second);
    for (std::size_t i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= local_range.first && slave_dofs[i] < local_range.second)
        b->set(&rhs_values_slave[i], 1, &slave_dofs[i]);
    }
    b->apply("insert");
  }
}
//-----------------------------------------------------------------------------
