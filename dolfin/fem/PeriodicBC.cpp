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
#include <vector>
#include <boost/unordered_map.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/log.h>
#include <dolfin/la/GenericMatrix.h>
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
// and each master dof/slave dof is a pair of the degree of freedom index AND its parallel owner.
typedef std::pair<int, int> dof_data;
typedef std::pair<dof_data, dof_data> dof_pair;
typedef std::map<std::vector<double>, dof_pair, lt_coordinate> coordinate_map;
typedef coordinate_map::iterator coordinate_iterator;

// To apply periodic BCs in parallel, we'll need to reduce the coordinate map onto each
// processor, and have that apply the BCs. This function below provides the reduction
// operator used in the MPI reduce.
struct merge_coordinate_map
{
  coordinate_map operator() (coordinate_map x, coordinate_map y)
  {
    coordinate_map z;

    for (coordinate_iterator it = x.begin(); it != x.end(); ++it)
    {
      z[it->first] = it->second;
    }

    for (coordinate_iterator it = y.begin(); it != y.end(); ++it)
    {
      coordinate_iterator match = z.find(it->first);
      if (match != z.end())
      {
        // Copy the degree of freedom indices and their parallel owners
        match->second.first.first  = std::max(it->second.first.first, match->second.first.first);
        match->second.first.second  = std::max(it->second.first.second, match->second.first.second);
        match->second.second.first  = std::max(it->second.second.first, match->second.second.first);
        match->second.second.second  = std::max(it->second.second.second, match->second.second.second);
      }
      else
      {
        z[it->first] = it->second;
      }
    }

    return z;
  }
};

//-----------------------------------------------------------------------------
PeriodicBC::PeriodicBC(const FunctionSpace& V,
                       const SubDomain& sub_domain)
  : BoundaryCondition(V), sub_domain(reference_to_no_delete_pointer(sub_domain)),
    num_dof_pairs(0), master_dofs(0), slave_dofs(0),
    rhs_values_master(0), rhs_values_slave(0)
{
  // Build mapping
  rebuild();
}
//-----------------------------------------------------------------------------
PeriodicBC::PeriodicBC(boost::shared_ptr<const FunctionSpace> V,
                       boost::shared_ptr<const SubDomain> sub_domain)
  : BoundaryCondition(V), sub_domain(sub_domain),
    num_dof_pairs(0), master_dofs(0), slave_dofs(0),
    rhs_values_master(0), rhs_values_slave(0)
{
  // Build mapping
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
void PeriodicBC::apply(GenericMatrix& A,
                       GenericVector& b,
                       const GenericVector& x) const
{
  apply(&A, &b, &x);
}
//-----------------------------------------------------------------------------
void PeriodicBC::rebuild()
{
  dolfin_assert(_function_space);

  cout << "Building mapping between periodic degrees of freedom." << endl;

  // Build list of dof pairs
  std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > > dof_pairs;
  extract_dof_pairs(*_function_space, dof_pairs);

  // Resize arrays
  num_dof_pairs = dof_pairs.size();
  master_dofs.resize(num_dof_pairs);
  slave_dofs.resize(num_dof_pairs);
  rhs_values_master.resize(num_dof_pairs);
  rhs_values_slave.resize(num_dof_pairs);
  if (MPI::num_processes() > 1)
  {
    master_owners.resize(num_dof_pairs);
    slave_owners.resize(num_dof_pairs);
  }

  // Store master and slave dofs
  for (uint i = 0; i < dof_pairs.size(); ++i)
  {
    master_dofs[i] = dof_pairs[i].first.first;
    slave_dofs[i] = dof_pairs[i].second.first;
  }

  if (MPI::num_processes() > 1)
  {
    master_owners.resize(num_dof_pairs);
    slave_owners.resize(num_dof_pairs);
    // Store master and slave dofs
    for (uint i = 0; i < dof_pairs.size(); ++i)
    {
      master_owners[i] = dof_pairs[i].first.second;
      slave_owners[i] = dof_pairs[i].second.second;
      dolfin_assert(master_owners[i] < MPI::num_processes());
      dolfin_assert(slave_owners[i] < MPI::num_processes());
    }
  }

  std::fill(rhs_values_master.begin(), rhs_values_master.end(), 0.0);
  std::fill(rhs_values_slave.begin(), rhs_values_slave.end(), 0.0);
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix* A,
                       GenericVector* b,
                       const GenericVector* x) const
{

  if (MPI::num_processes() > 1)
  {
    parallel_apply(A, b, x);
    return;
  }

  dolfin_assert(num_dof_pairs > 0);

  log(PROGRESS, "Applying periodic boundary conditions to linear system.");

  // Check arguments
  check_arguments(A, b, x);

  // Add slave rows to master rows
  for (uint i = 0; i < num_dof_pairs; ++i)
  {
    // Add slave row to master row in A
    if (A)
    {
      std::vector<uint> columns;
      std::vector<double> values;
      A->getrow(slave_dofs[i], columns, values);
      A->add(&values[0], 1, &master_dofs[i], columns.size(), &columns[0]);
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
    uint cols[2];
    double vals[2] = {1.0, -1.0};
    for (uint i = 0; i < num_dof_pairs; ++i)
    {
      const uint row = slave_dofs[i];
      cols[0] = master_dofs[i];
      cols[1] = slave_dofs[i];
      A->set(vals, 1, &row, 2, cols);
    }
    A->apply("insert");
  }

  // Modify boundary values for nonlinear problems
  if (x)
  {
    x->get_local(&rhs_values_master[0], num_dof_pairs, &master_dofs[0]);
    x->get_local(&rhs_values_slave[0],  num_dof_pairs, &slave_dofs[0]);
    for (uint i = 0; i < num_dof_pairs; i++)
      rhs_values_slave[i] = rhs_values_master[i] - rhs_values_slave[i];
  }
  else
    std::fill(rhs_values_slave.begin(), rhs_values_slave.end(), 0.0);

  // Zero slave rows in right-hand side
  if (b)
  {
    b->set(&rhs_values_slave[0], num_dof_pairs, &slave_dofs[0]);
    b->apply("insert");
  }
}
//-----------------------------------------------------------------------------
void PeriodicBC::extract_dof_pairs(const FunctionSpace& function_space,
                                   std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >& dof_pairs)
{
  dolfin_assert(function_space.element());

  // Call recursively for subspaces, should work for arbitrary nesting
  const uint num_sub_spaces = function_space.element()->num_sub_elements();
  if (num_sub_spaces > 0)
  {
    for (uint i = 0; i < num_sub_spaces; ++i)
    {
      cout << "Extracting matching degrees of freedom for sub space " << i << "." << endl;
      extract_dof_pairs((*function_space[i]), dof_pairs);
    }
    return;
  }

  // Assuming we have a non-mixed element
  dolfin_assert(function_space.element()->num_sub_elements() == 0);

  // Get mesh and dofmap
  dolfin_assert(function_space.mesh());
  dolfin_assert(function_space.dofmap());
  const Mesh& mesh = *function_space.mesh();
  const GenericDofMap& dofmap = *function_space.dofmap();

  // Get dimensions
  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();

  // Set geometric dimension (needed for SWIG interface)
  sub_domain->_geometric_dimension = gdim;

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

  // Iterate over all facets of the mesh (not only the boundary)
  Progress p("Applying periodic boundary conditions", mesh.size(tdim - 1));
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Get cell (there may be two, but pick first) and local facet index
    Cell cell(mesh, facet->entities(tdim)[0]);
    const uint local_facet = cell.index(*facet);

    // Tabulate dofs and coordinates on cell
    const std::vector<uint>& cell_dofs = dofmap.cell_dofs(cell.index());
    dofmap.tabulate_coordinates(data.coordinates, cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(&data.facet_dofs[0], local_facet);

    // Iterate over facet dofs
    for (uint i = 0; i < dofmap.num_facet_dofs(); ++i)
    {
      // Get dof and coordinate of dof
      const uint local_dof = data.facet_dofs[i];
      const int global_dof = cell_dofs[local_dof];
      for (uint j = 0; j < gdim; ++j)
        x[j] = data.coordinates[local_dof][j];

      // Set y = x
      y.assign(x.begin(), x.end());

      // Map coordinate from H to G
      sub_domain->map(_x, _y);

      // Check if coordinate is inside the domain G or in H
      const bool on_boundary = facet->num_entities(tdim) == 1;
      if (sub_domain->inside(_x, on_boundary))
      {
        // Check if coordinate exists from before
        coordinate_iterator it = coordinate_dof_pairs.find(x);
        if (it != coordinate_dof_pairs.end())
        {
          // Exists from before, so set dof associated with x
          it->second.first = dof_data(global_dof, MPI::process_number());
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal second value)
          dof_data g_dofs(global_dof, MPI::process_number());
          dof_data l_dofs(-1, -1);
          dof_pair pair(g_dofs, l_dofs);
          coordinate_dof_pairs[x] = pair;
        }
      }
      else if (sub_domain->inside(_y, on_boundary))
      {
        // Copy coordinate to std::vector
        x.assign(y.begin(), y.end());

        // Check if coordinate exists from before
        coordinate_iterator it = coordinate_dof_pairs.find(x);
        if (it != coordinate_dof_pairs.end())
        {
          // Exists from before, so set dof associated with y
          it->second.second = dof_data(global_dof, MPI::process_number());
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal first value)
          dof_data g_dofs(-1, -1);
          dof_data l_dofs(global_dof, MPI::process_number());
          dof_pair pair(g_dofs, l_dofs);
          coordinate_dof_pairs[x] = pair;
        }
      }
    }

    p++;
  }

#ifdef HAS_MPI
  coordinate_map final_coordinate_dof_pairs = MPI::all_reduce(coordinate_dof_pairs, merge_coordinate_map());
#else
  coordinate_map final_coordinate_dof_pairs = coordinate_dof_pairs;
#endif

  // Fill up list of dof pairs
  for (coordinate_iterator it = final_coordinate_dof_pairs.begin(); it != final_coordinate_dof_pairs.end(); ++it)
  {
    // Check dofs
    if (it->second.first.first == -1 || it->second.second.first == -1)
    {
      cout << "At coordinate: x =";
      for (uint j = 0; j < gdim; ++j)
        cout << " " << it->first[j];
      cout << endl;
      dolfin_error("PeriodicBC.cpp",
                   "apply periodic boundary condition",
                   "Could not find a pair of matching degrees of freedom");
    }

    if (it->second.first.second >= (int) MPI::num_processes() || it->second.first.second < 0)
    {
      cout << "At coordinate: x =";
      for (uint j = 0; j < gdim; ++j)
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
      for (uint j = 0; j < gdim; ++j)
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
void PeriodicBC::parallel_apply(GenericMatrix* A,
                       GenericVector* b,
                       const GenericVector* x) const
{

  dolfin_assert(num_dof_pairs > 0);

  log(PROGRESS, "Applying periodic boundary conditions to linear system.");

  // Check arguments
  check_arguments(A, b, x);

  // Add the slave equations to its corresponding master equation
  if (A)
  {
    // Data for a row -- master dof, cols, vals
    typedef boost::tuples::tuple<uint, std::vector<uint>, std::vector<double> > row_type;

    // All rows to be sent to a particular process
    typedef std::vector<row_type> rows_type;

    // Which processor should the row data be sent to?
    typedef std::map<uint, rows_type> row_map_type;

    row_map_type row_map;
    std::pair<uint, uint> dof_range;
    dof_range = A->local_range(0);
    std::set<uint> communicating_processors;

    for (uint i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
      {
        std::vector<uint> columns;
        std::vector<double> values;
        A->getrow(slave_dofs[i], columns, values);
        row_type row = row_type(master_dofs[i], columns, values);

        uint master_owner = master_owners[i];

        if (row_map.find(master_owner) == row_map.end())
        {
          rows_type rows_list; rows_list.push_back(row);
          row_map[master_owner] = rows_list;
        }
        else
        {
          row_map[master_owner].push_back(row);
        }

        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
      {
        uint slave_owner = slave_owners[i];

        communicating_processors.insert(slave_owner);
      }
    }

    row_map_type received_rows;
    MPI::distribute(communicating_processors, row_map, received_rows);

    for (row_map_type::const_iterator proc_it = received_rows.begin(); proc_it != received_rows.end(); ++proc_it)
    {
      rows_type rows = proc_it->second;
      for (rows_type::const_iterator row_it = rows.begin(); row_it != rows.end(); ++row_it)
      {
        row_type row = *row_it;
        uint master_dof = row.get<0>();
        std::vector<uint> columns = row.get<1>();
        std::vector<double> values = row.get<2>();

        A->add(&values[0], 1, &master_dof, columns.size(), &columns[0]);
      }
    }

    A->apply("add"); // effect the adding on of the slave rows
  }

  if (A) // now set the slave rows
  {
    std::pair<uint, uint> dof_range;
    dof_range = A->local_range(0);

    A->zero(num_dof_pairs, &slave_dofs[0]);
    A->apply("insert");

    // Insert 1 and -1 in the master and slave column of each slave row
    uint cols[2];
    double vals[2] = {1.0, -1.0};
    for (uint i = 0; i < num_dof_pairs; ++i)
    {
      if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
      {
        const uint row = slave_dofs[i];
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
  if (x)
  {
    typedef boost::tuples::tuple<uint, double> x_type; // dof index, value
    typedef std::vector<x_type> xs_type; // all x-information to be sent to a particular process
    typedef std::map<uint, xs_type> x_map_type; // processor to send it to

    std::set<uint> communicating_processors;
    x_map_type x_map;

    std::pair<uint, uint> dof_range;
    dof_range = x->local_range(0);

    for (uint i = 0; i < num_dof_pairs; ++i)
    {
      if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
      {
        double value;
        x->get_local(&value, 1, &slave_dofs[i]);

        x_type data = x_type(i, value);

        uint master_owner = master_owners[i];

        if (x_map.find(master_owner) == x_map.end())
        {
          xs_type x_list; x_list.push_back(data);
          x_map[master_owner] = x_list;
        }
        else
        {
          x_map[master_owner].push_back(data);
        }

        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
      {
        uint slave_owner = slave_owners[i];

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
        x_type x_data = *x_it;
        uint dof_idx = x_data.get<0>();
        double slave_value = x_data.get<1>();
        double master_value;
        x->get_local(&master_value, 1, &master_dofs[dof_idx]);
        rhs_values_slave[dof_idx] = master_value - slave_value;
      }
    }
  }
  else
    std::fill(rhs_values_slave.begin(), rhs_values_slave.end(), 0.0);

  // Add the slave equations to the master equations, then set the slave rows to the
  // appropriate values.
  if (b)
  {
    typedef boost::tuples::tuple<uint, double> vec_type; // master dof, value that should be added
    typedef std::vector<vec_type> vecs_type; // all vec_types that should be sent to a particular process
    typedef std::map<uint, vecs_type> vec_map_type; // which processor should the vec_data be sent to?

    vec_map_type vec_map;

    std::pair<uint, uint> dof_range;
    dof_range = b->local_range(0);
    std::set<uint> communicating_processors;

    for (uint i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
      {
        double value;
        b->get_local(&value, 1, &slave_dofs[i]);
        vec_type data = vec_type(master_dofs[i], value);

        uint master_owner = master_owners[i];
        if (vec_map.find(master_owner) == vec_map.end())
        {
          vecs_type list; list.push_back(data);
          vec_map[master_owner] = list;
        }
        else
        {
          vec_map[master_owner].push_back(data);
        }

        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
      {
        uint slave_owner = slave_owners[i];

        communicating_processors.insert(slave_owner);
      }
    }

    vec_map_type received_vecs;
    MPI::distribute(communicating_processors, vec_map, received_vecs);

    for (vec_map_type::const_iterator proc_it = received_vecs.begin(); proc_it != received_vecs.end(); ++proc_it)
    {
      vecs_type vecs = proc_it->second;
      for (vecs_type::const_iterator vec_it = vecs.begin(); vec_it != vecs.end(); ++vec_it)
      {
        vec_type vec_data = *vec_it;
        uint master_dof = vec_data.get<0>();
        double value    = vec_data.get<1>();

        b->add(&value, 1, &master_dof);
      }
    }

    b->apply("add"); // effect the adding on of the slave rows
  }

  if (b) // now zero the slave rows
  {
    std::pair<uint, uint> dof_range;
    dof_range = b->local_range(0);

    for (uint i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
      {
        b->set(&rhs_values_slave[i], 1, &slave_dofs[i]);
      }
    }
    b->apply("insert");
  }

}
//-----------------------------------------------------------------------------
