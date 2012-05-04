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
typedef std::map<std::vector<double>, std::pair<int, int>, lt_coordinate> coordinate_map;
typedef coordinate_map::iterator coordinate_iterator;

// To apply periodic BCs in parallel, we'll need to reduce the coordinate map onto the root
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
        match->second.first  = std::max(it->second.first, match->second.first);
        match->second.second = std::max(it->second.second, match->second.second);
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
  std::vector<std::pair<uint, uint> > dof_pairs;
  extract_dof_pairs(*_function_space, dof_pairs);

  // Resize arrays
  num_dof_pairs = dof_pairs.size();
  master_dofs.resize(num_dof_pairs);
  slave_dofs.resize(num_dof_pairs);
  rhs_values_master.resize(num_dof_pairs);
  rhs_values_slave.resize(num_dof_pairs);

  // Store master and slave dofs
  for (uint i = 0; i < dof_pairs.size(); ++i)
  {
    master_dofs[i] = dof_pairs[i].first;
    slave_dofs[i] = dof_pairs[i].second;
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
      for (int idx = 0; idx < columns.size(); idx++)
      {
        cout << "A:  (" << columns[idx] << ", " << values[idx] << ")" << endl;
      }
      A->add(&values[0], 1, &master_dofs[i], columns.size(), &columns[0]);
      A->apply("add");
    }

    // Add slave row to master row in b
    if (b)
    {
      double value;
      b->get(&value, 1, &slave_dofs[i]);
      cout << "b: (" << master_dofs[i] << ", " << value << ")" << endl;
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
    x->get(&rhs_values_master[0], num_dof_pairs, &master_dofs[0]);
    x->get(&rhs_values_slave[0],  num_dof_pairs, &slave_dofs[0]);
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
                                   std::vector<std::pair<uint, uint> >& dof_pairs)
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
          it->second.first = global_dof;
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal second value)
          std::pair<int, int> dofs(global_dof, -1);
          coordinate_dof_pairs[x] = dofs;
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
          it->second.second = global_dof;
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal first value)
          std::pair<int, int> dofs(-1, global_dof);
          coordinate_dof_pairs[x] = dofs;
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

  // Print out the before-and-after maps, to check
  for (coordinate_iterator it = coordinate_dof_pairs.begin(); it != coordinate_dof_pairs.end(); ++it)
  {
    cout << MPI::process_number() << " before reduction: (" << it->second.first << ", " << it->second.second << ")" << endl;
  }

  // Print out the before-and-after maps, to check
  for (coordinate_iterator it = final_coordinate_dof_pairs.begin(); it != final_coordinate_dof_pairs.end(); ++it)
  {
    cout << MPI::process_number() << " after  reduction: (" << it->second.first << ", " << it->second.second << ")" << endl;
  }


  // Fill up list of dof pairs
  for (coordinate_iterator it = final_coordinate_dof_pairs.begin(); it != final_coordinate_dof_pairs.end(); ++it)
  {
    // Check dofs
    if (it->second.first == -1 || it->second.second == -1)
    {
      cout << "At coordinate: x =";
      for (uint j = 0; j < gdim; ++j)
        cout << " " << it->first[j];
      cout << endl;
      dolfin_error("PeriodicBC.cpp",
                   "apply periodic boundary condition",
                   "Could not find a pair of matching degrees of freedom");
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

  if (A)
  {
    typedef std::pair<std::vector<uint>, std::vector<double> > row_type; // the data for a row
    typedef std::map<uint, row_type> row_data;                           // what's the master dof for that row -- to which row should it be added
    typedef std::map<uint, row_data> proc_row;                           // which processor should the row_data be sent to?

    proc_row row_map;
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
        row_type row = row_type(columns, values);
        row_data data; data[master_dofs[i]] = row;

        // FIXME: how do I get the owner of master_dofs[i]?
        // for now, just check if I own it, and if not assume it's
        // the other (only works on 2 processors)
        uint master_owner;
        if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
          master_owner = MPI::process_number();
        else
          master_owner = !MPI::process_number();

        row_map[master_owner] = data;
        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
      {
        // FIXME: how do I get the owner of slave_dofs[i]?
        // for now, just check if I own it, and if not assume it's
        // the other (only works on 2 processors)
        uint slave_owner;
        if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
          slave_owner = MPI::process_number();
        else
          slave_owner = !MPI::process_number();

        communicating_processors.insert(slave_owner);
      }
    }

    proc_row received_rows;
    MPI::distribute(communicating_processors, row_map, received_rows);

    for (proc_row::const_iterator it = received_rows.begin(); it != received_rows.end(); ++it)
    {
      row_data data = it->second;
      uint master_dof = data.begin()->first;
      row_type row = data.begin()->second;
      std::vector<uint> columns = row.first;
      std::vector<double> values = row.second;

      cout << "master_dof: " << master_dof << endl;

      for (int i = 0; i < columns.size(); i++)
      {
        cout << "A:  (" << columns[i] << ", " << values[i] << ")" << endl;
      }
      A->add(&values[0], 1, &master_dof, columns.size(), &columns[0]);
    }

    A->apply("add"); // effect the adding on of the slave rows
  }

  if (A) // now zero the slave rows
  {
    std::pair<uint, uint> dof_range;
    dof_range = A->local_range(0);

    A->zero(num_dof_pairs, &slave_dofs[0]);
    A->apply("insert");
  }

  // Do something similar for b and x here

  // not handling x yet
  dolfin_assert(!x);

  std::fill(rhs_values_slave.begin(), rhs_values_slave.end(), 0.0);

  if (b)
  {
    typedef std::map<uint, double> vec_data;                            // what's the master dof for that row -- to which entry should it be added
    typedef std::map<uint, vec_data> proc_vec;                          // which processor should the vec_data be sent to?

    proc_vec vec_map;
    std::pair<uint, uint> dof_range;
    dof_range = b->local_range(0);
    std::set<uint> communicating_processors;

    for (uint i = 0; i < num_dof_pairs; i++)
    {
      if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
      {
        double value;
        b->get(&value, 1, &slave_dofs[i]);
        vec_data data; data[master_dofs[i]] = value;

        // FIXME: how do I get the owner of master_dofs[i]?
        // for now, just check if I own it, and if not assume it's
        // the other (only works on 2 processors)
        uint master_owner;
        if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
          master_owner = MPI::process_number();
        else
          master_owner = !MPI::process_number();

        vec_map[master_owner] = data;
        communicating_processors.insert(master_owner);
      }
      else if (master_dofs[i] >= dof_range.first && master_dofs[i] < dof_range.second)
      {
        // FIXME: how do I get the owner of slave_dofs[i]?
        // for now, just check if I own it, and if not assume it's
        // the other (only works on 2 processors)
        uint slave_owner;
        if (slave_dofs[i] >= dof_range.first && slave_dofs[i] < dof_range.second)
          slave_owner = MPI::process_number();
        else
          slave_owner = !MPI::process_number();

        communicating_processors.insert(slave_owner);
      }
    }

    proc_vec received_vecs;
    MPI::distribute(communicating_processors, vec_map, received_vecs);

    for (proc_vec::const_iterator it = received_vecs.begin(); it != received_vecs.end(); ++it)
    {
      vec_data data = it->second;
      uint master_dof = data.begin()->first;
      double value = data.begin()->second;
      cout << "b: (" << master_dof << ", " << value << ")" << endl;
      b->add(&value, 1, &master_dof);
    }

    b->apply("add"); // effect the adding on of the slave rows
  }

  if (b) // now zero the slave rows
  {
    std::pair<uint, uint> dof_range;
    dof_range = b->local_range(0);

    b->set(&rhs_values_slave[0], num_dof_pairs, &slave_dofs[0]);
    b->apply("insert");
  }

}
//-----------------------------------------------------------------------------
