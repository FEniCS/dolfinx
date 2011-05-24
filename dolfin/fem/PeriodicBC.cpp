// Copyright (C) 2007-2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2007
// Modified by Johan Hake 2009
//
// First added:  2007-07-08
// Last changed: 2011-03-17

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

//-----------------------------------------------------------------------------
PeriodicBC::PeriodicBC(const FunctionSpace& V,
                       const SubDomain& sub_domain)
  : BoundaryCondition(V), sub_domain(reference_to_no_delete_pointer(sub_domain)),
    num_dof_pairs(0), master_dofs(0), slave_dofs(0),
    rhs_values_master(0), rhs_values_slave(0)
{
  not_working_in_parallel("Periodic boundary conditions");

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
  not_working_in_parallel("Periodic boundary conditions");

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
  assert(_function_space);

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
  assert(num_dof_pairs > 0);

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
      b->get(&value, 1, &slave_dofs[i]);
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
  // Call recursively for subspaces, should work for arbitrary nesting
  const uint num_sub_spaces = function_space.element().num_sub_elements();
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
  assert(function_space.element().num_sub_elements() == 0);

  // Get mesh and dofmap
  const Mesh& mesh = function_space.mesh();
  const GenericDofMap& dofmap = function_space.dofmap();

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
  std::vector<double> xx(gdim);
  Array<double> y(gdim);

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
    dofmap.tabulate_dofs(&data.cell_dofs[0], cell);
    dofmap.tabulate_coordinates(data.coordinates, cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(&data.facet_dofs[0], local_facet);

    // Iterate over facet dofs
    for (uint i = 0; i < dofmap.num_facet_dofs(); ++i)
    {
      // Get dof and coordinate of dof
      const uint local_dof = data.facet_dofs[i];
      const int global_dof = data.cell_dofs[local_dof];
      const Array<double>& x = data.array_coordinates[local_dof];

      // Set y = x
      for (uint j = 0; j < gdim; ++j)
        y[j] = x[j];

      // Map coordinate from H to G
      sub_domain->map(x, y);

      // Check if coordinate is inside the domain G or in H
      const bool on_boundary = facet->num_entities(tdim) == 1;
      if (sub_domain->inside(x, on_boundary))
      {
        // Copy coordinate to std::vector
        for (uint j = 0; j < gdim; ++j)
          xx[j] = x[j];

        // Check if coordinate exists from before
        coordinate_iterator it = coordinate_dof_pairs.find(xx);
        if (it != coordinate_dof_pairs.end())
        {
          // Exists from before, so set dof associated with x
          it->second.first = global_dof;
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal second value)
          std::pair<int, int> dofs(global_dof, -1);
          coordinate_dof_pairs[xx] = dofs;
        }
      }
      else if (sub_domain->inside(y, on_boundary))
      {
        // Copy coordinate to std::vector
        for (uint j = 0; j < gdim; ++j)
          xx[j] = y[j];

        // Check if coordinate exists from before
        coordinate_iterator it = coordinate_dof_pairs.find(xx);
        if (it != coordinate_dof_pairs.end())
        {
          // Exists from before, so set dof associated with y
          it->second.second = global_dof;
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal first value)
          std::pair<int, int> dofs(-1, global_dof);
          coordinate_dof_pairs[xx] = dofs;
        }
      }
    }

    p++;
  }

  // Fill up list of dof pairs
  for (coordinate_iterator it = coordinate_dof_pairs.begin(); it != coordinate_dof_pairs.end(); ++it)
  {
    // Check dofs
    if (it->second.first == -1 || it->second.second == -1)
    {
      cout << "At coordinate: x =";
      for (uint j = 0; j < gdim; ++j)
        cout << " " << it->first[j];
      cout << endl;
      error("Unable to find a pair of matching dofs for periodic boundary condition.");
    }

    // Store dofs
    dof_pairs.push_back(it->second);
  }
}
//-----------------------------------------------------------------------------
