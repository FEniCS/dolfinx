// Copyright (C) 2014 Mikael Mortensen
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
//
// First added:  2014-02-12
// Last changed:

#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/common/RangedIndexSet.h>
#include "Expression.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "LagrangeInterpolator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LagrangeInterpolator::interpolate(Function& u, const Expression& u0)
{
  // Get function space and element interpolating to
  dolfin_assert(u.function_space());
  const FunctionSpace& V = *u.function_space();
  dolfin_assert(V.element());
  const FiniteElement& element = *V.element();

  // Check that function ranks match
  if (element.value_rank() != u0.value_rank())
  {
    dolfin_error("LagrangeInterpolator.cpp",
                 "interpolate Expression into function space",
                 "Rank of Expression (%d) does not match rank of function space (%d)",
                 u0.value_rank(), element.value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < element.value_rank(); ++i)
  {
    if (element.value_dimension(i) != u0.value_dimension(i))
    {
      dolfin_error("LagrangeInterpolator.cpp",
                   "interpolate Expression into function space",
                   "Dimension %d of Expression (%d) does not match dimension %d of function space (%d)",
                   i, u0.value_dimension(i), i, element.value_dimension(i));
    }
  }

  // Get mesh and dimension of FunctionSpace interpolating to
  dolfin_assert(V.mesh());
  const Mesh& mesh = *V.mesh();
  const std::size_t gdim = mesh.geometry().dim();

  // Create arrays used to evaluate one point
  std::vector<double> x(gdim);
  std::vector<double> values(u0.value_size());
  Array<double> _x(gdim, x.data());
  Array<double> _values(u0.value_size(), values.data());

  // Create vector to hold all local values of u
  std::vector<double> local_u_vector(u.vector()->local_size());

  // Create map from coordinates to dofs sharing that coordinate
  const std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs = tabulate_coordinates_to_dofs(V);

  // Get a map from global dofs to component number in mixed space
  std::unordered_map<std::size_t, std::size_t> dof_component_map;
  int component = -1;
  extract_dof_component_map(dof_component_map, V, &component);

  // Evaluate all points
  for (const auto &map_it : coords_to_dofs)
  {
    // Place interpolation point in x
    std::copy(map_it.first.begin(), map_it.first.end(), x.begin());

    u0.eval(_values, _x);
    for (const auto &d : map_it.second)
    {
      dolfin_assert(d < local_u_vector.size());
      local_u_vector[d] = values[dof_component_map[d]];
    }
  }

  // Set and finalize
  u.vector()->set_local(local_u_vector);
  u.vector()->apply("insert");
}
//-----------------------------------------------------------------------------
void LagrangeInterpolator::interpolate(Function& u, const Function& u0)
{
  // Interpolate from Function u0 to Function u.
  // This mesh of u0 may be different from that of u
  //
  // The algorithm is briefly
  //
  //   1) Create a map from all different coordinates of u's dofs to
  //      the dofs living on that coordinate. This is done such that
  //      one only need to visit (and distribute) each interpolation
  //      point once.
  //   2) Create a map from dof to component index in Mixed Space.
  //   3) Create bounding boxes for the partitioned mesh of u0 and
  //      distribute to all processors.
  //   4) Using bounding boxes, compute the processes that *may* own
  //      the dofs of u.
  //   5) Distribute interpolation points to potential owners who
  //      subsequently tries to evaluate u0. If successful, return
  //      values of u0 to owner.

  // Get function spaces of Functions interpolating to/from
  dolfin_assert(u0.function_space());
  dolfin_assert( u.function_space());
  const FunctionSpace& V0 = *u0.function_space();
  const FunctionSpace& V1 =  *u.function_space();

  // Get element interpolating to
  dolfin_assert(V1.element());
  const FiniteElement& element = *V1.element();

  // Check that function ranks match
  if (element.value_rank() != u0.value_rank())
  {
    dolfin_error("LagrangeInterpolator.cpp",
                 "interpolate Function into function space",
                 "Rank of Function (%d) does not match rank of function space (%d)",
                 u0.value_rank(), element.value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < element.value_rank(); ++i)
  {
    if (element.value_dimension(i) != u0.value_dimension(i))
    {
      dolfin_error("LagrangeInterpolator.cpp",
                   "interpolate Function into function space",
                   "Dimension %d of Function (%d) does not match dimension %d of function space (%d)",
                   i, u0.value_dimension(i), i, element.value_dimension(i));
    }
  }

  // Get mesh and dimension of FunctionSpace interpolating to/from
  dolfin_assert(V0.mesh());
  dolfin_assert(V1.mesh());
  const Mesh& mesh0 = *V0.mesh();
  const Mesh& mesh1 = *V1.mesh();
  const std::size_t gdim0 = mesh0.geometry().dim();
  const std::size_t gdim1 = mesh1.geometry().dim();

  // Get communicator
  const MPI_Comm mpi_comm = V1.mesh()->mpi_comm();

  // Create bounding box of mesh0
  std::vector<double> x_min_max(2*gdim0);
  std::vector<double> coordinates = mesh0.coordinates();
  for (std::size_t i = 0; i < gdim0; ++i)
  {
    for (auto it = coordinates.begin() + i; it < coordinates.end(); it += gdim0)
    {
      x_min_max[i]         = std::min(x_min_max[i], *it);
      x_min_max[gdim0 + i] = std::max(x_min_max[gdim0 + i], *it);
    }
  }

  // Communicate bounding boxes
  std::vector<std::vector<double>> bounding_boxes;
  MPI::all_gather(mpi_comm, x_min_max, bounding_boxes);

  // Create arrays used to evaluate one point
  std::vector<double> x(gdim0);
  std::vector<double> values(u0.value_size());
  Array<double> _x(gdim0, x.data());
  Array<double> _values(u0.value_size(), values.data());

  // Create vector to hold all local values of u
  std::vector<double> local_u_vector(u.vector()->local_size());

  // Create map from coordinates to dofs sharing that coordinate
  std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs = tabulate_coordinates_to_dofs(V1);

  // Get a map from global dofs to component number in mixed space
  std::unordered_map<std::size_t, std::size_t> dof_component_map;
  int component = -1;
  extract_dof_component_map(dof_component_map, V1, &component);

  // Search this process first for all coordinates in u's local mesh
  std::vector<double> points_not_found;
  for (const auto &map_it : coords_to_dofs)
  {
    // Place interpolation point in x
    std::copy(map_it.first.begin(), map_it.first.end(), x.begin());

    try
    { // Store values when point is found
      u0.eval(_values, _x);
      std::vector<std::size_t> dofs = map_it.second;
      for (const auto &d : map_it.second)
        local_u_vector[d] = values[dof_component_map[d]];
    }
    catch (std::exception &e)
    {
      // If not found then it must be searched on the other processes
      points_not_found.insert(points_not_found.end(), x.begin(), x.end());
    }
  }

  // Get number of MPI processes
  std::size_t num_processes = MPI::size(mpi_comm);

  // Remaining interpolation points must be found through MPI
  // communication.  Check first using bounding boxes which process
  // may own the points
  std::vector<std::vector<double>> potential_points(num_processes);
  for (std::size_t i = 0; i < points_not_found.size(); i += gdim1)
  {
    std::copy(points_not_found.begin() + i,
              points_not_found.begin() + i + gdim1, x.begin());

    // Find potential owners
    for (std::size_t p = 0; p < num_processes; p++)
    {
      if (p == MPI::rank(mpi_comm))
        continue;

      // Check if in bounding box
      if (in_bounding_box(x, bounding_boxes[p], 1e-12))
      {
        potential_points[p].insert(potential_points[p].end(),
                                   x.begin(), x.end());
      }
    }
  }

  // Communicate all potential points
  std::vector<std::vector<double>> potential_points_recv;
  MPI::all_to_all(mpi_comm, potential_points, potential_points_recv);

  // Now try to eval u0 for the received points
  std::vector<std::vector<double>> coefficients_found(num_processes);
  std::vector<std::vector<double>> points_found(num_processes);

  for (std::size_t p = 0; p < num_processes; ++p)
  {
    if (p == MPI::rank(mpi_comm))
      continue;

    std::vector<double>& points = potential_points_recv[p];
    for (std::size_t j = 0; j < points.size()/gdim1; ++j)
    {
      std::copy(points.begin() + j*gdim1, points.begin() + (j + 1)*gdim1,
                x.begin());

      try
      {
        // push back when point is found
        u0.eval(_values, _x);
        coefficients_found[p].insert(coefficients_found[p].end(),
                                     values.begin(), values.end());
        points_found[p].insert(points_found[p].end(), x.begin(), x.end());
      }
      catch (std::exception &e)
      {
        // If not found then do nothing
      }
    }
  }

  // Send back the found coefficients and points
  std::vector<std::vector<double>> coefficients_recv;
  std::vector<std::vector<double>> points_recv;
  MPI::all_to_all(mpi_comm, coefficients_found, coefficients_recv);
  MPI::all_to_all(mpi_comm, points_found, points_recv);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    if (p == MPI::rank(mpi_comm))
      continue;

    // Get the new values and points
    const std::vector<double>& vals = coefficients_recv[p];
    const std::vector<double>& pts = points_recv[p];

    // Move all found coefficients into the local_u_vector
    for (std::size_t j = 0; j < pts.size()/gdim1; ++j)
    {
      std::copy(pts.begin() + j*gdim1, pts.begin() + (j + 1)*gdim1, x.begin());

      // Get the owned dofs sharing x
      const std::vector<std::size_t>& dofs = coords_to_dofs[x];

      // Place result in local_u_vector
      for (const auto &d : dofs)
      {
        dolfin_assert(d <  local_u_vector.size());
        local_u_vector[d]
          = vals[j*u0.value_size() + dof_component_map[d]];
      }
    }
  }

  // Set and finalize
  u.vector()->set_local(local_u_vector);
  u.vector()->apply("insert");
}
//-----------------------------------------------------------------------------
std::map<std::vector<double>, std::vector<std::size_t>,
         LagrangeInterpolator::lt_coordinate>
LagrangeInterpolator::tabulate_coordinates_to_dofs(const FunctionSpace& V)
{
  std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs(lt_coordinate(1.0e-12));

  // Extract mesh, dofmap and element
  dolfin_assert(V.dofmap());
  dolfin_assert(V.element());
  dolfin_assert(V.mesh());
  const GenericDofMap& dofmap = *V.dofmap();
  const FiniteElement& element = *V.element();
  const Mesh& mesh = *V.mesh();

  // Geometric dimension
  const std::size_t gdim = mesh.geometry().dim();

  // Loop over cells and tabulate dofs
  boost::multi_array<double, 2> coordinates;
  std::vector<double> coordinate_dofs;
  std::vector<double> coors(gdim);

  // Speed up the computations by only visiting (most) dofs once
  const std::size_t local_size = dofmap.ownership_range().second
    - dofmap.ownership_range().first;
  RangedIndexSet already_visited(std::make_pair(0, local_size));

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    cell->get_coordinate_dofs(coordinate_dofs);

    // Get local-to-global map
    const ArrayView<const dolfin::la_index> dofs
      = dofmap.cell_dofs(cell->index());

    // Tabulate dof coordinates on cell
    element.tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);

    // Map dofs into coords_to_dofs
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      const std::size_t dof = dofs[i];
      if (dof < local_size)
      {
        // Skip already checked dofs
        if (!already_visited.insert(dof))
          continue;

        // Put coordinates in coors
        std::copy(coordinates[i].begin(), coordinates[i].end(), coors.begin());

        // Add dof to list at this coord
        const auto ins = coords_to_dofs.insert
          (std::make_pair(coors, std::vector<std::size_t>{dof}));
        if (!ins.second)
          ins.first->second.push_back(dof);
      }
    }
  }
  return coords_to_dofs;
}
//-----------------------------------------------------------------------------
void
LagrangeInterpolator::extract_dof_component_map(std::unordered_map<std::size_t,
                                                std::size_t>& dof_component_map,
                                                const FunctionSpace& V,
                                                int* component)
{
  // Extract sub dofmaps recursively and store dof to component map
  if (V.element()->num_sub_elements() == 0)
  {
    std::unordered_map<std::size_t, std::size_t> collapsed_map;
    std::shared_ptr<GenericDofMap> dummy
      = V.dofmap()->collapse(collapsed_map, *V.mesh());
    (*component)++;
    for (const auto &map_it : collapsed_map)
      dof_component_map[map_it.second] = (*component);
  }
  else
  {
    for (std::size_t i = 0; i < V.element()->num_sub_elements(); ++i)
    {
      const std::vector<std::size_t> comp = {i};
      std::shared_ptr<FunctionSpace> Vs = V.extract_sub_space(comp);
      extract_dof_component_map(dof_component_map, *Vs, component);
    }
  }
}
//-----------------------------------------------------------------------------
bool
LagrangeInterpolator::in_bounding_box(const std::vector<double>& point,
                                      const std::vector<double>& bounding_box,
                                      const double tol)
{
  // Return false if bounding box is empty
  if (bounding_box.empty())
    return false;

  const std::size_t gdim = point.size();
  dolfin_assert(bounding_box.size() == 2*gdim);
  for (std::size_t i = 0; i < gdim; ++i)
  {
    if (!(point[i] >= (bounding_box[i] - tol)
          && point[i] <= (bounding_box[gdim + i] + tol)))
    {
      return false;
    }
  }
  return true;
}
//-----------------------------------------------------------------------------
