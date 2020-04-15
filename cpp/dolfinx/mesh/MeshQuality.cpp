// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshQuality.h"
#include "Geometry.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
std::array<double, 6> MeshQuality::dihedral_angles(const MeshEntity& cell)
{
  throw std::runtime_error("MeshQuality::dihedral_angles requires updating for "
                           "proper geometry handling");

  if (cell.dim() != 3)
  {
    throw std::runtime_error(
        "Calculation of dihedral angle only supported for 3D cells.");
  }

  static const std::size_t edges[6][2]
      = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
  const Mesh& mesh = cell.mesh();

  std::array<double, 6> dh_angle;
  for (std::uint32_t i = 0; i < 6; ++i)
  {
    const std::size_t i0 = cell.entities(0)[edges[i][0]];
    const std::size_t i1 = cell.entities(0)[edges[i][1]];
    const std::size_t i2 = cell.entities(0)[edges[5 - i][0]];
    const std::size_t i3 = cell.entities(0)[edges[5 - i][1]];

    const Eigen::Vector3d p0 = mesh.geometry().node(i0);
    Eigen::Vector3d v1 = mesh.geometry().node(i1) - p0;
    Eigen::Vector3d v2 = mesh.geometry().node(i2) - p0;
    Eigen::Vector3d v3 = mesh.geometry().node(i3) - p0;

    v1 /= v1.norm();
    v2 /= v2.norm();
    v3 /= v3.norm();
    double cphi = (v2.dot(v3) - v1.dot(v2) * v1.dot(v3))
                  / (v1.cross(v2).norm() * v1.cross(v3).norm());
    dh_angle[i] = acos(cphi);
  }

  return dh_angle;
}
//-----------------------------------------------------------------------------
std::array<double, 2> MeshQuality::dihedral_angles_min_max(const Mesh& mesh)
{
  // Get start min/max
  double d_ang_min = 3.14 + 1.0;
  double d_ang_max = -1.0;

  const int tdim = mesh.topology().dim();
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get the angles from the next cell
    std::array<double, 6> angles = dihedral_angles(MeshEntity(mesh, tdim, c));

    // And then update the min and max
    d_ang_min
        = std::min(d_ang_min, *std::min_element(angles.begin(), angles.end()));
    d_ang_max
        = std::max(d_ang_max, *std::max_element(angles.begin(), angles.end()));
  }
  return {{d_ang_min, d_ang_max}};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<std::int64_t>>
MeshQuality::dihedral_angle_histogram_data(const Mesh& mesh, int num_bins)
{
  std::vector<double> bins(num_bins);
  std::vector<std::int64_t> values(num_bins, 0);

  // Currently min value is 0.0 and max is Pi
  const double interval = M_PI / (static_cast<double>(num_bins));

  for (int i = 0; i < num_bins; ++i)
    bins[i] = static_cast<double>(i) * interval + interval / 2.0;

  const int tdim = mesh.topology().dim();
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // this one should return the value of the angle
    std::array<double, 6> angles = dihedral_angles(MeshEntity(mesh, tdim, c));

    // Iterate through the collected vector
    for (std::size_t i = 0; i < angles.size(); i++)
    {
      // Compute 'bin' index, and handle special case that angle = Pi
      const int slot
          = std::min(static_cast<int>(angles[i] / interval), num_bins - 1);
      values[slot] += 1;
    }
  }

  // FIXME: This is terrible. Avoid MPI calls inside loop.
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    std::int64_t value = values[i];
    MPI_Allreduce(&values[i], &value, 1, MPI_INT64_T, MPI_SUM, mesh.mpi_comm());
  }

  return {bins, values};
}
//-----------------------------------------------------------------------------
