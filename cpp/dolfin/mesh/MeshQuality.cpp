// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshQuality.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshFunction.h"
#include "MeshIterator.h"
#include "Vertex.h"
#include <dolfin/common/MPI.h>
#include <math.h>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
std::array<double, 6> MeshQuality::dihedral_angles(const Cell& cell)
{
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
    const Eigen::Vector3d p0 = Vertex(mesh, i0).x();
    Eigen::Vector3d v1 = Vertex(mesh, i1).x() - p0;
    Eigen::Vector3d v2 = Vertex(mesh, i2).x() - p0;
    Eigen::Vector3d v3 = Vertex(mesh, i3).x() - p0;
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

  for (auto& cell : MeshRange<Cell>(mesh))
  {
    // Get the angles from the next cell
    std::array<double, 6> angles = dihedral_angles(cell);

    // And then update the min and max
    d_ang_min
        = std::min(d_ang_min, *std::min_element(angles.begin(), angles.end()));
    d_ang_max
        = std::max(d_ang_max, *std::max_element(angles.begin(), angles.end()));
  }

  d_ang_min = MPI::min(mesh.mpi_comm(), d_ang_min);
  d_ang_max = MPI::max(mesh.mpi_comm(), d_ang_max);

  return {{d_ang_min, d_ang_max}};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<std::size_t>>
MeshQuality::dihedral_angle_histogram_data(const Mesh& mesh,
                                           std::size_t num_bins)
{
  std::vector<double> bins(num_bins);
  std::vector<std::size_t> values(num_bins, 0);

  // Currently min value is 0.0 and max is Pi
  const double interval = M_PI / (static_cast<double>(num_bins));

  for (std::size_t i = 0; i < num_bins; ++i)
    bins[i] = static_cast<double>(i) * interval + interval / 2.0;

  for (auto& cell : MeshRange<Cell>(mesh))
  {
    // this one should return the value of the angle
    std::array<double, 6> angles = dihedral_angles(cell);

    // Iterate through the collected vector
    for (std::size_t i = 0; i < angles.size(); i++)
    {
      // Compute 'bin' index, and handle special case that angle = Pi
      const std::size_t slot = std::min(
          static_cast<std::size_t>(angles[i] / interval), num_bins - 1);

      values[slot] += 1;
    }
  }

  // FIXME: This is terrible. Avoid MPI calls inside loop.
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = MPI::sum(mesh.mpi_comm(), values[i]);

  return {bins, values};
}
//-----------------------------------------------------------------------------
