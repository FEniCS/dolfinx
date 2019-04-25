// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <petscsys.h>
#include <string>
#include <vector>

namespace dolfin
{
namespace function
{
class Function;
}
namespace mesh
{
class Mesh;
}

namespace io
{

/// Write VTK mesh::Mesh representation

class VTKWriter
{
public:
  /// mesh::Mesh writer
  static void write_mesh(const mesh::Mesh& mesh, std::size_t cell_dim,
                         std::string file);

  /// Cell data writer
  static void write_cell_data(const function::Function& u, std::string file);
};
} // namespace io
} // namespace dolfin