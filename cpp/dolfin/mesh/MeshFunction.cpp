// Copyright (C) 2009-2011 Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshFunction.h"
#include "LocalMeshValueCollection.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshPartitioning.h"
#include "MeshValueCollection.h"
#include <sstream>
#include <string>

namespace dolfin
{
namespace mesh
{
//-----------------------------------------------------------------------------
template <>
std::string MeshFunction<double>::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (std::size_t i = 0; i < _size; i++)
      s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
  }
  else
    s << "<MeshFunction of topological dimension " << _dim << " containing "
      << _size << " values>";
  return s.str();
}
//-----------------------------------------------------------------------------
template <>
std::string MeshFunction<std::size_t>::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (std::size_t i = 0; i < _size; i++)
      s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
  }
  else
    s << "<MeshFunction of topological dimension " << _dim << " containing "
      << _size << " values>";
  return s.str();
}
//-----------------------------------------------------------------------------
}
}