// Copyright (C) 2009-2011 Ola Skavhaug and Garth N. Wells
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
// First added:  2009-09-25
// Last changed: 2011-09-22

#include <sstream>
#include <string>
#include "LocalMeshValueCollection.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshPartitioning.h"
#include "MeshFunction.h"
#include "MeshValueCollection.h"

namespace dolfin
{

//-----------------------------------------------------------------------------
template<> std::string MeshFunction<double>::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (std::size_t i = 0; i < _size; i++)
      s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
  }
  else
    s << "<MeshFunction of topological dimension " << _dim << " containing " << _size << " values>";
  return s.str();
}
//-----------------------------------------------------------------------------
template<> std::string MeshFunction<std::size_t>::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (std::size_t i = 0; i < _size; i++)
      s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
  }
  else
    s << "<MeshFunction of topological dimension " << _dim << " containing " << _size << " values>";
  return s.str();
}
//-----------------------------------------------------------------------------

}
