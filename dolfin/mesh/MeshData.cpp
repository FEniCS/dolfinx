// Copyright (C) 2008-2011 Anders Logg
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
// Modified by Niclas Jansson 2008
//
// First added:  2008-05-19
// Last changed: 2011-11-14

#include <sstream>
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include "MeshData.h"

using namespace dolfin;

typedef std::map<std::string, std::vector<std::size_t>>::iterator a_iterator;
typedef std::map<std::string, std::vector<std::size_t>>::const_iterator
a_const_iterator;

//-----------------------------------------------------------------------------
MeshData::MeshData() : _arrays(5)
{
  // Add list of deprecated names
  _deprecated_names.push_back("boundary_facet_cells");
  _deprecated_names.push_back("boundary_facet_numbers");
  _deprecated_names.push_back("boundary_indicators");
  _deprecated_names.push_back("material_indicators");
  _deprecated_names.push_back("cell_domains");
  _deprecated_names.push_back("interior_facet_domains");
  _deprecated_names.push_back("exterior_facet_domains");
}
//-----------------------------------------------------------------------------
MeshData::~MeshData()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshData& MeshData::operator= (const MeshData& data)
{
  // Copy arrays
  _arrays = data._arrays;

  return *this;
}
//-----------------------------------------------------------------------------
bool MeshData::exists(std::string name, std::size_t dim) const
{
  if (_arrays.size() < dim)
    return false;

  // Check is named array has been created
  if (dim < _arrays.size())
    if (_arrays[dim].find(name) != _arrays[dim].end())
      return true;

  return false;
}
//-----------------------------------------------------------------------------
void MeshData::clear()
{
  _arrays.clear();
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>& MeshData::create_array(std::string name,
                                                 std::size_t dim)
{
  // Check if array needs to be re-sized
  if (_arrays.size() < dim + 1)
    _arrays.resize(dim + 1);

  // Check if data already exists
  dolfin_assert(dim < _arrays.size());
  a_iterator it = _arrays[dim].find(name);
  if (it != _arrays[dim].end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Check if name is deprecated
  check_deprecated(name);

  // Add empty vector to map
  std::pair<a_iterator, bool> ins
    = _arrays[dim].insert(std::make_pair(name, std::vector<std::size_t>(0)));

  // Return vector
  return ins.first->second;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>& MeshData::array(std::string name, std::size_t dim)
{
  dolfin_assert(dim < _arrays.size());

  // Check if data exists
  a_iterator it = _arrays[dim].find(name);
  if (it == _arrays[dim].end())
  {
    dolfin_error("MeshData.cpp",
                 "access mesh data",
                 "Mesh data array named \"%s\" does not exist",
                 name.c_str());
  }
  return it->second;
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& MeshData::array(std::string name,
                                                std::size_t dim) const
{
  // Check if data exists
  dolfin_assert(dim < _arrays.size());
  a_const_iterator it = _arrays[dim].find(name);
  if (it == _arrays[dim].end())
  {
    dolfin_error("MeshData.cpp",
                 "access mesh data",
                 "Mesh data array named \"%s\" does not exist",
                 name.c_str());
  }
  return it->second;
}
//-----------------------------------------------------------------------------
void MeshData::erase_array(const std::string name, std::size_t dim)
{
  dolfin_assert(dim < _arrays.size());
  a_iterator it = _arrays[dim].find(name);
  if (it != _arrays[dim].end())
    _arrays[dim].erase(it);
  else
    warning("Mesh data named \"%s\" does not exist.", name.c_str());
}
//-----------------------------------------------------------------------------
std::string MeshData::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    s << "  std::vector<std::size_t>" << std::endl;
    s << "  -----------------" << std::endl;
    for (std::size_t d = 0; d < _arrays.size(); ++d)
    {
      for (a_const_iterator it = _arrays[d].begin(); it
             != _arrays[d].end(); ++it)
      {
        s << "  " << it->first << " ( dim = " << d
          << ", size = " << it->second.size() << ")" << std::endl;
      }
    }
    s << std::endl;
  }
  else
  {
    std::size_t size = 0;
    for (std::size_t d = 0; d < _arrays.size(); ++d)
      size += _arrays[d].size();
    s << "<MeshData containing " << size << " objects>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void MeshData::check_deprecated(std::string name) const
{
  for (std::size_t i = 0; i < _deprecated_names.size(); i++)
  {
    if (name == _deprecated_names[i])
    {
      dolfin_error("MeshData.cpp",
                   "access mesh data",
                   "Mesh data named \"%s\" is no longer recognized by DOLFIN",
                   name.c_str());
    }
  }
}
//-----------------------------------------------------------------------------
