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
#include "MeshFunction.h"
#include "MeshData.h"

using namespace dolfin;

typedef std::map<std::string, std::vector<std::size_t> >::iterator a_iterator;
typedef std::map<std::string, std::vector<std::size_t> >::const_iterator
a_const_iterator;

//-----------------------------------------------------------------------------
MeshData::MeshData()
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
  // Clear all data
  clear();

  // Copy ararys
  _arrays = data._arrays;

  return *this;
}
//-----------------------------------------------------------------------------
bool MeshData::exists(std::string name) const
{
  // Check is named array has been created
  if (_arrays.find(name) != _arrays.end())
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void MeshData::clear()
{
  _arrays.clear();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<dolfin::MeshFunction<std::size_t> >
MeshData::create_mesh_function(std::string name)
{
  dolfin_error("MeshData.cpp",
               "create a MeshFunction via mesh data",
               "MeshFunctions can no longer be stored in MeshData. Use arrays instead");

  return boost::shared_ptr<MeshFunction<std::size_t> >();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<dolfin::MeshFunction<std::size_t> >
MeshData::create_mesh_function(std::string name, std::size_t dim)
{
  dolfin_error("MeshData.cpp",
               "create a MeshFunction via mesh data",
               "MeshFunctions can no longer be stored in MeshData. Use arrays instead");

  return boost::shared_ptr<MeshFunction<std::size_t> >();
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>& MeshData::create_array(std::string name)
{
  return create_array(name, 0);
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>& MeshData::create_array(std::string name,
                                                 std::size_t size)
{
  // Check if data already exists
  a_iterator it = _arrays.find(name);
  if (it != _arrays.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Check if name is deprecated
  check_deprecated(name);

  // Add to map
  _arrays[name] = std::vector<std::size_t>(size, 0);

  return _arrays[name];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<MeshFunction<std::size_t> >
MeshData::mesh_function(const std::string name) const
{
  dolfin_error("MeshData.cpp",
               "access a MeshFunction via mesh data",
               "MeshFunctions can no longer be stored in MeshData. Use arrays instead");

  return boost::shared_ptr<MeshFunction<std::size_t> >();
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>& MeshData::array(std::string name)
{
  // Check if data exists
  a_iterator it = _arrays.find(name);
  if (it == _arrays.end())
  {
    dolfin_error("MeshData.cpp",
                 "access mesh data",
                 "Mesh data array named \"%s\" does not exist",
                 name.c_str());
  }
  return it->second;
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& MeshData::array(std::string name) const
{
  // Check if data exists
  a_const_iterator it = _arrays.find(name);
  if (it == _arrays.end())
  {
    dolfin_error("MeshData.cpp",
                 "access mesh data",
                 "Mesh data array named \"%s\" does not exist",
                 name.c_str());
  }
  return it->second;
}
//-----------------------------------------------------------------------------
void MeshData::erase_array(const std::string name)
{
  a_iterator it = _arrays.find(name);
  if (it != _arrays.end())
    _arrays.erase(it);
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
    for (a_const_iterator it = _arrays.begin(); it != _arrays.end(); ++it)
    {
      s << "  " << it->first << " (size = " << it->second.size() << ")"
        << std::endl;
    }
    s << std::endl;
  }
  else
  {
    s << "<MeshData containing " << _arrays.size() << " objects>";
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
