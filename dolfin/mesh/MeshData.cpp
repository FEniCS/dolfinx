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
// Modified by Niclas Jansson, 2008.
//
// First added:  2008-05-19
// Last changed: 2010-11-18

#include <sstream>
#include <dolfin/common/utils.h>
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshData.h"

using namespace dolfin;

typedef std::map<std::string, boost::shared_ptr<MeshFunction<unsigned int> > >::iterator mf_iterator;
typedef std::map<std::string, boost::shared_ptr<MeshFunction<unsigned int> > >::const_iterator mf_const_iterator;

typedef std::map<std::string, std::vector<dolfin::uint>*>::iterator a_iterator;
typedef std::map<std::string, std::vector<dolfin::uint>*>::const_iterator a_const_iterator;

typedef std::map<std::string, std::map<dolfin::uint, dolfin::uint>*>::iterator m_iterator;
typedef std::map<std::string, std::map<dolfin::uint, dolfin::uint>*>::const_iterator m_const_iterator;

typedef std::map<std::string, std::map<dolfin::uint, std::vector<dolfin::uint> > *>::iterator mvec_iterator;
typedef std::map<std::string, std::map<dolfin::uint, std::vector<dolfin::uint> > *>::const_iterator mvec_const_iterator;

//-----------------------------------------------------------------------------
MeshData::MeshData(Mesh& mesh) : mesh(mesh)
{
  // Do nothing
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

  // Copy MeshFunctions
  for (mf_const_iterator it = data.mesh_functions.begin(); it != data.mesh_functions.end(); ++it)
  {
    boost::shared_ptr<MeshFunction<unsigned int> > f = create_mesh_function(it->first, it->second->dim());
    *f = *it->second;
  }

  // Copy arrays
  for (a_const_iterator it = data.arrays.begin(); it != data.arrays.end(); ++it)
  {
    std::vector<uint>* a = create_array( it->first, static_cast<uint>(it->second->size()) );
    *a = *it->second;
  }

  return *this;
}
//-----------------------------------------------------------------------------
void MeshData::clear()
{
  mesh_functions.clear();

  for (a_iterator it = arrays.begin(); it != arrays.end(); ++it)
    delete it->second;
  arrays.clear();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<MeshFunction<unsigned int> > MeshData::create_mesh_function(std::string name)
{
  // Check if data already exists
  mf_iterator it = mesh_functions.find(name);
  if (it != mesh_functions.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Create new data
  boost::shared_ptr<MeshFunction<unsigned int> > f(new MeshFunction<uint>(mesh));
  assert(f);

  // Add to map
  mesh_functions[name] = f;

  return f;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<MeshFunction<unsigned int> > MeshData::create_mesh_function(std::string name, uint dim)
{
  boost::shared_ptr<MeshFunction<unsigned int> > f = create_mesh_function(name);
  f->init(dim);

  return f;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>* MeshData::create_array(std::string name)
{
  return create_array(name, 0);
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>* MeshData::create_array(std::string name, uint size)
{
  // Check if data already exists
  a_iterator it = arrays.find(name);
  if (it != arrays.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Create new data
  std::vector<uint>* a = new std::vector<uint>(size);
  std::fill(a->begin(), a->end(), 0);

  // Add to map
  arrays[name] = a;

  return a;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<MeshFunction<unsigned int> > MeshData::mesh_function(const std::string name) const
{
  // Check if data exists
  mf_const_iterator it = mesh_functions.find(name);
  if (it == mesh_functions.end())
    return boost::shared_ptr<MeshFunction<unsigned int> >();

  return it->second;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>* MeshData::array(const std::string name) const
{
  // Check if data exists
  a_const_iterator it = arrays.find(name);
  if (it == arrays.end())
    return 0;

  return it->second;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>* MeshData::array(const std::string name, uint i) const
{
  return array(name + " " + to_string(i));
}
//-----------------------------------------------------------------------------
void MeshData::erase_mesh_function(const std::string name)
{
  mf_iterator it = mesh_functions.find(name);
  if (it != mesh_functions.end())
    mesh_functions.erase(it);
  else
    warning("Mesh data named \"%s\" doesn't exist.", name.c_str());
}
//-----------------------------------------------------------------------------
void MeshData::erase_array(const std::string name)
{
  a_iterator it = arrays.find(name);
  if (it != arrays.end())
  {
    delete it->second;
    arrays.erase(it);
  }
  else
  {
    warning("Mesh data named \"%s\" doesn't exist.", name.c_str());
  }
}
//-----------------------------------------------------------------------------
std::string MeshData::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    // Mesh functions
    s << "  MeshFunction<uint>" << std::endl;
    s << "  ------------------" << std::endl;
    for (mf_const_iterator it = mesh_functions.begin(); it != mesh_functions.end(); ++it)
      s << "  " << it->first << " (size = " << it->second->size() << ")" << std::endl;
    s << std::endl;

    // Arrays
    s << "  std::vector<uint>" << std::endl;
    s << "  -----------------" << std::endl;
    for (a_const_iterator it = arrays.begin(); it != arrays.end(); ++it)
      s << "  " << it->first << " (size = " << it->second->size() << ")" << std::endl;
    s << std::endl;
  }
  else
  {
    const uint num_objects =
      mesh_functions.size() + arrays.size();
    s << "<MeshData containing " << num_objects << " objects>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
