// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
// 
// First added:  2008-05-19
// Last changed: 2008-11-14

#include "MeshFunction.h"
#include "MeshData.h"

using namespace dolfin;

typedef std::map<std::string, MeshFunction<dolfin::uint>*>::iterator mf_iterator;
typedef std::map<std::string, MeshFunction<dolfin::uint>*>::const_iterator mf_const_iterator;

typedef std::map<std::string, Array<dolfin::uint>*>::iterator a_iterator;
typedef std::map<std::string, Array<dolfin::uint>*>::const_iterator a_const_iterator;

typedef std::map<std::string, std::map<dolfin::uint, dolfin::uint>*>::iterator m_iterator;
typedef std::map<std::string, std::map<dolfin::uint, dolfin::uint>*>::const_iterator m_const_iterator;

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
  for (mf_const_iterator it = data.meshfunctions.begin(); it != data.meshfunctions.end(); ++it)
  {
    MeshFunction<uint> *f = createMeshFunction( it->first, it->second->dim() );
    for (uint i = 0; i < it->second->size(); i++)
      f->values()[i] = it->second->values()[i];   
  }

  // Copy Arrays
  for (a_const_iterator it = data.arrays.begin(); it != data.arrays.end(); ++it)
  {
    Array<uint>* a = createArray( it->first, static_cast<uint>(it->second->size()) );
    for (uint i = 0; i < it->second->size(); i++)
      (*a)[i] = (*(it->second))[i];   
  }     

  //Copy Mappings
  for (m_const_iterator it = data.maps.begin(); it != data.maps.end(); ++it)
  {
    std::map<uint, uint>* m = createMapping( it->first );
    std::map<uint, uint>::const_iterator i;
    for (i = it->second->begin(); i != it->second->end(); ++i)
      (*m)[i->first] = i->second;
  }

  return *this;
}
//-----------------------------------------------------------------------------
void MeshData::clear()
{ 
  for (mf_iterator it = meshfunctions.begin(); it != meshfunctions.end(); ++it)
    delete it->second;
  meshfunctions.clear();

  for (a_iterator it = arrays.begin(); it != arrays.end(); ++it)
    delete it->second;
  arrays.clear();

  for (m_iterator it = maps.begin(); it != maps.end(); ++it)
    delete it->second;
  maps.clear();
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>* MeshData::createMeshFunction(std::string name)
{
  // Check if data already exists
  mf_iterator it = meshfunctions.find(name);
  if (it != meshfunctions.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Create new data
  MeshFunction<uint>* f = new MeshFunction<uint>(mesh);
  dolfin_assert(f);

  // Add to map
  meshfunctions[name] = f;

  return f;
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>* MeshData::createMeshFunction(std::string name, uint dim)
{
  MeshFunction<uint>* f = createMeshFunction(name);
  f->init(dim);

  return f;
}
//-----------------------------------------------------------------------------
Array<dolfin::uint>* MeshData::createArray(std::string name, uint size)
{
  // Check if data already exists
  a_iterator it = arrays.find(name);
  if (it != arrays.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Create new data
  Array<uint>* a = new Array<uint>(size);
  *a = 0;

  // Add to map
  arrays[name] = a;

  return a;
}
//-----------------------------------------------------------------------------
std::map<dolfin::uint, dolfin::uint>* MeshData::createMapping(std::string name)
{
  // Check if data already exists
  m_iterator it = maps.find(name);
  if (it != maps.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Create new data
  std::map<uint, uint>* m = new std::map<uint, uint>;
  
  // Add to map
  maps[name] = m;

  return m;
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>* MeshData::meshFunction(const std::string name) const
{
  // Check if data exists
  mf_const_iterator it = meshfunctions.find(name);
  if (it == meshfunctions.end())
    return 0;
  
  return it->second;
}
//-----------------------------------------------------------------------------
Array<dolfin::uint>* MeshData::array(const std::string name) const
{
  // Check if data exists
  a_const_iterator it = arrays.find(name);
  if (it == arrays.end())
    return 0;
  
  return it->second;
}
//-----------------------------------------------------------------------------
std::map<dolfin::uint, dolfin::uint>* MeshData::mapping(const std::string name) const
{
  // Check if data exists
  m_const_iterator it = maps.find(name);
  if (it == maps.end())
    return 0;

  return it->second;
}
//-----------------------------------------------------------------------------
void MeshData::disp() const
{
  // Begin indentation
  begin("Auxiliary mesh data");

  for (mf_const_iterator it = meshfunctions.begin(); it != meshfunctions.end(); ++it)
  {
    cout << "MeshFunction<uint> of size "
         << it->second->size()
         << " on entities of topological dimension "
         << it->second->dim()
         << ": \"" << it->first << "\"" << endl;
  }

  for (a_const_iterator it = arrays.begin(); it != arrays.end(); ++it)
    cout << "Array<uint> of size " << static_cast<uint>(it->second->size())
         << ": \"" << it->first << "\"" << endl;

  for (m_const_iterator it = maps.begin(); it != maps.end(); ++it)
    cout << "map<uint, uint> of size " << static_cast<uint>(it->second->size())
         << ": \"" << it->first << "\"" << endl;

  // End indentation
  end();
}
//-----------------------------------------------------------------------------
