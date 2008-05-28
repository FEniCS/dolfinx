// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-19
// Last changed: 2008-05-21

#include "MeshData.h"

using namespace dolfin;

typedef std::map<std::string, MeshFunction<dolfin::uint>*>::iterator mf_iterator;
typedef std::map<std::string, MeshFunction<dolfin::uint>*>::const_iterator mf_const_iterator;

typedef std::map<std::string, Array<dolfin::uint>*>::iterator a_iterator;
typedef std::map<std::string, Array<dolfin::uint>*>::const_iterator a_const_iterator;

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
void MeshData::clear()
{ 
  for (mf_iterator it = meshfunctions.begin(); it != meshfunctions.end(); ++it)
    delete it->second;
  meshfunctions.clear();

  for (a_iterator it = arrays.begin(); it != arrays.end(); ++it)
    delete it->second;
  arrays.clear();
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>* MeshData::createMeshFunction(std::string name, uint dim)
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
  f->init(mesh, dim);

  // Add to map
  meshfunctions[name] = f;

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
MeshFunction<dolfin::uint>* MeshData::meshFunction(std::string name)
{
  // Check if data exists
  mf_iterator it = meshfunctions.find(name);
  if (it == meshfunctions.end())
    return 0;
  
  return it->second;
}
//-----------------------------------------------------------------------------
Array<dolfin::uint>* MeshData::array(std::string name)
{
  // Check if data exists
  a_iterator it = arrays.find(name);
  if (it == arrays.end())
    return 0;
  
  return it->second;
}
//-----------------------------------------------------------------------------
void MeshData::disp() const
{
  // Begin indentation
  cout << "Auxiliary mesh data" << endl;
  begin("-------------------");
  cout << endl;

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

  // End indentation
  end();
}
//-----------------------------------------------------------------------------
