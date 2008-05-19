// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-19
// Last changed: 2008-05-19

#include "MeshData.h"

using namespace dolfin;

typedef std::map<std::string, MeshFunction<dolfin::uint>*>::iterator iterator;
typedef std::map<std::string, MeshFunction<dolfin::uint>*>::const_iterator const_iterator;

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
  for (iterator it = data.begin(); it != data.end(); ++it)
    delete it->second;
  data.clear();
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>* MeshData::create(std::string name, uint dim)
{
  // Check if data already exists
  iterator it = data.find(name);
  if (it != data.end())
  {
    warning("Mesh data named \"%s\" already exists.", name.c_str());
    return it->second;
  }

  // Create new data
  MeshFunction<uint>* f = new MeshFunction<uint>;
  f->init(mesh, dim);

  // Add to map
  data[name] = f;

  return f;
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>* MeshData::operator[] (std::string name)
{
  // Check if data exists
  iterator it = data.find(name);
  if (it == data.end())
    error("No mesh data named \"%s\" exists.", name.c_str());
  
  return it->second;
}
//-----------------------------------------------------------------------------
void MeshData::disp() const
{
  // Begin indentation
  cout << "Auxiliary mesh data" << endl;
  begin("-------------------");
  cout << endl;

  for (const_iterator it = data.begin(); it != data.end(); ++it)
  {
    cout << "MeshFunction<uint> of size "
         << it->second->size()
         << " on entities of topological dimension "
         << it->second->dim()
         << ": \"" << it->first << "\"" << endl;
  }
  
  // End indentation
  end();
}
//-----------------------------------------------------------------------------
