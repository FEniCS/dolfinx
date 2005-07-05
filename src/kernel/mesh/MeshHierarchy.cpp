// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005

#include <dolfin/Mesh.h>
#include <dolfin/MeshHierarchy.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshHierarchy::MeshHierarchy()
{
  clear();
}
//-----------------------------------------------------------------------------
MeshHierarchy::MeshHierarchy(Mesh& mesh)
{
  clear();
  init(mesh);
}
//-----------------------------------------------------------------------------
MeshHierarchy::~MeshHierarchy()
{
  clear();
}
//-----------------------------------------------------------------------------
void MeshHierarchy::init(Mesh& mesh)
{
  // Clear previous mesh hierarchy
  clear();

  // Find top mesh (level 0)
  Mesh* top = &mesh;
  for (; top->_parent; top = top->_parent);
  
  // Count the number of meshes
  int count = 0;
  for (Mesh* g = top; g; g = g->_child)
    count++;

  // Allocate memory for the meshes
  meshes.init(count);

  // Put the meshes in the list
  int pos = 0;
  for (Mesh* g = top; g; g = g->_child)
    meshes(pos++) = g;

  // Write a message
  cout << "Creating mesh hierarchy: found " << count << " mesh(s)." << endl;
}
//-----------------------------------------------------------------------------
void MeshHierarchy::clear()
{
  meshes.clear();
}
//-----------------------------------------------------------------------------
void MeshHierarchy::add(Mesh& mesh)
{
  

}
//-----------------------------------------------------------------------------
Mesh& MeshHierarchy::operator() (int level) const
{
  if ( empty() )
    dolfin_error("Mesh hierarchy is empty.");
  
  if ( level < 0 || level >= meshes.size() )
    dolfin_error1("No mesh at level %d.", level);
  
  return *(meshes(level));
}
//-----------------------------------------------------------------------------
Mesh& MeshHierarchy::coarse() const
{
  if ( empty() )
    dolfin_error("Mesh hierarchy is empty.");
  
  return *(meshes(0));
}
//-----------------------------------------------------------------------------
Mesh& MeshHierarchy::fine() const
{
  if ( empty() )
    dolfin_error("Mesh hierarchy is empty.");

  return *(meshes(meshes.size() - 1));
}
//-----------------------------------------------------------------------------
int MeshHierarchy::size() const
{
  return meshes.size();
}
//-----------------------------------------------------------------------------
bool MeshHierarchy::empty() const
{
  return meshes.empty();
}
//-----------------------------------------------------------------------------
