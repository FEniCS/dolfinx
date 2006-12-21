// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-12-20
// Last changed: 2006-12-20

#include <dolfin/MeshHierarchy.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshHierarchy::MeshHierarchy(const Mesh& mesh) 
  : meshes(0), num_meshes(0)
{
  // Allocate data
  num_meshes = 1;
  meshes = new Mesh[num_meshes];
  
  // Initialize mesh hierarchy 
  meshes[0] = mesh; 
}
//-----------------------------------------------------------------------------
MeshHierarchy::MeshHierarchy() 
  : meshes(0), num_meshes(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshHierarchy::~MeshHierarchy()
{
  clear();
}
//-----------------------------------------------------------------------------
void MeshHierarchy::init(const Mesh& mesh) 
{
  // Clear old data if any
  clear();

  // Allocate data 
  num_meshes = 1;
  meshes = new Mesh[num_meshes];
  
  // Initialize mesh hierarchy 
  meshes[0] = mesh; 
}
//-----------------------------------------------------------------------------
void MeshHierarchy::clear()
{
  num_meshes = 0;

  if ( meshes )
    delete [] meshes;
  meshes = 0;
}
//-----------------------------------------------------------------------------
void MeshHierarchy::add(const Mesh& mesh) 
{
  // Add mesh to mesh hierarchy if non empty 
  if ( num_meshes > 0 )
  {
    // Create temporary array 
    Mesh* tmp_array; 
    tmp_array = new Mesh[num_meshes];

    // Copy data
    for (uint i = 0; i < num_meshes; i++)
      tmp_array[i] = meshes[i];
    
    // Clear old mesh hierarchy 
    delete [] meshes;
    
    // Add new mesh to mesh hierarchy 
    num_meshes++;
    meshes = new Mesh[num_meshes];
    for (uint i = 0; i < num_meshes-1; i++)
      meshes[i] = tmp_array[i];
    meshes[num_meshes-1] = mesh; 
  
    // Clear temporary array 
    delete [] tmp_array;
  }
  else
  {
    init(mesh);
  }
}
//-----------------------------------------------------------------------------
void MeshHierarchy::remove() 
{
  // Remove finest mesh from mesh hierarchy if non empty 
  if ( num_meshes > 0 )
  {
    // Create temporary array 
    Mesh* tmp_array; 
    tmp_array = new Mesh[num_meshes-1];

    // Copy data
    for (uint i = 0; i < num_meshes-1; i++)
      tmp_array[i] = meshes[i];
    
    // Clear old mesh hierarchy 
    delete [] meshes;
    
    // Remove finest mesh from mesh hierarchy 
    num_meshes--;
    meshes = new Mesh[num_meshes];
    for (uint i = 0; i < num_meshes; i++)
      meshes[i] = tmp_array[i];

    // Clear temporary array 
    delete [] tmp_array;
  }
  else
  {
    dolfin_error("Cannot remove mesh from empty mesh hierarcy");
  }
}
//-----------------------------------------------------------------------------
Mesh MeshHierarchy::operator[](uint k) const 
{
  dolfin_error("not implemented.");
  Mesh mesh;
  return mesh;
}
//-----------------------------------------------------------------------------
Mesh MeshHierarchy::operator()(uint k) const 
{
  dolfin_error("not implemented.");
  Mesh mesh;
  return mesh;
}
//-----------------------------------------------------------------------------


