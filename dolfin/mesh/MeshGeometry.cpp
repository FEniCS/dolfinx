// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-05-19
// Last changed: 2008-10-08

#include <dolfin/log/dolfin_log.h>
#include "MeshGeometry.h"
#include <dolfin/function/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry() : _dim(0), _size(0), coordinates(0), 
                               mesh_coordinates(0), affine_cell(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry(const MeshGeometry& geometry) : _dim(0), _size(0), 
                            coordinates(0), mesh_coordinates(0), affine_cell(0)
{
  *this = geometry;
}
//-----------------------------------------------------------------------------
MeshGeometry::~MeshGeometry()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshGeometry& MeshGeometry::operator= (const MeshGeometry& geometry)
{
  // Clear old data if any
  clear();

  // Allocate data
  _dim = geometry._dim;
  _size = geometry._size;
  const uint n = _dim*_size;
  coordinates = new real[n];

  // Copy data
  for (uint i = 0; i < n; i++)
    coordinates[i] = geometry.coordinates[i];

  return *this;
}
//-----------------------------------------------------------------------------
Point MeshGeometry::point(uint n) const
{
  real _x = 0.0;
  real _y = 0.0;
  real _z = 0.0;
  
  if ( _dim > 0 )
    _x = x(n, 0);
  if ( _dim > 1 )
    _y = x(n, 1);
  if ( _dim > 2 )
    _z = x(n, 2);
  
  Point p(_x, _y, _z);
  return p;
}
//-----------------------------------------------------------------------------
void MeshGeometry::clear()
{
  _dim = 0;
  _size = 0;
  if ( coordinates )
    delete [] coordinates;
  coordinates = 0;
  if ( mesh_coordinates )
    delete(mesh_coordinates);
  mesh_coordinates = 0;
  if ( affine_cell )
    delete [] affine_cell;
  affine_cell = 0;
}
//-----------------------------------------------------------------------------
void MeshGeometry::init(uint dim, uint size)
{
  // Delete old data if any
  clear();

  // Allocate new data
  coordinates = new real[dim*size];

  // Save dimension and size
  _dim = dim;
  _size = size;
}
//-----------------------------------------------------------------------------
void MeshGeometry::initAffineIndicator(uint num_cells)
{
  // Clear it if it was already allocated
  if ( affine_cell )
    delete [] affine_cell;

  // Allocate new data
  affine_cell = new bool[num_cells];

  // Initialize all cells to be affine
  for (uint i = 0; i < num_cells; i++)
    affine_cell[i] = true;
}
//-----------------------------------------------------------------------------
void MeshGeometry::setAffineIndicator(uint i, bool value)
{
  affine_cell[i] = value;
}
//-----------------------------------------------------------------------------
void MeshGeometry::set(uint n, uint i, real x)
{
  coordinates[n*_dim + i] = x;
}
//-----------------------------------------------------------------------------
void MeshGeometry::setMeshCoordinates(Mesh& mesh, Vector& mesh_coord_vec,
                                        const std::string FE_signature,
                                        const std::string dofmap_signature)
{
  // If the mesh.xml file contained higher order coordinate data,
  //    then store this in the MeshGeometry class

  // FIXME: Shouldn't this function just take a plain Function?
  //if ( mesh_coord_vec )
  // {
  //  error("MeshGeometry::setMeshCoordinates needs to be updated for the Function interface.");
  //  mesh_coordinates = new Function(mesh, mesh_coord_vec, FE_signature, dofmap_signature);
  // }
  //else
    mesh_coordinates = new Function(); // an empty function
}
//-----------------------------------------------------------------------------
void MeshGeometry::disp() const
{
  // Begin indentation
  cout << "Mesh geometry" << endl;
  begin("-------------");
  cout << endl;

  // Check if empty
  if ( _dim == 0 )
  {
    cout << "empty" << endl << endl;
    end();
    return;
  }
  
  // Display coordinates for all vertices
  for (uint i = 0; i < _size; i++)
  {
    cout << i << ":";
    for (uint d = 0; d < _dim; d++)
      cout << " " << x(i, d);
    cout << endl;
  }
  cout << endl;

  // End indentation
  end();
}
//-----------------------------------------------------------------------------
