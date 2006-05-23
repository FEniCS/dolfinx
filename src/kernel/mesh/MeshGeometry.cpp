// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-19
// Last changed: 2006-05-19

#include <dolfin/dolfin_log.h>
#include <dolfin/MeshGeometry.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry() : coordinates(0), _dim(0), _size(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshGeometry::~MeshGeometry()
{
  clear();
}
//-----------------------------------------------------------------------------
void MeshGeometry::clear()
{
  if ( coordinates )
    delete [] coordinates;
  coordinates = 0;
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
void MeshGeometry::set(uint n, uint d, real x)
{
  coordinates[d*_size + n] = x;
}
//-----------------------------------------------------------------------------
void MeshGeometry::disp() const
{
  cout << "Mesh geometry" << endl;
  cout << "-------------" << endl << endl;
  
  // Begin indentation
  dolfin_begin();
  
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
  dolfin_end();
}
//-----------------------------------------------------------------------------
