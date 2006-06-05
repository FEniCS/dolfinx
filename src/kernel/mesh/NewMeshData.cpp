// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-06-05

#include <dolfin/NewMeshData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewMeshData::NewMeshData() : cell_type(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewMeshData::~NewMeshData()
{
  clear();
}
//-----------------------------------------------------------------------------
void NewMeshData::clear()
{ 
  // Clear mesh topology
  topology.clear();

  // Clear mesh geometry
  geometry.clear();

  // Clear cell type
  if ( cell_type )
    delete cell_type;
  cell_type = 0;
}
//-----------------------------------------------------------------------------
void NewMeshData::disp() const
{
  cout << "Mesh data" << endl;
  cout << "---------" << endl << endl;
  
  // Begin indentation
  dolfin_begin();

  // Display topology and geometry
  topology.disp();
  geometry.disp();

  // Display cell type
  cout << "Cell type" << endl;
  cout << "---------" << endl << endl;
  dolfin_begin();
  if ( cell_type )
    cout << cell_type->description() << endl;
  else
    cout << "undefined" << endl;
  dolfin_end();
  cout << endl;
  
  // End indentation
  dolfin_end();
}
//-----------------------------------------------------------------------------
