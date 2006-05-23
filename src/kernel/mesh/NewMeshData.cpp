// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-05-18

#include <dolfin/NewMeshData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewMeshData::NewMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewMeshData::~NewMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewMeshData::clear()
{ 
  // Clear mesh topology
  topology.clear();

  // Clear mesh geometry
  geometry.clear();
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
  
  // End indentation
  dolfin_end();
}
//-----------------------------------------------------------------------------
