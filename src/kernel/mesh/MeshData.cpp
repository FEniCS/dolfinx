// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-08
// Last changed: 2006-06-22

#include <dolfin/MeshData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshData::MeshData() : cell_type(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshData::MeshData(const MeshData& data) : cell_type(0)
{
  *this = data;
}
//-----------------------------------------------------------------------------
MeshData::~MeshData()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshData& MeshData::operator= (const MeshData& data)
{
  // Clear old data if any
  clear();

  // Assign data
  topology = data.topology;
  geometry = data.geometry;

  // Create new cell type
  if ( data.cell_type )
    cell_type = CellType::create(data.cell_type->cellType());
  else
    cell_type = 0;

  return *this;
}
//-----------------------------------------------------------------------------
void MeshData::clear()
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
void MeshData::disp() const
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
