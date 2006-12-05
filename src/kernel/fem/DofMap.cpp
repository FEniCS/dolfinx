// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-12-05
// Last changed: 2006

#include <algorithm>  
#include <dolfin/Cell.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/DofMap.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(Mesh& mesh, const FiniteElement* element_0, 
               const FiniteElement* element_1) : mesh(&mesh)
{  
  // Initialise data
  element[0] = element_0;
  element[1] = element_1;

  _size[0] = 0;
  _size[1] = 0;

  map[0].clear();
  map[1].clear();

  vector_sparsity_pattern.clear();
  matrix_sparsity_pattern.clear();

  // Intialise mesh connectivity if not already done
  for (uint i = 0; i < mesh.topology().dim(); i++)
    mesh.init(i);

  // This is needed for higher order Lagrange elements (degree >= 4)
  // to compute the alignment of faces
  if ( mesh.topology().dim() == 3 )
    mesh.init(2, 1);
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMap::attach(const FiniteElement* element_0, const FiniteElement* element_1)
{
  element[0]= element_0;
  element[1]= element_1;
}
//-----------------------------------------------------------------------------
void DofMap::dofmap(int dof_map[], const Cell& cell, const uint e) const
{
  dolfin_assert( e < 2 );
  if ( !element[e] )
    dolfin_error("No FiniteElement associated with DofMap.");

  element[e]->nodemap(dof_map, cell, *mesh);
}
//-----------------------------------------------------------------------------
//const Array< Array<int> >& DofMap::getMap(const uint e) const
//{
//  return map[e];
//}
//-----------------------------------------------------------------------------
const dolfin::uint DofMap::size(const uint e)
{
  dolfin_assert( e < 2);

  if ( !element[e] )
    dolfin_error("No FiniteElement associated with DofMap.");
  
  int num_dof = 0;
  int* dof_map_array = new int[ element[e]->spacedim() ];
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    element[e]->nodemap(dof_map_array, *cell, *mesh);
    for (uint i = 0; i < element[e]->spacedim(); i++)
      num_dof = std::max(dof_map_array[i], num_dof);
  }
  _size[e] = num_dof + 1;
  delete [] dof_map_array;  

  return _size[e];
}
//-----------------------------------------------------------------------------
dolfin::uint DofMap::numNonZeroes()
{
  // Compute matrix sparsity pattern 
  computeMatrixSparsityPattern();  

  uint nzmax = 0;
  for(int i = 0; i < _size[0]; ++i)
    nzmax = std::max(matrix_sparsity_pattern[i].size(), nzmax);

  return nzmax;
}
//-----------------------------------------------------------------------------
void DofMap::numNonZeroesRow(int nz_row[])
{
  // Compute matrix sparsity pattern 
  computeMatrixSparsityPattern();  

  const int nrow = size(0);
  // Insert number of non-zeroes per row into integer array 
  for(int i = 0; i < nrow; ++i)
    nz_row[i] = matrix_sparsity_pattern[i].size();
}
//-----------------------------------------------------------------------------
void DofMap::build(const uint e)
{
  dolfin_assert( e < 2 );

  if ( !element[e] )
    dolfin_error("No FiniteElement associated with DofMap.");

  map[e].clear();
  map[e].reserve( mesh->numCells() );

  int* dof_map_array = new int[ element[e]->spacedim() ];
  Array<int> dof_map_cell( element[e]->spacedim());

  int num_dof = 0;
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    // Get map for cell from FFC
    element[e]->nodemap(dof_map_array, *cell, *mesh);

    // Copy map to dof_map_cell 
    for (uint i = 0; i < element[e]->spacedim(); i++)
    {
      dof_map_cell[i] = dof_map_array[i]; 
      num_dof = std::max(dof_map_array[i], num_dof);
    }

    // Add dof_map_cell to global map
    map[e].push_back(dof_map_cell);
  }
  _size[e] = num_dof + 1;
  delete [] dof_map_array;
}
//-----------------------------------------------------------------------------
void DofMap::computeVectorSparsityPattern(const uint e)
{
  dolfin_assert(e < 2);

  if ( !element[e] )
    dolfin_error("No FiniteElement associated with DofMap.");

  // Initialise sparsity pattern
  vector_sparsity_pattern.clear();

  int* dof = new int[ element[e]->spacedim() ];
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    element[e]->nodemap(dof, *cell, *mesh);
    for (uint i = 0; i < element[e]->spacedim(); ++i)
      vector_sparsity_pattern.insert( dof[i] );
  }
  delete [] dof;
}
//-----------------------------------------------------------------------------
void DofMap::computeMatrixSparsityPattern()
{
  if( !element[0] && !element[1] ) 
    dolfin_error("Two finite elements must be associated with DofMap object to build matrix sparsity pattern.");

  // Initialise sparsity pattern
  matrix_sparsity_pattern.clear();
  matrix_sparsity_pattern.resize( size(0) );

  int* row_dof    = new int[ element[0]->spacedim()];
  int* column_dof = new int[ element[1]->spacedim()];

  // Build sparsity pattern
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    element[0]->nodemap(row_dof, *cell, *mesh);
    for (uint i = 0; i < element[0]->spacedim(); ++i)
    {
      element[1]->nodemap(column_dof, *cell, *mesh);
      for (uint j = 0; j < element[1]->spacedim(); ++j)
        matrix_sparsity_pattern[ row_dof[i] ].insert( column_dof[j] );
    }
  }
  delete [] row_dof;
  delete [] column_dof;
}
//-----------------------------------------------------------------------------
