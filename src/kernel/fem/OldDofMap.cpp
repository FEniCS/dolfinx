// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2007.
//
// First added:  2006-12-05
// Last changed: 2007-01-16

#include <algorithm>

#include <dolfin/Cell.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/OldDofMap.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
OldDofMap::OldDofMap(Mesh& mesh, const FiniteElement* element_0, 
               const FiniteElement* element_1) : mesh(&mesh)
{  
  // Initialise data
  element[0] = element_0;
  element[1] = element_1;

  // Only one element, Cuthill-McKee needs symmetric adjacency matrix
  finite_element = element_0;

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

  adjacency_graph = 0;
}
//-----------------------------------------------------------------------------
OldDofMap::OldDofMap(Mesh& mesh, const ufc::dof_map& dof_map)
{
  adjacency_graph = 0;

  // Not implemented

}
//-----------------------------------------------------------------------------
OldDofMap::~OldDofMap()
{
  if ( adjacency_graph )
    delete adjacency_graph;
}
//-----------------------------------------------------------------------------
void OldDofMap::attach(const FiniteElement* element_0, const FiniteElement* element_1)
{
  // FIXME: Remove this function and require elements in constructor
  element[0] = element_0;
  element[1] = element_1;

  finite_element = element_0;
}
//-----------------------------------------------------------------------------
void OldDofMap::dofmap(int dof_map[], const Cell& cell, const uint e) const
{
  dolfin_assert( e < 2 );
  if ( !element[e] )
    dolfin_error("No FiniteElement associated with OldDofMap.");

  element[e]->nodemap(dof_map, cell, *mesh);
}
//-----------------------------------------------------------------------------
//const Array< Array<int> >& OldDofMap::getMap(const uint e) const
//{
//  return map[e];
//}
//-----------------------------------------------------------------------------
const dolfin::uint OldDofMap::size(const uint e)
{
  // FIXME: Here the size is computed my checking the maximum index of
  // FIXME: global dofs mapped to by the FFC-generated nodemap.
  // FIXME: This will be replaced ufc::dof_map::global_dimension()
  
  dolfin_assert( e < 2);

  if ( !element[e] )
    dolfin_error("No FiniteElement associated with OldDofMap.");
  
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
dolfin::uint OldDofMap::numNonZeroes()
{
  if ( matrix_sparsity_pattern.size() == 0 )
    computeMatrixSparsityPattern();

  

  if ( _size[0] == 0 )
    size(0);

  uint num_nonzero = 0;
  for (int i = 0; i < _size[0]; ++i)
    num_nonzero += matrix_sparsity_pattern[i].size();

  return num_nonzero;
}
//-----------------------------------------------------------------------------
dolfin::uint OldDofMap::numNonZeroesRowMax()
{
  if ( matrix_sparsity_pattern.size() == 0 )
    computeMatrixSparsityPattern();

  if ( _size[0] == 0 )
    size(0);

  // nzmax needs to of type size_t since matrix_sparsity_pattern[i].size()
  // is of type size_t, otherwise std::max might complain.
  size_t nzmax = 0;
  for (int i = 0; i < _size[0]; ++i)
    nzmax = std::max(matrix_sparsity_pattern[i].size(), nzmax);

  return nzmax;
}
//-----------------------------------------------------------------------------
void OldDofMap::numNonZeroesRow(int nz_row[])
{
  if ( matrix_sparsity_pattern.size() == 0 )
    computeMatrixSparsityPattern();

  if( _size[0] == 0 )
    size(0);

  const int num_rows = _size[0];
  for (int i = 0; i < num_rows; ++i)
    nz_row[i] = matrix_sparsity_pattern[i].size();
}
//-----------------------------------------------------------------------------
void OldDofMap::build(const uint e)
{
  dolfin_assert( e < 2 );

  if ( !element[e] )
    dolfin_error("No FiniteElement associated with OldDofMap.");

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
void OldDofMap::computeVectorSparsityPattern(const uint e)
{
  dolfin_assert(e < 2);

  if ( !element[e] )
    dolfin_error("No FiniteElement associated with OldDofMap.");

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
void OldDofMap::computeMatrixSparsityPattern()
{
  if ( !element[0] || !element[1] ) 
    dolfin_error("Two finite elements must be associated with OldDofMap object to build matrix sparsity pattern.");

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
void OldDofMap::createCSRLayout()
{
  if( matrix_sparsity_pattern.size() == 0 )
    computeMatrixSparsityPattern();

  Array<int> row_pointer( size(0) +1 );

  Array<int> column_index;
  column_index.clear();
  column_index.reserve( numNonZeroes() );

  std::set<int>::iterator column;

  row_pointer[ 0 ] = 0; 
  for(uint row = 0; row < size(0); ++row)
  {
    const std::set<int>& row_set = matrix_sparsity_pattern[ row ];
    for(column = row_set.begin(); column != row_set.end(); ++column )
      column_index.push_back(*column);

    row_pointer[ row + 1 ] = row_pointer[ row ] + row_set.size(); 
  }
}
//-----------------------------------------------------------------------------
void OldDofMap::computeAdjacencyGraph()
{
  cout << "Computing adjacency graph..." << endl;
  
  // Delete old graph if previously computed
  if ( adjacency_graph )
    delete adjacency_graph;

  // Create new adjacency graph with given number of vertices
  adjacency_graph = new AdjacencyGraph(size(0));

  // Iterate over the cells and build the graph
  uint n = finite_element->spacedim();
  int* row_dofs = new int[n];
  int* col_dofs = new int[n];
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    finite_element->nodemap(row_dofs, *cell, *mesh);
    for (uint i = 0; i < n; ++i)
    {
      finite_element->nodemap(col_dofs, *cell, *mesh);
      for (uint j = 0; j < element[1]->spacedim(); ++j)
        boost::add_edge(row_dofs[i], row_dofs[j], *adjacency_graph);
    }
  }
  delete [] row_dofs;
  delete [] col_dofs;
}
//-----------------------------------------------------------------------------
