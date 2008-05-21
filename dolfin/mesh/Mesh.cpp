// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman, 2007.
// Modified by Garth N. Wells 2007.
//
// First added:  2006-05-09
// Last changed: 2008-05-19

#include <sstream>

#include <dolfin/io/File.h>
#include <dolfin/main/MPI.h>
#include "ALE.h"
#include "UniformMeshRefinement.h"
#include "LocalMeshRefinement.h"
#include "LocalMeshCoarsening.h"
#include "TopologyComputation.h"
#include "MeshOrdering.h"
#include "MeshFunction.h"
#include "MeshPartition.h"
#include "BoundaryMesh.h"
#include "Cell.h"
#include "Vertex.h"
#include "MPIMeshCommunicator.h"
#include "MeshData.h"
#include "Mesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Mesh::Mesh()
  : Variable("mesh", "DOLFIN mesh"), _data(0), _cell_type(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
  : Variable("mesh", "DOLFIN mesh"), _data(0), _cell_type(0)
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
Mesh::Mesh(std::string filename)
  : Variable("mesh", "DOLFIN mesh"), _data(0), _cell_type(0)
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Mesh::~Mesh()
{
  clear();
}
//-----------------------------------------------------------------------------
const Mesh& Mesh::operator=(const Mesh& mesh)
{
  clear();

  _topology = mesh._topology;
  _geometry = mesh._geometry;
  
  if (mesh._cell_type)
    _cell_type = CellType::create(mesh._cell_type->cellType());
  
  rename(mesh.name(), mesh.label());

  return *this;
}
//-----------------------------------------------------------------------------
MeshData& Mesh::data()
{
  if (!_data)
    _data = new MeshData(*this);

  return *_data;
}
//-----------------------------------------------------------------------------
dolfin::uint Mesh::init(uint dim)
{
  return TopologyComputation::computeEntities(*this, dim);
}
//-----------------------------------------------------------------------------
void Mesh::init(uint d0, uint d1)
{
  TopologyComputation::computeConnectivity(*this, d0, d1);
}
//-----------------------------------------------------------------------------
void Mesh::init()
{
  // Compute all entities
  for (uint d = 0; d <= topology().dim(); d++)
    init(d);

  // Compute all connectivity
  for (uint d0 = 0; d0 <= topology().dim(); d0++)
    for (uint d1 = 0; d1 <= topology().dim(); d1++)
      init(d0, d1);
}
//-----------------------------------------------------------------------------
void Mesh::clear()
{
  _topology.clear();
  _geometry.clear();
  delete _cell_type;
  _cell_type = 0;
  delete _data;
  _data = 0;
}
//-----------------------------------------------------------------------------
void Mesh::order()
{
  MeshOrdering::order(*this);
}
//-----------------------------------------------------------------------------
void Mesh::refine()
{
  message("No cells marked for refinement, assuming uniform mesh refinement.");
  UniformMeshRefinement::refine(*this);
}
//-----------------------------------------------------------------------------
void Mesh::refine(MeshFunction<bool>& cell_markers, bool refine_boundary)
{
  LocalMeshRefinement::refineMeshByEdgeBisection(*this, cell_markers,
                                                 refine_boundary);
}
//-----------------------------------------------------------------------------
void Mesh::coarsen()
{
  // FIXME: Move implementation to separate class and just call function here

  message("No cells marked for coarsening, assuming uniform mesh coarsening.");
  MeshFunction<bool> cell_marker(*this);
  cell_marker.init(this->topology().dim());
  for (CellIterator c(*this); !c.end(); ++c)
    cell_marker.set(c->index(),true);

  LocalMeshCoarsening::coarsenMeshByEdgeCollapse(*this,cell_marker);
}
//-----------------------------------------------------------------------------
void Mesh::coarsen(MeshFunction<bool>& cell_markers, bool coarsen_boundary)
{
  LocalMeshCoarsening::coarsenMeshByEdgeCollapse(*this, cell_markers,
                                                 coarsen_boundary);
}
//-----------------------------------------------------------------------------
void Mesh::move(Mesh& boundary, const MeshFunction<uint>& vertex_map,
                ALEMethod method)
{
  ALE::move(*this, boundary, vertex_map, method);
}
//-----------------------------------------------------------------------------
void Mesh::smooth() 
{
  // FIXME: Move implementation to separate class and just call function here

  MeshFunction<bool> bnd_vertex(*this); 
  bnd_vertex.init(0); 
  for (VertexIterator v(*this); !v.end(); ++v)
    bnd_vertex.set(v->index(),false);

  MeshFunction<uint> bnd_vertex_map; 
  MeshFunction<uint> bnd_cell_map; 
  BoundaryMesh boundary(*this,bnd_vertex_map,bnd_cell_map);

  for (VertexIterator v(boundary); !v.end(); ++v)
    bnd_vertex.set(bnd_vertex_map.get(v->index()),true);

  Point midpoint = 0.0; 
  uint num_neighbors = 0;
  for (VertexIterator v(*this); !v.end(); ++v)
  {
    if ( !bnd_vertex.get(v->index()) )
    {
      midpoint = 0.0;
      num_neighbors = 0;
      for (VertexIterator vn(*v); !vn.end(); ++vn)
      {
        if ( v->index() != vn->index() )
        {
          midpoint += vn->point();
          num_neighbors++;
        }
      }
      midpoint /= real(num_neighbors);

      for (uint sd = 0; sd < this->geometry().dim(); sd++)
        this->geometry().set(v->index(), sd, midpoint[sd]);
    }
  }
}
//-----------------------------------------------------------------------------
void Mesh::partition(MeshFunction<uint>& partitions)
{
  partition(partitions, MPI::numProcesses());
}
//-----------------------------------------------------------------------------
void Mesh::partition(MeshFunction<uint>& partitions, uint num_partitions)
{
  // Receive mesh partition function according to parallel policy
  if (MPI::receive()) { MPIMeshCommunicator::receive(partitions); return; }

  // Partition mesh
  MeshPartition::partition(*this, partitions, num_partitions);

  // Broadcast mesh according to parallel policy
  if (MPI::broadcast()) { MPIMeshCommunicator::broadcast(partitions); }
}
//-----------------------------------------------------------------------------
void Mesh::disp() const
{
  // Begin indentation
  cout << "Mesh data" << endl;
  begin("---------");
  cout << endl;

  // Display topology and geometry
  _topology.disp();
  _geometry.disp();

  // Display cell type
  cout << "Cell type" << endl;
  cout << "---------" << endl;
  begin("");
  if (_cell_type)
    cout << _cell_type->description() << endl;
  else
    cout << "undefined" << endl;
  end();

  // Display mesh data
  if (_data)
  {
    cout << endl;
    _data->disp();
  }
  
  // End indentation
  end();
}
//-----------------------------------------------------------------------------
std::string Mesh::str() const
{
  std::ostringstream stream;
  stream << "[Mesh of topological dimension "
         << numVertices()
         << " and "
         << numCells()
         << " cells]";
  return stream.str();
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Mesh& mesh)
{
  stream << mesh.str();
  return stream;
}
//-----------------------------------------------------------------------------
