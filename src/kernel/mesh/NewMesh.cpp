// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-06-21

#include <dolfin/File.h>
#include <dolfin/UniformMeshRefinement.h>
#include <dolfin/TopologyComputation.h>
#include <dolfin/NewMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewMesh::NewMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewMesh::NewMesh(const NewMesh& mesh)
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
NewMesh::NewMesh(std::string filename)
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
NewMesh::~NewMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const NewMesh& NewMesh::operator=(const NewMesh& mesh)
{
  data = mesh.data;
  return *this;
}
//-----------------------------------------------------------------------------
dolfin::uint NewMesh::init(uint dim)
{
  return TopologyComputation::computeEntities(*this, dim);
}
//-----------------------------------------------------------------------------
void NewMesh::init(uint d0, uint d1)
{
  TopologyComputation::computeConnectivity(*this, d0, d1);
}
//-----------------------------------------------------------------------------
void NewMesh::init()
{
  // Compute all entities
  for (uint d = 0; d <= dim(); d++)
    init(d);

  // Compute all connectivity
  for (uint d0 = 0; d0 <= dim(); d0++)
    for (uint d1 = 0; d1 <= dim(); d1++)
      init(d0, d1);
}
//-----------------------------------------------------------------------------
void NewMesh::disp() const
{
  data.disp();
}
//-----------------------------------------------------------------------------
void NewMesh::refine()
{
  dolfin_info("No cells marked for refinement, assuming uniform mesh refinement.");
  UniformMeshRefinement::refine(*this);
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const NewMesh& mesh)
{
  stream << "[ Mesh of dimension "
	 << mesh.dim()
	 << " with "
	 << mesh.numVertices()
	 << " vertices and "
	 << mesh.numCells()
	 << " cells ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
