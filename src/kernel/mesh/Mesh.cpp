// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-06-22

#include <dolfin/File.h>
#include <dolfin/UniformMeshRefinement.h>
#include <dolfin/TopologyComputation.h>
#include <dolfin/Mesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Mesh::Mesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
Mesh::Mesh(std::string filename)
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Mesh::~Mesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Mesh& Mesh::operator=(const Mesh& mesh)
{
  data = mesh.data;
  return *this;
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
void Mesh::disp() const
{
  data.disp();
}
//-----------------------------------------------------------------------------
void Mesh::refine()
{
  dolfin_info("No cells marked for refinement, assuming uniform mesh refinement.");
  UniformMeshRefinement::refine(*this);
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Mesh& mesh)
{
  stream << "[ Mesh of topological dimension "
	 << mesh.topology().dim()
	 << " with "
	 << mesh.numVertices()
	 << " vertices and "
	 << mesh.numCells()
	 << " cells ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
