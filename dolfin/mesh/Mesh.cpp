// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman, 2007.
// Modified by Garth N. Wells 2007.
// Modified by Niclas Jansson 2008.
// Modified by Kristoffer Selim 2008.
// Modified by Andre Massing, 2009-2010.
//
// First added:  2006-05-09
// Last changed: 2010-11-25

#include <sstream>
#include <vector>

#include <dolfin/log/log.h>
#include <dolfin/io/File.h>
#include <dolfin/main/MPI.h>
#include <dolfin/ale/ALE.h>
#include <dolfin/io/File.h>
#include <dolfin/common/utils.h>
#include <dolfin/common/Timer.h>
#include "IntersectionOperator.h"
#include "TopologyComputation.h"
#include "MeshSmoothing.h"
#include "MeshOrdering.h"
#include "MeshFunction.h"
#include "LocalMeshData.h"
#include "MeshPartitioning.h"
#include "MeshColoring.h"
#include "BoundaryMesh.h"
#include "Cell.h"
#include "Vertex.h"
#include "Mesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Mesh::Mesh() : Variable("mesh", "DOLFIN mesh"), _data(*this), _cell_type(0),
               _intersection_operator(*this), _ordered(false), _colored(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh) : Variable("mesh", "DOLFIN mesh"), _data(*this),
                               _cell_type(0), _intersection_operator(*this),
                               _ordered(false), _colored(-1)
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
Mesh::Mesh(std::string filename) : Variable("mesh", "DOLFIN mesh"),
                                   _data(*this), _cell_type(0),
                                   _intersection_operator(*this),
                                   _ordered(false), _colored(-1)
{
  if (MPI::num_processes() > 1)
  {
    // Read local mesh data
    Timer timer("PARALLEL 0: Parse local mesh data");
    File file(filename);
    LocalMeshData data;
    file >> data;
    timer.stop();

    // Partition data
    MeshPartitioning::partition(*this, data);
  }
  else
  {
    File file(filename);
    file >> *this;
  }
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
  _data = mesh._data;

  if (mesh._cell_type)
    _cell_type = CellType::create(mesh._cell_type->cell_type());

  rename(mesh.name(), mesh.label());

  _ordered = mesh._ordered;

   return *this;
}
//-----------------------------------------------------------------------------
dolfin::uint Mesh::init(uint dim) const
{
  // This function is obviously not const since it may potentially compute
  // new connectivity. However, in a sense all connectivity of a mesh always
  // exists, it just hasn't been computed yet. The const_cast is also needed
  // to allow iterators over a const Mesh to create new connectivity.

  // Skip if already computed
  if (_topology.size(dim) > 0)
    return _topology.size(dim);

  // Skip vertices and cells (should always exist)
  if (dim == 0 || dim == _topology.dim())
    return _topology.size(dim);

  // Check that mesh is ordered
  if (!ordered())
    error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_entities(*mesh, dim);

  // Order mesh if necessary
  if (!ordered())
    mesh->order();

  return _topology.size(dim);
}
//-----------------------------------------------------------------------------
void Mesh::init(uint d0, uint d1) const
{
  // This function is obviously not const since it may potentially compute
  // new connectivity. However, in a sense all connectivity of a mesh always
  // exists, it just hasn't been computed yet. The const_cast is also needed
  // to allow iterators over a const Mesh to create new connectivity.

  // Skip if already computed
  if (_topology(d0, d1).size() > 0)
    return;

  // Check that mesh is ordered
  if (!ordered())
    error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_connectivity(*mesh, d0, d1);

  // Order mesh if necessary
  if (!ordered())
    mesh->order();
}
//-----------------------------------------------------------------------------
void Mesh::init() const
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
  _data.clear();
  delete _cell_type;
  _cell_type = 0;
  _intersection_operator.clear();
  _ordered = false;
}
//-----------------------------------------------------------------------------
void Mesh::clean()
{
  const uint D = topology().dim();
  for (uint d0 = 0; d0 <= D; d0++)
  {
    for (uint d1 = 0; d1 <= D; d1++)
    {
      if (!(d0 == D && d1 == 0))
        topology.clear(d0, d1);
    }
  }
}
//-----------------------------------------------------------------------------
void Mesh::order()
{
  // Order mesh
  MeshOrdering::order(*this);

  // Remember that the mesh has been ordered
  _ordered = true;
}
//-----------------------------------------------------------------------------
bool Mesh::ordered() const
{
  // Don't check if we know (or think we know) that the mesh is ordered
  if (_ordered)
    return true;

  _ordered = MeshOrdering::ordered(*this);
  return _ordered;
}
//-----------------------------------------------------------------------------
void Mesh::move(BoundaryMesh& boundary, ALEType method)
{
  ALE::move(*this, boundary, method);
}
//-----------------------------------------------------------------------------
void Mesh::move(Mesh& mesh, ALEType method)
{
  ALE::move(*this, mesh, method);
}
//-----------------------------------------------------------------------------
void Mesh::move(const Function& displacement)
{
  ALE::move(*this, displacement);
}
//-----------------------------------------------------------------------------
void Mesh::smooth(uint num_iterations)
{
  MeshSmoothing::smooth(*this, num_iterations);
}
//-----------------------------------------------------------------------------
void Mesh::smooth_boundary(uint num_iterations, bool harmonic_smoothing)
{
  MeshSmoothing::smooth_boundary(*this, num_iterations, harmonic_smoothing);
}
//-----------------------------------------------------------------------------
void Mesh::snap_boundary(const SubDomain& sub_domain, bool harmonic_smoothing)
{
  MeshSmoothing::snap_boundary(*this, sub_domain, harmonic_smoothing);
}
//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>& Mesh::color(std::string coloring_type) const
{
  // Check if mesh has already been colored
  const uint dim = MeshColoring::type_to_dim(coloring_type, *this);
  if (static_cast<int>(dim) == _colored)
  {
    info("Mesh has already been colored, not coloring again.");
    MeshFunction<uint>* colors = _data.mesh_function("cell colors");
    assert(colors);
    return *colors;
  }

  // We do the same const-cast trick here as in the init() functions
  // since we are not really changing the mesh, just attaching some
  // auxiliary data to it.
  Mesh* _mesh = const_cast<Mesh*>(this);
  _colored = dim;
  return MeshColoring::color_cells(*_mesh, coloring_type);
}
//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>& Mesh::color(uint dim) const
{
  // Check if mesh has already been colored
  if (static_cast<int>(dim) == _colored)
  {
    info("Mesh has already been colored, not coloring again.");
    MeshFunction<uint>* colors = _data.mesh_function("cell colors");
    assert(colors);
    return *colors;
  }

  // We do the same const-cast trick here as in the init() functions
  // since we are not really changing the mesh, just attaching some
  // auxiliary data to it.
  Mesh* _mesh = const_cast<Mesh*>(this);
  _colored = dim;
  return MeshColoring::color_cells(*_mesh, dim);
}
//-----------------------------------------------------------------------------
void Mesh::all_intersected_entities(const Point& point,
                                    uint_set& ids_result) const
{
  _intersection_operator.all_intersected_entities(point, ids_result);
}
//-----------------------------------------------------------------------------
void Mesh::all_intersected_entities(const std::vector<Point>& points,
                                    uint_set& ids_result) const
{
  _intersection_operator.all_intersected_entities(points, ids_result);
}
//-----------------------------------------------------------------------------
void Mesh::all_intersected_entities(const MeshEntity & entity,
                                    std::vector<uint>& ids_result) const
{
  _intersection_operator.all_intersected_entities(entity, ids_result);
}
//-----------------------------------------------------------------------------
void Mesh::all_intersected_entities(const std::vector<MeshEntity>& entities,
                                    uint_set& ids_result) const
{
  _intersection_operator.all_intersected_entities(entities, ids_result);
}
//-----------------------------------------------------------------------------
void Mesh::all_intersected_entities(const Mesh& another_mesh,
                                    uint_set& ids_result) const
{
  _intersection_operator.all_intersected_entities(another_mesh, ids_result);
}
//-----------------------------------------------------------------------------
int Mesh::any_intersected_entity(const Point& point) const
{
  return _intersection_operator.any_intersected_entity(point);
}
//-----------------------------------------------------------------------------
Point Mesh::closest_point(const Point& point) const
{
  return _intersection_operator.closest_point(point);
}
//-----------------------------------------------------------------------------
dolfin::uint Mesh::closest_cell(const Point & point) const
{
  return _intersection_operator.closest_cell(point);
}
//-----------------------------------------------------------------------------
std::pair<Point,dolfin::uint> Mesh::closest_point_and_cell(const Point & point) const
{
  return _intersection_operator.closest_point_and_cell(point);
}
//-----------------------------------------------------------------------------
IntersectionOperator& Mesh::intersection_operator()
{
  return _intersection_operator;
}
//-----------------------------------------------------------------------------
const IntersectionOperator& Mesh::intersection_operator() const
{
  return _intersection_operator;
}
//-----------------------------------------------------------------------------
double Mesh::hmin() const
{
  CellIterator cell(*this);
  double h = cell->diameter();
  for (; !cell.end(); ++cell)
    h = std::min(h, cell->diameter());

  return h;
}
//-----------------------------------------------------------------------------
double Mesh::hmax() const
{
  CellIterator cell(*this);
  double h = cell->diameter();
  for (; !cell.end(); ++cell)
    h = std::max(h, cell->diameter());

  return h;
}
//-----------------------------------------------------------------------------
std::string Mesh::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << indent(_geometry.str(true));
    s << indent(_topology.str(true));
    s << indent(_data.str(true));
  }
  else
  {
    s << "<Mesh of topological dimension "
      << topology().dim() << " ("
      << _cell_type->description(true) << ") with "
      << num_vertices() << " vertices and "
      << num_cells() << " cells, "
      << (_ordered ? "ordered" : "unordered") << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
