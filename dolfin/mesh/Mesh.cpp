// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Hoffman, 2007.
// Modified by Garth N. Wells 2007-2011.
// Modified by Niclas Jansson 2008.
// Modified by Kristoffer Selim 2008.
// Modified by Andre Massing, 2009-2010.
//
// First added:  2006-05-09
// Last changed: 2011-02-07

#include <dolfin/ale/ALE.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/UniqueIdGenerator.h>
#include <dolfin/common/utils.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include "BoundaryMesh.h"
#include "Cell.h"
#include "LocalMeshData.h"
#include "MeshColoring.h"
#include "MeshData.h"
#include "MeshFunction.h"
#include "MeshOrdering.h"
#include "MeshPartitioning.h"
#include "MeshRenumbering.h"
#include "MeshSmoothing.h"
#include "ParallelData.h"
#include "TopologyComputation.h"
#include "Vertex.h"
#include "Mesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Mesh::Mesh() : Variable("mesh", "DOLFIN mesh"),
               Hierarchical<Mesh>(*this),
               _data(*this),
               _parallel_data(new ParallelData(*this)),
               _cell_type(0),
               unique_id(UniqueIdGenerator::id()),
               _intersection_operator(*this),
               _ordered(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh) : Variable("mesh", "DOLFIN mesh"),
                               Hierarchical<Mesh>(*this),
                               _data(*this),
                               _parallel_data(new ParallelData(*this)),
                               _cell_type(0),
                               unique_id(UniqueIdGenerator::id()),
                               _intersection_operator(*this),
                               _ordered(false)
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
Mesh::Mesh(std::string filename) : Variable("mesh", "DOLFIN mesh"),
                                   Hierarchical<Mesh>(*this),
                                   _data(*this),
                                   _parallel_data(new ParallelData(*this)),
                                   _cell_type(0),
                                   unique_id(UniqueIdGenerator::id()),
                                   _intersection_operator(*this),
                                   _ordered(false)
{
  if (MPI::num_processes() > 1)
  {
    // Read local mesh data
    Timer timer("PARALLEL 0: Parse local mesh data");
    File file(filename);
    LocalMeshData local_data;
    file >> local_data;
    timer.stop();

    // Partition data
    MeshPartitioning::partition(*this, local_data);
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
  // Clear all data
  clear();

  // Assign data
  _topology = mesh._topology;
  _geometry = mesh._geometry;
  _data = mesh._data;
  _parallel_data.reset(new ParallelData(*mesh._parallel_data));
  if (mesh._cell_type)
    _cell_type = CellType::create(mesh._cell_type->cell_type());

  // Rename
  rename(mesh.name(), mesh.label());

  // Call assignment operator for base class
  Hierarchical<Mesh>::operator=(mesh);

  return *this;
}
//-----------------------------------------------------------------------------
MeshData& Mesh::data()
{
  return _data;
}
//-----------------------------------------------------------------------------
const MeshData& Mesh::data() const
{
  return _data;
}
//-----------------------------------------------------------------------------
ParallelData& Mesh::parallel_data()
{
  assert(_parallel_data);
  return *_parallel_data;
}
//-----------------------------------------------------------------------------
const ParallelData& Mesh::parallel_data() const
{
  assert(_parallel_data);
  return *_parallel_data;
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
        _topology.clear(d0, d1);
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
dolfin::Mesh Mesh::renumber_by_color() const
{
  std::vector<uint> coloring_type;
  const uint D = topology().dim();
  coloring_type.push_back(D); coloring_type.push_back(0); coloring_type.push_back(D);
  return MeshRenumbering::renumber_by_color(*this, coloring_type);
}
//-----------------------------------------------------------------------------
void Mesh::move(BoundaryMesh& boundary)
{
  ALE::move(*this, boundary);
}
//-----------------------------------------------------------------------------
void Mesh::move(Mesh& mesh)
{
  ALE::move(*this, mesh);
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
const dolfin::MeshFunction<dolfin::uint>&
Mesh::color(std::string coloring_type) const
{
  // Define graph type
  const uint dim = MeshColoring::type_to_dim(coloring_type, *this);
  std::vector<uint> _coloring_type;
  _coloring_type.push_back(topology().dim());
  _coloring_type.push_back(dim);
  _coloring_type.push_back(topology().dim());

  return color(_coloring_type);
}
//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>&
Mesh::color(std::vector<uint> coloring_type) const
{
  // Find color data
  std::map<const std::vector<uint>, std::pair<MeshFunction<uint>,
           std::vector<std::vector<uint> > > >::const_iterator coloring_data;
  coloring_data = this->parallel_data().coloring.find(coloring_type);

  if (coloring_data != this->parallel_data().coloring.end())
  {
    dolfin_debug("Mesh has already been colored, not coloring again.");
    return coloring_data->second.first;
  }

  // We do the same const-cast trick here as in the init() functions
  // since we are not really changing the mesh, just attaching some
  // auxiliary data to it.
  Mesh* _mesh = const_cast<Mesh*>(this);
  return MeshColoring::color(*_mesh, coloring_type);
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const Point& point,
                             std::set<uint>& cells) const
{
  _intersection_operator.all_intersected_entities(point, cells);
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const std::vector<Point>& points,
                             std::set<uint>& cells) const
{
  _intersection_operator.all_intersected_entities(points, cells);
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const MeshEntity & entity,
                             std::vector<uint>& cells) const
{
  _intersection_operator.all_intersected_entities(entity, cells);
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const std::vector<MeshEntity>& entities,
                             std::set<uint>& cells) const
{
  _intersection_operator.all_intersected_entities(entities, cells);
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const Mesh& another_mesh,
                             std::set<uint>& cells) const
{
  _intersection_operator.all_intersected_entities(another_mesh, cells);
}
//-----------------------------------------------------------------------------
int Mesh::intersected_cell(const Point& point) const
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
std::pair<Point,dolfin::uint>
Mesh::closest_point_and_cell(const Point & point) const
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
void Mesh::initialize_exterior_facet_domains()
{
  // Do nothing if mesh function "exterior_facet_domains" is present
  if (_data.mesh_function("exterior_facet_domains"))
    return;

  // Extract data for boundary indicators
  boost::shared_ptr<const std::vector<uint> >
    boundary_indicators = _data.array("boundary_indicators");
  boost::shared_ptr<const std::vector<uint> >
    boundary_facet_cells = _data.array("boundary_facet_cells");
  boost::shared_ptr<const std::vector<uint> >
    boundary_facet_numbers = _data.array("boundary_facet_numbers");

  // Do nothing if there are no indicators
  if (!boundary_indicators)
    return;

  // Need facet cells and numbers if indicators are present
  if (!boundary_facet_cells || !boundary_facet_numbers)
    dolfin_error("Mesh.cpp",
                 "initialize boundary indicators",
                 "Mesh has boundary indicators but missing data for \"boundary_facet_cells\" and \"boundary_facet_numbers\"");
  const uint num_facets = boundary_indicators->size();
  assert(num_facets > 0);
  assert(boundary_facet_cells->size() == num_facets);
  assert(boundary_facet_numbers->size() == num_facets);

  // Initialize facets
  const uint D = _topology.dim();
  order();
  init(D - 1);

  // Create mesh function "exterior_facet_domains"
  boost::shared_ptr<MeshFunction<unsigned int> > exterior_facet_domains
    = _data.create_mesh_function("exterior_facet_domains", D - 1);
  assert(exterior_facet_domains);

  // Initialize meshfunction to zero
  exterior_facet_domains->set_all(0);

  // Assign domain numbers for each facet
  for (uint i = 0; i < num_facets; i++)
  {
    // Get cell index and local facet index
    const uint cell_index = (*boundary_facet_cells)[i];
    const uint local_facet = (*boundary_facet_numbers)[i];

    // Get global facet index
    const uint global_facet = _topology(D, D - 1)(cell_index)[local_facet];

    // Set boundary indicator for facet
    (*exterior_facet_domains)[global_facet] = (*boundary_indicators)[i];
  }
}
//-----------------------------------------------------------------------------
