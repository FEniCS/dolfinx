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
// Modified by Johan Hoffman 2007
// Modified by Garth N. Wells 2007-2011
// Modified by Niclas Jansson 2008
// Modified by Kristoffer Selim 2008
// Modified by Andre Massing 2009-2010
// Modified by Johannes Ring 2012
// Modified by Marie E. Rognes 2012
// Modified by Mikael Mortensen 2012
// Modified by Jan Blechta 2013
//
// First added:  2006-05-09
// Last changed: 2013-03-05

#include <boost/serialization/map.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/ale/ALE.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/common/Array.h>
#include <dolfin/generation/CSGMeshGenerator.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/function/Expression.h>
#include "BoundaryMesh.h"
#include "Cell.h"
#include "LocalMeshData.h"
#include "MeshColoring.h"
#include "MeshData.h"
#include "MeshFunction.h"
#include "MeshValueCollection.h"
#include "MeshOrdering.h"
#include "MeshPartitioning.h"
#include "MeshRenumbering.h"
#include "MeshSmoothing.h"
#include "MeshTransformation.h"
#include "TopologyComputation.h"
#include "Vertex.h"
#include "Mesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Mesh::Mesh() : Variable("mesh", "DOLFIN mesh"),
               Hierarchical<Mesh>(*this),
               _domains(*this),
               _data(*this),
               _cell_type(0),
               _intersection_operator(*this),
               _ordered(false),
               _cell_orientations(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh) : Variable("mesh", "DOLFIN mesh"),
                               Hierarchical<Mesh>(*this),
			       _domains(*this),
                               _data(*this),
                               _cell_type(0),
                               _intersection_operator(*this),
                               _ordered(false),
                               _cell_orientations(0)
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
Mesh::Mesh(std::string filename) : Variable("mesh", "DOLFIN mesh"),
                                   Hierarchical<Mesh>(*this),
				   _domains(*this),
                                   _data(*this),
                                   _cell_type(0),
                                   _intersection_operator(*this),
                                   _ordered(false),
                                   _cell_orientations(0)
{
  File file(filename);
  file >> *this;

  _cell_orientations.resize(this->num_cells(), -1);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(LocalMeshData& local_mesh_data)
                                 : Variable("mesh", "DOLFIN mesh"),
                                   Hierarchical<Mesh>(*this),
				   _domains(*this),
                                   _data(*this),
                                   _cell_type(0),
                                   _intersection_operator(*this),
                                   _ordered(false),
                                   _cell_orientations(0)
{
  MeshPartitioning::build_distributed_mesh(*this, local_mesh_data);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const CSGGeometry& geometry, std::size_t mesh_resolution)
  : Variable("mesh", "DOLFIN mesh"),
    Hierarchical<Mesh>(*this),
    _domains(*this),
    _data(*this),
    _cell_type(0),
    _intersection_operator(*this),
    _ordered(false),
    _cell_orientations(0)

{
  // Build mesh on process 0
  if (MPI::process_number() == 0)
    CSGMeshGenerator::generate(*this, geometry, mesh_resolution);

  // Build distributed mesh
  if (MPI::num_processes() > 1)
    MeshPartitioning::build_distributed_mesh(*this);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(boost::shared_ptr<const CSGGeometry> geometry, std::size_t resolution)
  : Variable("mesh", "DOLFIN mesh"),
    Hierarchical<Mesh>(*this),
    _domains(*this),
    _data(*this),
    _cell_type(0),
    _intersection_operator(*this),
    _ordered(false),
    _cell_orientations(0)
{
  assert(geometry);

  // Build mesh on process 0
  if (MPI::process_number() == 0)
    CSGMeshGenerator::generate(*this, *geometry, resolution);

  // Build distributed mesh
  if (MPI::num_processes() > 1)
    MeshPartitioning::build_distributed_mesh(*this);
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
  _domains = mesh._domains;
  _data = mesh._data;
  if (mesh._cell_type)
    _cell_type = CellType::create(mesh._cell_type->cell_type());
  _cell_orientations = mesh._cell_orientations;

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
std::size_t Mesh::init(std::size_t dim) const
{
  // This function is obviously not const since it may potentially compute
  // new connectivity. However, in a sense all connectivity of a mesh always
  // exists, it just hasn't been computed yet. The const_cast is also needed
  // to allow iterators over a const Mesh to create new connectivity.

  // Skip if mesh is empty
  if (num_cells() == 0)
  {
    warning("Mesh is empty, unable to create entities of dimension %d.", dim);
    return 0;
  }

  // Skip if already computed
  if (_topology.size(dim) > 0)
    return _topology.size(dim);

  // Skip vertices and cells (should always exist)
  if (dim == 0 || dim == _topology.dim())
    return _topology.size(dim);

  // Check that mesh is ordered
  if (!ordered())
  {
    dolfin_error("Mesh.cpp",
                 "initialize mesh entities",
                 "Mesh is not ordered according to the UFC numbering convention. Consider calling mesh.order()");
  }

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_entities(*mesh, dim);

  // Order mesh if necessary
  if (!ordered())
    mesh->order();

  return _topology.size(dim);
}
//-----------------------------------------------------------------------------
void Mesh::init(std::size_t d0, std::size_t d1) const
{
  // This function is obviously not const since it may potentially compute
  // new connectivity. However, in a sense all connectivity of a mesh always
  // exists, it just hasn't been computed yet. The const_cast is also needed
  // to allow iterators over a const Mesh to create new connectivity.

  // Skip if mesh is empty
  if (num_cells() == 0)
  {
    warning("Mesh is empty, unable to create connectivity %d --> %d.", d0, d1);
    return;
  }

  // Skip if already computed
  if (!_topology(d0, d1).empty())
    return;

  // Check that mesh is ordered
  if (!ordered())
  {
    dolfin_error("Mesh.cpp",
                 "initialize mesh connectivity",
                 "Mesh is not ordered according to the UFC numbering convention. Consider calling mesh.order()");
  }

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
  for (std::size_t d = 0; d <= topology().dim(); d++)
    init(d);

  // Compute all connectivity
  for (std::size_t d0 = 0; d0 <= topology().dim(); d0++)
    for (std::size_t d1 = 0; d1 <= topology().dim(); d1++)
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
  _cell_orientations.clear();
}
//-----------------------------------------------------------------------------
void Mesh::clean()
{
  const std::size_t D = topology().dim();
  for (std::size_t d0 = 0; d0 <= D; d0++)
  {
    for (std::size_t d1 = 0; d1 <= D; d1++)
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

  // Clear cell_orientations (as these depend on the ordering)
  _cell_orientations.clear();
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
  std::vector<std::size_t> coloring_type;
  const std::size_t D = topology().dim();
  coloring_type.push_back(D); coloring_type.push_back(0); coloring_type.push_back(D);
  return MeshRenumbering::renumber_by_color(*this, coloring_type);
}
//-----------------------------------------------------------------------------
void Mesh::rotate(double angle, std::size_t axis)
{
  MeshTransformation::rotate(*this, angle, axis);
}
//-----------------------------------------------------------------------------
void Mesh::rotate(double angle, std::size_t axis, const Point& p)
{
  MeshTransformation::rotate(*this, angle, axis, p);
}
//-----------------------------------------------------------------------------
MeshDisplacement Mesh::move(BoundaryMesh& boundary)
{
  return ALE::move(*this, boundary);
}
//-----------------------------------------------------------------------------
MeshDisplacement Mesh::move(Mesh& mesh)
{
  return ALE::move(*this, mesh);
}
//-----------------------------------------------------------------------------
void Mesh::move(const GenericFunction& displacement)
{
  ALE::move(*this, displacement);
}
//-----------------------------------------------------------------------------
void Mesh::smooth(std::size_t num_iterations)
{
  MeshSmoothing::smooth(*this, num_iterations);
}
//-----------------------------------------------------------------------------
void Mesh::smooth_boundary(std::size_t num_iterations, bool harmonic_smoothing)
{
  MeshSmoothing::smooth_boundary(*this, num_iterations, harmonic_smoothing);
}
//-----------------------------------------------------------------------------
void Mesh::snap_boundary(const SubDomain& sub_domain, bool harmonic_smoothing)
{
  MeshSmoothing::snap_boundary(*this, sub_domain, harmonic_smoothing);
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& Mesh::color(std::string coloring_type) const
{
  // Define graph type
  const std::size_t dim = MeshColoring::type_to_dim(coloring_type, *this);
  std::vector<std::size_t> _coloring_type;
  _coloring_type.push_back(topology().dim());
  _coloring_type.push_back(dim);
  _coloring_type.push_back(topology().dim());

  return color(_coloring_type);
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& Mesh::color(std::vector<std::size_t> coloring_type) const
{
  // Find color data
  std::map<const std::vector<std::size_t>, std::pair<std::vector<std::size_t>,
           std::vector<std::vector<std::size_t> > > >::const_iterator coloring_data;
  coloring_data = this->topology().coloring.find(coloring_type);

  if (coloring_data != this->topology().coloring.end())
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
void Mesh::intersected_cells(const Point& point, std::set<std::size_t>& cells) const
{
  // CGAL needs mesh with more than 1 cell
  if (num_cells() > 1)
    _intersection_operator.all_intersected_entities(point, cells);
  else
  {
    // Num cells == 1
    const Cell cell(*this, 0);
    if (cell.intersects(point))
      cells.insert(0);
  }
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const std::vector<Point>& points,
                             std::set<std::size_t>& cells) const
{
  // CGAL needs mesh with more than 1 cell
  if (num_cells() > 1)
    _intersection_operator.all_intersected_entities(points, cells);
  else
  {
    // Num cells == 1
    const Cell cell(*this, 0);
    for (std::vector<Point>::const_iterator p = points.begin(); p != points.end(); ++p)
    {
      if (cell.intersects(*p))
        cells.insert(0);
    }
  }
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const MeshEntity & entity,
                             std::vector<std::size_t>& cells) const
{
  // CGAL needs mesh with more than 1 cell
  if (num_cells() > 1)
    _intersection_operator.all_intersected_entities(entity, cells);
  else
  {
    // Num cells == 1
    const Cell cell(*this, 0);
    if (cell.intersects(entity))
      cells.push_back(0);
  }
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const std::vector<MeshEntity>& entities,
                             std::set<std::size_t>& cells) const
{
  // CGAL needs mesh with more than 1 cell
  if (num_cells() > 1)
    _intersection_operator.all_intersected_entities(entities, cells);
  else
  {
    // Num cells == 1
    const Cell cell(*this, 0);
    for (std::vector<MeshEntity>::const_iterator entity = entities.begin();
            entity != entities.end(); ++entity)
    {
      if (cell.intersects(*entity))
        cells.insert(0);
    }
  }
}
//-----------------------------------------------------------------------------
void Mesh::intersected_cells(const Mesh& another_mesh,
                             std::set<std::size_t>& cells) const
{
  _intersection_operator.all_intersected_entities(another_mesh, cells);
}
//-----------------------------------------------------------------------------
int Mesh::intersected_cell(const Point& point) const
{
  // CGAL needs mesh with more than 1 cell
  if (num_cells() > 1)
    return  _intersection_operator.any_intersected_entity(point);

  // Num cells == 1
  const Cell cell(*this, 0);
  return cell.intersects(point) ? 0 : -1;
}
//-----------------------------------------------------------------------------
Point Mesh::closest_point(const Point& point) const
{
  return _intersection_operator.closest_point(point);
}
//-----------------------------------------------------------------------------
std::size_t Mesh::closest_cell(const Point & point) const
{
  // CGAL exits with an assertion error whilst performing
  // the closest cell query if num_cells() == 1
  if (num_cells() > 1)
    return _intersection_operator.closest_cell(point);

  // Num cells == 1
  return 0;
}
//-----------------------------------------------------------------------------
std::pair<Point, std::size_t>
Mesh::closest_point_and_cell(const Point & point) const
{
  return _intersection_operator.closest_point_and_cell(point);
}
//-----------------------------------------------------------------------------
double Mesh::distance(const Point& point) const
{
  return _intersection_operator.distance(point);
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
double Mesh::rmin() const
{
  CellIterator cell(*this);
  double r = cell->inradius();
  for (; !cell.end(); ++cell)
    r = std::min(r, cell->inradius());

  return r;
}
//-----------------------------------------------------------------------------
double Mesh::rmax() const
{
  CellIterator cell(*this);
  double r = cell->inradius();
  for (; !cell.end(); ++cell)
    r = std::max(r, cell->inradius());

  return r;
}
//-----------------------------------------------------------------------------
double Mesh::radius_ratio_min() const
{
  CellIterator cell(*this);
  double q = cell->radius_ratio();
  for (; !cell.end(); ++cell)
    q = std::min(q, cell->radius_ratio());

  return q;
}
//-----------------------------------------------------------------------------
double Mesh::radius_ratio_max() const
{
  CellIterator cell(*this);
  double q = cell->radius_ratio();
  for (; !cell.end(); ++cell)
    q = std::max(q, cell->radius_ratio());

  return q;
}
//-----------------------------------------------------------------------------
std::size_t Mesh::hash() const
{
  // Compute hash based on the Cantor pairing function
  const std::size_t k1 = _topology.hash();
  const std::size_t k2 = _geometry.hash();
  return (k1 + k2)*(k1 + k2 + 1)/2 + k2;
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
    std::string cell_type("undefined cell type");
    if (_cell_type)
      cell_type = _cell_type->description(true);

   s << "<Mesh of topological dimension "
      << topology().dim() << " ("
      << cell_type << ") with "
      << num_vertices() << " vertices and "
      << num_cells() << " cells, "
      << (_ordered ? "ordered" : "unordered") << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
const std::vector<int>& Mesh::cell_orientations() const
{
  return _cell_orientations;
}
//-----------------------------------------------------------------------------
std::vector<int>& Mesh::cell_orientations()
{
  return _cell_orientations;
}
//-----------------------------------------------------------------------------
void Mesh::init_cell_orientations(const Expression& global_normal)
{
  // Check that global_normal has the right size
  if (global_normal.value_size() != 3)
  {
     dolfin_error("Mesh.cpp",
                  "initialize cell orientations",
                  "Global normal value size is assumed to be 3 (not %d)",
                  global_normal.value_size());
  }

  Array<double> values(3);
  Point up;
  for (CellIterator cell(*this); !cell.end(); ++cell)
  {
    // Extract cell midpoint as Array
    const Array<double> x(3, cell->midpoint().coordinates());

    // Evaluate global normal at cell midpoint
    global_normal.eval(values, x);

    // Extract values as Point
    for (unsigned int i = 0; i < 3; i++)
      up[i] = values[i];

    // Set orientation as orientation relative to up direction.
    _cell_orientations[cell->index()] = cell->orientation(up);
  }
}
//-----------------------------------------------------------------------------
