// Copyright (C) 2012 Fredrik Valdmanis
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
// Modified by Joachim B Haga 2012
//
// First added:  2012-06-20
// Last changed: 2012-08-31

#ifdef HAS_VTK

#include <vtkStringArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkPointSetToLabelHierarchy.h>
#include <vtkTextProperty.h>
#include <vtkLabelPlacementMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkIdFilter.h>
#include <vtkLabeledDataMapper.h>
#include <vtkCellCenters.h>
#include <vtkSelectVisiblePoints.h>
#include <vtkRenderer.h>

#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/CellType.h>
#include "VTKPlottableMesh.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableMesh::VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh, uint entity_dim) :
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()),
  _full_grid(vtkSmartPointer<vtkUnstructuredGrid>::New()),
  _geometryFilter(vtkSmartPointer<vtkGeometryFilter>::New()),
  _mesh(mesh),
  _entity_dim(entity_dim)
{
  // Do nothing
}
//----------------------------------------------------------------------------
VTKPlottableMesh::VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh) :
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()),
  _full_grid(vtkSmartPointer<vtkUnstructuredGrid>::New()),
  _geometryFilter(vtkSmartPointer<vtkGeometryFilter>::New()),
  _mesh(mesh),
  _entity_dim(mesh->topology().dim())
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::init_pipeline(const Parameters &parameters)
{
  dolfin_assert(_geometryFilter);

  _geometryFilter->SetInput(_grid);
  _geometryFilter->Update();
}
//----------------------------------------------------------------------------
bool VTKPlottableMesh::requires_depthsort() const
{
  if (_entity_dim < 2 || dim() < 3)
  {
    return false;
  }

  vtkFloatArray *pointdata = dynamic_cast<vtkFloatArray*>(_grid->GetPointData()->GetScalars());
  if (pointdata && pointdata->GetNumberOfComponents() == 1)
  {
    for (uint i = 0; i < pointdata->GetNumberOfTuples(); i++)
    {
      if (isnan(pointdata->GetValue(i)))
        return true;
    }
  }

  vtkFloatArray *celldata = dynamic_cast<vtkFloatArray*>(_grid->GetCellData()->GetScalars());
  if (celldata && celldata->GetNumberOfComponents() == 1)
  {
    for (uint i = 0; i < celldata->GetNumberOfTuples(); i++)
    {
      if (isnan(celldata->GetValue(i)))
        return true;
    }
  }

  return false;
}
//----------------------------------------------------------------------------
bool VTKPlottableMesh::is_compatible(const Variable &var) const
{
  return dynamic_cast<const Mesh*>(&var);
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int framecounter)
{
  if (var)
  {
    _mesh = boost::dynamic_pointer_cast<const Mesh>(var);
  }
  dolfin_assert(_grid);
  dolfin_assert(_full_grid);
  dolfin_assert(_mesh);

  Timer t("VTK construct grid");

  //
  // Construct VTK point array from DOLFIN mesh vertices
  //

  // Create point array
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->SetNumberOfPoints(_mesh->num_vertices());

  // Iterate vertices and add to point array
  Point p;
  for (VertexIterator vertex(*_mesh); !vertex.end(); ++vertex)
  {
    p = vertex->point();
    points->SetPoint(vertex->index(), p.x(), p.y(), p.z());
  }

  // Insert points, vertex labels and cells in VTK unstructured grid
  _full_grid->SetPoints(points);

  //
  // Construct VTK cells from DOLFIN facets
  //

  build_grid_cells(_full_grid, dim());
  if (_entity_dim == dim())
  {
    _grid->ShallowCopy(_full_grid);
  }
  else
  {
    _grid->SetPoints(points);
    build_grid_cells(_grid, _entity_dim);
  }
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::build_grid_cells(vtkSmartPointer<vtkUnstructuredGrid> &grid, uint entity_dim)
{
  // Add mesh cells to VTK cell array. Note: Preallocation of storage
  // in cell array did not give speedups when testing during development
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  const MeshTopology &topo = _mesh->topology();
  const uint *connectivity = topo(_entity_dim, 0)();
  const uint vertices_per_entity = _mesh->type().num_vertices(_entity_dim);
  const uint num_entities = topo.size(_entity_dim);

  if (_entity_dim > 0)
  {
    for (uint i = 0; i < num_entities; ++i)
    {
      // Insert all vertex indices for a given cell. For a simplex cell in nD,
      // n+1 indices are inserted. The connectivity array must be indexed at
      // (nv x cell_number + idx_offset)
      cells->InsertNextCell(vertices_per_entity);
      for(uint j = 0; j < vertices_per_entity; ++j) {
        cells->InsertCellPoint(connectivity[i*vertices_per_entity + j]);
      }
    }
  }
  else
  {
    for (uint i = 0; i < num_entities; ++i)
    {
      // Cells equals vertices, connectivity is NULL
      cells->InsertNextCell(1);
      cells->InsertCellPoint(i);
    }
  }

  // Free unused memory in cell array
  // (automatically allocated during cell insertion)
  cells->Squeeze();

  switch (_entity_dim)
  {
    case 0:
      grid->SetCells(VTK_VERTEX, cells);
      break;
    case 1:
      grid->SetCells(VTK_LINE, cells);
      break;
    case 2:
      grid->SetCells(VTK_TRIANGLE, cells);
      break;
    case 3:
      grid->SetCells(VTK_TETRA, cells);
      break;
    default:
      dolfin_error("VTKPlottableMesh.cpp", "initialise cells", "Not implemented for dim>3");
      break;
  }
  
  // Is this needed?
  // _grid->Modified();
  // _geometryFilter->Modified();
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::update_range(double range[2])
{
  _grid->GetScalarRange(range);
}
//----------------------------------------------------------------------------
dolfin::uint VTKPlottableMesh::dim() const
{
  return _mesh->topology().dim();
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkAlgorithmOutput> VTKPlottableMesh::get_output() const
{
  return _geometryFilter->GetOutputPort();
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::build_id_filter()
{
  if (!_idFilter)
  {
    _idFilter = vtkSmartPointer<vtkIdFilter>::New();
    if (_entity_dim == dim() || _entity_dim == 0)
    {
      // Kludge to get to the unwarped mesh in relevant cases
      _idFilter->SetInputConnection(get_mesh_actor()->GetMapper()->GetInputConnection(0,0));
    }
    else
    {
      _idFilter->SetInputConnection(_geometryFilter->GetOutputPort());
    }
    _idFilter->PointIdsOn();
    _idFilter->CellIdsOn();
    _idFilter->FieldDataOn();
  }
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkActor2D> VTKPlottableMesh::get_vertex_label_actor(vtkSmartPointer<vtkRenderer> renderer)
{
  // Return actor if already created
  if (!_vertexLabelActor)
  {
    build_id_filter();

    vtkSmartPointer<vtkSelectVisiblePoints> vis = vtkSmartPointer<vtkSelectVisiblePoints>::New();
    vis->SetInputConnection(_idFilter->GetOutputPort());
    // If the tolerance is too high, too many labels are visible (especially at
    // a distance).  If set too low, some labels are invisible. There isn't a
    // "correct" value, it should really depend on distance and resolution.
    vis->SetTolerance(1e-4);
    vis->SetRenderer(renderer);
    //vis->SelectionWindowOn();
    //vis->SetSelection(0, 0.3, 0, 0.3);

    vtkSmartPointer<vtkLabeledDataMapper> ldm = vtkSmartPointer<vtkLabeledDataMapper>::New();
    ldm->SetInputConnection(vis->GetOutputPort());
    ldm->SetLabelModeToLabelFieldData();
    ldm->GetLabelTextProperty()->SetColor(0.0, 0.0, 0.0);
    ldm->GetLabelTextProperty()->ItalicOff();
    ldm->GetLabelTextProperty()->ShadowOff();

    _vertexLabelActor = vtkSmartPointer<vtkActor2D>::New();
    _vertexLabelActor->SetMapper(ldm);
  }
  return _vertexLabelActor;
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkActor2D> VTKPlottableMesh::get_cell_label_actor(vtkSmartPointer<vtkRenderer> renderer)
{
  if (!_cellLabelActor)
  {
    build_id_filter();

    vtkSmartPointer<vtkCellCenters> cc = vtkSmartPointer<vtkCellCenters>::New();
    cc->SetInputConnection(_idFilter->GetOutputPort());

    vtkSmartPointer<vtkSelectVisiblePoints> vis = vtkSmartPointer<vtkSelectVisiblePoints>::New();
    vis->SetTolerance(1e-4); // See comment for vertex labels
    vis->SetInputConnection(cc->GetOutputPort());
    vis->SetRenderer(renderer);

    vtkSmartPointer<vtkLabeledDataMapper> ldm = vtkSmartPointer<vtkLabeledDataMapper>::New();
    ldm->SetInputConnection(vis->GetOutputPort());
    ldm->SetLabelModeToLabelFieldData();
    ldm->GetLabelTextProperty()->SetColor(0.3, 0.3, 0.0);
    ldm->GetLabelTextProperty()->ShadowOff();

    _cellLabelActor = vtkSmartPointer<vtkActor2D>::New();
    _cellLabelActor->SetMapper(ldm);
  }

  return _cellLabelActor;
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkActor> VTKPlottableMesh::get_mesh_actor()
{
  if (!_meshActor)
  {
    vtkSmartPointer<vtkGeometryFilter> geomfilter = vtkSmartPointer<vtkGeometryFilter>::New();
    geomfilter->SetInput(_full_grid);
    geomfilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(geomfilter->GetOutputPort());

    _meshActor = vtkSmartPointer<vtkActor>::New();
    _meshActor->SetMapper(mapper);
    _meshActor->GetProperty()->SetRepresentationToWireframe();
    _meshActor->GetProperty()->SetColor(0.7, 0.7, 0.3);
    _meshActor->GetProperty()->SetOpacity(0.5);
    vtkMapper::SetResolveCoincidentTopologyToPolygonOffset();

  }
  return _meshActor;
}
//----------------------------------------------------------------------------
boost::shared_ptr<const Mesh> VTKPlottableMesh::mesh() const
{
  return _mesh;
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkPointSet> VTKPlottableMesh::grid() const
{
  return _grid;
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::insert_filter(vtkSmartPointer<vtkPointSetAlgorithm> filter)
{
  filter->SetInput(_grid);
  _geometryFilter->SetInput(filter->GetOutput());
  _geometryFilter->Update();
}
//----------------------------------------------------------------------------
VTKPlottableMesh *dolfin::CreateVTKPlottable(boost::shared_ptr<const Mesh> mesh)
{
  return new VTKPlottableMesh(mesh);
}

#endif
