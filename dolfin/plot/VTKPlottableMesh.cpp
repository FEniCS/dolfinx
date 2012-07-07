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
// First added:  2012-06-20
// Last changed: 2012-06-26

#ifdef HAS_VTK

#include <vtkStringArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkPointSetToLabelHierarchy.h>
#include <vtkTextProperty.h>
#include <vtkLabelPlacementMapper.h>

#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Vertex.h>
#include "VTKPlottableMesh.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableMesh::VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh) :
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()),
  _geometryFilter(vtkSmartPointer<vtkGeometryFilter>::New()),
  _mesh(mesh)
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::init_pipeline()
{
  dolfin_assert(_geometryFilter);

  _geometryFilter->SetInput(_grid);
  _geometryFilter->Update();
}

//----------------------------------------------------------------------------
void VTKPlottableMesh::update(const Parameters& parameters)
{
  dolfin_assert(_grid);

  Timer t("Construct VTK grid");

  // Construct VTK point array from DOLFIN mesh vertices

  // Create pint array
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->SetNumberOfPoints(_mesh->num_vertices());

  // Create array to hold index labels
  vtkSmartPointer<vtkStringArray> labels = vtkSmartPointer<vtkStringArray>::New();
  std::stringstream label;
  labels->SetNumberOfValues(_mesh->num_vertices());
  labels->SetName("indices");

  // Iterate vertices and add to point and label array
  Point p;
  for (VertexIterator vertex(*_mesh); !vertex.end(); ++vertex)
  {
    p = vertex->point();
    points->SetPoint(vertex->index(), p.x(), p.y(), p.z());

    // Reset label, convert integer index to string and add to array
    label.str("");
    label << vertex->index();
    labels->SetValue(vertex->index(), label.str().c_str());
  }

  // Add mesh cells to VTK cell array. Note: Preallocation of storage
  // in cell array did not give speedups when testing during development
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  const uint *connectivity = _mesh->cells();
  uint spatial_dim = _mesh->topology().dim();

  for (uint i = 0; i < _mesh->num_cells(); ++i)
  {
    // Insert all vertex indices for a given cell. For a simplex cell in nD,
    // n+1 indices are inserted. The connectivity array must be indexed at
    // ((n+1) x cell_number + idx_offset)
    cells->InsertNextCell(spatial_dim+1);
    for(uint j = 0; j <= spatial_dim; ++j) {
      cells->InsertCellPoint(connectivity[(spatial_dim+1)*i + j]);
    }
  }
  // Free unused memory in cell array
  // (automatically allocated during cell insertion)
  cells->Squeeze();

  // Insert points, vertex labels and cells in VTK unstructured grid
  _grid->SetPoints(points);
  _grid->GetPointData()->AddArray(labels);
  switch (spatial_dim)
  {
    case 1:
      _grid->SetCells(VTK_LINE, cells);
      break;
    case 2:
      _grid->SetCells(VTK_TRIANGLE, cells);
      break;
    case 3:
      _grid->SetCells(VTK_TETRA, cells);
      break;
    default:
      // Should never be reached
      break;
  }
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::update_range(double range[2])
{
  _grid->GetScalarRange(range);
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkAlgorithmOutput> VTKPlottableMesh::get_output() const
{
  return _geometryFilter->GetOutputPort();
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkActor2D> VTKPlottableMesh::get_vertex_label_actor()
{
  // Return actor if already created
  if (_vertexLabelActor)
    return _vertexLabelActor;

  // We create the actor on the first call to the method

  // TODO: Should we use vtkLabeledDataMapper here instead? Together with
  // vtkSelectVisiblePoints to only label visible points, and use vtkCellCenters
  // to generate points at the center of cells to label cells. See
  // http://www.vtk.org/doc/release/5.8/html/a01117.html

  // Generate the label hierarchy.
  vtkSmartPointer<vtkPointSetToLabelHierarchy> pointSetToLabelHierarchyFilter
    = vtkSmartPointer<vtkPointSetToLabelHierarchy>::New();
  pointSetToLabelHierarchyFilter->SetInput(_grid);
  pointSetToLabelHierarchyFilter->SetLabelArrayName("indices"); // This name must match the one set in "update"
  // NOTE: One may set an integer array with priorites on the hierarchy filter.
  // These priorities will indicate which labels will be shown when there is
  // limited space.
  //pointSetToLabelHierarchyFilter->SetPriorityArrayName("priorities");
  pointSetToLabelHierarchyFilter->GetTextProperty()->SetColor(0, 0, 0);
  pointSetToLabelHierarchyFilter->Update();

  // Create a mapper and actor for the labels.
  vtkSmartPointer<vtkLabelPlacementMapper> labelMapper
    = vtkSmartPointer<vtkLabelPlacementMapper>::New();
  labelMapper->SetInputConnection(
    pointSetToLabelHierarchyFilter->GetOutputPort());
  _vertexLabelActor = vtkSmartPointer<vtkActor2D>::New();
  _vertexLabelActor->SetMapper(labelMapper);

  return _vertexLabelActor;
}
//----------------------------------------------------------------------------

#endif
