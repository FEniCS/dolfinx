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
// Last changed: 2012-06-20

#ifdef HAS_VTK

#include <vtkCellArray.h>

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
  _geometryFilter->SetInput(_grid);
  _geometryFilter->Update();
}
//----------------------------------------------------------------------------
void VTKPlottableMesh::update()
{
  dolfin_assert(_grid);

  Timer t("Construct VTK grid");

  // Construct VTK point array from DOLFIN mesh vertices
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->SetNumberOfPoints(_mesh->num_vertices());
  Point p;

  for (VertexIterator vertex(*_mesh); !vertex.end(); ++vertex) {
    p = vertex->point();
    points->SetPoint(vertex->index(), p.x(), p.y(), p.z());
  }

  // Add mesh cells to VTK cell array. Note: Preallocation of storage
  // in cell array did not give speedups when testing during development
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  const uint *connectivity = _mesh->cells();
  uint spatial_dim = _mesh->topology().dim();

  for (uint i = 0; i < _mesh->num_cells(); ++i) {

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

  // Insert points and cells in VTK unstructured grid
  _grid->SetPoints(points);
  switch (spatial_dim) {
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
#endif
