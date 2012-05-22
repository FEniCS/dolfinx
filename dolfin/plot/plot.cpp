// Copyright (C) 2007-2009 Anders Logg
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
// Modified by Joachim Berdal Haga, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-05-02
// Last changed: 2009-10-07

#include <cstdlib>
#include <sstream>

#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <vtkDataSetMapper.h>
#include <vtkGeometryFilter.h>
#include <vtkCellType.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkIdList.h>
#include <vtkProperty.h>

#include <dolfin/common/MPI.h>
#include <dolfin/common/utils.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Vertex.h>
#include "FunctionPlotData.h"
#include "plot.h"

using namespace dolfin;

// Template function for plotting objects
template <typename T>
void plot_object(const T& t, std::string title, std::string mode)
{
  info("Plotting %s (%s).",
          t.name().c_str(), t.label().c_str());

  // Don't plot when running in parallel
  if (dolfin::MPI::num_processes() > 1)
  {
    warning("Plotting disabled when running in parallel; see \
https://bugs.launchpad.net/dolfin/+bug/427534");
    return;
  }

  // Get filename prefix
  std::string prefix = parameters["plot_filename_prefix"];

  // Modify prefix and title when running in parallel
  if (dolfin::MPI::num_processes() > 1)
  {
    const dolfin::uint p = dolfin::MPI::process_number();
    prefix += std::string("_p") + to_string(p);
    title += " (process " + to_string(p) + ")";
  }

  // Save to file
  std::string filename = prefix + std::string(".xml");
  File file(filename);
  file << t;

  // Build command string
  std::stringstream command;
  command << "viper --mode=" << mode << " "
          << "--title=\"" << title
          << "\" " << filename;

  // Call Viper from command-line
  if (system(command.str().c_str()) != 0)
      warning("Unable to plot.");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Function& v,
                  std::string title, std::string mode)
{
  // Duplicate test here since FunctionPlotData may fail in parallel
  // as it does for the eigenvalue demo when vector is not initialized
  // correctly.
  if (dolfin::MPI::num_processes() > 1)
  {
    warning("Plotting disabled when running in parallel; see \
https://bugs.launchpad.net/dolfin/+bug/427534");
    return;
  }

  dolfin_assert(v.function_space()->mesh());
  FunctionPlotData w(v, *v.function_space()->mesh());
  plot_object(w, title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Expression& v, const Mesh& mesh,
                  std::string title, std::string mode)
{
  FunctionPlotData w(v, mesh);
  plot_object(w, title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Mesh& mesh,
                  std::string title)
{

uint spatial_dim = mesh.topology().dim();
  std::cout << "Initializing mesh in " << spatial_dim << " spatial dimensions." << std::endl;

  vtkUnstructuredGrid *grid = vtkUnstructuredGrid::New();
  vtkPoints *points = vtkPoints::New(); 

  // Iterate over points and add to array
  uint num_points = mesh.num_vertices();
  points->Allocate(num_points);
  Point p;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) {
    p = vertex->point();
    // DOLFIN only stores (x,y)-coordinates for 2D meshes. Must add z-coordinate manually
    //points->InsertNextPoint(vertex->x()[0], vertex->x()[1], 0.0);
    points->InsertNextPoint(p.x(), p.y(), p.z());
  }

  // Iterate over cells and add to array
  uint num_cells = mesh.num_cells();
  // Allocate storage in VTK grid. Number of cells times spatial dim + 1, 
  // since 2D triangles have 3 vertices, 3D tetrahedrons have 4 vertices, etc.
  grid->Allocate(num_cells*(spatial_dim+1));
  //cells->Allocate(num_cells*3);
  const uint *connectivity = mesh.cells();
  vtkIdList *ids = vtkIdList::New();
  for (uint i = 0; i < num_cells; ++i) {
    //cells->InsertNextCell(3, (vtkIdType*) &connectivity[3*i]);
    ids->Initialize();
    // Insert all vertex ids for a given cell. For a simplex in nD, n+1 ids are inserted.
    // The connectivity array must be indexed at ((n+1) x cell_number + id_offset)
    for(uint j = 0; j <= spatial_dim; ++j) {
      ids->InsertNextId((vtkIdType) connectivity[(spatial_dim+1)*i + j]);
    }
    
    switch (spatial_dim) {
      case 1:
        grid->InsertNextCell(VTK_LINE, ids);
        break;
      case 2:
        grid->InsertNextCell(VTK_TRIANGLE, ids);
        break;
      case 3:
        grid->InsertNextCell(VTK_TETRA, ids);
        break;
      default:
        // Error handling!
        break;
    }
  }

  grid->SetPoints(points);
  points->Delete();
  // Free unused memory
  grid->Squeeze();

  std::cout << "Num cells: " << grid->GetNumberOfCells() << std::endl;
  std::cout << "Num vertices: " << grid->GetNumberOfPoints() << std::endl;

  // Create filter and attach grid to it
  vtkGeometryFilter *geometryFilter = vtkGeometryFilter::New();
  geometryFilter->SetInput(grid);
  geometryFilter->Update();

  // Create mapper and actor
  vtkPolyDataMapper *mapper = vtkPolyDataMapper::New();
  mapper->SetInputConnection(geometryFilter->GetOutputPort());

  /*
  vtkDataSetMapper *mapper = vtkDataSetMapper::New();
  mapper->SetInput(grid);
  mapper->ImmediateModeRenderingOn();
  */

  vtkActor *actor = vtkActor::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetRepresentationToWireframe();

  vtkRenderer *ren1 = vtkRenderer::New();
  ren1->AddActor(actor);
  ren1->SetBackground(0,0,0);

  vtkRenderWindow *window = vtkRenderWindow::New();
  window->AddRenderer(ren1);
  window->SetSize(600,600);
  window->SetWindowName("halla");

  vtkRenderWindowInteractor *inter = vtkRenderWindowInteractor::New();
  inter->SetRenderWindow(window);
  vtkInteractorStyleTrackballCamera *style = vtkInteractorStyleTrackballCamera::New();
  inter->SetInteractorStyle(style);
  inter->Initialize();
  inter->Start();
 
  grid->Delete();
  geometryFilter->Delete();
  mapper->Delete();
  actor->Delete();
  ren1->Delete();
  window->Delete();
  inter->Delete();
  style->Delete();
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<uint>& f,
                  std::string title)
{
  plot_object(f, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<double>& f,
                  std::string title)
{
  plot_object(f, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<bool>& f,
                  std::string title)
{
  plot_object(f, title, "auto");
}
//-----------------------------------------------------------------------------
