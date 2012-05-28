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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-05-23
// Last changed: 2012-05-28

#ifndef __VTKPLOTTER_H
#define __VTKPLOTTER_H

#ifdef HAS_VTK

#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkPointSet.h>
#include <vtkActor.h>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class VTKPlotter : public Variable
  {
  public:

    explicit VTKPlotter(const Mesh& mesh);
    explicit VTKPlotter(const Function& function);
    explicit VTKPlotter(const Expression& expression, const Mesh& mesh);
    explicit VTKPlotter(const MeshFunction<uint>& mesh_function);
    explicit VTKPlotter(const MeshFunction<double>& mesh_function);
    explicit VTKPlotter(const MeshFunction<bool>& mesh_function);

    void plot();

    ~VTKPlotter();

    static Parameters default_parameters()
    {
      Parameters p("vtk_plotter");
      p.add("vector_mode", "glyphs");
      p.add("title", "Plot");
      p.add("title_suffix", " - DOLFIN VTK Plotter");
      return p;
    }

  private:

    void construct_vtk_grid();
    
    void plot_scalar_function();
    
    void plot_vector_function();

    void plot_warp();

    void plot_glyphs();

    void filter_and_map(vtkSmartPointer<vtkPointSet> point_set);
    
    void render(vtkSmartPointer<vtkActor> actor);

    boost::shared_ptr<const Mesh> _mesh;

    boost::shared_ptr<const GenericFunction> _function;

    vtkSmartPointer<vtkUnstructuredGrid> _grid;

  };

}

#endif

#endif
