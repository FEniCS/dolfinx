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
// Last changed: 2012-09-13

#ifndef __GENERIC_VTK_PLOTTABLE_H
#define __GENERIC_VTK_PLOTTABLE_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>
#include <vtkAlgorithmOutput.h>
#include <vtkActor.h>
#include <vtkActor2D.h>

namespace dolfin
{

  class Parameters;
  class Variable;
  class VTKWindowOutputStage;

  /// This class defines a common interface for objects that can be
  /// plotted by the VTKPlotter class

  class GenericVTKPlottable
  {
  public:

    // This destructor should be uncommented, but it causes a problem
    // in parallel with MPI calls being made after MPI is shutdown. Needs
    // further investigation.
    //virtual ~GenericVTKPlottable() {}

    /// To be redefined in classes that require special parameters. Called
    /// once with the default parameters.
    virtual void modify_default_parameters(Parameters& p) = 0;

    /// To be redefined in classes that require special parameters. Called
    /// once with user-specified parameters, but before init_pipeline.
    virtual void modify_user_parameters(Parameters& p) {}

    /// Initialize the parts of the pipeline that this class controls
    virtual void init_pipeline(const Parameters& p) = 0;

    /// Connect or reconnect to the output stage.
    virtual void connect_to_output(VTKWindowOutputStage& output) = 0;

    /// Update the plottable data. The variable may be empty, or it may
    /// be a new variable to plot. is_compatible(var) must be true.
    virtual void update(boost::shared_ptr<const Variable> var,
                        const Parameters& p, int framecounter) = 0;

    /// Return whether this plottable is compatible with the variable
    virtual bool is_compatible(const Variable &var) const = 0;

    /// Update the scalar range of the plottable data
    virtual void update_range(double range[2]) = 0;

    /// Inform the plottable about the range. Most plottables don't care,
    /// since this is handled in the output stage.
    virtual void rescale(double range[2], const Parameters& p) {}

    /// Return geometric dimension
    virtual std::size_t dim() const = 0;

    /// Get an actor for showing vertex labels
    virtual vtkSmartPointer<vtkActor2D> get_vertex_label_actor(vtkSmartPointer<vtkRenderer>)
    {
      warning("Plotting of vertex labels is not implemented by the current"
              " VTK plottable type.");
      // Return empty actor to have something (invisible) to render
      return vtkSmartPointer<vtkActor2D>::New();
    }

    /// Get an actor for showing cell labels
    virtual vtkSmartPointer<vtkActor2D> get_cell_label_actor(vtkSmartPointer<vtkRenderer>)
    {
      warning("Plotting of cell labels is not implemented by the current"
              " VTK plottable type.");
      // Return empty actor to have something (invisible) to render
      return vtkSmartPointer<vtkActor2D>::New();
    }

    /// Get an actor for showing the mesh
    virtual vtkSmartPointer<vtkActor> get_mesh_actor()
    {
      warning("Plotting of mesh is not implemented by the current"
          " VTK plottable type.");
      // Return empty actor to have something (invisible) to render
      return vtkSmartPointer<vtkActor>::New();
    }

  };
}

#endif

#endif
