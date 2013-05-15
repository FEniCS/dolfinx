// Copyright (C) 2013 Nico Schlömer
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
// First added:  2013-02-27

#ifndef __EXODUS_FILE_H
#define __EXODUS_FILE_H

#ifdef HAS_VTK
#ifdef HAS_VTK_EXODUS

#include "GenericFile.h"

#include <vtkSmartPointer.h>

// forward declarations
class vtkUnstructuredGrid;
class vtkExodusIIWriter;

namespace dolfin
{

  /// This class supports the output of meshes and functions
  /// Exodus format for visualistion purposes. It is not suitable to
  /// checkpointing as it may decimate some data.

  class ExodusFile : public GenericFile
  {
  public:

    ExodusFile(const std::string filename);
    ~ExodusFile();

    void operator<< (const Mesh& mesh);
    void operator<< (const MeshFunction<unsigned int>& meshfunction);
    void operator<< (const MeshFunction<int>& meshfunction);
    void operator<< (const MeshFunction<double>& meshfunction);
    void operator<< (const Function& u);
    void operator<< (const std::pair<const Function*, double> u);

private:

    void write_function(const Function& u, double time) const;

    vtkSmartPointer<vtkUnstructuredGrid> create_vtk_mesh(const Mesh& mesh) const;

    void perform_write(const vtkSmartPointer<vtkUnstructuredGrid> & vtk_mesh) const;

    //
    const vtkSmartPointer<vtkExodusIIWriter> _writer;

  };

}

#endif // HAS_VTK_EXODUS
#endif // HAS_VTK
#endif // __EXODUS_FILE_H
