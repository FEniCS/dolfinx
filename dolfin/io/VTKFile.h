// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
//
// First added:  2005-07-05
// Last changed: 2007-05-08

#ifndef __VTK_FILE_H
#define __VTK_FILE_H

#include <fstream>
#include "GenericFile.h"

namespace dolfin
{

  class VTKFile : public GenericFile
  {
  public:

    VTKFile(const std::string filename);
    ~VTKFile();

    void operator<< (const Mesh& mesh);
    void operator<< (const MeshFunction<int>& meshfunction);
    void operator<< (const MeshFunction<unsigned int>& meshfunction);
    void operator<< (const MeshFunction<double>& meshfunction);
    void operator<< (const Function& u);

  private:

    void mesh_write(const Mesh& mesh) const;
    void results_write(const Function& u) const;
    void pvd_file_write(uint u);
    void vtk_header_open(const Mesh& mesh) const;
    void vtk_header_close() const;
    void vtu_name_update(const int counter);

    template<class T>
    void mesh_function_write(T& meshfunction);

    // Most recent position in pvd file
    std::ios::pos_type mark;

    // vtu filename
    std::string vtu_filename;

  };

}

#endif
