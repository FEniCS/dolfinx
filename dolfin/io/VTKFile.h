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
#include <string>
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

  protected:

    void mesh_write(const Mesh& mesh) const;
    void results_write(const Function& u) const;
    void pvd_file_write(uint u);

    void pvtu_file_write();

    void vtk_header_open(uint num_vertices, uint num_cells) const;
    void vtk_header_close() const;

    std::string vtu_name(const int process, const int num_processes,
                         const int counter, std::string ext) const;
    std::string pvtu_name(const int counter);
    
    void clear_file(std::string file) const;

    template<class T>
    void mesh_function_write(T& meshfunction);

    // vtu filename
    std::string vtu_filename;
    std::string pvtu_filename;

  private:

    // Most recent position in pvd file
    std::ios::pos_type mark;

  };

}

#endif
