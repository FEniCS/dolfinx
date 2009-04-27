// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2005-07-05
// Last changed: 2009-03-11

#ifndef __PVTK_FILE_H
#define __PVTK_FILE_H

#include <fstream>
#include "GenericFile.h"

namespace dolfin
{

  class PVTKFile : public GenericFile
  {
  public:
    
    PVTKFile(const std::string filename);
    ~PVTKFile();
    
    void operator<< (const Mesh& mesh);
    void operator<< (const MeshFunction<int>& meshfunction);
    void operator<< (const MeshFunction<unsigned int>& meshfunction);
    void operator<< (const MeshFunction<double>& meshfunction);
    void operator<< (const Function& u);
    
  private:

    void mesh_write(const Mesh& mesh) const;
    void results_write(const Function& u) const;
    void pvd_file_write(uint u);
    void pvtu_file_write();
    void pvtu_file_write_func(const Function& u);
    void vtk_header_open(const Mesh& mesh) const;
    void vtk_header_close() const;
    void vtu_name_update(const int counter);
    void pvtu_name_update(const int counter);

    template<class T>
    void mesh_function_write(T& meshfunction);    
    
    // Most recent position in pvd file
    std::ios::pos_type mark;
    
    // vtu filename
    std::string vtu_filename;

    // pvtu filename
    std::string pvtu_filename;

  };
  
}

#endif
