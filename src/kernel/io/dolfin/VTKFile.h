// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-07-05
// Last changed: 2005-09-15

#ifndef __VTK_FILE_H
#define __VTK_FILE_H

#include <dolfin/GenericFile.h>
#include <fstream>
#include <iostream>

namespace dolfin
{

  class VTKFile : public GenericFile
  {
  public:
    
    VTKFile(const std::string filename);
    ~VTKFile();
    
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    
  private:

    void MeshWrite(const Mesh& mesh) const;
    void ResultsWrite(Function& u) const;
    void pvdFileWrite(Function& u);
    void VTKHeaderOpen(const Mesh& mesh) const;
    void VTKHeaderClose() const;
    void vtuNameUpdate(const int counter);
    
    // Most recent position in pvd file
    std::ios::pos_type mark;
    
    // vtu filename
    std::string vtu_filename;

  };
  
}

#endif
