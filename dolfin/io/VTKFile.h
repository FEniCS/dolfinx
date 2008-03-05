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
    
    void operator<< (Mesh& mesh);
    void operator<< (MeshFunction<int>& meshfunction);
    void operator<< (MeshFunction<unsigned int>& meshfunction);
    void operator<< (MeshFunction<double>& meshfunction);
    void operator<< (Function& u);
    
  private:

    void MeshWrite(Mesh& mesh) const;
    void ResultsWrite(Function& u) const;
    void pvdFileWrite(uint u);
    void VTKHeaderOpen(Mesh& mesh) const;
    void VTKHeaderClose() const;
    void vtuNameUpdate(const int counter);

    template<class T>
    void MeshFunctionWrite(T& meshfunction);    
    
    // Most recent position in pvd file
    std::ios::pos_type mark;
    
    // vtu filename
    std::string vtu_filename;

  };
  
}

#endif
