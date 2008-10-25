// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Niclas Jansson 2008.
//
// First added:  2005-07-05
// Last changed: 2008-06-26

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
    
    void operator<< (Mesh& mesh);
    void operator<< (MeshFunction<int>& meshfunction);
    void operator<< (MeshFunction<unsigned int>& meshfunction);
    void operator<< (MeshFunction<double>& meshfunction);
    void operator<< (Function& u);
    
    void write();
  private:

    void MeshWrite(const Mesh& mesh) const;
    void ResultsWrite(Function& u) const;
    void pvdFileWrite(uint u);
    void pvtuFileWrite();
    void pvtuFileWrite_func(Function& u);
    void VTKHeaderOpen(const Mesh& mesh) const;
    void VTKHeaderClose() const;
    void vtuNameUpdate(const int counter);
    void pvtuNameUpdate(const int counter);

    template<class T>
    void MeshFunctionWrite(T& meshfunction);    
    
    // Most recent position in pvd file
    std::ios::pos_type mark;
    
    // vtu filename
    std::string vtu_filename;

    // pvtu filename
    std::string pvtu_filename;
  };
  
}

#endif
