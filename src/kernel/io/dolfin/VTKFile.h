// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-07-05
// Last changed: 2005-07-05

#ifndef __VTK_FILE_H
#define __VTK_FILE_H

#include <dolfin/GenericFile.h>

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
    void VTKHeaderOpen(const Mesh& mesh) const;
    void VTKHeaderClose() const;
    
  };
  
}

#endif
