// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005
//
// First added:  2002-11-12
// Last changed: 2005-10-02

#ifndef __FILE_H
#define __FILE_H

#include <string>

namespace dolfin
{

  class Vector;
  class Matrix;
  class Mesh;
  class Function;
  class Sample;
  class ParameterList;
  class GenericFile;
  class Form;
  
  class File
  {
  public:
    
    enum Type { xml, matlab, octave, opendx, gid, tecplot, vtk };
    
    File(const std::string& filename);
    File(const std::string& filename, Type type);
    ~File();
    
    // Input
  
    void operator>> (Vector& x);
    void operator>> (Matrix& A);
    void operator>> (Mesh& mesh);
    void operator>> (Function& u);
    void operator>> (Sample& sample);
    void operator>> (ParameterList& parameters);
    void operator>> (Form& form);
  
    // Output
    
    void operator<< (Vector& x);
    void operator<< (Matrix& A);
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (Sample& sample);
    void operator<< (ParameterList& parameters);
    void operator<< (Form& form);
    
  private:
    
    GenericFile* file;
    
  };
  
}

#endif
