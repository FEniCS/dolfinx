// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FILE_H
#define __FILE_H

#include <string>
#include <dolfin/Function.h>

namespace dolfin {

  class Vector;
  class Matrix;
  class Mesh;
  class Function;
  class Function::Vector;
  class Sample;
  class ParameterList;
  class GenericFile;
  
  class File {
  public:
    
    enum Type { XML, MATLAB, OCTAVE, OPENDX, GID };
    
    File(const std::string& filename);
    File(const std::string& filename, Type type);
    ~File();
    
    // Input
    
    void operator>> (Vector& x);
    void operator>> (Matrix& A);
    void operator>> (Mesh& mesh);
    void operator>> (Function& u);
    void operator>> (Function::Vector& u);
    void operator>> (Sample& sample);
    void operator>> (ParameterList& parameters);
    
    // Output
    
    void operator<< (Vector& x);
    void operator<< (Matrix& A);
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (Function::Vector& u);
    void operator<< (Sample& sample);
    void operator<< (ParameterList& parameters);
    
  private:
    
    GenericFile* file;
    
  };
  
}

#endif
