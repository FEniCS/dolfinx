// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>

#ifndef __FILE_H
#define __FILE_H

namespace dolfin {

  class Vector;
  class Matrix;
  class Mesh;
  class Function;
  class TimeSlabSample;
  class GenericFile;
  
  class File {
  public:
    
    enum Type { XML, MATLAB, OCTAVE, OPENDX };
    
    File(const std::string& filename);
    File(const std::string& filename, Type type);
    ~File();
    
    // Input
    
    void operator>> (Vector& x);
    void operator>> (Matrix& A);
    void operator>> (Mesh& mesh);
    void operator>> (Function& u);
    void operator>> (TimeSlabSample& sample);
    
    // Output
    
    void operator<< (Vector& x);
    void operator<< (Matrix& A);
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (TimeSlabSample& sample);
    
  private:
    
    GenericFile* file;
    
  };
  
}

#endif
