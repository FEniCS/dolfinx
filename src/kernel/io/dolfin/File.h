// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>

#ifndef __FILE_H
#define __FILE_H

namespace dolfin {

  class Vector;
  class Matrix;
  class Grid;
  class Function;
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
    void operator>> (Grid& grid);
    void operator>> (Function& u);
    
    // Output
    
    void operator<< (Vector& x);
    void operator<< (Matrix& A);
    void operator<< (Grid& grid);
    void operator<< (Function& u);
    
  private:
    
    GenericFile* file;
    
  };
  
}

#endif
