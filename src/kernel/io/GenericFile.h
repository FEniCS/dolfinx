// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <string>
#include <fstream>


namespace dolfin {
  
  class Vector;
  class Matrix;
  class Grid;
  class Function;
  
  class GenericFile {
  public:
    
    GenericFile(const std::string filename);
    virtual ~GenericFile();
    
    // Input
    
    virtual void operator>> (Vector& x)   = 0;
    virtual void operator>> (Matrix& A)   = 0;
    virtual void operator>> (Grid& grid)  = 0;
    virtual void operator>> (Function& u) = 0;
    
    // Output
    
    virtual void operator<< (Vector& x)   = 0;
    virtual void operator<< (Matrix& A)   = 0;
    virtual void operator<< (Grid& grid)  = 0;
    virtual void operator<< (Function& u) = 0;
    
    void read();
    void write();
    
  protected:
    
    std::string filename;
    
    bool opened_read;
    bool opened_write;

    bool  check_header;     // True if we have written a header
    
  };
  
}

#endif
