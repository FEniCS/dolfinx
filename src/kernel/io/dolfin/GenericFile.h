// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <string>
#include <dolfin/Function.h>

namespace dolfin {
  
  class Vector;
  class Matrix;
  class Mesh;
  class Function;
  class Function::Vector;
  class ParameterList;
  class Sample;
  
  class GenericFile {
  public:
    
    GenericFile(const std::string filename);
    virtual ~GenericFile();
    
    // Input
    
    virtual void operator>> (Vector& x);
    virtual void operator>> (Matrix& A);
    virtual void operator>> (Mesh& mesh);
    virtual void operator>> (Function& u);
    virtual void operator>> (Function::Vector& u);
    virtual void operator>> (Sample& sample);
    virtual void operator>> (ParameterList& parameters);
    
    // Output
    
    virtual void operator<< (Vector& x);
    virtual void operator<< (Matrix& A);
    virtual void operator<< (Mesh& mesh);
    virtual void operator<< (Function& u);
    virtual void operator<< (Function::Vector& u);
    virtual void operator<< (Sample& sample);
    virtual void operator<< (ParameterList& parameters);
    
    void read();
    void write();
    
  protected:
    
    void read_not_impl(const std::string object);
    void write_not_impl(const std::string object);

    std::string filename;
    std::string type;
    
    bool opened_read;
    bool opened_write;

    bool check_header; // True if we have written a header

    int no_meshes;
    int no_frames;
    
  };
  
}

#endif
