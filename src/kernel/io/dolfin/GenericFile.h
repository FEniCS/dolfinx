// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <string>

namespace dolfin
{
  
  class Vector;
  class Matrix;
  class Mesh;
  class NewFunction;
  class Sample;
  class NewSample;
  class ParameterList;
  
  class GenericFile {
  public:
    
    GenericFile(const std::string filename);
    virtual ~GenericFile();
    
    // Input
    
    virtual void operator>> (Vector& x);
    virtual void operator>> (Matrix& A);
    virtual void operator>> (Mesh& mesh);
    virtual void operator>> (NewFunction& u);
    virtual void operator>> (Sample& sample);
    virtual void operator>> (NewSample& sample);
    virtual void operator>> (ParameterList& parameters);
    
    // Output
    
    virtual void operator<< (Vector& x);
    virtual void operator<< (Matrix& A);
    virtual void operator<< (Mesh& mesh);
    virtual void operator<< (NewFunction& u);
    virtual void operator<< (Sample& sample);
    virtual void operator<< (NewSample& sample);
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
