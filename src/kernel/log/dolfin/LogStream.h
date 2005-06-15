// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005.

#ifndef __LOG_STREAM_H
#define __LOG_STREAM_H

#include <string>
#include <dolfin/constants.h>

namespace dolfin {

  class LogStream {
  public:
    
    enum Type {COUT, ENDL};
    
    LogStream(Type type);
    ~LogStream();
    
    LogStream& operator<<(const char* s);
    LogStream& operator<<(const std::string& s);
    LogStream& operator<<(int a);
    LogStream& operator<<(unsigned int a);
    LogStream& operator<<(real a);
    LogStream& operator<<(complex z);
    LogStream& operator<<(const LogStream& stream);

    void show() const;
    
  private:
    
    void add(const char* msg);
    
    Type type;
    char* buffer;
    int current;
    
  };
  
  extern LogStream cout;
  extern LogStream endl;
  
}

#endif
