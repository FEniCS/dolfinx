// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BUFFER_H
#define __BUFFER_H

namespace dolfin {

  class Buffer {
  public:

    enum Type { INFO, DEBUG, WARNING, ERROR };

    Buffer();
    Buffer(int lines, int cols);
    ~Buffer();

    void init(int lines, int cols);
    
    int size() const;
    void add(const char* msg, Type type = INFO);
    const char* get(int line) const;
    Type type(int line) const;

  private:

    void clear();

    int lines;     // Number of lines
    int cols;      // Number of columns
    
    int first;     // Position of first row
    int last;      // Position of last row
    bool full;     // True if the buffer is full

    char** buffer; // The buffer
    Type*  types;  // Line types
    
  };

}

#endif
