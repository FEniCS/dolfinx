// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BUFFER_H
#define __BUFFER_H

namespace dolfin {

  class Buffer {
  public:

    Buffer();
    Buffer(int lines, int cols);
    ~Buffer();

    void init(int lines, int cols);
    
    int size();
    void add(const char* msg);
    const char* get(int line);

  private:

    void clear();

    int lines;     // Number of lines
    int cols;      // Number of columns
    
    int first;     // Position of first row
    int last;      // Position of last row
    bool full;     // True if the buffer is full

    char** buffer; // The buffer
    
  };

}

#endif
