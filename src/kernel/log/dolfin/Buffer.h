// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-26
// Last changed: 2005

#ifndef __BUFFER_H
#define __BUFFER_H

namespace dolfin {

  class Buffer {
  public:

    enum Type { info, debug, warning, error };

    Buffer();
    Buffer(int lines, int cols);
    ~Buffer();

    void init(int lines, int cols);
    
    int size() const;
    void add(const char* msg, Type type = info, int level = 0);
    const char* get(int line) const;
    Type type(int line) const;
    int level(int line) const;

  private:

    void clear();

    int lines;     // Number of lines
    int cols;      // Number of columns
    
    int first;     // Position of first row
    int last;      // Position of last row
    bool full;     // True if the buffer is full

    char** buffer; // The buffer
    Type*  types;  // Line types
    int*   levels; // Levels
    
  };

}

#endif
