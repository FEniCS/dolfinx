// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_TMP_FILE_H
#define __ELEMENT_TMP_FILE_H

#include <stdio.h>

namespace dolfin {

  class ElementBlock;

  /// Temporary file used for storing element data in binary format
  /// when data is to large to fit into memory.

  class ElementTmpFile {
  public:

    /// Constructor
    ElementTmpFile();

    /// Destructor
    ~ElementTmpFile();

    /// Write an element block to the file
    void write(const ElementBlock& block);
   
  private:

    // Compute size of block in bytes
    unsigned int bytes(const ElementBlock& block);

    FILE* fp;
    const char* filename;

  };

}

#endif
