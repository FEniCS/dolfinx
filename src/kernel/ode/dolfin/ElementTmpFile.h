// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_TMP_FILE_H
#define __ELEMENT_TMP_FILE_H

#include <stdio.h>

namespace dolfin {

  class ElementBlock;

  /// Temporary file used for storing element data in binary format
  /// when data is to large to fit into memory. It is assumed that
  /// blocks are written sequentially to the file.
  
  class ElementTmpFile {
  public:

    /// Constructor
    ElementTmpFile();

    /// Destructor
    ~ElementTmpFile();

    /// Write an element block to the file
    void write(const ElementBlock& block);

    // Read an element block from the file
    void read(ElementBlock& block, real t);

    // Read first element block from the file
    void readFirst(ElementBlock& block);

    // Read last element block from the file
    void readLast(ElementBlock& block);

    /// Check if the file is empty
    bool empty() const;
   
  private:

    // Step forward in file to given time
    bool searchForward(real t);

    // Step backward in file to given time
    bool searchBackward(real t);

    // Read head of current block (and return)
    bool readHead(unsigned int& size, real& t0, real &t1);

    // Read tail of current block (and return)
    bool readTail(unsigned int& size);
    
    // Read block starting at the current position
    void readBlock(ElementBlock& block);
      
    // Compute size of block in bytes
    unsigned int bytes(const ElementBlock& block);

    // Pointer to the temporary file
    FILE* fp;
    
    // True if the file is empty
    bool _empty;

  };

}

#endif
