// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <string>

namespace dolfin {
  
  class Vector;
  
  class GenericFile {
  public:
	 
	 GenericFile(const std::string filename);
	 virtual ~GenericFile();
	 
	 // Input
	 
	 virtual void operator>> (Vector& vector) = 0;
	 
	 // Output
	 
	 virtual void operator<< (const Vector &vector) = 0;
	 
  private:
	 
	 std::string filename;
	 
  };

}

#endif
