// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_FILE_H
#define __XML_FILE_H

#include "GenericFile.h"

namespace dolfin {
  
  class XMLFile : public GenericFile {
  public:

	 XMLFile(const std::string filename);
	 ~XMLFile();
	 
	 // Input
	 
	 void operator>> (Vector &vector);
	 
	 // Output
	 
	 void operator<< (const Vector &vector);
	 	 
  private:
	 
  };

}

#endif
