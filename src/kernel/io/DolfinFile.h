// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DOLFIN_FILE_H
#define __DOLFIN_FILE_H

#include "GenericFile.h"

namespace dolfin {
  
  class DolfinFile : public GenericFile {
  public:

	 DolfinFile(const std::string filename);
	 ~DolfinFile();
	 
	 // Input
	 
	 void operator>> (Vector &vector);
	 
	 // Output
	 
	 void operator<< (const Vector &vector);
	 	 
  private:

	 void createIndex();

	 int no_objects;
	 long *objects;
	 
  };

}

#endif
