// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>

#ifndef __FILE_H
#define __FILE_H

namespace dolfin {

  class Vector;
  class GenericFile;
  
  class File {
  public:

	 File(const std::string& filename);
	 ~File();

	 // Input
	 
	 void operator>> (Vector& vector);
	 
	 // Output

	 void operator<< (const Vector &vector);

  private:

	 GenericFile *file;
	 
  };

}

#endif
