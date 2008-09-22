// Copyright (C) 2005-2007 Garth N.Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008.
//
// First added:  2008-05-29


#ifndef __RAW_FILE_H
#define __RAW_FILE_H

#include <fstream>
#include "GenericFile.h"

namespace dolfin
{
  
  class RAWFile : public GenericFile
    {
    public:
      
      //Write results in raw format
      //Much lighter than the other formats
      //A Xd3d format for instance
      
      RAWFile(const std::string filename);
      ~RAWFile();
      
      
      void operator<< (MeshFunction<int>& meshfunction);
      void operator<< (MeshFunction<unsigned int>& meshfunction);
      void operator<< (MeshFunction<double>& meshfunction);
      void operator<< (Function& u);
      
    private:
      void ResultsWrite(Function& u) const;
      void rawNameUpdate(const int counter);
      
      template<class T>
        void MeshFunctionWrite(T& meshfunction);    
      
      // raw filename
      std::string raw_filename;

    };
  
}

#endif
