// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2005

#ifndef __OPEN_DX_FILE_H
#define __OPEN_DX_FILE_H

#include <string>
#include <stdio.h>
#include <dolfin/common/types.h>
#include <dolfin/log/Event.h>
#include <dolfin/common/Array.h>
#include "GenericFile.h"

namespace dolfin
{
    
  class OpenDXFile : public GenericFile
  {
  public:
    
    OpenDXFile(const std::string filename);
    ~OpenDXFile();
    
    // Input
    
    // Output
    
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);

  private:

    void writeHeader   (FILE* fp);
    void writeMesh     (FILE* fp, Mesh& mesh);
    void writeMeshData (FILE* fp, Mesh& mesh);
    void writeFunction (FILE* fp, Function& u);
    void writeSeries   (FILE* fp, Function& u);
  
    void removeSeries  (FILE* fp);

    // Data for each frame
    class Frame {
    public:

      Frame(real time);
      ~Frame();
      
      real time;

    };

    // Frame data
    Array<Frame> frames;

    // Start position for latest time series
    long series_pos;

    // Check if we should save each mesh
    bool save_each_mesh;

    // Events
    Event event_saving_mesh;
    Event event_saving_function;
    
  };
  
}

#endif
