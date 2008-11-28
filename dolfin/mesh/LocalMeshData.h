// Copyright (C) 2006 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-11-28

#ifndef __LOCALMESHDATA_H
#define __LOCALMESHDATA_H




namespace dolfin
{

  class LocalMeshData
  {
      public:

        /// Constructor
        LocalMeshData();

        /// Destructor
        ~LocalMeshData();

      private:
        std::vector<std::vector<double> > vertex_coordinates;
        std::vector<uint> vertex_indices;
        std::vector<std::vector<uint> > cell_vertices;

        friend XMLLocalMeshData;

  };

}
#endif
