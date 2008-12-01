// Copyright (C) 2006 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-11-28

#ifndef __XMLLOCALMESHDATA_H
#define __XMLLOCALMESHDATA_H



/// Documentation of class XMLLocalMeshData

namespace dolfin
{

  class XMLLocalMeshData: public XMLObject
  {
      public:

        /// Constructor
        XMLLocalMeshData(LocalMeshData& mesh_data);

        /// Destructor
        ~XMLLocalMeshData();

        void startElement (const xmlChar* name, const xmlChar** attrs);
        void endElement   (const xmlChar* name);
        
        void open(std::string filename);
        bool close();
   
      private:

        enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY,
                      DONE};
        // Callbacks for reading XML data
        void readMesh        (const xmlChar* name, const xmlChar** attrs);
        void readVertices    (const xmlChar* name, const xmlChar** attrs);
        void readVertex      (const xmlChar* name, const xmlChar** attrs);
        void readInterval    (const xmlChar* name, const xmlChar** attrs);
        void readTriangle    (const xmlChar* name, const xmlChar** attrs);
        void readTetrahedron (const xmlChar* name, const xmlChar** attrs);
        void readMeshFunction(const xmlChar* name, const xmlChar** attrs);
        void readArray       (const xmlChar* name, const xmlChar** attrs);
        void readMeshEntity  (const xmlChar* name, const xmlChar** attrs);
        void readArrayElement(const xmlChar* name, const xmlChar** attrs);
        
        // Partition parsed vertices, called when finished reading vertices
        void closeVertices();
        
        // Close mesh, called when finished reading data
        void closeMesh();

        uint vertex_index_start, vertex_index_stop;

        //uint* vertex_distribution;
        LocalMeshData& mesh_data;

  };

}
#endif
