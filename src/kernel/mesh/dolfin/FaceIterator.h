// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FACE_ITERATOR_H
#define __FACE_ITERATOR_H

#include <dolfin/Array.h>
#include <dolfin/List.h>
#include <dolfin/Table.h>

namespace dolfin {

  class Mesh;
  class Cell;
  class Face;
  class Boundary;

  typedef Face* FacePointer;
  
  class FaceIterator {
  public:
	 
    FaceIterator(const Mesh& mesh);
    FaceIterator(const Mesh* mesh);

    FaceIterator(const Boundary& boundary);

    FaceIterator(const Cell& cell);
    FaceIterator(const CellIterator& cellIterator);

    ~FaceIterator();
    
    operator FacePointer() const;
    
    FaceIterator& operator++();
    bool end();
    bool last();
    int index();
	 
    Face& operator*() const;
    Face* operator->() const;
    bool  operator==(const FaceIterator& f) const;
    bool  operator!=(const FaceIterator& f) const;
    bool  operator==(const Face& f) const;
    bool  operator!=(const Face& f) const;
	 
    // Base class for face iterators
    class GenericFaceIterator {
    public:
		
      virtual void operator++() = 0;
      virtual bool end() = 0;
      virtual bool last() = 0;
      virtual int index() = 0;
		
      virtual Face& operator*() const = 0;
      virtual Face* operator->() const = 0;
      virtual Face* pointer() const = 0;
		
    };
	 
    // Iterator for the faces in a mesh
    class MeshFaceIterator : public GenericFaceIterator {
    public:
		
      MeshFaceIterator(const Mesh& mesh); 
		
      void operator++();
      bool end();
      bool last();
      int index();
		
      Face& operator*() const;
      Face* operator->() const;
      Face* pointer() const;

      Table<Face>::Iterator face_iterator;
      Table<Face>::Iterator at_end;
		
    };

    // Iterator for the faces on a boundary
    class BoundaryFaceIterator : public GenericFaceIterator {
    public:

      BoundaryFaceIterator(const Boundary& boundary);
      void operator++();
      bool end();
      bool last();
      int index();

      Face& operator*() const;
      Face* operator->() const;
      Face* pointer() const;
		
    private:

      List<Face*>::Iterator face_iterator;
      
    };

    // Iterator for the faces in a cell
    class CellFaceIterator : public GenericFaceIterator {
    public:

      CellFaceIterator(const Cell& cell);
      void operator++();
      bool end();
      bool last();
      int index();

      Face& operator*() const;
      Face* operator->() const;
      Face* pointer() const;
		
    private:

      Array<Face*>::Iterator face_iterator;
		
    };

  private:

    GenericFaceIterator* f;
	 
  };

}

#endif
