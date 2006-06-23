// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-12
// Last changed: 2006-06-21

#include <dolfin/MeshEntityIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const MeshEntityIterator& it)
{
  stream << "[ Mesh entity iterator at position "
	 << it.pos
	 << " stepping from 0 to "
	 << it.pos_end - 1
	 << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
