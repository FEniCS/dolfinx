// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-11
// Last changed: 2006-06-16

#include <dolfin/dolfin_log.h>
#include <dolfin/MeshEntity.h>

//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const MeshEntity& entity)
{
  stream << "[ Mesh entity " << entity.index()
	 << " of topological dimension " << entity.dim() << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
