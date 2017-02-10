/* -*- C -*- */
// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-05-10
// Last changed: 2014-05-20

// ===========================================================================
// SWIG directives for the DOLFIN geometry kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Point ignores
//-----------------------------------------------------------------------------
%ignore dolfin::Point::operator=;
%ignore dolfin::Point::operator[];
%ignore dolfin::Point::coordinates;
%rename ("coordinates") dolfin::Point::array() const;

//-----------------------------------------------------------------------------
// Ignore reference (to FunctionSpaces) constructors of BoundingBoxTree
//-----------------------------------------------------------------------------
%ignore dolfin::BoundingBoxTree::BoundingBoxTree(const Mesh&);
%ignore dolfin::BoundingBoxTree::BoundingBoxTree(const Mesh&, unsigned int);

//-----------------------------------------------------------------------------
// Ignore nested classes. They are not supported by SWIG
//-----------------------------------------------------------------------------
%warnfilter(325) dolfin::GenericBoundingBoxTree::BBox;
%warnfilter(325) dolfin::GenericBoundingBoxTree::less_x_point;
%warnfilter(325) dolfin::GenericBoundingBoxTree::less_y_point;
%warnfilter(325) dolfin::GenericBoundingBoxTree::less_z_point;
%warnfilter(325) dolfin::GenericBoundingBoxTree::less_x_bbox;
%warnfilter(325) dolfin::GenericBoundingBoxTree::less_y_bbox;
%warnfilter(325) dolfin::GenericBoundingBoxTree::less_z_bbox;
