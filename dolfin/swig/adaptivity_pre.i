/* -*- C -*- */
// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3 or any later version
//
// First added: 2011-02-23
// Last changed: 2011-02-23

// ===========================================================================
// SWIG directives for the DOLFIN adaptivity kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

namespace dolfin {
  class ErrorControl;
}

%template (HierarchicalErrorControl) dolfin::Hierarchical<dolfin::ErrorControl>;
