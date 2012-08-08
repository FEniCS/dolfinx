/* -*- C -*- */
// ===========================================================================
// SWIG directives for the DOLFIN graph kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
// ===========================================================================

// ---------------------------------------------------------------------------
// Instantiate template classes
// ---------------------------------------------------------------------------
%template(Graph) std::vector<dolfin::graph_set_type>;
