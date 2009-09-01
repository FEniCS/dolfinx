// Ignore str and reemplement as the special method __str__.
// (done in dolfin_common_post.i)
// We cannot use rename as the extend directive in dolfin_shared_ptr_classes.i
// will confuse swig.
%ignore dolfin::Variable::str;
