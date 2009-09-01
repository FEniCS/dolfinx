//--- Extend mesh entity iterators to work as Python iterators ---

%feature("docstring")  dolfin::MeshFunction::fill "

Set all values to given value";

%extend dolfin::MeshFunction {

void fill(const T& value)
{
  (*self) = value;
}

}

%extend dolfin::MeshEntityIterator {
%pythoncode
%{
def __iter__(self):
  self.first = True
  return self

def next(self):
  if not self.first:
    self.increment()
  if self.end():
    raise StopIteration
  self.first = False
  return self.dereference()
%}
}

//--- Map MeshFunction template to Python ---

%template(MeshFunctionInt) dolfin::MeshFunction<int>;
%template(MeshFunctionUInt) dolfin::MeshFunction<unsigned int>;
%template(MeshFunctionDouble) dolfin::MeshFunction<double>;
%template(MeshFunctionBool) dolfin::MeshFunction<bool>;

%pythoncode
%{
_doc_string = MeshFunctionInt.__doc__
_doc_string += """     
    Arguments
    ---------
    @param tp:
      String defining the type of the MeshFunction
      Allowed: 'int', 'uint', 'double', and 'bool'
    @param mesh:
      A DOLFIN mesh.
      Optional.
    @param dim:
      The topological dimension of the MeshFunction.
      Optional.
    @param filename:
      A filename with a stored MeshFunction.
      Optional.

"""
class MeshFunction(object):
    __doc__ = _doc_string
    def __new__(self, tp, *args):
        if tp == "int":
            return MeshFunctionInt(*args)
        if tp == "uint":
            return MeshFunctionUInt(*args)
        elif tp == "double":
            return MeshFunctionDouble(*args)
        elif tp == "bool":
            return MeshFunctionBool(*args)
        else:
            raise RuntimeError, "Cannot create a MeshFunction of type '%s'." % (tp,)

del _doc_string

MeshFunctionInt.__call__    = MeshFunctionInt.get
MeshFunctionUInt.__call__   = MeshFunctionUInt.get
MeshFunctionDouble.__call__ = MeshFunctionDouble.get
MeshFunctionBool.__call__   = MeshFunctionBool.get


%}

//%extend dolfin::Mesh {
//  dolfin::MeshFunction<uint>* partition(dolfin::uint n) {
//    dolfin::MeshFunction<dolfin::uint>* partitions = new dolfin::MeshFunction<dolfin::uint>;
//   self->partition(*partitions, n);
//    return partitions;
//  }
//}

//--- Extend Point interface with Python selectors ---

%extend dolfin::Point {
  double get(int i) { return (*self)[i]; }
  void set(int i, double val) { (*self)[i] = val; }
}

%pythoncode
%{
  def __getitem__(self, i):
      return self.get(i)
  def __setitem__(self, i, val):
      self.set(i, val)

  Point.__getitem__ = __getitem__
  Point.__setitem__ = __setitem__
%}
