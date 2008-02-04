//--- Extend mesh entity iterators to work as Python iterators ---

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
%template(MeshFunctionReal) dolfin::MeshFunction<real>;
%template(MeshFunctionBool) dolfin::MeshFunction<bool>;

%pythoncode
%{
class MeshFunction(object):
    def __new__(self, tp, *args):
        if tp == "int":
            return MeshFunctionInt(*args)
        if tp == "uint":
            return MeshFunctionUInt(*args)
        elif tp == "real":
            return MeshFunctionReal(*args)
        elif tp == "bool":
            return MeshFunctionBool(*args)
        else:
            raise RuntimeError, "Cannot create a MeshFunction of %s" % (tp,)

MeshFunctionInt.__call__  = MeshFunctionInt.get
MeshFunctionUInt.__call__ = MeshFunctionUInt.get
MeshFunctionReal.__call__ = MeshFunctionReal.get
MeshFunctionBool.__call__ = MeshFunctionBool.get

%}

%extend dolfin::Mesh {
  dolfin::MeshFunction<uint>* partition(dolfin::uint n) {
    dolfin::MeshFunction<dolfin::uint>* partitions = new dolfin::MeshFunction<dolfin::uint>;
    self->partition(*partitions, n);
    return partitions;
  }
}

//--- Extend Point interface with Python selectors ---

%extend dolfin::Point {
  real get(int i) { return (*self)[i]; }
  void set(int i, real val) { (*self)[i] = val; }
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
