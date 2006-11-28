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
%template(MeshFunctionFloat) dolfin::MeshFunction<double>;
%template(MeshFunctionBool) dolfin::MeshFunction<bool>;

%pythoncode
%{
class MeshFunction(object):
    def __new__(self, tp):
        if tp == int:
            return MeshFunctionInt()
        elif tp == float:
            return MeshFunctionFloat()
        elif tp == bool:
            return MeshFunctionBool()
        else:
            raise RuntimeError, "Cannot create a MeshFunction of %s" % (tp,)

MeshFunctionInt.__call__   = MeshFunctionInt.get
MeshFunctionFloat.__call__ = MeshFunctionFloat.get
MeshFunctionBool.__call__  = MeshFunctionBool.get

%}

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
