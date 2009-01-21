%extend dolfin::Parametrized {
void dolfin_set(std::string key, std::string value) {
    self->set(key, value);
}

void dolfin_set(std::string key, double value) {
    self->set(key, value);
}

void dolfin_set(std::string key, int value) {
    self->set(key, value);
}

%pythoncode %{
def set(self, key, value):
    """
    Set parameter key to value.
    """
    return self.dolfin_set(key, value)
%}

}

%extend dolfin::FunctionSpace {
%pythoncode %{
def __contains__(self,u):
    " Check whether a function is in the FunctionSpace"
    assert(isinstance(u,cpp_Function))
    return u._in(self)
%}
}

%extend dolfin::Function {
%pythoncode %{
def sub(self,i):
    """ Return a sub function
    
    A sub function can be extracted from a discrete function that is in a
    a MixedFunctionSpace or in a VectorFunctionSpace. The sub function is a
    function that resides in a sub space of the mixed space.
    
    The sub functions are numbered from i = 0..N-1, where N is the total number
    of sub spaces. 
    
    @param i : The number of the sub function
    """
    num_sub_space = self.function_space().element().num_sub_elements()
    if not num_sub_space > 1:
        raise RuntimeError, "No subfunctions to extract"
    if not i < num_sub_space:
        raise RuntimeError, "Can only extract subfunctions with i = 0..%d"%num_sub_space
    return DiscreteFunction(self._sub(i))

def split(self):
    " Extract any sub functions"
    num_sub_space = self.function_space().element().num_sub_elements()
    if not num_sub_space > 1:
        raise RuntimeError, "No subfunctions to extract"
    return tuple(self.sub(i) for i in xrange(num_sub_space))
%}
}
%extend dolfin::Data {
  double get_coordinate(uint i) {
    if (i < self->cell().dim())  
      return self->x[i];  
    else 
      throw std::out_of_range("index out of range");
    return 0.0; 
  }
}

