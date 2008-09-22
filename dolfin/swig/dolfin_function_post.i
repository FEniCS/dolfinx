// Trick to expose protected member cell() in Python
%extend dolfin::Function{
const dolfin::Cell& new_cell() const{
    return self->cell();
}
}

// Trick to expose protected member normal() in Python
%extend dolfin::Function{
dolfin::Point new_normal() const{
    return self->normal();
}
}

%extend dolfin::Parametrized {
void dolfin_set(std::string key, std::string value) {
    self->set(key, value);
}

void dolfin_set(std::string key, dolfin::real value) {
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
