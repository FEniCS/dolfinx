// Source term
class Source2D : public dolfin::Function
{
public:

  Source2D(dolfin::Mesh& mesh) : dolfin::Function(mesh)
  {
    C = 1.0;
  }

  double eval(const double* x) const
  {

    double vx = -exp(x[0])*(x[1]*cos(x[1]) + sin(x[1]));
    double vy = exp(x[0])*(x[1]*sin(x[1]));

    double ux = 5.0*DOLFIN_PI*cos(5.0*DOLFIN_PI*x[0])*sin(5.0*DOLFIN_PI*x[1]);
    double uy = 5.0*DOLFIN_PI*sin(5.0*DOLFIN_PI*x[0])*cos(5.0*DOLFIN_PI*x[1]);
    double uxx = -25.0*DOLFIN_PI*DOLFIN_PI*sin(5.0*DOLFIN_PI*x[0])*sin(5.0*DOLFIN_PI*x[1]);
    double uyy = -25.0*DOLFIN_PI*DOLFIN_PI*sin(5.0*DOLFIN_PI*x[0])*sin(5.0*DOLFIN_PI*x[1]);

    return vx*ux + vy*uy - C*(uxx + uyy);
  }
  double C;
};

// Velocity
class Velocity2D : public dolfin::Function
{
public:

  Velocity2D(dolfin::Mesh& mesh) : dolfin::Function(mesh) {}

  void eval(double* values, const double* x) const
  {
    values[0] = -exp(x[0])*(x[1]*cos(x[1]) + sin(x[1]));
    values[1] = exp(x[0])*(x[1]*sin(x[1]));
  }

  dolfin::uint rank() const
  {
    return 1;
  }

  dolfin::uint dim(dolfin::uint i) const
  {
    if(i == 0)
    {
      return 2;
    }
    throw std::runtime_error("Invalid dimension i in dim(i).");
  }

};

class OutflowFacet2D : public dolfin::Function
{
public:

  OutflowFacet2D(dolfin::Mesh& mesh) : dolfin::Function(mesh)
  {
    velocity = 0;
  }

  double eval(const double* x) const
  {
    // If there is no facet (assembling on interior), return 0.0
    if (facet() < 0)
      return 0.0;
    else
    {
      if (!velocity)
        dolfin::error("Attach a velocity function.");
      double normal_vector[2];
      double velocities[2] = {0.0, 0.0};

      // Compute facet normal
      for (dolfin::uint i = 0; i < cell().dim(); i++)
        normal_vector[i] = cell().normal(facet(), i);

      // Get velocities
      velocity->eval(velocities, x);

      // Compute dot product of the facet outward normal and the velocity vector
      double dot = 0.0;
      for (dolfin::uint i = 0; i < cell().dim(); i++)
        dot += normal_vector[i]*velocities[i];

      // If dot product is positive the facet is an outflow facet,
      // meaning the contribution from this cell is on the upwind side.
      if (dot > DOLFIN_EPS)
      {
        return 1.0;
      }
      else
      {
        return 0.0;
      }
    }
  }
  dolfin::Function* velocity;
};
