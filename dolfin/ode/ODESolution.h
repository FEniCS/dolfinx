// Copyright (C) 2008 Benjamin Kehlet
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-06-11
// Last changed: 2011-02-16

#ifndef __ODESOLUTION_H
#define __ODESOLUTION_H

#include <vector>
#include <tr1/tuple>
#include <iterator>
#include <utility>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/common/Array.h>
#include "Method.h"

namespace dolfin
{

  /// ODESolution stores the solution from the ODE solver, primarily to
  /// be able to solve the dual problem. A number of interpolated values
  /// is cached, since the ODE solver repeatedly requests evaluation of
  /// the same t.
  ///
  /// The samples are stored in memory if possible, otherwise stored
  /// in a temporary file and fetched from disk in blocks when needed.
  ///
  /// Since GMP at the moment doesn't support saving binary operands
  /// on disk this class uses ascii files for storage.
  /// Fortunately storing operands on disk in binary is planned in
  /// the next major release of GMP.


  class Lagrange;
  class ODESolutionIterator;

//-----------------------------------------------------------------------------
//-------------------------------- timeslabdata -------------------------------
//-----------------------------------------------------------------------------
  class ODESolutionData
  {

  public:

  ODESolutionData(const real& a, const real& k, uint nodal_size, uint N, const real* values) :
    a(a), k(k), N(N), nodal_size(nodal_size)
    {
      nv = new real[nodal_size * N];
      real_set(N * nodal_size, nv, values);
    }

    //copy constructor
    ODESolutionData(const ODESolutionData& cp)
    {
      a = cp.a;
      k = cp.k;
      N = cp.N;
      nodal_size = cp.nodal_size;
      nv = new real[N * nodal_size];
      real_set(N * nodal_size, nv, cp.nv);
    }

    ~ODESolutionData()
     { delete [] nv; }

    real b() {return a+k;}

    // Evaluate the solution at t = a (first nodal point)
    void eval_a(real* u)
    {
      for (uint i = 0; i < N; i++)
        u[i] = nv[i * nodal_size];
    }

    real a;
    real k;
    real* nv;
    uint N;
    uint nodal_size;

  };

//-----------------------------------------------------------------------------
//-------------------------------- ODESolution---------------------------------
//-----------------------------------------------------------------------------

  class ODESolution
  {
  public:

    /// Create solution data for given ODE
    ODESolution();
    ODESolution(std::string filename, uint number_of_files = 1); //load data from file

    /// Destructor
    ~ODESolution();

    //must be called before starting to add data
    void init(uint N, const Lagrange& trial, const real* quad_weights);

    // Add solution data. Must be in write_mode
    void add_timeslab(const real& a, const real& b, const real* values);

    /// Make object ready for evaluating, set to read mode
    void flush();

    /// Evaluate (interpolate) value of solution at given time
    void eval(const real& t, Array<real>& y);

    /// Get timeslab (used when iterating)
    ODESolutionData& get_timeslab(uint index);

    /// Get pointer to weights
    const real* get_weights() const;

    void set_filename(std::string filename);
    void save_to_file();

    std::string str(bool verbose) const;

    uint size() const {return no_timeslabs;}
    uint nsize() const {return nodal_size;}
    real endtime() const {return T;}

    // iterator
    typedef ODESolutionIterator iterator;
    iterator begin();
    iterator end();

  private:
    Lagrange* trial;
    uint N; //number of components
    uint nodal_size;
    real T; //endtime. Updated when new timeslabs are added

    uint no_timeslabs;
    std::vector<ODESolutionData> data; //data in memory

    real* quadrature_weights;

    //cache
    std::pair<real, real*>* cache;
    uint cache_size;
    uint ringbufcounter;

    bool initialized;
    bool read_mode;

    bool use_exact_interpolation;

    // Stuff related to file storage
    static const uint max_filesize = 3000000000u;     // approx 3GB
    std::vector< std::pair<real, uint> > file_table;  // table mapping t values and index to files
    uint fileno_in_memory;                            // which file is currently in memory
    bool data_on_disk;                                //
    uint max_timeslabs;                               // number of timeslabs pr file and in memory
    bool dirty;                                       // all data written to disk
    std::string filename;

    uint get_file_index(const real& t); //find which file stores timeslab containing given t
    void read_file(uint file_number);
    dolfin::uint open_and_read_header(std::ifstream& file, uint filenumber);

    void add_data(const real& a, const real& b, const real* data);

    void interpolate_exact(Array<real>& y, ODESolutionData& timeslab, real tau);

    //Evaluate linearly between closest nodal points
    void interpolate_linear(Array<real>& y, ODESolutionData& timeslab, real tau);

    uint get_buffer_index(const real& t);
    int buffer_index_cache;

    // some functions for convenience
    real a_in_memory() {return data[0].a;}
    real b_in_memory() {return data[data.size()-1].a + data[data.size()-1].k;}
    uint a_index_in_memory() {return data_on_disk ? file_table[fileno_in_memory].second : 0;}
    uint b_index_in_memory() {return a_index_in_memory() + data.size()-1;}

  };

//-----------------------------------------------------------------------------
//-------------------------------- iterator -----------------------------------
//-----------------------------------------------------------------------------
  class ODESolutionIterator :
    public std::iterator<std::input_iterator_tag, ODESolutionData*>
  {

  public:
    ODESolutionIterator(ODESolution& u) : u(u), index(0) {}
    ODESolutionIterator(ODESolution& u, int index) : u(u), index(index) {}
    ODESolutionIterator(const ODESolutionIterator& it) :
      u(it.get_ODESolution()), index(it.get_index()) {}

    ODESolutionIterator& operator++() {++index;return *this;}
    void operator++(int) {++index;}

    uint get_index() const {return index;}

    ODESolution& get_ODESolution() const {return u;};

    bool operator==(const ODESolutionIterator& rhs) {return index == rhs.get_index();}
    bool operator!=(const ODESolutionIterator& rhs) {return index != rhs.get_index();}

    ODESolutionData& operator*() {return u.get_timeslab(index);}

  private:
    ODESolution& u;
    uint index;

  };

}
#endif
