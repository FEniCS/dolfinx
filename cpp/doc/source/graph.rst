Graph (``dolfinx::graph``)
==========================

Adjacency list
--------------

.. doxygenclass:: dolfinx::graph::AdjacencyList
   :project: DOLFINx
   :members:


Adjacency list builders
-----------------------

.. doxygenfunction:: dolfinx::graph::regular_adjacency_list
   :project: DOLFINx

.. doxygenfunction:: dolfinx::graph::create_adjacency_data
   :project: DOLFINx


Re-ordering
-----------

.. doxygenfunction:: dolfinx::graph::reorder_gps
   :project: DOLFINx


Partitioning
------------

.. doxygenfunction:: dolfinx::graph::partition_graph
   :project: DOLFINx

.. doxygenfunction:: dolfinx::graph::scotch::partitioner
   :project: DOLFINx

.. doxygenfunction:: dolfinx::graph::parmetis::partitioner
   :project: DOLFINx

.. doxygenfunction:: dolfinx::graph::kahip::partitioner
   :project: DOLFINx


Enumerations and typedefs
-------------------------

.. doxygentypedef:: dolfinx::graph::partition_fn
   :project: DOLFINx


.. doxygenenum:: dolfinx::graph::scotch::strategy
   :project: DOLFINx


Functions for building distributed graphs
-----------------------------------------

.. doxygennamespace:: dolfinx::graph::build
   :project: DOLFINx
