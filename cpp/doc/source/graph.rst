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


Re-ordering
-----------

.. doxygenfunction:: dolfinx::graph::reorder_rcm
   :project: DOLFINx

.. doxygenfunction:: dolfinx::graph::reorder_sfc_morton
   :project: DOLFINx

.. doxygenfunction:: dolfinx::graph::reorder_sfc_hilbert
   :project: DOLFINx

.. doxygentypedef:: dolfinx::graph::Reorder
   :project: DOLFINx

.. doxygentypedef:: dolfinx::graph::reorder_graph_fn
   :project: DOLFINx

.. doxygentypedef:: dolfinx::graph::reorder_geom_fn
   :project: DOLFINx

Partitioning
------------

.. doxygenfunction:: dolfinx::graph::partition_graph
   :project: DOLFINx

.. doxygenstruct:: dolfinx::graph::Partitioner
   :project: DOLFINx
   :members:

.. doxygenfunction:: dolfinx::graph::has_partitioner
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

.. doxygentypedef:: dolfinx::graph::geom_partition_fn
   :project: DOLFINx

.. doxygentypedef:: dolfinx::graph::hybrid_partition_fn
   :project: DOLFINx

.. doxygentypedef:: dolfinx::graph::AnyPartitionFunction
   :project: DOLFINx


.. doxygenenum:: dolfinx::graph::scotch::strategy
   :project: DOLFINx


Functions for building distributed graphs
-----------------------------------------

.. doxygennamespace:: dolfinx::graph::build
   :project: DOLFINx
