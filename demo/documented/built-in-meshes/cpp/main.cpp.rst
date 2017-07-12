Built-in meshes
===============

This demo illustrates:

* How to define some of the different built-in meshes in DOLFIN
* Writing meshes to the XDMF format for viewing in Paraview.

Implementation
--------------

Running this demo requires the files: :download:`main.cpp` and
:download:`CMakeLists.txt`.


Under construction

.. code-block:: cpp

   #include <dolfin.h>

   using namespace dolfin;

   int main()
   {
     if (dolfin::MPI::size(MPI_COMM_WORLD) == 1)
     {
       UnitIntervalMesh interval(10);
       XDMFFile("interval.xdmf").write(interval);
     }

     UnitSquareMesh square_default(10, 10);
     XDMFFile("square_default.xdmf").write(square_default);

     UnitSquareMesh square_left(10, 10, "left");
     XDMFFile("square_left.xdmf").write(square_left);

     UnitSquareMesh square_crossed(10, 10, "crossed");
     XDMFFile("square_crossed.xdmf").write(square_crossed);

     UnitSquareMesh square_right_left(10, 10, "right/left");
     XDMFFile("square_right_left.xdmf").write(square_right_left);

     RectangleMesh rectangle_default(Point(0.0, 0.0), Point(10.0, 4.0), 10, 10);
     XDMFFile("rectangle_default.xdmf").write(rectangle_default);

     RectangleMesh rectangle_right_left(Point(-3.0, 2.0), Point(7.0, 6.0), 10, 10, "right/left");
     XDMFFile("rectangle_right_left.xdmf").write(rectangle_right_left);

     UnitCubeMesh cube(10, 10, 10);
     XDMFFile("cube.xdmf").write(cube);

     BoxMesh box(Point(0.0, 0.0, 0.0), Point(10.0, 4.0, 2.0), 10, 10, 10);
     XDMFFile("box.xdmf").write(box);

     return 0;
   }
