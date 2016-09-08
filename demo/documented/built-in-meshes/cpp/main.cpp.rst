Built-in meshes
===============

This demo illustrates:

* How to define some of the different built-in meshes in DOLFIN


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
       info("Plotting a UnitIntervalMesh");
       plot(interval, "Unit interval");
     }

     UnitSquareMesh square_default(10, 10);
     info("Plotting a UnitSquareMesh");
     plot(square_default, "Unit square");

     UnitSquareMesh square_left(10, 10, "left");
     info("Plotting a UnitSquareMesh");
     plot(square_left, "Unit square (left)");

     UnitSquareMesh square_crossed(10, 10, "crossed");
     info("Plotting a UnitSquareMesh");
     plot(square_crossed, "Unit square (crossed)");

     UnitSquareMesh square_right_left(10, 10, "right/left");
     info("Plotting a UnitSquareMesh");
     plot(square_right_left, "Unit square (right/left)");

     RectangleMesh rectangle_default(Point(0.0, 0.0), Point(10.0, 4.0), 10, 10);
     info("Plotting a RectangleMesh");
     plot(rectangle_default, "Rectangle");

     RectangleMesh rectangle_right_left(Point(-3.0, 2.0), Point(7.0, 6.0), 10, 10, "right/left");
     info("Plotting a RectangleMesh");
     plot(rectangle_right_left, "Rectangle (right/left)");

     UnitCubeMesh cube(10, 10, 10);
     info("Plotting a UnitCubeMesh");
     plot(cube, "Unit cube");

     BoxMesh box(Point(0.0, 0.0, 0.0), Point(10.0, 4.0, 2.0), 10, 10, 10);
     info("Plotting a BoxMesh");
     plot(box, "Box");

     interactive();

     return 0;
   }
