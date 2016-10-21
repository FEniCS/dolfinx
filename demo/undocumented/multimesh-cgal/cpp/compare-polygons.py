from decimal import Decimal
import sys

intersections1 = {}

# Shift coordinates in a consistent way, so that polygons can be compared
# No longer relevant. cgal demo now outputs consistently with multimesh implementation.
# def shift_polygon(coordinates) :
#     min_index = 0
#     min_coordinate = coordinates[0]
#     for i in range(1, len(coordinates)) :
#         if coordinates[i][0] < min_coordinate[0] or (coordinates[i][0] == min_coordinate[0] and coordinates[i][1] < min_coordinate[1]) :
#             min_index = i
#             min_coordinate = coordinates[i]
#     return coordinates[min_index : len(coordinates)+min_index]

# def dot(a, b) :
#     return a[0]*b[0] + a[1]*b[1]

# def polygon_is_counterclockwise(coordinates) :
#     # assuming polygon is convex
#     a = (coordinates[1][0]-coordinates[0][0], coordinates[1][1]-coordinates[0][1])
#     t = dot(coordinates[2], a)/dot(a,a)
#     proj = (coordinates[0][0]+t*a[0], coordinates[0][1]+t*a[1])


with open(sys.argv[1], "r") as file :
    for line in file :
        cells, numbers = line.split(":")
        cells = cells.strip().split()
        cell0 = eval(cells[0])
        cell1 = eval(cells[1])
        points = numbers.strip().strip(",").split(",")
        # print(points)
        coordinates = [(Decimal(x.split()[0]), Decimal(x.split()[1])) for x in points]
        assert len(coordinates) % 3 == 0
        intersections1[(cell0, cell1)] = coordinates

max_diff = 0.
with open(sys.argv[2], "r") as file :
    for line in file :
        cells, numbers = line.split(":")
        cells = cells.strip().split()
        cell0 = eval(cells[0])
        cell1 = eval(cells[1])
        points = numbers.strip().strip(",").split(",")
        coordinates = [(Decimal(x.split()[0]), Decimal(x.split()[1])) for x in points]
        assert (cell0, cell1) in intersections1
        diff = [(abs(x1[0]-x2[0]), abs(x1[1]-x2[1])) for x1, x2, in zip(coordinates, intersections1[(cell0, cell1)])]

        for d in diff :
            if max(d) > max_diff :
                max_diff = max(d)
                
            if d[0] > 1e-15 or d[1] > 1e-15 :
                print("Large diff : (", cell0, ", ", cell1, ")")
                print(diff)
        # flatten diff array
        
print(max_diff)
#print(intersections1)
#for item in intersections1.items() :
#    print(item)

# with open(sys.argv[2], "r") as file2 :
#     for line1, line2 in zip(file1, file2) :
#         cells1, coordinates1 = line1.split(":")
#         cells2, coordinates2 = line2.split(":")
#         if cells1.strip() != cells2.strip() :
#             print "--- ", cells1, " <---> ", cells2
#             #print "Comparing ", coordinates1, " and ", coordinates2
