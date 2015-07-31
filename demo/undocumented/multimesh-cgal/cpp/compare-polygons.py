from decimal import Decimal
import sys

intersections1 = {}

# Shift coordinates in a consistent way, so that polygons can be compared
def shift_polygon(coordinates) :
    min_index = 0
    min_coordinate = coordinates[0]
    for i in range(1, len(coordinates)) :
        if coordinates[i][0] < min_coordinate[0] or (coordinates[i][0] == min_coordinate[0] and coordinates[i][1] < min_coordinate[1]) :
            min_index = i
            min_coordinate = coordinates[i]
    return coordinates[min_index : len(coordinates)+min_index]

def dot(a, b) :
    return a[0]*b[0] + a[1]*b[1]

def polygon_is_counterclockwise(coordinates) :
    # assuming polygon is convex
    a = (coordinates[1][0]-coordinates[0][0], coordinates[1][1]-coordinates[0][1])
    t = dot(coordinates[2], a)/dot(a,a)
    proj = (coordinates[0][0]+t*a[0], coordinates[0][1]+t*a[1])


with open(sys.argv[1], "r") as file1 :
    for line in file1 :
        cells, numbers = line.split(":")
        cells = cells.strip().split()
        cell0 = eval(cells[0])
        cell1 = eval(cells[1])
        points = numbers.strip().strip(",").split(",")
        # print(points)
        coordinates = [(Decimal(x.split()[0]), Decimal(x.split()[1])) for x in points]
        intersections1[(cell0, cell1)] = shift_polygon(coordinates)

for item in intersections1.items() :
    print(item)

# with open(sys.argv[2], "r") as file2 :
#     for line1, line2 in zip(file1, file2) :
#         cells1, coordinates1 = line1.split(":")
#         cells2, coordinates2 = line2.split(":")
#         if cells1.strip() != cells2.strip() :
#             print "--- ", cells1, " <---> ", cells2
#             #print "Comparing ", coordinates1, " and ", coordinates2
