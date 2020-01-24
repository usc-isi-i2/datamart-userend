from math import *


class GeospatialRelated:
    def __init__(self, x: float, y: float) -> None:
        """
        Initial method of coordinate.
        :param x: X-axis coordinate, usually is latitude
        :param y: Y-axis coordinate, usually is longitude
        :return:
        """
        self.x = x
        self.y = y

    def coordinate_transform(self):
        """
        This function is used to do the axis transformation, in order to adapt to the Wikidata query service.
        So, x-axis coordinate should be longitude, y-axis should be latitude
        """
        temp = self.x
        self.x = self.y
        self.y = temp

    def distinguish_two_points(self, g: 'GeospatialRelated'):
        """
        This function is used to distinguish top left point and right bottom point in a bounding box.
        :param g: an instance of class GeospatialRelated
        :return: two tuples, one is top_left_point(x, y), another is right_bottom_point(x, y)
        """
        if self.x < g.x and self.y > g.y:
            return (self.x, self.y), (g.x, g.y)
        elif g.x < self.x and g.y > self.y:
            return (g.x, g.y), (self.x, self.y)
        else:
            return None, None


    def get_coordinate(self):
        return self.x, self.y

    @staticmethod
    def get_distance(point_A, point_B, unit="km") -> float:
        """
        function used to calculate the distance between two coordinate in latitude, longitude format
        :param point_A: [0] = latitude, [1] = longitude
        :param point_B: [0] = latitude, [1] = longitude
        :param unit: the output distance unit
        :return: a float indicate the distance
        """
        try:
            if point_A == point_B:
                return 0
            ra = 6378.140
            rb = 6356.755
            flatten = (ra - rb) / ra
            lat_A = point_A[0]
            lat_B = point_B[0]
            lng_A = point_A[1]
            lng_B = point_B[1]
            rad_lat_A = radians(lat_A)
            rad_lng_A = radians(lng_A)
            rad_lat_B = radians(lat_B)
            rad_lng_B = radians(lng_B)
            pA = atan(rb / ra * tan(rad_lat_A))
            pB = atan(rb / ra * tan(rad_lat_B))
            xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
            c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
            c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
            dr = flatten / 8 * (c1 - c2)
            distance = ra * (xx + dr)
            if unit == "km":
                pass
            elif unit == "m":
                distance *= 1000
            elif distance == "mile":
                distance *= 1.6
        except ZeroDivisionError:
            print("get zero division on calculating pair {} {}, treat value as 0".format(str(point_A), str(point_B)))
            return 0
        except:
            print("get error on calculating pair {} {}".format(str(point_A), str(point_B)))
            raise
        return distance


'''
    # not necessary code
import pyproj
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from functools import partial

    @staticmethod
    def get_area(coordinates):
        """
        transform the cordinates in km and then use shape to calculate the area
        :param coordinates:
        :return:
        """
        res = [[0, 0]]
        start_p = coordinates[0]
        for each_point in coordinates[1:-1]:
            dist_x = GeospatialRelated.get_distance(start_p, [each_point[0], start_p[1]])
            dist_y = GeospatialRelated.get_distance(start_p, [start_p[0], each_point[1]])
            # start_p = each_point
            res.append([dist_x, dist_y])
        cop = {"type": "Polygon", "coordinates": [res]}
        return shape(cop).area

    @staticmethod
    def get_area2(coordinates):
        """
        use build-in functions to get the area
        :param coordinates:
        :return:
        """
        geom = Polygon(coordinates)
        geom_area = ops.transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init='EPSG:4326'),
                pyproj.Proj(
                    proj='aea',
                    lat_1=geom.bounds[1],
                    lat_2=geom.bounds[3])),
            geom)
        return geom_area.area / 1000000
'''