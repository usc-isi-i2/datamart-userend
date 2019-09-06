class GeospatialRelated:
    def __init__(self, x: float, y: float) -> None:
        """
        Initial method of coordinate.
        :param x: X-axis coordinate point, usually is latitude
        :param y: Y-axis coordinate point, usually is longitude
        :return:
        """
        self.x = x
        self.y = y

    def coordinate_transform(self):
        """
        This function is used to change latitude into longitude, and longitude into latitude.
        So, x-axis coordinate point becomes longitude, y-axis becomes latitude
        """
        temp = self.x
        self.x = self.y
        self.y = temp

    def distinguish_two_points(self, g: 'GeospatialRelated'):
        """
        This function is used to distinguish top left point and right bottom point in bounding box.
        :param g: an instance of class GeospatialRelated
        :return: two tuples, one is top_left_point(x, y), another is right_bottom_point(x, y)
        """
        if self.x < g.x and self.y > g.y:
            return (self.x, self.y), (g.x, g.y)
        elif g.x < self.x and g.y > self.y:
            return (g.x, g.y), (self.x, self.y)
        else:
            return None, None
