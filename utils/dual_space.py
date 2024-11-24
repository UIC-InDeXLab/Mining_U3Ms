import numpy as np
import numpy.linalg as linalg


class GeoUtility:
    def get_intersect_in_dual(point_a, point_b):
        value = linalg.solve(np.array([point_a, point_b]), np.ones(2))
        return np.array([round(value[0], 5), round(value[1], 5)])

    def sort_points_by_polar(points):
        keys = points.keys()
        return sorted(keys, key=lambda x: np.arctan(x[1] / x[0]))

    def normalize_vector(vector: tuple):
        return np.array(vector) / sum(vector)

    def max_skew_given_median(q, median):
        direction = np.dot(linalg.inv(np.dot(q.T, q)), median)  # Direction of max skew
        return direction / sum(direction)  # Unit vector

    def is_in_between(query_vector, start_vector, end_vector):
        return (
            np.arctan(start_vector[1] / start_vector[0])
            < np.arctan(query_vector[1] / query_vector[0])
            < np.arctan(end_vector[1] / end_vector[0])
        )
