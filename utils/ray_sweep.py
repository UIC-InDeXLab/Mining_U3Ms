from utils.linkedlist import *
from utils.dual_space import *
import pandas as pd
import math
import heapq
import numpy as np


class MedianRegion:
    def __init__(self, start: LinkedList, end: LinkedList, median: tuple):
        self.start = start
        self.end = end
        self.median = median  # The median point inside the original points list, not the projected one!

    def __str__(self):
        return (
            f"start: {self.start.point}, end: {self.end.point}, median: {self.median}"
        )


class SD:
    def __init__(self, points, mean):
        self.points = points
        self.n = len(points)
        self.mean = mean
        self.sum = sum(points)
        self.x_2_sum = sum([p[0] ** 2 for p in points])
        self.y_2_sum = sum([p[1] ** 2 for p in points])
        self.xy_sum = sum([p[0] * p[1] for p in points])

    def get_sd(self, f):
        mean_f = np.dot(self.mean, f)
        sd2 = (
            self.n * mean_f**2
            - 2 * mean_f * np.dot(self.sum, f)
            + f[0] ** 2 * self.x_2_sum
            + f[1] ** 2 * self.y_2_sum
            + 2 * f[0] * f[1] * self.xy_sum
        )
        return math.sqrt(sd2 / self.n)


class MaxSkewCalculator:
    def __init__(self, points: "pd.DataFrame", heap, vector_transfer, epsilon):
        points[0] = points[0] - points[0].min()
        points[1] = points[1] - points[1].min()
        self.points = np.array(points)
        self.q = self.points - np.mean(self.points, axis=0)
        self.intersects = {}  # {intersect --> [points]}
        self.intersect_keys = []  # sorted list of intersect points
        self.line_intersects = {}  # {line --> [LinkedList[intersect]]}
        self.sd = SD(self.points, np.mean(self.points, axis=0))
        self.heap = heap
        self.vector_transfer = vector_transfer
        self.epsilon = epsilon

    def get_angel(self, vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    def _get_first_median_on_x(self):
        x_sorted = sorted(self.points, key=lambda x: x[0])
        return tuple(x_sorted[len(self.points) // 2])

    def _calc_skew(self, f, median, verbose=False):
        m_point = np.array(median)
        mean_f = np.dot(self.sd.mean, f)

        if verbose:
            print(np.median(np.dot(self.points, f)), np.dot(m_point, f))
            p_f_1 = np.dot(self.points, f)
            mean_1 = np.mean(p_f_1)
            sd_1 = np.std(p_f_1)
            median_1 = np.median(p_f_1)
            skew_1 = (mean_1 - median_1) / sd_1

        skew = abs((mean_f - np.dot(m_point, f)) / self.sd.get_sd(f))

        if verbose:
            if round(skew, 3) != round(abs(skew_1), 3):
                print(f"Error: {skew}, {skew_1}")

        return skew

    def _get_next_median(self, intersection, candidate_points, prev_median):
        candidate_points = sorted(
            candidate_points, key=lambda x: np.arctan(x[1] / x[0])
        )
        index = candidate_points.index(prev_median)
        return candidate_points[len(candidate_points) - index - 1]

    def _get_intersects(self):
        self.intersects = {}
        for i in range(len(self.points) - 1):
            point_a = self.points[i]
            for point_b in self.points[i + 1 :]:
                try:
                    point_a, point_b = tuple(point_a), tuple(point_b)
                    intr = tuple(GeoUtility.get_intersect_in_dual(point_a, point_b))
                    if intr in self.intersects:
                        self.intersects[intr].add(point_a)
                        self.intersects[intr].add(point_b)
                    else:
                        self.intersects[intr] = set([point_a, point_b])
                except:
                    continue
        self.intersect_keys = GeoUtility.sort_points_by_polar(self.intersects)
        # Keep intersection in the upper halfspace
        self.intersect_keys = list(
            filter(lambda x: x[1] > 0 and x[0] > 0, self.intersect_keys)
        )

    def _get_line_intersects(self):
        self.line_intersects = {}

        for key in self.intersect_keys:
            intersect = key
            occurs = self.intersects[intersect]

            links = [LinkedList(intersect, [], point, None) for point in occurs]
            for link in links:
                link.neighbours = links

            for i, point in enumerate(occurs):
                if point in self.line_intersects:
                    l = self.line_intersects[point]
                    LinkedList.append_to_end(l, links[i])
                else:
                    self.line_intersects[point] = links[i]

    def preprocess(self):
        self._get_intersects()
        self._get_line_intersects()
        print(f"Number of intersects: {len(self.intersects)}")

    def train(self, verbose=False):
        first_median = self._get_first_median_on_x()
        last_vec = None
        # Main Loop
        finish = False
        median_region = MedianRegion(
            LinkedList((1 / first_median[0], 0), [], first_median, None),
            self.line_intersects[first_median],
            first_median,
        )
        while not finish:
            if verbose:
                print(median_region)
            skew_vector_start = GeoUtility.normalize_vector(median_region.start.point)
            # skew_vector_end = normalize_vector(median_region.end)
            # skew_vector_max = max_universal_skew(self.q, mean - median_region.median)
            # if lies_between(skew_vector_max, skew_vector_start, skew_vector_end):
            # heapq.heappush(skew_heap, (-calc_skew(skew_vector_max, median_region.median, sd), tuple(skew_vector_max)))
            if (
                last_vec is not None
                and self.get_angel(skew_vector_start, last_vec) > self.epsilon
            ):
                heapq.heappush(
                    self.heap,
                    (
                        -self._calc_skew(
                            skew_vector_start, median_region.median, verbose
                        ),
                        self.vector_transfer(tuple(skew_vector_start)),
                    ),
                )
                last_vec = skew_vector_start
            if last_vec is None:
                last_vec = skew_vector_start
            # heapq.heappush(skew_heap, (calc_skew(points, skew_vector_end, mean, median_region.median), tuple(skew_vector_end)))

            if median_region.end.point[0] == 0:
                print("Reached Y axis, finish.")
                break  # Reached the Y axis!

            # Next median region
            try:
                current_points = self.intersects[median_region.end.point]
            except:
                print("Didn't find end of median region, quit.")
                break

            line_b = self._get_next_median(
                median_region.end.point, current_points, median_region.median
            )
            next_neighbour = list(
                filter(lambda n: n.line == line_b, median_region.end.neighbours)
            )[0]

            if verbose:
                print(
                    f"nextneighbour: {next_neighbour.point}, next: {next_neighbour.next}"
                )

            if next_neighbour.next is None:
                new_end = LinkedList((0, 1 / line_b[1]), [], line_b, None)
            else:
                new_end = next_neighbour.next  # change of line

            median_region = MedianRegion(
                median_region.end, new_end, line_b
            )  # median changes to line_b!

    def get_high_skews(self, top_k=10):
        count = 0
        heap = self.heap.copy()
        while True:
            count = count + 1
            try:
                yield heapq.heappop(heap)
                if count == top_k:
                    break
            except:
                break

    def get_tail_accuracy(self, f, model):
        pass
