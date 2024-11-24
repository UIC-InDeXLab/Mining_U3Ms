"""
Linked list of intersect points of a line
"""


class LinkedList:
    def __init__(self, point, neighbours, line, next=None):
        self.point = point  # Intersect point
        self.next = next
        self.neighbours = (
            neighbours  # Other linked lists, for the lines, lying on this intersect
        )
        self.line = line

    def append_to_end(start, l):
        cur = start
        while cur.next is not None:
            cur = cur.next
        cur.next = l

    def __str__(self):
        return f"value: {self.point} --> {self.next is None}"
