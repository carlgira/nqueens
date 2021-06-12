from abc import ABC

import nqueens
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely import affinity
import random

phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l


def polar_2_cartesian(theta, r, x0=0.0, y0=0.0):
    return r*np.cos(theta) + x0 , r*np.sin(theta) + y0


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(line1, line2):
    v1 = np.array(line1)
    v2 = np.array(line2)
    v1 = v1[1]-v1[0]
    v2 = v2[1]-v2[0]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
'''
def get_angle(p0, p1, p2):
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)
'''

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def create_polygon(sol, x, y):
    tx = []
    ty = []
    for e in range(len(sol)):
        i = np.where(sol == e)[0][0]
        tx.append(round(x[i], 10))
        ty.append(round(y[i], 10))

    poly = Polygon(list(zip(tx, ty)))
    return poly

def get_inverse_poly(poly):
    l = list(poly.exterior.coords)[:-1]
    return Polygon(np.array(l).dot([[1, 0], [0, -1]]))

def get_sides(polygon):
    x, y = zip(*list(polygon.exterior.coords))
    d = []
    for i in range(len(x)-1):
        p1 = (x[i], y[i])
        p2 = (x[i+1], y[i+1])
        dist = distance(p1, p2)
        d.append(round(dist, 8))
    return d

from super_queens import SuperQueens
class Jigsaw:
    def __init__(self, n):
        self.n = n

        sols = nqueens.n_queens(self.n).all_solutions
        #sq = SuperQueens(n)
        #sols = list(sq.other_sols(sq.init_sols(), sols))[0]
        #print(sols[0])

        c = math.pow(phi, 2/math.pi)
        a = [(math.pi*2*i)/self.n for i in range(self.n)]
        r = [spiral1(c, 10*math.pi + math.pi*2/self.n, aa) for aa in a]

        a = np.array(a)
        r = np.array(r)

        self.pieces = []
        self.upieces = []
        areas = []
        for index, sol in enumerate(sols):
            tt = [a[v] for v in sol]
            x, y = polar_2_cartesian(tt, r)
            poly = create_polygon(np.array(sol), x, y)
            if round(poly.area, 8) not in areas:
                areas.append(round(poly.area, 8))
                self.upieces.append(poly)
            self.pieces.append(poly)

        print('areas', areas)

    def draw_line(self, line, pax):
        a, b = line
        ax, ay = a
        bx, by = b
        pax.plot([ax, bx], [ay, by], c='red', linewidth=4)

    def draw_poly(self, poly, line, pax, annotate=True):
        coords = list(poly.exterior.coords)
        x, y = zip(*coords)
        pax.plot(x, y)

        #pax.annotate(str(round(poly.area, 8)), (0, 0))

        if line is not None:
            self.draw_line(line, pax)

        if annotate:
            pax.annotate(str(round(poly.area, 4)), (0, 0))
            for i in range(len(x)-1):
                p1 = coords[i]
                p2 = coords[i+1]
                txt = str(round(distance(p1, p2), 4))
                xm, ym = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2
                pax.annotate(txt, (xm, ym))

    def draw_pair_polys(self, poly1, poly2, tpoly2, line1, line2):
        fig, axs = plt.subplots(1, 4)

        self.draw_poly(poly1, line1, axs[0])
        self.draw_poly(poly2, line2, axs[1])

        self.draw_poly(poly1, line1, axs[2], annotate=False)
        self.draw_poly(poly2, line2, axs[2], annotate=False)

        self.draw_poly(poly1, None, axs[3], annotate=False)
        self.draw_poly(tpoly2, None, axs[3], annotate=False)

        #plt.axis('off')
        plt.show()

    def transform_poly(self, poly, angle, rotate_point, off):
        if angle != 0:
            poly = affinity.rotate(poly, angle, rotate_point)

        x_off, y_off = off
        transformed_poly = affinity.translate(poly, xoff=x_off, yoff=y_off)

        coords = [(round(x, 10), round(y, 10)) for x,y in list(transformed_poly.exterior.coords)]
        transformed_poly = Polygon(coords[:-1])

        return transformed_poly

    def join_polys_aux(self, poly1, poly2, line1, line2, cut_point, angle):
        a, b = line1

        rotate_line = affinity.rotate(LineString(line2), angle, cut_point).coords

        if round(distance(a, rotate_line[0]), 8) == round(distance(b, rotate_line[1]), 8):
            transformed_poly = self.transform_poly(poly2, angle, cut_point, (a[0]-rotate_line[0][0], a[1]-rotate_line[0][1]))
        else:
            transformed_poly = self.transform_poly(poly2, angle, cut_point, (a[0]-rotate_line[1][0], a[1]-rotate_line[1][1]))

        #print('area', poly1.intersection(transformed_poly).area)
        #print('intersection', transformed_poly.intersection(LineString(line1)))
        #print('poly1', poly1)
        #print('transformed_poly', transformed_poly)
        #print('lines', LineString(line1), transformed_poly.intersection(LineString(line1)) == LineString(line1))
        #self.draw_pair_polys(poly1, poly2, transformed_poly, line1, line2)

        return transformed_poly

    def join_polys(self, poly1, poly2, line1, line2, co=None):
        transformed_poly = None
        cut_point = None
        angle = 0
        try:
            cut_point = line_intersection(line1, line2)
            angle = angle_between(line1, line2)
        except:
            return None

        transformed_poly = self.join_polys_aux(poly1, poly2, line1, line2, cut_point, angle)

        if co > 3:
            print('co', co)
            #self.draw_pair_polys(poly1, poly2, transformed_poly, line1, line2)

        line1_ls = LineString(line1)

        if poly1.intersection(transformed_poly).area != 0 or \
                transformed_poly.intersection(line1_ls) != line1_ls:
            transformed_poly = self.join_polys_aux(poly1, poly2, line1, line2, cut_point, -angle)

            if poly1.intersection(transformed_poly).area != 0 or \
                    transformed_poly.intersection(line1_ls) != line1_ls:
                transformed_poly = self.join_polys_aux(poly1, poly2, line1, line2, cut_point, 180-angle)

                if poly1.intersection(transformed_poly).area != 0 or \
                        transformed_poly.intersection(line1_ls) != line1_ls:
                    transformed_poly = self.join_polys_aux(poly1, poly2, line1, line2, cut_point, angle-180)

                    if poly1.intersection(transformed_poly).area != 0 or \
                            transformed_poly.intersection(line1_ls) != line1_ls:
                        return None

        union = poly1.union(transformed_poly)

        if type(union) is MultiPolygon:
            return None

        return union

    def find_line_similar_btw_polys(self, poly1, poly2, inv_poly2):
        sides1 = get_sides(poly1)
        sides2 = get_sides(poly2)

        options = [value for value in sides1 if value in sides2]

        if len(options) == 0:
            return None, None, None

        option = random.choice(options)
        pi1 = random.choice([i for i, value in enumerate(sides1) if option == value])
        pi2 = random.choice([i for i, value in enumerate(sides2) if option == value])

        poly1_coords = list(poly1.exterior.coords)
        poly2_coords = list(poly2.exterior.coords)
        inv_poly2_coords = list(inv_poly2.exterior.coords)

        line1 = [poly1_coords[pi1], poly1_coords[pi1+1]]
        line2 = [poly2_coords[pi2], poly2_coords[pi2+1]]
        inv_line2 = [inv_poly2_coords[pi2], inv_poly2_coords[pi2+1]]

        return line1, line2, inv_line2

    def solve_once(self, pieces):
        lp = list(range(len(pieces)))
        random.shuffle(lp)
        sol = []
        pi = lp.pop()
        sol.append(pi)
        poly = pieces[pi]

        co = 0
        while len(lp) > 0:
            pi = lp.pop()
            sol.append(pi)
            new_poly = pieces[pi]
            inv_new_poly = get_inverse_poly(new_poly)
            line1, line2, inv_line2 = self.find_line_similar_btw_polys(poly, new_poly, inv_new_poly)

            if line1 is None:
                return None

            union_poly = None

            try:
                union_poly = self.join_polys(poly, new_poly, line1, line2, co)
            except Exception as e:
                print(e)

            try:
                if union_poly is None or type(union_poly) is MultiPolygon:
                    union_poly = self.join_polys(poly, inv_new_poly, line1, inv_line2, co)
            except Exception as e:
                print(e)
                return None

            if union_poly is None or type(union_poly) is MultiPolygon:
                return None
            else:
                poly = union_poly

            co += 1

        return poly

n = 8
jigsaw = Jigsaw(n)

sols = []
print(len(jigsaw.upieces))

for _ in range(1000):
    r = jigsaw.solve_once(jigsaw.upieces)
    if r is not None:
        if r not in sols:
            fig, axs = plt.subplots(1)
            sols.append(tuple(list(r.exterior.coords)))
            jigsaw.draw_poly(r, None, axs)
            plt.show()


sols = set(sols)

for sol in sols:
    print(sol)




