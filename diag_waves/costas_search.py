"""Busqueda de arrays de Costas y anatomia de su arbol."""

import numpy as np


def search(n, count_nodes=True, prune=None):
	"""DFS columna a columna. prune(cur, usados, n) -> True si podar."""
	nodes = [0]; found = [0]
	cur = []; usados = set(); libres = list(range(n))

	def rec():
		nodes[0] += 1
		r = len(cur)
		if r == n:
			found[0] += 1; return
		if prune is not None and prune(cur, usados, n):
			return
		for k in range(n):
			c = libres[k] if k < len(libres) else None
			if c is None: break
		for c in [x for x in range(n) if x not in cur]:
			nuevos = set()
			ok = True
			for i in range(r):
				v = (r - i, c - cur[i])
				if v in usados or v in nuevos: ok = False; break
				nuevos.add(v)
			if not ok: continue
			cur.append(c); usados.update(nuevos)
			rec()
			cur.pop(); usados.difference_update(nuevos)
	rec()
	return nodes[0], found[0]


def harvest(n, dmin=None, dmax=None):
	"""Estados parciales etiquetados por si su subarbol tiene algun Costas."""
	dmin = dmin or n // 3; dmax = dmax or n - 1
	rows = []; cur = []; usados = set()

	def rec():
		r = len(cur)
		if r == n: return 1
		tot = 0
		for c in [x for x in range(n) if x not in cur]:
			nuevos = set(); ok = True
			for i in range(r):
				v = (r - i, c - cur[i])
				if v in usados or v in nuevos: ok = False; break
				nuevos.add(v)
			if not ok: continue
			cur.append(c); usados.update(nuevos)
			tot += rec()
			cur.pop(); usados.difference_update(nuevos)
		if dmin <= r <= dmax:
			rows.append((tuple(cur), tot > 0))
		return tot
	rec()
	return rows


def available(cur, usados, n):
	"""Para cada columna pendiente, que filas siguen siendo posibles."""
	r = len(cur); libres = [v for v in range(n) if v not in cur]
	out = []
	for j in range(r, n):
		opts = []
		for v in libres:
			ok = True; nuevos = set()
			for i in range(r):
				d = (j - i, v - cur[i])
				if d in usados or d in nuevos: ok = False; break
				nuevos.add(d)
			if ok: opts.append(v)
		out.append(opts)
	return out


def kuhn(adj, vals):
	idx = {v: k for k, v in enumerate(vals)}
	match = [-1] * len(vals)
	def try_row(u, seen):
		for v in adj[u]:
			k = idx[v]
			if seen[k]: continue
			seen[k] = True
			if match[k] == -1 or try_row(match[k], seen):
				match[k] = u; return True
		return False
	for u in range(len(adj)):
		if not try_row(u, [False] * len(vals)): return False
	return True


def dead_count(cur, usados, n):
	"""condicion barata: alguna columna pendiente sin opciones, o Hall falla"""
	av = available(cur, usados, n)
	if any(not o for o in av): return True
	vals = sorted({v for o in av for v in o})
	if len(vals) < len(av): return True
	return not kuhn(av, vals)
