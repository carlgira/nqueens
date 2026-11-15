"""Genera index.html (explorador 3D) inyectando charpoly.json y chambers.json en template.html."""
import json
import numpy as np

chi = {r['N']: r for r in json.load(open('charpoly.json'))}
ch = json.load(open('chambers.json'))
chambers = {int(N): {''.join(map(str, r['sol'])): [r['facets'], round(r['inradius'], 4)] for r in rows}
            for N, rows in ch.items()}
walls = {N: 3 * N * (N - 1) // 2 for N in range(2, 12)}

def mean_facets(N): return np.mean([r['facets'] for r in ch[str(N)]])
Ns = sorted(int(n) for n in ch)
allsol = sum(len(ch[str(n)]) for n in Ns)
per = [f"N={n}: {chi[n]['regions'] // chi[n]['Q']:,}".replace(',', '.') for n in sorted(chi) if chi[n]['Q']]
facet_line = ', '.join(f"{mean_facets(n):.1f}" for n in Ns)
n7 = ch['7']
tor = np.mean([r['facets'] for r in n7 if r['toroidal']]); non = np.mean([r['facets'] for r in n7 if not r['toroidal']])

findings = [
  dict(big='100 %', t='Todas las cámaras de soluciones están acotadas',
       p=f"Lo cumplen las {allsol} soluciones de N={Ns[0]}…{Ns[-1]}, cuando entre todas las cámaras solo lo está entre el 43 y el 46 % (N=4…8). "
         "Criterio exacto: el grafo de pares «cercanos» (|x<sub>i</sub>−x<sub>j</sub>| &lt; |i−j|) es conexo. Las filas 0 y N−1 siempre son cercanas."),
  dict(big='1/√2', t='Pegadas a la pared',
       p="Cada solución está a 1/√2 de su pared más cercana, lo mínimo posible para un punto entero. Casi siempre ese es también el radio "
         "de la mayor bola que cabe dentro: las cámaras son finas. Excepción: en N=6, dos de las 4 soluciones tienen inradio 0,884 y sus traspuestas no. "
         "La disposición distingue filas de columnas, así que rompe la simetría de rotación del tablero."),
  dict(big=f"{mean_facets(Ns[0]):.0f} → {mean_facets(Ns[-1]):.0f}", t='Pocas paredes tocan cada cámara',
       p=f"Media de paredes que tocan la cámara, N={Ns[0]}…{Ns[-1]}: {facet_line}. Crece como ~2,4·N, "
         f"mientras que el total crece como 1,5·N². En N=7 las soluciones toroidales tocan más paredes ({tor:.1f}) que las no toroidales ({non:.1f})."),
  dict(big='Q(N) ≪ cámaras', t='La topología cuenta cámaras, no soluciones',
       p="Cámaras por cada solución: " + ', '.join(per) + ". Contar cámaras es un problema limpio (un polinomio). "
         "Q(N) pregunta cuántas contienen un punto entero dentro del tablero, y esa capa aritmética es la difícil."),
]
data = dict(chi={str(k): v for k, v in chi.items()}, chambers=chambers, walls=walls, findings=findings)
html = open('template.html').read().replace('__DATA__', json.dumps(data, ensure_ascii=False))
open('index.html', 'w').write(html)
print('ok', len(html), 'bytes;', 'chi N =', sorted(chi))
