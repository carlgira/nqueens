"""
Consulta la API JSON de OEIS (https://oeis.org/search?q=...&fmt=json) para ver si
las secuencias que derivamos son conocidas.

Uso:  python oeis_query.py            # corre las consultas de abajo
      python oeis_query.py 10,28,88   # consulta una secuencia concreta

Aviso: secuencias de 3-4 terminos devuelven MUCHOS falsos positivos; se necesitan
>=5-6 terminos distintivos para un resultado fiable.
"""
import sys
import json
import subprocess
import urllib.parse


def query(seq, label=""):
    q = urllib.parse.quote(seq)
    url = f"https://oeis.org/search?q={q}&fmt=json"
    out = subprocess.run(["curl", "-s", "--max-time", "20", url],
                         capture_output=True, text=True).stdout
    print(f"\n=== {label or seq} ===")
    print(f"    (query: {seq})")
    try:
        data = json.loads(out)
    except Exception:
        print("  (sin respuesta JSON)"); return
    if data is None:
        results = []
    elif isinstance(data, list):
        results = data
    else:
        results = data.get("results") or []
    if not results:
        print("  NADA en OEIS para esta secuencia.")
        return
    for r in results[:3]:
        print(f"  A{r['number']:06d}: {r['name']}")
        print(f"     datos: {r['data'][:80]}...")


DEFAULT_QUERIES = [
    ("2, 10, 4, 40, 92, 352, 724", "total de soluciones N-reinas (confirma A000170)"),
    ("10, 28, 88, 130, 238, 304", "nuestras espirales (N-3)*N en primos 5,7,11,13,17,19"),
    ("0, 4, 10, 18, 28, 40, 54, 70, 88, 108", "n(n-3) general -> A028552"),
    ("1, 0, 0, 0, 10, 0, 28, 0, 0, 0, 88, 0, 4524", "queens TOROIDALES -> A051906 (=espirales hasta N=11)"),
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        for seq, label in DEFAULT_QUERIES:
            query(seq, label)
