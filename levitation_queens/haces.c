/* Censo de haces n+m en C (misma busqueda que haces.py, sin reducir por giros).
 * Para cada pareja ordenada de pendientes (a, b) (a < b si n == m) y cada colocacion normalizada
 * (una recta de cada pendiente por el origen, la segunda paralela a distancia 1) busca todas las
 * toroidales que en cada fila toman la reina de una de las n+m rectas y usan TODAS las rectas.
 * Salida: una linea "a b T(0) ... T(p-1)" por solucion base.
 * Orbitas de traslacion de la pareja = base * (p-1) / (n*(n-1)*m)   (n >= 2).
 * Uso: ./haces p n m [k K]   (solo las parejas con a % K == k, para repartir en procesos)
 *      PAR="a b" ./haces p n m   (solo esa pareja de pendientes)
 * cc -O3 -o haces haces.c
 */
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long u64;
static int p, n, m, nl, la[16], lc[16], T[64], cand[64][16], nc[64];
static int A, B;
static u64 found;

/* Misma busqueda exhaustiva, pero en cada paso se rellena la fila con menos candidatas vivas
 * (si alguna fila se queda sin ninguna se corta). No cambia el conjunto de soluciones. */
static void rec(int hechas, u64 filas, u64 cols, u64 sm, u64 df) {
    if (hechas == p) {
        int solo[16] = {0};
        for (int x = 0; x < p; x++) {
            int q = -1, k = 0;
            for (int i = 0; i < nl; i++) if ((la[i] * x + lc[i]) % p == T[x]) { q = i; k++; }
            if (k == 1) solo[q] = 1;
        }
        for (int i = 0; i < nl; i++) if (!solo[i]) return;
        found++;
        printf("%d %d", A, B);
        for (int x = 0; x < p; x++) printf(" %d", T[x]);
        printf("\n");
        return;
    }
    int br = -1, bn = 99;
    for (int r = 0; r < p && bn > 1; r++) {
        if (filas >> r & 1) continue;
        int k = 0;
        for (int i = 0; i < nc[r]; i++) {
            int c = cand[r][i];
            if (!((cols >> c & 1) || (sm >> ((c + r) % p) & 1) || (df >> ((c - r + p) % p) & 1))) k++;
        }
        if (k == 0) return;
        if (k < bn) { bn = k; br = r; }
    }
    int r = br;
    for (int i = 0; i < nc[r]; i++) {
        int c = cand[r][i];
        u64 b1 = 1ULL << c, b2 = 1ULL << ((c + r) % p), b3 = 1ULL << ((c - r + p) % p);
        if ((cols & b1) || (sm & b2) || (df & b3)) continue;
        T[r] = c;
        rec(hechas + 1, filas | 1ULL << r, cols | b1, sm | b2, df | b3);
    }
}

static void lanza(void) {
    for (int r = 0; r < p; r++) {
        nc[r] = 0;
        for (int i = 0; i < nl; i++) {
            int c = (la[i] * r + lc[i]) % p, rep = 0;
            for (int j = 0; j < nc[r]; j++) if (cand[r][j] == c) rep = 1;
            if (!rep) cand[r][nc[r]++] = c;
        }
    }
    rec(0, 0, 0, 0, 0);
}

/* combinaciones crecientes para los cortes libres: primero las de A (desde 2), luego las de B (desde 1) */
static void elige(int idx, int desde) {
    if (idx == nl) { lanza(); return; }
    int esA = idx < n;
    if (idx == 0) { lc[0] = 0; elige(1, 0); return; }
    if (esA && idx == 1) { lc[1] = 1; elige(2, 2); return; }
    if (!esA && idx == n) { lc[n] = 0; elige(n + 1, 1); return; }
    for (int c = desde; c < p; c++) { lc[idx] = c; elige(idx + 1, (idx + 1 == n) ? 0 : c + 1); }
}

int main(int argc, char **argv) {
    p = atoi(argv[1]); n = atoi(argv[2]); m = atoi(argv[3]);
    int k = argc > 5 ? atoi(argv[4]) : 0, K = argc > 5 ? atoi(argv[5]) : 1;
    nl = n + m;
    int pa = 0, pb = 0;
    if (getenv("PAR")) sscanf(getenv("PAR"), "%d %d", &pa, &pb);
    for (A = 2; A <= p - 2; A++) {
        if (A % K != k) continue;
        for (B = 2; B <= p - 2; B++) {
            if (A == B || (n == m && A > B)) continue;
            if (pa && (A != pa || B != pb)) continue;
            for (int i = 0; i < nl; i++) la[i] = i < n ? A : B;
            elige(0, 0);
        }
    }
    fprintf(stderr, "p=%d %d+%d parte %d/%d: %llu base\n", p, n, m, k, K, found);
    return 0;
}
