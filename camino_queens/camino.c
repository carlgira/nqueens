// camino.c — ordenar los N^N tableros como un árbol y ver dónde caen las soluciones.
//
// Un tablero es x = (x_0..x_{N-1}), x_i = fila de la reina en la columna i.
// Un "camino" es un orden de los N^N tableros dado por N funcionales lineales
// mod N (f_1 el más significativo): cada nivel del árbol se decide por f_k(x).
// Las soluciones quedan juntas si comparten un prefijo largo (mismo subárbol).
//
// Uso: ./camino N        (N primo, 5..13)
//
// Conjuntos: ALL (todas), TOR (toroidales), LIN (x_i = a·i + b), TORNL
// (toroidales no lineales), NOTOR (no toroidales).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXN 16
typedef unsigned char vec[MAXN];

static int N;
static int inv[MAXN];

typedef struct { const char *name; int n; vec *v; } Set;

static int md(int a) { a %= N; return a < 0 ? a + N : a; }

// ---------- soluciones ----------
static vec *sols; static int nsol, capsol;
static vec cur;
static void bt(int c, unsigned rows, unsigned d1, unsigned d2) {
    if (c == N) {
        if (nsol == capsol) { capsol = capsol ? 2 * capsol : 1024; sols = realloc(sols, capsol * sizeof(vec)); }
        memcpy(sols[nsol++], cur, MAXN); return;
    }
    for (int r = 0; r < N; r++) {
        unsigned a = 1u << r, b = 1u << (r + c), e = 1u << (r - c + N);
        if ((rows & a) || (d1 & b) || (d2 & e)) continue;
        cur[c] = r; bt(c + 1, rows | a, d1 | b, d2 | e);
    }
}
static int is_tor(const unsigned char *x) {
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        int d = md(x[j] - x[i]), k = j - i;
        if (d == md(k) || d == md(-k)) return 0;
    }
    return 1;
}
static int is_lin(const unsigned char *x) {
    for (int i = 0; i + 2 < N; i++) if (md(x[i] - 2 * x[i + 1] + x[i + 2])) return 0;
    return 1;
}

// ---------- álgebra mod p ----------
// Reduce v contra una base en forma escalonada (piv[r] = columna pivote, filas normalizadas).
static void reduce(int *v, int (*B)[MAXN], int *piv, int r) {
    for (int k = 0; k < r; k++) { int c = v[piv[k]]; if (c) for (int j = 0; j < N; j++) v[j] = md(v[j] - c * B[k][j]); }
}
static int add_basis(int *v, int (*B)[MAXN], int *piv, int r) {
    reduce(v, B, piv, r);
    int p = -1; for (int j = 0; j < N; j++) if (v[j]) { p = j; break; }
    if (p < 0) return r;
    int s = inv[v[p]]; for (int j = 0; j < N; j++) v[j] = md(v[j] * s);
    for (int k = 0; k < r; k++) { int c = B[k][p]; if (c) for (int j = 0; j < N; j++) B[k][j] = md(B[k][j] - c * v[j]); }
    memcpy(B[r], v, sizeof(int) * MAXN); piv[r] = p; return r + 1;
}
static int affine_dim(Set *S) {
    int B[MAXN][MAXN], piv[MAXN], r = 0;
    for (int i = 1; i < S->n && r < N; i++) {
        int v[MAXN] = {0}; for (int j = 0; j < N; j++) v[j] = md(S->v[i][j] - S->v[0][j]);
        r = add_basis(v, B, piv, r);
    }
    return r;
}


// Funcionales constantes sobre S (anulador de las diferencias): base del espacio
// de w con w·x igual para todo x ∈ S. Son los niveles que S comparte en el árbol.
static void annihilator(Set *S) {
    int B[MAXN][MAXN], piv[MAXN], r = 0;
    for (int i = 1; i < S->n && r < N; i++) {
        int v[MAXN] = {0}; for (int j = 0; j < N; j++) v[j] = md(S->v[i][j] - S->v[0][j]);
        r = add_basis(v, B, piv, r);
    }
    int isp[MAXN] = {0}; for (int k = 0; k < r; k++) isp[piv[k]] = 1;
    printf("    funcionales constantes (%d):", N - r);
    for (int f = 0; f < N; f++) if (!isp[f]) {
        int w[MAXN] = {0}; w[f] = 1;
        for (int k = 0; k < r; k++) w[piv[k]] = md(-B[k][f]);
        int c = 0; for (int j = 0; j < N; j++) c += w[j] * S->v[0][j];
        printf("  ("); for (int j = 0; j < N; j++) printf("%d%s", w[j], j + 1 < N ? "," : "");
        printf(")=%d", c % N);
    }
    printf("\n");
}

// ---------- órdenes ----------
// LEX: f_k = x_{k-1}.  D2: f_k = Δ²x_{k-1} (k=1..N-2), f_{N-1} = x_1 - x_0, f_N = x_0.
// MOM: f_1..f_3 = Σ i^k x_i (k = 0,1,2), luego filas Δ² independientes hasta N-2
// niveles, luego x_1 - x_0 y x_0. Anidado: toroidales a profundidad 3, lineales a N-2.
static int Mmom[MAXN][MAXN];
static void build_mom(void) {
    int B[MAXN][MAXN], piv[MAXN], r = 0, rows = 0;
    for (int k = 0; k < 3; k++) {
        int v[MAXN]; for (int i = 0; i < N; i++) { int t = 1; for (int e = 0; e < k; e++) t = t * i % N; v[i] = t; }
        memcpy(Mmom[rows++], v, sizeof v); r = add_basis(v, B, piv, r);
    }
    for (int k = 0; k + 2 < N && rows < N - 2; k++) {
        int v[MAXN] = {0}; v[k] = 1; v[k + 1] = N - 2; v[k + 2] = 1;
        int w[MAXN]; memcpy(w, v, sizeof w);
        int r2 = add_basis(w, B, piv, r); if (r2 > r) { memcpy(Mmom[rows++], v, sizeof v); r = r2; }
    }
    int v[MAXN] = {0}; v[0] = N - 1; v[1] = 1; memcpy(Mmom[rows++], v, sizeof v);
    int u[MAXN] = {0}; u[0] = 1; memcpy(Mmom[rows++], u, sizeof u);
    int Bc[MAXN][MAXN], pc[MAXN], rc = 0;
    for (int k = 0; k < N; k++) { int w[MAXN]; memcpy(w, Mmom[k], sizeof w); rc = add_basis(w, Bc, pc, rc); }
    if (rc != N) { fprintf(stderr, "MOM no invertible (rango %d)\n", rc); exit(1); }
}
static void key_digits(int ord, const unsigned char *x, int *f) {
    if (ord == 2) { for (int k = 0; k < N; k++) { int s = 0; for (int j = 0; j < N; j++) s += Mmom[k][j] * x[j]; f[k] = s % N; } return; }
    if (ord == 0) { for (int k = 0; k < N; k++) f[k] = x[k]; return; }
    for (int k = 0; k + 2 < N; k++) f[k] = md(x[k] - 2 * x[k + 1] + x[k + 2]);
    f[N - 2] = md(x[1] - x[0]); f[N - 1] = x[0];
}
static int cmpu64(const void *a, const void *b) { uint64_t x = *(uint64_t *)a, y = *(uint64_t *)b; return x < y ? -1 : x > y; }

static void report_order(Set *S, int ord) {
    const char *on = ord == 2 ? "MOM" : ord ? "D2 " : "LEX";
    uint64_t *idx = malloc(S->n * sizeof(uint64_t));
    int (*F)[MAXN] = malloc(S->n * sizeof(*F));
    for (int i = 0; i < S->n; i++) {
        key_digits(ord, S->v[i], F[i]);
        uint64_t k = 0; for (int j = 0; j < N; j++) k = k * N + F[i][j]; idx[i] = k;
    }
    // perfil: nº de prefijos distintos de longitud k (1 = todas en el mismo subárbol)
    printf("    %s  perfil prefijos:", on);
    for (int k = 1; k <= N; k++) {
        uint64_t *pre = malloc(S->n * sizeof(uint64_t)); uint64_t div = 1;
        for (int j = k; j < N; j++) div *= N;
        for (int i = 0; i < S->n; i++) pre[i] = idx[i] / div;
        qsort(pre, S->n, sizeof(uint64_t), cmpu64);
        int d = S->n ? 1 : 0; for (int i = 1; i < S->n; i++) d += pre[i] != pre[i - 1];
        printf(" %d", d); free(pre);
    }
    qsort(idx, S->n, sizeof(uint64_t), cmpu64);
    uint64_t span = idx[S->n - 1] - idx[0] + 1; int runs = 1;
    for (int i = 1; i < S->n; i++) runs += idx[i] != idx[i - 1] + 1;
    printf("\n         inicio %llu  extensión %llu  densidad %.4g  tramos %d\n",
           (unsigned long long)idx[0], (unsigned long long)span, (double)S->n / span, runs);
    if (S->n <= 40) {
        printf("         posiciones:"); for (int i = 0; i < S->n; i++) printf(" %llu", (unsigned long long)idx[i]);
        printf("\n         saltos:    "); for (int i = 1; i < S->n; i++) printf(" %llu", (unsigned long long)(idx[i] - idx[i - 1]));
        printf("\n");
    }
    free(idx); free(F);
}

// ---------- todos los funcionales del primer nivel (N <= 7) ----------
// Para cada w ≠ 0: cuántos valores distintos toma w·x sobre S (= cuántos subárboles
// del primer nivel ocupa S) y el tamaño del mayor grupo.
static void all_functionals(Set *S) {
    long hist[MAXN + 1] = {0}; long tot = 1; for (int j = 0; j < N; j++) tot *= N;
    int bestgrp = 0; long nbest = 0;
    int w[MAXN] = {0};
    for (long t = 1; t < tot; t++) {
        long u = t; for (int j = 0; j < N; j++) { w[j] = u % N; u /= N; }
        int cnt[MAXN] = {0};
        for (int i = 0; i < S->n; i++) { int s = 0; for (int j = 0; j < N; j++) s += w[j] * S->v[i][j]; cnt[s % N]++; }
        int d = 0, g = 0; for (int c = 0; c < N; c++) { d += cnt[c] > 0; if (cnt[c] > g) g = cnt[c]; }
        hist[d]++;
        if (d > 1) { if (g > bestgrp) { bestgrp = g; nbest = 0; } if (g == bestgrp) nbest++; }
    }
    printf("    funcionales del 1er nivel (%ld): nº de subárboles ocupados →", tot - 1);
    for (int d = 1; d <= N; d++) if (hist[d]) printf("  %d:%ld", d, hist[d]);
    printf("\n         si no caben en uno, mayor grupo en un subárbol: %d de %d (%ld funcionales)\n", bestgrp, S->n, nbest);
}

// ---------- rectas: mayor nº de soluciones alineadas ----------
static void lines(Set *S) {
    if (S->n < 2 || S->n > 20000) return;
    int cap = 1; while (cap < 2 * S->n) cap <<= 1;
    unsigned char (*K)[MAXN] = malloc(cap * MAXN); int *cnt = malloc(cap * sizeof(int)); char *low = malloc(cap); char *used = malloc(cap);
    long hist[64] = {0}; int best = 1;
    for (int p = 0; p < S->n; p++) {
        memset(used, 0, cap);
        for (int q = 0; q < S->n; q++) if (q != p) {
            unsigned char d[MAXN] = {0}; int f = -1;
            for (int j = 0; j < N; j++) { d[j] = md(S->v[q][j] - S->v[p][j]); if (f < 0 && d[j]) f = j; }
            int s = inv[d[f]]; uint64_t h = 1469598103934665603ULL;
            for (int j = 0; j < N; j++) { d[j] = md(d[j] * s); h = (h ^ d[j]) * 1099511628211ULL; }
            int k = h & (cap - 1);
            while (used[k] && memcmp(K[k], d, MAXN)) k = (k + 1) & (cap - 1);
            if (!used[k]) { used[k] = 1; memcpy(K[k], d, MAXN); cnt[k] = 0; low[k] = 0; }
            cnt[k]++; if (q < p) low[k] = 1;
        }
        for (int k = 0; k < cap; k++) if (used[k] && !low[k]) {
            int m = cnt[k] + 1; if (m > best) best = m; if (m >= 3) hist[m < 63 ? m : 63]++;
        }
    }
    printf("    rectas: máximo %d soluciones alineadas;", best);
    for (int m = 3; m < 64; m++) if (hist[m]) printf("  %d en recta:%ld", m, hist[m]);
    printf("\n");
    free(K); free(cnt); free(low); free(used);
}

// ---------- caminos dentro del subárbol de las lineales ----------
// LIN = {a·id + b·1 : a ∈ 2..N-2, b ∈ Z_N}. Con los N-2 niveles de arriba
// constantes sobre LIN, la posición dentro del subárbol es c1·N + c0 con
// (c1, c0) = (α1 a + β1 b + t1, α0 a + β0 b + t0). Se cuentan todas las
// elecciones (invertibles) y qué patrón dejan.
static void lin_paths(void) {
    int nA = N - 3, n = nA * N;
    long tot = 0, contig = 0, ap = 0, contig0 = 0; long startcnt[MAXN * MAXN] = {0};
    int *pos = malloc(n * sizeof(int));
    for (int a1 = 0; a1 < N; a1++) for (int b1 = 0; b1 < N; b1++) for (int a0 = 0; a0 < N; a0++) for (int b0 = 0; b0 < N; b0++) {
        if (md(a1 * b0 - b1 * a0) == 0) continue;
        for (int t1 = 0; t1 < N; t1++) for (int t0 = 0; t0 < N; t0++) {
            unsigned char occ[MAXN * MAXN] = {0};
            for (int a = 2; a <= N - 2; a++) for (int b = 0; b < N; b++)
                occ[md(a1 * a + b1 * b + t1) * N + md(a0 * a + b0 * b + t0)] = 1;
            int m = 0; for (int c = 0; c < N * N; c++) if (occ[c]) pos[m++] = c;
            tot++;
            int g = pos[1] - pos[0], isap = 1; for (int i = 2; i < n; i++) if (pos[i] - pos[i - 1] != g) { isap = 0; break; }
            if (isap) ap++;
            if (pos[n - 1] - pos[0] + 1 == n) { contig++; startcnt[pos[0]]++; if (t1 == 0 && t0 == 0) contig0++; }
        }
    }
    printf("  Caminos dentro del subárbol (%d lineales, %ld elecciones de los 2 últimos niveles):\n", n, tot);
    printf("    en progresión aritmética: %ld   en un bloque contiguo: %ld   (sin traslación: %ld)\n", ap, contig, contig0);
    printf("    inicio del bloque dentro del subárbol:");
    for (int s = 0; s < N * N; s++) if (startcnt[s]) printf("  %d:%ld", s, startcnt[s]);
    printf("\n");
    free(pos);
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 5;
    if (N < 5 || N > 13) { fprintf(stderr, "N entre 5 y 13\n"); return 1; }
    for (int p = 2; p < N; p++) if (N % p == 0) { fprintf(stderr, "N debe ser primo\n"); return 1; }
    for (int a = 1; a < N; a++) for (int b = 1; b < N; b++) if (a * b % N == 1) inv[a] = b;
    bt(0, 0, 0, 0);
    build_mom();

    Set sets[5] = {{"ALL", 0, 0}, {"TOR", 0, 0}, {"LIN", 0, 0}, {"TORNL", 0, 0}, {"NOTOR", 0, 0}};
    for (int s = 0; s < 5; s++) sets[s].v = malloc(nsol * sizeof(vec));
    for (int i = 0; i < nsol; i++) {
        int t = is_tor(sols[i]), l = is_lin(sols[i]);
        memcpy(sets[0].v[sets[0].n++], sols[i], MAXN);
        if (t) memcpy(sets[1].v[sets[1].n++], sols[i], MAXN);
        if (l) memcpy(sets[2].v[sets[2].n++], sols[i], MAXN);
        if (t && !l) memcpy(sets[3].v[sets[3].n++], sols[i], MAXN);
        if (!t) memcpy(sets[4].v[sets[4].n++], sols[i], MAXN);
    }
    printf("N = %d   tableros %.0f   soluciones %d\n\n", N, (double)0 + __builtin_powi(N, N), nsol);
    for (int s = 0; s < 5; s++) {
        Set *S = &sets[s];
        if (S->n == 0) { printf("[%s] vacío\n\n", S->name); continue; }
        if (s == 1 && sets[3].n == 0) { printf("[TOR] = LIN (%d)\n\n", S->n); continue; }
        if (s == 0 && sets[4].n == 0) { printf("[ALL] = TOR (%d)\n\n", S->n); continue; }
        printf("[%s] %d soluciones   dimensión afín %d (de %d)\n", S->name, S->n, affine_dim(S), N);
        annihilator(S);
        report_order(S, 0);
        report_order(S, 1);
        report_order(S, 2);
        if (N <= 7) all_functionals(S);
        lines(S);
        printf("\n");
    }
    lin_paths();
    return 0;
}
