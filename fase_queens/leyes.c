/*
 * leyes.c — Saca la "ley" de cada cadena guardada por parabolas.c (modo aceleración común).
 *
 * Para cada cadena: polígono factible (v,a) por reina (igual que parabolas.c), intersección
 * de las proyecciones en a entre reinas -> intervalo(s) de aceleración común.
 * Se elige la fracción MÁS SIMPLE dentro (árbol de Stern-Brocot): si la ley es natural,
 * debería salir una fracción pequeña. Con esa a fija, para cada reina se elige la velocidad
 * más simple v_r y luego la fase más simple beta_r.
 *
 * Salida (una línea por cadena):
 *   K | a=p/q | v_0..v_{N-1} | beta_0..beta_{N-1} | índices de la cadena
 *
 * uso: ./leyes N tableros.txt cadenas.txt > leyes.txt
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXV 48
#define MAXB 8
#define MAXK 64
typedef struct { int n; double v[MAXV], a[MAXV]; long x[MAXK]; } Br;
typedef struct { int nb; Br b[MAXB]; } Set;
static int N, Q; static int *sol;
static const double EPS = 1e-9;

static int cut(Br *B, double p, double q, double c) {
    double nv[MAXV], na[MAXV]; int n = 0;
    for (int i = 0; i < B->n; i++) {
        int j = (i + 1) % B->n;
        double fi = p * B->v[i] + q * B->a[i] - c, fj = p * B->v[j] + q * B->a[j] - c;
        if (fi <= 0 && n < MAXV) { nv[n] = B->v[i]; na[n] = B->a[i]; n++; }
        if ((fi <= 0) != (fj <= 0) && n < MAXV) { double t = fi / (fi - fj); nv[n] = B->v[i] + t * (B->v[j] - B->v[i]); na[n] = B->a[i] + t * (B->a[j] - B->a[i]); n++; }
    }
    B->n = n; memcpy(B->v, nv, sizeof(double) * n); memcpy(B->a, na, sizeof(double) * n);
    if (n < 3) return 0;
    double s = 0; for (int i = 0; i < n; i++) { int j = (i + 1) % n; s += B->v[i] * B->a[j] - B->v[j] * B->a[i]; }
    return fabs(s) > 1e-12;
}
static int addpt(const Set *S, Set *T, int k, int c) {
    T->nb = 0;
    for (int b = 0; b < S->nb; b++) {
        const Br *B = &S->b[b]; double fmin = 1e18, fmax = -1e18;
        for (int i = 0; i < B->n; i++) { double f = B->v[i] * k + B->a[i] * k * k / 2.0; if (f < fmin) fmin = f; if (f > fmax) fmax = f; }
        double lo = B->x[0] + fmin - 1, hi = B->x[0] + fmax + 1;
        long m0 = (long)floor((lo - c) / N), m1 = (long)ceil((hi - c) / N);
        for (long m = m0; m <= m1; m++) {
            long xk = c + m * N; if (xk <= lo - 1e-12 || xk >= hi + 1e-12) continue;
            if (T->nb >= MAXB) return T->nb > 0;
            Br R = *B; R.x[k] = xk; int ok = 1;
            for (int i = 0; i < k && ok; i++) { double p = k - i, q = (double)(k * k - i * i) / 2.0, d = (double)(xk - R.x[i]); ok = cut(&R, p, q, d + 1 - EPS) && cut(&R, -p, -q, -(d - 1 + EPS)); }
            if (ok) T->b[T->nb++] = R;
        }
    }
    return T->nb > 0;
}
/* fracción más simple en (lo,hi) abierto */
static void simplest(double lo, double hi, long *pp, long *qq) {
    if (lo < 0 && hi > 0) { *pp = 0; *qq = 1; return; }
    if (hi <= 0) { simplest(-hi, -lo, pp, qq); *pp = -*pp; return; }
    long fl = (long)floor(lo);
    if (fl + 1 < hi) { long n = (long)ceil(lo + 1e-12); if (n <= lo) n++; if (n < hi) { *pp = n; *qq = 1; return; } }
    /* Stern-Brocot */
    long a = (long)floor(lo), b = 1; /* izquierda a/1 */
    long pl = a, ql = 1, pr = a + 1, qr = 1;
    for (int it = 0; it < 200000; it++) {
        long pm = pl + pr, qm = ql + qr; double m = (double)pm / qm;
        if (m <= lo) { pl = pm; ql = qm; }
        else if (m >= hi) { pr = pm; qr = qm; }
        else { *pp = pm; *qq = qm; return; }
    }
    (void)b; *pp = (long)llround((lo + hi) / 2 * 1000); *qq = 1000;
}
/* rango de v en el polígono para a fija (corte vertical) */
static int slice(const Br *B, double a, double *vlo, double *vhi) {
    double lo = 1e18, hi = -1e18; int hit = 0;
    for (int i = 0; i < B->n; i++) {
        int j = (i + 1) % B->n; double ai = B->a[i], aj = B->a[j];
        if ((ai - a) * (aj - a) <= 0 && fabs(ai - aj) > 1e-15) { double t = (a - ai) / (aj - ai); double v = B->v[i] + t * (B->v[j] - B->v[i]); if (v < lo) lo = v; if (v > hi) hi = v; hit = 1; }
        else if (fabs(ai - a) < 1e-15) { if (B->v[i] < lo) lo = B->v[i]; if (B->v[i] > hi) hi = B->v[i]; hit = 1; }
    }
    *vlo = lo; *vhi = hi; return hit && hi > lo;
}

static Set st[MAXK][32];
int main(int argc, char **argv) {
    N = atoi(argv[1]);
    FILE *ft = fopen(argv[2], "r"); int cap = 1024; sol = malloc(sizeof(int) * N * cap); Q = 0;
    while (1) { if (Q == cap) { cap *= 2; sol = realloc(sol, sizeof(int) * N * cap); } int ok = 1; for (int r = 0; r < N; r++) if (fscanf(ft, "%d", &sol[(long)Q * N + r]) != 1) { ok = 0; break; } if (!ok) break; Q++; }
    fclose(ft);
    FILE *fc = fopen(argv[3], "r"); char line[4096];
    while (fgets(line, sizeof line, fc)) {
        if (line[0] == '#') continue;
        int ch[MAXK], K = 0; char *p = line; int v;
        while (sscanf(p, "%d", &v) == 1) { ch[K++] = v; while (*p == ' ') p++; while (*p && *p != ' ' && *p != '\n') p++; }
        if (K < 2) continue;
        for (int r = 0; r < N; r++) { Set *S = &st[0][r]; S->nb = 1; Br *B = &S->b[0]; B->n = 4; B->v[0] = 0; B->a[0] = -N; B->v[1] = N; B->a[1] = -N; B->v[2] = N; B->a[2] = N; B->v[3] = 0; B->a[3] = N; B->x[0] = sol[(long)ch[0] * N + r]; }
        int ok = 1;
        for (int k = 1; k < K && ok; k++) for (int r = 0; r < N && ok; r++) ok = addpt(&st[k - 1][r], &st[k][r], k, sol[(long)ch[k] * N + r]);
        if (!ok) { printf("# cadena no factible?\n"); continue; }
        Set *F = st[K - 1];
        /* intersección de proyecciones en a: barrido sobre candidatos (extremos) */
        double bestlo = 0, besthi = 0; long bp = 0, bq = 0; int found = 0;
        double ev[MAXB * 32 * 2]; int ne = 0;
        for (int r = 0; r < N; r++) for (int b = 0; b < F[r].nb; b++) { double mn = 1e18, mx = -1e18; for (int i = 0; i < F[r].b[b].n; i++) { double a = F[r].b[b].a[i]; if (a < mn) mn = a; if (a > mx) mx = a; } ev[ne++] = mn; ev[ne++] = mx; }
        /* ordenar extremos y probar el centro de cada hueco */
        for (int i = 0; i < ne; i++) for (int j = i + 1; j < ne; j++) if (ev[j] < ev[i]) { double t = ev[i]; ev[i] = ev[j]; ev[j] = t; }
        for (int s = 0; s + 1 < ne; s++) {
            double lo = ev[s], hi = ev[s + 1]; if (hi - lo < 1e-12) continue; double mid = (lo + hi) / 2; int good = 1;
            for (int r = 0; r < N && good; r++) { int any = 0; for (int b = 0; b < F[r].nb; b++) { double mn = 1e18, mx = -1e18; for (int i = 0; i < F[r].b[b].n; i++) { double a = F[r].b[b].a[i]; if (a < mn) mn = a; if (a > mx) mx = a; } if (mid > mn && mid < mx) any = 1; } if (!any) good = 0; }
            if (!good) continue;
            long pp, qq; simplest(lo, hi, &pp, &qq);
            if (!found || qq < bq || (qq == bq && labs(pp) < labs(bp))) { found = 1; bp = pp; bq = qq; bestlo = lo; besthi = hi; }
        }
        if (!found) { printf("# sin aceleracion comun\n"); continue; }
        double a = (double)bp / bq;
        printf("%d | a=%ld/%ld [%.4f,%.4f] | v=", K, bp, bq, bestlo, besthi);
        long vp[32], vq[32]; double vv[32], VL[32], VH[32];
        for (int r = 0; r < N; r++) {
            double lo = 1e18, hi = -1e18; for (int b = 0; b < F[r].nb; b++) { double l, h; if (slice(&F[r].b[b], a, &l, &h)) { if (l < lo) lo = l; if (h > hi) hi = h; } }
            VL[r]=lo; VH[r]=hi; simplest(lo, hi, &vp[r], &vq[r]); vv[r] = (double)vp[r] / vq[r];
            printf("%ld/%ld%s", vp[r], vq[r], r < N - 1 ? "," : "");
        }
        printf(" | beta=");
        for (int r = 0; r < N; r++) {
            /* beta en [x_k - v k - a k^2/2, x_k+1 - ...) con la rama que contiene (v,a): buscamos la rama */
            double blo = -1e18, bhi = 1e18;
            for (int b = 0; b < F[r].nb; b++) {
                double l = -1e18, h = 1e18; const Br *B = &F[r].b[b];
                for (int k = 0; k < K; k++) { double y = B->x[k] - vv[r] * k - a * k * k / 2.0; if (y > l) l = y; if (y + 1 < h) h = y + 1; }
                if (l < h) { blo = l; bhi = h; break; }
            }
            long pp, qq; if (blo < bhi) simplest(blo, bhi, &pp, &qq); else { pp = 0; qq = 0; }
            printf("%ld/%ld%s", pp, qq, r < N - 1 ? "," : "");
        }
        printf(" | vint="); for (int r = 0; r < N; r++) printf("%.4f:%.4f%s", VL[r], VH[r], r < N - 1 ? "," : "");
        printf(" | ");
        for (int k = 0; k < K; k++) printf("%d%c", ch[k], k < K - 1 ? ' ' : '\n');
    }
    return 0;
}
