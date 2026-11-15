/*
 * acelera.c — Velocidades que cambian, pero con aceleración acotada.
 *
 * Cada reina r tiene una fase continua th_k (en unidades de columna) en cada muestra k.
 * Lectura: floor(th_k) = c_r(k)  (mod N).
 * Velocidad v_k = th_{k+1} - th_k, libre al principio en [0, N).
 * Restricción física: |v_{k+1} - v_k| = |th_{k+2} - 2 th_{k+1} + th_k| <= a.
 *     a = 0  -> velocidad constante (lo de cadenas_frac.c)
 *     a grande -> cualquier orden vale
 *
 * Exacto: para cada reina se propaga el conjunto factible de (th_{k-1}, th_k), que es una
 * unión de polígonos convexos. Paso: (t,u) -> (u, 2u - t), suma de Minkowski con el
 * segmento vertical [-a, a], y recorte con la franja w en [c + mN, c + mN + 1) para cada m.
 *
 * Búsqueda: DFS exhaustiva de cadenas de soluciones distintas.
 * Control: el mismo conjunto de tamaño Q pero de permutaciones al azar (sin diagonales).
 *
 * uso: ./acelera N a [control_seed] [limite_segundos]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXV 64
#define MAXP 6
#define MAXK 96
typedef struct { int n; double x[MAXV], y[MAXV]; } Poly;
typedef struct { int np; Poly p[MAXP]; } Set;

static int N, Q, cap;
static int *sol;
static double A;
static const double EPS = 1e-7;

static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) {
        if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); }
        memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return;
    }
    for (int c = 0; c < N; c++) {
        if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue;
        cur[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N));
    }
}

/* ---- polígonos convexos ---- */
static double cross(double ox, double oy, double ax, double ay, double bx, double by) { return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox); }
static int cmpd(const void *a, const void *b) {
    const double *p = a, *q = b;
    if (p[0] != q[0]) return p[0] < q[0] ? -1 : 1;
    return p[1] < q[1] ? -1 : (p[1] > q[1]);
}
static void hull(double *pts, int n, Poly *out) { /* pts: 2n doubles */
    qsort(pts, n, 2 * sizeof(double), cmpd);
    double h[2 * MAXV * 2 + 4][2]; int k = 0;
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(h[k - 2][0], h[k - 2][1], h[k - 1][0], h[k - 1][1], pts[2 * i], pts[2 * i + 1]) <= 1e-15) k--;
        h[k][0] = pts[2 * i]; h[k][1] = pts[2 * i + 1]; k++;
    }
    for (int i = n - 2, t = k + 1; i >= 0; i--) {
        while (k >= t && cross(h[k - 2][0], h[k - 2][1], h[k - 1][0], h[k - 1][1], pts[2 * i], pts[2 * i + 1]) <= 1e-15) k--;
        h[k][0] = pts[2 * i]; h[k][1] = pts[2 * i + 1]; k++;
    }
    out->n = k > 1 ? k - 1 : k;
    if (out->n > MAXV) out->n = MAXV;
    for (int i = 0; i < out->n; i++) { out->x[i] = h[i][0]; out->y[i] = h[i][1]; }
}
/* recorta con y >= lo (s=+1) o y <= hi (s=-1) */
static void clip(Poly *P, double v, int s) {
    Poly R; R.n = 0;
    for (int i = 0; i < P->n; i++) {
        int j = (i + 1) % P->n;
        double yi = P->y[i], yj = P->y[j];
        int ini = s > 0 ? yi >= v : yi <= v, inj = s > 0 ? yj >= v : yj <= v;
        if (ini) { R.x[R.n] = P->x[i]; R.y[R.n] = yi; R.n++; }
        if (ini != inj) {
            double t = (v - yi) / (yj - yi);
            R.x[R.n] = P->x[i] + t * (P->x[j] - P->x[i]); R.y[R.n] = v; R.n++;
        }
        if (R.n >= MAXV - 2) break;
    }
    *P = R;
}
static double area(const Poly *P) {
    double s = 0; for (int i = 0; i < P->n; i++) { int j = (i + 1) % P->n; s += P->x[i] * P->y[j] - P->x[j] * P->y[i]; } return fabs(s) / 2;
}

/* paso: S (sobre (th_{k-1}, th_k)) -> T (sobre (th_k, th_{k+1})) con lectura c */
static int step(const Set *S, Set *T, int c) {
    T->np = 0;
    for (int i = 0; i < S->np; i++) {
        const Poly *P = &S->p[i];
        double pts[4 * MAXV]; int n = 0;
        double ylo = 1e18, yhi = -1e18;
        for (int v = 0; v < P->n; v++) {
            double u = P->y[v], w = 2 * P->y[v] - P->x[v];
            pts[2 * n] = u; pts[2 * n + 1] = w - A; n++;
            pts[2 * n] = u; pts[2 * n + 1] = w + A; n++;
            if (w - A < ylo) ylo = w - A;
            if (w + A > yhi) yhi = w + A;
        }
        Poly H; hull(pts, n, &H);
        /* franjas c + mN */
        int m0 = (int)floor((ylo - c - 1) / N), m1 = (int)ceil((yhi - c) / N);
        for (int m = m0; m <= m1; m++) {
            double lo = c + (double)m * N + EPS, hi = c + (double)m * N + 1 - EPS;
            if (hi < ylo || lo > yhi) continue;
            Poly R = H; clip(&R, lo, +1); if (R.n < 3) { if (R.n == 0) continue; }
            clip(&R, hi, -1);
            if (R.n == 0) continue;
            if (A > 0 && area(&R) < 1e-14) continue;
            if (T->np < MAXP) T->p[T->np++] = R;
        }
    }
    return T->np > 0;
}

static Set st[MAXK][32];
static int chain[MAXK], K, best, nbest;
static char *used;
static long nodes;
static double tlimit; static clock_t t0; static int timed_out;
static int bestchain[MAXK];

static void dfs(void) {
    if (timed_out) return;
    if ((++nodes & 0xffff) == 0 && (double)(clock() - t0) / CLOCKS_PER_SEC > tlimit) { timed_out = 1; return; }
    int ext = 0;
    if (K < MAXK) {
        for (int j = 0; j < Q && !timed_out; j++) {
            if (used[j]) continue;
            const int *T = sol + (long)j * N;
            int ok = 1;
            for (int r = 0; r < N && ok; r++) ok = step(&st[K - 1][r], &st[K][r], T[r]);
            if (!ok) continue;
            ext = 1; chain[K++] = j; used[j] = 1;
            dfs();
            used[j] = 0; K--;
        }
    }
    if (!ext && K > best) { best = K; nbest = 1; memcpy(bestchain, chain, sizeof(int) * K); }
    else if (!ext && K == best) nbest++;
}

static unsigned long long rs = 12345;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

int main(int argc, char **argv) {
    N = atoi(argv[1]); A = atof(argv[2]);
    int ctrl = argc > 3 ? atoi(argv[3]) : 0;
    tlimit = argc > 4 ? atof(argv[4]) : 120;
    bt(0, 0, 0, 0);
    if (ctrl) { /* Q permutaciones al azar distintas */
        rs ^= (unsigned long long)ctrl * 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < Q; i++) {
            int *p = sol + (long)i * N; int ok;
            do {
                for (int r = 0; r < N; r++) p[r] = r;
                for (int r = N - 1; r > 0; r--) { int j = (int)(ur() * (r + 1)); int t = p[r]; p[r] = p[j]; p[j] = t; }
                ok = 1; for (int q = 0; q < i && ok; q++) if (!memcmp(p, sol + (long)q * N, sizeof(int) * N)) ok = 0;
            } while (!ok);
        }
    }
    if (ctrl < 0) { /* control simétrico: órbitas D4 de permutaciones al azar, hasta >= Q */
        rs ^= (unsigned long long)(-ctrl) * 0x9E3779B97F4A7C15ULL;
        int want = Q; Q = 0; int p[32], g[32];
        while (Q < want) {
            for (int r = 0; r < N; r++) p[r] = r;
            for (int r = N - 1; r > 0; r--) { int j = (int)(ur() * (r + 1)); int t = p[r]; p[r] = p[j]; p[j] = t; }
            for (int e = 0; e < 8; e++) {
                /* e: bit0 espejo columnas, bit1 espejo filas, bit2 transponer (inversa) */
                int q[32]; memcpy(q, p, sizeof q);
                if (e & 4) for (int r = 0; r < N; r++) q[p[r]] = r;
                for (int r = 0; r < N; r++) g[r] = q[(e & 2) ? N - 1 - r : r];
                if (e & 1) for (int r = 0; r < N; r++) g[r] = N - 1 - g[r];
                int dup = 0; for (int k = 0; k < Q && !dup; k++) if (!memcmp(g, sol + (long)k * N, sizeof(int) * N)) dup = 1;
                if (!dup) { if (Q == cap) { cap *= 2; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, g, sizeof(int) * N); Q++; }
            }
        }
    }
    used = calloc(Q, 1);
    t0 = clock();
    for (int i = 0; i < Q && !timed_out; i++) {
        /* estado inicial tras dos muestras: th_0 en [c0, c0+1), th_1 con velocidad inicial en [0, N) */
        const int *S0 = sol + (long)i * N;
        chain[0] = i; used[i] = 1;
        for (int j = 0; j < Q && !timed_out; j++) {
            if (j == i) continue;
            const int *S1 = sol + (long)j * N;
            int ok = 1;
            for (int r = 0; r < N && ok; r++) {
                Set *T = &st[1][r]; T->np = 0;
                double a0 = S0[r] + EPS, a1 = S0[r] + 1 - EPS;
                for (int m = 0; m <= 1; m++) {
                    double b0 = S1[r] + (double)m * N + EPS, b1 = S1[r] + (double)m * N + 1 - EPS;
                    /* (t,u): t in [a0,a1], u in [b0,b1], 0 <= u - t < N */
                    Poly P; P.n = 4;
                    P.x[0] = a0; P.y[0] = b0; P.x[1] = a1; P.y[1] = b0; P.x[2] = a1; P.y[2] = b1; P.x[3] = a0; P.y[3] = b1;
                    /* recorta u - t en [0, N): usamos coordenadas rotadas aproximando con recorte por vértices */
                    double pts[2 * MAXV]; int n = 0;
                    for (int v = 0; v < 4; v++) { pts[2 * n] = P.x[v]; pts[2 * n + 1] = P.y[v] - P.x[v]; n++; }
                    Poly R; hull(pts, n, &R); clip(&R, 0, +1); clip(&R, N - EPS, -1);
                    if (R.n < 3) continue;
                    for (int v = 0; v < R.n; v++) R.y[v] += R.x[v];
                    if (T->np < MAXP) T->p[T->np++] = R;
                }
                if (!T->np) ok = 0;
            }
            if (!ok) continue;
            chain[1] = j; used[j] = 1; K = 2;
            dfs();
            used[j] = 0;
        }
        used[i] = 0;
    }
    printf("N=%d Q=%d a=%.3f %s: cadena maxima %d%s  (nodos %ld, %.1fs)\n", N, Q, A, ctrl > 0 ? "CONTROL azar" : (ctrl < 0 ? "CONTROL simetrico" : "soluciones"),
           best, timed_out ? " (cota inferior: tiempo agotado)" : "", nodes, (double)(clock() - t0) / CLOCKS_PER_SEC);
    printf("  ejemplo:");
    for (int k = 0; k < best; k++) {
        printf(" [");
        for (int r = 0; r < N; r++) printf("%d%s", sol[(long)bestchain[k] * N + r], r < N - 1 ? "," : "");
        printf("]");
    }
    printf("\n");
    return 0;
}
