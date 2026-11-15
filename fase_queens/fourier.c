/*
 * fourier.c — Reparto de soluciones en grupos recorridos por ondas de Fourier.
 *
 * Cada reina r de un grupo tiene su propia onda (fase, en columnas), muestreada en k = 0,1,2...
 * (cada muestra = un intervalo de pi):
 *     th_r(k) = beta + v k + sum_{m=1..M} [ a_m cos(w m k) + b_m sin(w m k) ],   w = 2 pi / P
 * Lectura: floor(th_r(k)) = c_r(k)  (mod N).   |a_m|, |b_m| <= AMAX.
 * M = 0 es la velocidad constante de antes. Cada armónico añade 2 números por reina.
 *
 * Comprobación exacta por programación lineal (símplex pequeño) para cada elección de vueltas
 * x_k = c_k + n_k N; las vueltas se ramifican en una ventana pequeña.
 *
 * Reparto voraz con búsqueda aleatoria: en cada ronda, R intentos de cadena (inicio al azar,
 * candidatos en orden aleatorio, se toma el primero que encaja); se queda la más larga, se
 * quita y se repite. Mismo procedimiento para soluciones y controles.
 *
 * uso: ./fourier N conjunto M [P=12] [AMAX=1] [R=300] [semilla=1]
 *      conjunto: 0 soluciones, s>0 azar, s<0 simétrico (igual que reparto.c)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAXK 64
#define MAXD 10
#define MAXBR 4
static int N, Q, cap, M, D; static double P = 12, AMAX = 1.0, W;
static int *sol;
static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) { if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return; }
    for (int c = 0; c < N; c++) { if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue; cur[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N)); }
}
static unsigned long long rs = 12345;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static int ri(int n) { return (int)(ur() * n); }
static void randperm(int *p) { for (int r = 0; r < N; r++) p[r] = r; for (int r = N - 1; r > 0; r--) { int j = ri(r + 1); int t = p[r]; p[r] = p[j]; p[j] = t; } }
static int have(const int *g, int n) { for (int k = 0; k < n; k++) if (!memcmp(g, sol + (long)k * N, sizeof(int) * N)) return 1; return 0; }
static void push(const int *g) { if (Q == cap) { cap = cap ? cap * 2 : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, g, sizeof(int) * N); Q++; }

/* ---------- símplex de fase 1: ¿existe z>=0 con G z <= h ? ---------- */
#define MAXR (2 * MAXK + 2 * MAXD + 4)
static double T[MAXR + 1][MAXD + MAXR + 3];
static int basis[MAXR];
static int feasible(int m, int n, double G[][MAXD], const double *h) {
    /* variables: z (n), t (1), holguras (m). Fila i: G z - t + s_i = h_i. Minimizar t. */
    int cols = n + 1 + m + 1; /* última = rhs */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < cols; j++) T[i][j] = 0;
        for (int j = 0; j < n; j++) T[i][j] = G[i][j];
        T[i][n] = -1; T[i][n + 1 + i] = 1; T[i][cols - 1] = h[i]; basis[i] = n + 1 + i;
    }
    /* fila objetivo: minimizar t  ->  guardamos coeficientes reducidos */
    for (int j = 0; j < cols; j++) T[m][j] = 0;
    T[m][n] = 1;
    /* entrada inicial de t en la fila con h más negativo */
    int rmin = -1; double hmin = 0;
    for (int i = 0; i < m; i++) if (h[i] < hmin) { hmin = h[i]; rmin = i; }
    if (rmin < 0) return 1;
    #define PIVOT(pr, pc) do { double pv = T[pr][pc]; for (int j = 0; j < cols; j++) T[pr][j] /= pv; \
        for (int i = 0; i <= m; i++) if (i != pr && fabs(T[i][pc]) > 1e-15) { double f = T[i][pc]; for (int j = 0; j < cols; j++) T[i][j] -= f * T[pr][j]; } basis[pr] = pc; } while (0)
    PIVOT(rmin, n);
    for (int it = 0; it < 500; it++) {
        /* columna que entra: coeficiente reducido más negativo (fila m es c - ...; minimizamos) */
        int pc = -1; double best = -1e-11;
        for (int j = 0; j < cols - 1; j++) if (T[m][j] < best) { best = T[m][j]; pc = j; }
        if (pc < 0) break;
        int pr = -1; double ratio = 1e300;
        for (int i = 0; i < m; i++) if (T[i][pc] > 1e-12) { double q = T[i][cols - 1] / T[i][pc]; if (q < ratio - 1e-15) { ratio = q; pr = i; } }
        if (pr < 0) break;
        PIVOT(pr, pc);
    }
    /* valor de t */
    double t = 0; for (int i = 0; i < m; i++) if (basis[i] == n) t = T[i][cols - 1];
    return t < 1e-9;
}

/* rama de una reina: historial de vueltas */
typedef struct { long x[MAXK]; } Br;
typedef struct { int nb; Br b[MAXBR]; } Set;
static double phi[MAXK][MAXD];

/* ¿la secuencia x[0..k] es factible? variables y = (beta, v, a1,b1,...). Usamos z = y - lo >= 0 */
static int check(const long *x, int k) {
    static double G[MAXR][MAXD], h[MAXR];
    double lo[MAXD], hi[MAXD];
    lo[0] = x[0]; hi[0] = x[0] + 1 - 1e-6;  /* beta en la celda de la primera muestra */
    lo[1] = 0; hi[1] = N;                 /* velocidad */
    for (int d = 2; d < D; d++) { lo[d] = -AMAX; hi[d] = AMAX; }
    int m = 0;
    for (int s = 0; s <= k; s++) {
        double base = 0; for (int d = 0; d < D; d++) base += phi[s][d] * lo[d];
        /* th(s) >= x_s  ->  -phi z <= base - x_s ;  th(s) <= x_s + 1 - eps -> phi z <= x_s + 1 - eps - base */
        for (int d = 0; d < D; d++) { G[m][d] = -phi[s][d]; G[m + 1][d] = phi[s][d]; }
        h[m] = base - x[s]; h[m + 1] = x[s] + 1 - 1e-6 - base; m += 2;
    }
    for (int d = 0; d < D; d++) { for (int e = 0; e < D; e++) G[m][e] = (e == d); h[m] = hi[d] - lo[d]; m++; }
    return feasible(m, D, G, h);
}

static Set st[MAXK][32];
/* extiende la reina r con la columna c en la posición k */
static int ext(int k, int r, int c) {
    Set *S = &st[k - 1][r], *Tt = &st[k][r]; Tt->nb = 0;
    long win = (long)ceil(2 * AMAX * 2 * M) + 2;
    for (int b = 0; b < S->nb; b++) {
        long xp = S->b[b].x[k - 1];
        long lo = xp - win, hi = xp + N + win;
        long n0 = (long)floor((double)(lo - c) / N), n1 = (long)ceil((double)(hi - c) / N);
        for (long n = n0; n <= n1; n++) {
            long xk = c + n * N; if (xk < lo || xk > hi) continue;
            Br B = S->b[b]; B.x[k] = xk;
            if (check(B.x, k)) { if (Tt->nb < MAXBR) Tt->b[Tt->nb++] = B; }
        }
    }
    return Tt->nb > 0;
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); int set = atoi(argv[2]); M = atoi(argv[3]);
    if (argc > 4) P = atof(argv[4]); if (argc > 5) AMAX = atof(argv[5]);
    int R = argc > 6 ? atoi(argv[6]) : 300; unsigned long long seed = argc > 7 ? strtoull(argv[7], 0, 10) : 1;
    D = 2 + 2 * M; W = 2 * M_PI / P;
    for (int s = 0; s < MAXK; s++) { phi[s][0] = 1; phi[s][1] = s; for (int mm = 1; mm <= M; mm++) { phi[s][2 * mm] = cos(W * mm * s); phi[s][2 * mm + 1] = sin(W * mm * s); } }
    bt(0, 0, 0, 0); int want = Q;
    if (set > 0) { rs ^= (unsigned long long)set * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32]; while (Q < want) { randperm(p); if (!have(p, Q)) push(p); } }
    else if (set < 0) {
        rs ^= (unsigned long long)(-set) * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32], g[32], q[32];
        while (Q < want) { randperm(p); for (int e = 0; e < 8; e++) { memcpy(q, p, sizeof q); if (e & 4) for (int r = 0; r < N; r++) q[p[r]] = r; for (int r = 0; r < N; r++) g[r] = q[(e & 2) ? N - 1 - r : r]; if (e & 1) for (int r = 0; r < N; r++) g[r] = N - 1 - g[r]; if (!have(g, Q)) push(g); } }
    }
    char tag[32]; if (set == 0) snprintf(tag, sizeof tag, "sol"); else if (set > 0) snprintf(tag, sizeof tag, "azar%d", set); else snprintf(tag, sizeof tag, "sim%d", -set);
    rs = seed * 0x9E3779B97F4A7C15ULL | 1;
    mkdir("datos", 0755);
    char fn[160]; snprintf(fn, sizeof fn, "datos/fourier_N%d_%s_M%d_P%g_A%g.txt", N, tag, M, P, AMAX);
    FILE *fo = fopen(fn, "w"); fprintf(fo, "# reparto con ondas de Fourier M=%d P=%g AMAX=%g (indices en datos/tableros_N%d_%s.txt)\n", M, P, AMAX, N, tag);
    char *gone = calloc(Q, 1), *used = calloc(Q, 1);
    int left = Q, ng = 0, sizes[4096];
    int chain[MAXK], bestc[MAXK], order[8192];
    while (left > 0) {
        int best = 0;
        for (int at = 0; at < R; at++) {
            int i; do i = ri(Q); while (gone[i]);
            memset(used, 0, Q); chain[0] = i; used[i] = 1; int K = 1;
            for (int r = 0; r < N; r++) { st[0][r].nb = 1; st[0][r].b[0].x[0] = sol[(long)i * N + r]; }
            for (;;) {
                int no = 0; for (int j = 0; j < Q; j++) if (!gone[j] && !used[j]) order[no++] = j;
                for (int a = no - 1; a > 0; a--) { int b = ri(a + 1); int t = order[a]; order[a] = order[b]; order[b] = t; }
                int got = -1;
                for (int a = 0; a < no && got < 0 && K < MAXK; a++) {
                    const int *Tb = sol + (long)order[a] * N; int ok = 1;
                    for (int r = 0; r < N && ok; r++) ok = ext(K, r, Tb[r]);
                    if (ok) got = order[a];
                }
                if (got < 0) break;
                chain[K++] = got; used[got] = 1;
            }
            if (K > best) { best = K; memcpy(bestc, chain, sizeof(int) * K); }
            if (best >= left) break;
        }
        for (int k = 0; k < best; k++) { gone[bestc[k]] = 1; fprintf(fo, "%d%c", bestc[k], k < best - 1 ? ' ' : '\n'); }
        sizes[ng++] = best; left -= best;
    }
    fclose(fo);
    double mean = (double)Q / ng;
    printf("N=%d %s M=%d (P=%g, AMAX=%g, R=%d): %d grupos, media %.2f  tamanos:", N, tag, M, P, AMAX, R, ng, mean);
    for (int g = 0; g < ng; g++) printf(" %d", sizes[g]);
    printf("\n");
    return 0;
}
