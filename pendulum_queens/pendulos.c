/*
 * pendulos.c — N-reinas como N péndulos que interfieren.
 *
 * Reina k (fila k) = péndulo con ángulo th_k. Su "sombra" horizontal es la
 * columna:   x_k = (N-1)(1 + sin th_k)/2   ∈ [0, N-1]   (sin paredes).
 *
 * Interferencia: cada par de reinas interfiere como una red de M rendijas
 * (núcleo de Fejér):
 *      F(d) = ( sin(pi d) / (M sin(pi d / M)) )^2
 *   F(0) = 1 (choque), F(d) = 0 exactamente en enteros d != 0, F >= 0 siempre.
 * Con M >= 2N-1 no hay aliasing en las diagonales, así que
 *      V = sum_{j<k} F(x_j-x_k) + F((x_j+j)-(x_k+k)) + F((x_j-j)-(x_k-k))
 * es >= 0 y vale 0  <=>  el tablero es una solución de N-reinas.
 * Las soluciones son los únicos "puntos oscuros" del patrón de interferencia.
 *
 * Dinámica: Hamiltoniana + cadena Nosé-Hoover por péndulo (determinista,
 * caótica, reparte la energía por igual y hace que la órbita se *quede* en
 * los valles). Temperatura T(t) = T0 (T1/T0)^(t/tot): con T1=T0 es una
 * órbita que salta entre soluciones; con T1 << T0 es un enfriamiento hacia
 * el equilibrio.
 *
 * uso: ./pendulos N runs steps T0 T1 [dt] [seed] [M]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int N, M;
static double A; /* (N-1)/2 */
#ifndef MAP
#define MAP 0 /* 0: sombra sin(th) (pendulo); 1: onda triangular (rebote uniforme) */
#endif
static inline double xmap(double t){
#if MAP==0
  return A*(1+sin(t));
#else
  return A*(1+(2/M_PI)*asin(sin(t)));
#endif
}
static inline double dxmap(double t){
#if MAP==0
  return A*cos(t);
#else
  return A*(2/M_PI)*(cos(t)>=0?1:-1);
#endif
}

static uint64_t rs;
static double urand(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (rs >> 11) * (1.0 / 9007199254740992.0);
}

/* núcleo de Fejér y su derivada */
static inline void fejer(double d, double *f, double *df) {
    double u = M_PI * d, w = u / M;
    double s = sin(u), t = sin(w);
    if (fabs(t) < 1e-7) { /* d ~ 0: F ~ 1 - c d^2 */
        double c = M_PI * M_PI * (M * M - 1.0) / (3.0 * M * M);
        *f = 1.0 - c * d * d; *df = -2.0 * c * d; return;
    }
    double r = s / (M * t);
    *f = r * r;
    double dr = (M_PI * cos(u) * t - s * (M_PI / M) * cos(w)) / (M * t * t);
    *df = 2.0 * r * dr;
}

/* V y dV/dth */
static double potential(const double *th, double *g, double *x) {
    for (int k = 0; k < N; k++) { x[k] = xmap(th[k]); g[k] = 0; }
    double V = 0;
    for (int j = 0; j < N; j++)
        for (int k = j + 1; k < N; k++) {
            double d0 = x[j] - x[k], f, df, gs = 0;
            fejer(d0, &f, &df);            V += f; gs += df;
            fejer(d0 + (j - k), &f, &df);  V += f; gs += df;
            fejer(d0 - (j - k), &f, &df);  V += f; gs += df;
            g[j] += gs; g[k] -= gs;
        }
    for (int k = 0; k < N; k++) g[k] *= dxmap(th[k]); /* regla de la cadena */
    return V;
}

/* ¿el tablero redondeado es solución? */
static int is_solution(const double *x, int *c) {
    static int col[256], dp[512], dm[512];
    memset(col, 0, sizeof(int) * N); memset(dp, 0, sizeof(int) * 2 * N); memset(dm, 0, sizeof(int) * 2 * N);
    for (int k = 0; k < N; k++) {
        int v = (int)lround(x[k]);
        if (v < 0) v = 0; if (v > N - 1) v = N - 1;
        c[k] = v;
        if (col[v]++ || dp[v + k]++ || dm[v - k + N]++) return 0;
    }
    return 1;
}

/* estado: th[N], p[N], xi1[N], xi2[N] */
static double *S, *K1, *K2, *K3, *K4, *TMP, *G, *X;
static double Q = 1.0, Tcur;

static void deriv(const double *s, double *ds) {
    const double *th = s, *p = s + N, *x1 = s + 2 * N, *x2 = s + 3 * N;
    potential(th, G, X);
    for (int k = 0; k < N; k++) {
        ds[k] = p[k];
        ds[N + k] = -G[k] - x1[k] * p[k];
        ds[2 * N + k] = (p[k] * p[k] - Tcur) / Q - x2[k] * x1[k];
        ds[3 * N + k] = (Q * x1[k] * x1[k] - Tcur) / Q;
    }
}

static void rk4(double dt) {
    int L = 4 * N;
    deriv(S, K1);
    for (int i = 0; i < L; i++) TMP[i] = S[i] + 0.5 * dt * K1[i];
    deriv(TMP, K2);
    for (int i = 0; i < L; i++) TMP[i] = S[i] + 0.5 * dt * K2[i];
    deriv(TMP, K3);
    for (int i = 0; i < L; i++) TMP[i] = S[i] + dt * K3[i];
    deriv(TMP, K4);
    for (int i = 0; i < L; i++) S[i] += dt / 6.0 * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]);
}

/* conjunto de soluciones distintas (hash simple) */
#define HS (1 << 16)
static uint64_t htab[HS];
static int hcount;
static int hinsert(const int *c) {
    uint64_t h = 1469598103934665603ULL;
    for (int k = 0; k < N; k++) { h ^= (uint64_t)c[k] + 1; h *= 1099511628211ULL; }
    if (!h) h = 1;
    uint64_t i = h & (HS - 1);
    while (htab[i]) { if (htab[i] == h) return 0; i = (i + 1) & (HS - 1); }
    if (hcount < HS / 2) { htab[i] = h; hcount++; }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "uso: %s N runs steps T0 T1 [dt] [seed] [M]\n", argv[0]); return 1; }
    N = atoi(argv[1]);
    int runs = atoi(argv[2]);
    long steps = atol(argv[3]);
    double T0 = atof(argv[4]), T1 = atof(argv[5]);
    double dt = argc > 6 ? atof(argv[6]) : 0.02;
    rs = argc > 7 ? strtoull(argv[7], 0, 10) : 12345;
    M = argc > 8 ? atoi(argv[8]) : 2 * N;
    A = (N - 1) / 2.0;
    S = malloc(sizeof(double) * 4 * N); K1 = malloc(sizeof(double) * 4 * N);
    K2 = malloc(sizeof(double) * 4 * N); K3 = malloc(sizeof(double) * 4 * N);
    K4 = malloc(sizeof(double) * 4 * N); TMP = malloc(sizeof(double) * 4 * N);
    G = malloc(sizeof(double) * N); X = malloc(sizeof(double) * N);
    int *c = malloc(sizeof(int) * N), *best = malloc(sizeof(int) * N);
    int solved_end = 0, solved_any = 0; long tfirst_sum = 0; long sol_steps = 0;
    int have_best = 0;

    for (int r = 0; r < runs; r++) {
        for (int k = 0; k < N; k++) {
            S[k] = 2 * M_PI * urand();
            S[N + k] = sqrt(T0) * (2 * urand() - 1) * 1.7;
            S[2 * N + k] = 0; S[3 * N + k] = 0;
        }
        long tfirst = -1; int in_sol = 0;
        for (long t = 0; t < steps; t++) {
            Tcur = T0 * pow(T1 / T0, (double)t / steps);
            rk4(dt);
            if ((t & 7) == 0) {
                double xs[256];
                for (int k = 0; k < N; k++) xs[k] = xmap(S[k]);
                in_sol = is_solution(xs, c);
                if (in_sol) {
                    sol_steps += 8;
                    hinsert(c);
                    if (tfirst < 0) { tfirst = t; memcpy(best, c, sizeof(int) * N); have_best = 1; }
                }
            }
        }
        double Vend = potential(S, G, X);
        if (tfirst >= 0) { solved_any++; tfirst_sum += tfirst; }
        if (in_sol) solved_end++;
        fprintf(stderr, "run %3d  V_final=%.4f  primera_sol=%ld  final_es_sol=%d\n", r, Vend, tfirst, in_sol);
    }
    printf("N=%d M=%d runs=%d steps=%ld T0=%g T1=%g dt=%g\n", N, M, runs, steps, T0, T1, dt);
    printf("  runs que tocaron solucion: %d/%d   terminan en solucion: %d/%d\n", solved_any, runs, solved_end, runs);
    printf("  fraccion de tiempo en solucion: %.4f   soluciones distintas: %d\n",
           (double)sol_steps / ((double)runs * steps), hcount);
    if (solved_any) printf("  paso medio de la primera: %.0f\n", (double)tfirst_sum / solved_any);
    if (have_best) { printf("  ejemplo:"); for (int k = 0; k < N; k++) printf(" %d", best[k]); printf("\n"); }
    return 0;
}
