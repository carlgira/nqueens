/*
 * kernel.c — Escenario A: el paisaje ideal para N-reinas, construido a lo bruto.
 *
 * Estado: una reina por fila, c_r en {0..N-1}  (N^N estados, enumerados todos).
 * Energía: suma sobre pares de filas de un núcleo que depende solo de la separación:
 *      E(c) = sum_{r<s} J(s-r, |c_s - c_r|)
 * J tiene (N-1)*N parámetros. El núcleo de "ataque" (1 si misma columna o diagonal, 0 si no)
 * da E=0 exactamente en las soluciones, pero deja muchos valles falsos.
 *
 * Medida exacta (sobre TODOS los estados): bajada por máxima pendiente moviendo una reina
 * dentro de su fila. P = fracción de estados cuya bajada termina en una solución.
 * L = número de mínimos locales que no son solución. Además se exige que las soluciones
 * sean los mínimos globales (margen > 0).
 *
 * Optimización: estrategia evolutiva (1+1) con paso adaptativo sobre J, maximizando P.
 *
 * uso: ./kernel N eval            -> núcleo de ataque
 *      ./kernel N opt iters seed  -> optimiza y escribe J en J_N.txt
 *      ./kernel N file J.txt      -> evalúa un núcleo leído de archivo
 *      ./kernel N sample J.txt starts seed -> estima P con bajadas desde estados al azar (N grande)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int N;
static long S; /* N^N */
static double J[32][32];
static float *E;
static int *nxt;
static char *issol;
static int *Cbuf;

static void decode(long s, int *c) { for (int r = 0; r < N; r++) { c[r] = s % N; s /= N; } }
static long pw[32];

static int is_solution(const int *c) {
    for (int r = 0; r < N; r++)
        for (int q = r + 1; q < N; q++) {
            int d = abs(c[q] - c[r]);
            if (d == 0 || d == q - r) return 0;
        }
    return 1;
}

static double energy(const int *c) {
    double e = 0;
    for (int r = 0; r < N; r++)
        for (int q = r + 1; q < N; q++) e += J[q - r][abs(c[q] - c[r])];
    return e;
}

/* bajada de un paso desde c: devuelve índice del mejor vecino o -1 si es mínimo local */
static long best_move(long s, const int *c, double *dEbest) {
    double best = -1e-9; long bs = -1;
    for (int r = 0; r < N; r++) {
        int c0 = c[r];
        for (int c1 = 0; c1 < N; c1++) {
            if (c1 == c0) continue;
            double d = 0;
            for (int q = 0; q < N; q++) {
                if (q == r) continue;
                int dr = abs(q - r);
                d += J[dr][abs(c[q] - c1)] - J[dr][abs(c[q] - c0)];
            }
            if (d < best) { best = d; bs = s + (long)(c1 - c0) * pw[r]; }
        }
    }
    if (dEbest) *dEbest = best;
    return bs;
}

typedef struct { double P, margin; long L, nsol; } Res;

static Res evaluate(void) {
    int c[32];
    Res R = {0};
    double maxsol = -1e30, minbad = 1e30;
    for (long s = 0; s < S; s++) {
        decode(s, c);
        E[s] = (float)energy(c);
        if (issol[s]) { if (E[s] > maxsol) maxsol = E[s]; }
        else if (E[s] < minbad) minbad = E[s];
        nxt[s] = (int)best_move(s, c, NULL);
    }
    /* resolver destino final con compresión de caminos: nxt -> terminal */
    long good = 0;
    for (long s = 0; s < S; s++) {
        long t = s;
        while (nxt[t] >= 0) t = nxt[t];
        long u = s;
        while (nxt[u] >= 0) { long w = nxt[u]; nxt[u] = (int)t; u = w; }
        /* ojo: nxt[t] = -1 se mantiene; los demás apuntan al terminal (y ya no son -1) */
        if (issol[t]) good++;
    }
    for (long s = 0; s < S; s++) {
        /* terminales: los que quedaron con nxt == -1 */
        if (nxt[s] == -1) { if (issol[s]) R.nsol++; else R.L++; }
    }
    R.P = (double)good / S;
    R.margin = minbad - maxsol;
    return R;
}

static void attack_kernel(void) {
    for (int dr = 1; dr < N; dr++)
        for (int dc = 0; dc < N; dc++) J[dr][dc] = (dc == 0 || dc == dr) ? 1.0 : 0.0;
}

static void save(const char *fn) {
    FILE *f = fopen(fn, "w");
    for (int dr = 1; dr < N; dr++) {
        for (int dc = 0; dc < N; dc++) fprintf(f, "%.6f ", J[dr][dc]);
        fprintf(f, "\n");
    }
    fclose(f);
}
static void load(const char *fn) {
    FILE *f = fopen(fn, "r");
    for (int dr = 1; dr < N; dr++)
        for (int dc = 0; dc < N; dc++) if (fscanf(f, "%lf", &J[dr][dc]) != 1) { fprintf(stderr, "J mal\n"); exit(1); }
    fclose(f);
}
static void print_J(void) {
    printf("      dc:");
    for (int dc = 0; dc < N; dc++) printf(" %6d", dc);
    printf("\n");
    for (int dr = 1; dr < N; dr++) {
        printf("  dr=%d  ", dr);
        for (int dc = 0; dc < N; dc++) printf(" %6.3f", J[dr][dc]);
        printf("\n");
    }
}

static unsigned long long rs = 88172645463325252ULL;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double gn(void) { return sqrt(-2 * log(ur())) * cos(6.283185307179586 * ur()); }

static double score(Res r) { return r.margin > 1e-6 ? r.P : r.P - 1.0 - fabs(r.margin); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "uso: ver cabecera\n"); return 1; }
    N = atoi(argv[1]);
    const char *mode = argv[2];
    if (!strcmp(mode, "sample")) {
        /* estimación por muestreo para N grande: no enumera */
        load(argv[3]);
        long starts = atol(argv[4]); rs ^= strtoull(argv[5], 0, 10) * 0x9E3779B97F4A7C15ULL;
        pw[0] = 1; for (int r = 1; r < N; r++) pw[r] = pw[r - 1] * N;
        long good = 0; int c[32];
        for (long k = 0; k < starts; k++) {
            for (int r = 0; r < N; r++) c[r] = (int)(ur() * N);
            for (;;) {
                double bestd = -1e-9; int br = -1, bc = -1;
                for (int r = 0; r < N; r++) for (int c1 = 0; c1 < N; c1++) {
                    if (c1 == c[r]) continue; double d = 0;
                    for (int q = 0; q < N; q++) { if (q == r) continue; int dr = abs(q - r); d += J[dr][abs(c[q] - c1)] - J[dr][abs(c[q] - c[r])]; }
                    if (d < bestd) { bestd = d; br = r; bc = c1; }
                }
                if (br < 0) break;
                c[br] = bc;
            }
            good += is_solution(c);
        }
        printf("N=%d muestreo: P = %.4f  (%ld/%ld bajadas terminan en solucion)\n", N, (double)good / starts, good, starts);
        return 0;
    }
    S = 1; pw[0] = 1;
    for (int r = 0; r < N; r++) { if (r) pw[r] = pw[r - 1] * N; S *= N; }
    E = malloc(sizeof(float) * S); nxt = malloc(sizeof(int) * S); issol = malloc(S);
    int c[32]; long nsol = 0;
    for (long s = 0; s < S; s++) { decode(s, c); issol[s] = (char)is_solution(c); nsol += issol[s]; }
    printf("N=%d  estados=%ld  soluciones=%ld\n", N, S, nsol);

    if (!strcmp(mode, "eval") || !strcmp(mode, "file")) {
        if (!strcmp(mode, "eval")) attack_kernel(); else load(argv[3]);
        Res r = evaluate();
        print_J();
        printf("  P(bajada termina en solucion) = %.4f   minimos falsos L = %ld   soluciones-minimo = %ld   margen = %.4f\n",
               r.P, r.L, r.nsol, r.margin);
        return 0;
    }
    if (!strcmp(mode, "opt")) {
        int iters = atoi(argv[3]);
        rs ^= strtoull(argv[4], 0, 10) * 0x9E3779B97F4A7C15ULL;
        attack_kernel();
        Res cur = evaluate(); double fc = score(cur);
        printf("inicio (ataque): P=%.4f L=%ld margen=%.3f\n", cur.P, cur.L, cur.margin);
        double sigma = 0.1, Jb[32][32];
        for (int it = 1; it <= iters; it++) {
            memcpy(Jb, J, sizeof J);
            for (int dr = 1; dr < N; dr++) for (int dc = 0; dc < N; dc++) J[dr][dc] += sigma * gn();
            /* normaliza: max |J| = 1 (la escala no importa) */
            double m = 0; for (int dr = 1; dr < N; dr++) for (int dc = 0; dc < N; dc++) if (fabs(J[dr][dc]) > m) m = fabs(J[dr][dc]);
            for (int dr = 1; dr < N; dr++) for (int dc = 0; dc < N; dc++) J[dr][dc] /= m;
            Res r = evaluate(); double f = score(r);
            if (f >= fc) { fc = f; cur = r; sigma *= 1.3; }
            else { memcpy(J, Jb, sizeof J); sigma *= 0.93; }
            if (sigma < 0.005) sigma = 0.05; /* reinicio suave del paso */
            if (it % 50 == 0) { printf("it %5d  P=%.4f  L=%ld  margen=%.3f  sigma=%.3f\n", it, cur.P, cur.L, cur.margin, sigma); fflush(stdout); }
        }
        char fn[64]; snprintf(fn, sizeof fn, "J_%d.txt", N); save(fn);
        print_J();
        printf("final: P=%.4f  L=%ld  margen=%.4f  -> %s\n", cur.P, cur.L, cur.margin, fn);
        return 0;
    }
    return 1;
}
