// zeta_queens.c — las soluciones de N-reinas (N=7..9) escritas como numeros con
// cifras 1..N, contadas como pi(x):  A(x) = #{solucion <= x}.
// Modelo tipo formula explicita de Riemann:
//   A(x) = r * P(x)  +  sum_k a_k cos(g_k u) + b_k sin(g_k u),   u = log x (o x)
// donde P(x) = #{permutaciones <= x} y r = Q/N! es la tendencia suave sin parametros.
// Las frecuencias g_k se buscan con un pursuit voraz (barrido + seccion aurea) y las
// amplitudes se reajustan conjuntamente por minimos cuadrados.  Se compara con
// subconjuntos aleatorios de permutaciones del mismo tamano (nulo).
//
// cc -O2 -o zeta_queens zeta_queens.c -lm && ./zeta_queens
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXP 362880
#define MAXQ 400
#define KMAX 12
#define NNULL 60

static int N, nperm, nsol;
static uint64_t perm[MAXP];
static int solrank[MAXQ];
static int used[10], cur[10];

static int valid(void) {
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (abs(cur[i] - cur[j]) == j - i) return 0;
    return 1;
}

// orden lexicografico de cifras = orden numerico, perm[] queda ordenado
static void gen(int r, uint64_t v) {
    if (r == N) {
        if (valid()) solrank[nsol++] = nperm;
        perm[nperm++] = v;
        return;
    }
    for (int d = 0; d < N; d++)
        if (!used[d]) {
            used[d] = 1; cur[r] = d;
            gen(r + 1, v * 10 + d + 1);
            used[d] = 0;
        }
}

static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static int cmpi(const void *a, const void *b) { return *(int *)a - *(int *)b; }

// resuelve M x = b (n x n) por eliminacion gaussiana con pivoteo
static void solve(int n, double *M, double *b) {
    for (int c = 0; c < n; c++) {
        int p = c;
        for (int r = c + 1; r < n; r++) if (fabs(M[r*n+c]) > fabs(M[p*n+c])) p = r;
        for (int k = 0; k < n; k++) { double t = M[c*n+k]; M[c*n+k] = M[p*n+k]; M[p*n+k] = t; }
        double t = b[c]; b[c] = b[p]; b[p] = t;
        for (int r = c + 1; r < n; r++) {
            double f = M[r*n+c] / M[c*n+c];
            for (int k = c; k < n; k++) M[r*n+k] -= f * M[c*n+k];
            b[r] -= f * b[c];
        }
    }
    for (int c = n - 1; c >= 0; c--) {
        for (int k = c + 1; k < n; k++) b[c] -= M[c*n+k] * b[k];
        b[c] /= M[c*n+c];
    }
}

// reduccion de SS al anadir la onda de frecuencia g al residuo res
static double gain(const double *u, const double *res, int q, double g) {
    double cc = 0, cs = 0, ss = 0, cr = 0, sr = 0;
    for (int i = 0; i < q; i++) {
        double c = cos(g * u[i]), s = sin(g * u[i]);
        cc += c*c; cs += c*s; ss += s*s; cr += c*res[i]; sr += s*res[i];
    }
    double det = cc*ss - cs*cs;
    if (det < 1e-12) return 0;
    double a = (ss*cr - cs*sr) / det, b = (cc*sr - cs*cr) / det;
    return a*cr + b*sr;
}

// pursuit: devuelve R^2 tras k=1..K ondas; g[] y amp[] de la ultima ronda
static void pursuit(const double *u, const double *y, int q, int K,
                    double *R2, double *g, double *amp) {
    double umin = u[0], umax = u[0], mean = 0, sst = 0;
    for (int i = 0; i < q; i++) { if (u[i] < umin) umin = u[i]; if (u[i] > umax) umax = u[i]; mean += y[i]; }
    mean /= q;
    for (int i = 0; i < q; i++) sst += (y[i] - mean) * (y[i] - mean);
    double W = umax - umin, dg = M_PI / (4 * W), gmax = M_PI * q / W;
    double res[MAXQ], uc[MAXQ];
    for (int i = 0; i < q; i++) { uc[i] = u[i] - 0.5 * (umin + umax); res[i] = y[i] - mean; }

    for (int k = 0; k < K; k++) {
        double best = -1, bg = dg;
        for (double t = dg; t <= gmax; t += dg) {
            double v = gain(uc, res, q, t);
            if (v > best) { best = v; bg = t; }
        }
        double lo = bg - dg, hi = bg + dg, phi = 0.6180339887;
        for (int it = 0; it < 40; it++) {
            double m1 = hi - phi * (hi - lo), m2 = lo + phi * (hi - lo);
            if (gain(uc, res, q, m1) > gain(uc, res, q, m2)) hi = m2; else lo = m1;
        }
        g[k] = 0.5 * (lo + hi);

        // reajuste conjunto: columnas 1, cos(g_j u), sin(g_j u)
        int n = 2 * (k + 1) + 1;
        double M[(2*KMAX+1)*(2*KMAX+1)], b[2*KMAX+1], col[2*KMAX+1];
        memset(M, 0, sizeof(double) * n * n); memset(b, 0, sizeof(double) * n);
        for (int i = 0; i < q; i++) {
            col[0] = 1;
            for (int j = 0; j <= k; j++) { col[1+2*j] = cos(g[j]*uc[i]); col[2+2*j] = sin(g[j]*uc[i]); }
            for (int r = 0; r < n; r++) {
                b[r] += col[r] * y[i];
                for (int c = 0; c < n; c++) M[r*n+c] += col[r] * col[c];
            }
        }
        for (int r = 0; r < n; r++) M[r*n+r] += 1e-9;
        solve(n, M, b);
        double ssr = 0;
        for (int i = 0; i < q; i++) {
            double f = b[0];
            for (int j = 0; j <= k; j++) f += b[1+2*j]*cos(g[j]*uc[i]) + b[2+2*j]*sin(g[j]*uc[i]);
            res[i] = y[i] - f; ssr += res[i] * res[i];
        }
        R2[k] = 1 - ssr / sst;
        for (int j = 0; j <= k; j++) amp[j] = hypot(b[1+2*j], b[2+2*j]);
    }
}

// construye u,y para un conjunto de rangos ordenados
static void build(const int *rk, int q, int mode, double *u, double *y) {
    double r = (double)q / nperm, scale = pow(10, N - 1);
    for (int i = 0; i < q; i++) {
        double x = (double)perm[rk[i]];
        u[i] = mode == 0 ? log(x) : x / scale;
        y[i] = (i + 0.5) - r * (rk[i] + 0.5);   // convencion de medio salto
    }
}

int main(void) {
    for (N = 7; N <= 9; N++) {
        nperm = nsol = 0;
        memset(used, 0, sizeof used);
        gen(0, 0);
        int K = nsol / 5 < KMAX ? nsol / 5 : KMAX;

        int hist[10] = {0};
        for (int i = 0; i < nsol; i++) hist[perm[solrank[i]] / (uint64_t)pow(10, N - 1)]++;
        printf("\n==== N=%d  Q=%d  N!=%d  r=Q/N!=%.5f\n", N, nsol, nperm, (double)nsol / nperm);
        printf("primera cifra:");
        for (int d = 1; d <= N; d++) printf(" %d:%d", d, hist[d]);
        printf("\n");

        for (int mode = 0; mode < 2; mode++) {
            double u[MAXQ], y[MAXQ], R2[KMAX], g[KMAX], amp[KMAX];
            double gS[KMAX], aS[KMAX], R2S[KMAX];
            build(solrank, nsol, mode, u, y);
            pursuit(u, y, nsol, K, R2S, gS, aS);

            double m[KMAX] = {0}, m2[KMAX] = {0};
            int rk[MAXQ], *idx = malloc(sizeof(int) * nperm);
            for (int t = 0; t < NNULL; t++) {
                for (int i = 0; i < nperm; i++) idx[i] = i;
                for (int i = 0; i < nsol; i++) {
                    int j = i + xr() % (nperm - i);
                    int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                    rk[i] = idx[i];
                }
                qsort(rk, nsol, sizeof(int), cmpi);
                build(rk, nsol, mode, u, y);
                pursuit(u, y, nsol, K, R2, g, amp);
                for (int k = 0; k < K; k++) { m[k] += R2[k]; m2[k] += R2[k] * R2[k]; }
            }
            free(idx);

            printf("-- variable %s\n", mode == 0 ? "u = log x" : "u = x / 10^(N-1)");
            printf("   K   R2(sol)  R2(nulo)±sd     z\n");
            for (int k = 0; k < K; k++) {
                double mu = m[k] / NNULL, sd = sqrt(fmax(m2[k] / NNULL - mu * mu, 1e-12));
                printf("  %2d   %.3f    %.3f±%.3f  %6.1f\n", k + 1, R2S[k], mu, sd, (R2S[k] - mu) / sd);
            }
            printf("   frecuencias (sol): ");
            for (int k = 0; k < (K < 6 ? K : 6); k++) {
                if (mode == 0) printf("g=%.2f(amp %.2f, razon e^(2pi/g)=%.3f) ", gS[k], aS[k], exp(2 * M_PI / gS[k]));
                else printf("g=%.2f(amp %.2f, periodo %.3f cifras) ", gS[k], aS[k], 2 * M_PI / gS[k]);
            }
            printf("\n");
        }
    }
    return 0;
}
