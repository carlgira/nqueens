// zeta_big.c — misma busqueda de ondas que zeta_queens.c, con dos codificados:
//   godel : n = prod_r p_r^{c_r}  ->  u = log n = sum_r c_r log p_r   (N <= 10)
//   base  : cifras 1..N en base N+1 (sin cero) -> u = x / (N+1)^(N-1)  o  u = log x
// En base N+1 el orden numerico es el lexicografico, asi que P(x) = rango de Lehmer
// y no hace falta enumerar las N! permutaciones.  Nulo: Q permutaciones al azar.
//
// cc -O2 -o zeta_big zeta_big.c -lm && ./zeta_big godel 7 10 && ./zeta_big base 7 13
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXN 16
#define KMAX 40
#define GRID 3000

static int N, nsol, cap;
static unsigned char *sols;        // nsol x N columnas 1..N
static int cur[MAXN];
static double logp[MAXN];
static double *gkeys;              // claves godel ordenadas de las N! permutaciones
static double fact;

static void addsol(void) {
    if (nsol == cap) { cap = cap ? 2 * cap : 1024; sols = realloc(sols, (size_t)cap * N); }
    for (int r = 0; r < N; r++) sols[(size_t)nsol * N + r] = cur[r] + 1;
    nsol++;
}

static void queens(int r, unsigned cols, unsigned d1, unsigned d2) {
    if (r == N) { addsol(); return; }
    unsigned free = ~(cols | d1 | d2) & ((1u << N) - 1);
    while (free) {
        unsigned b = free & -free; free ^= b;
        cur[r] = __builtin_ctz(b);
        queens(r + 1, cols | b, (d1 | b) << 1, (d2 | b) >> 1);
    }
}

static double lehmer(const unsigned char *c) {
    double rk = 0;
    for (int i = 0; i < N; i++) {
        int less = 0;
        for (int j = i + 1; j < N; j++) less += c[j] < c[i];
        rk = rk * (N - i) + less;
    }
    return rk;
}

static int mode;  // 0 godel log, 1 base lineal, 2 base log
static double key(const unsigned char *c) {
    if (mode == 0) { double s = 0; for (int r = 0; r < N; r++) s += c[r] * logp[r]; return s; }
    double x = 0;
    for (int r = 0; r < N; r++) x = x * (N + 1) + c[r];
    x /= pow(N + 1, N - 1);
    return mode == 1 ? x : log(x);
}

static double rankof(const unsigned char *c) {
    if (mode != 0) return lehmer(c);
    double k = key(c);
    long lo = 0, hi = (long)fact;
    while (lo < hi) { long m = (lo + hi) / 2; if (gkeys[m] < k - 1e-12) lo = m + 1; else hi = m; }
    return lo;
}

static int used[MAXN]; static long gi;
static void allperm(int r, double s) {
    if (r == N) { gkeys[gi++] = s; return; }
    for (int d = 0; d < N; d++)
        if (!used[d]) { used[d] = 1; allperm(r + 1, s + (d + 1) * logp[r]); used[d] = 0; }
}

static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

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

static double gain(const double *u, const double *res, int q, double g) {
    double cc = 0, cs = 0, ss = 0, cr = 0, sr = 0;
    for (int i = 0; i < q; i++) {
        double c = cos(g * u[i]), s = sin(g * u[i]);
        cc += c*c; cs += c*s; ss += s*s; cr += c*res[i]; sr += s*res[i];
    }
    double det = cc*ss - cs*cs;
    if (det < 1e-12) return 0;
    return ((ss*cr - cs*sr) * cr + (cc*sr - cs*cr) * sr) / det;
}

// pursuit voraz; el barrido avanza cos/sin por rotacion (sin trigonometria por paso)
static void pursuit(const double *u, const double *y, int q, int K, double *R2, double *g, double *amp) {
    double umin = u[0], umax = u[q-1], mean = 0, sst = 0;
    for (int i = 0; i < q; i++) mean += y[i];
    mean /= q;
    for (int i = 0; i < q; i++) sst += (y[i] - mean) * (y[i] - mean);
    double W = umax - umin, dg = M_PI / (4 * W), gmax = fmin(M_PI * q / W, GRID * dg);
    double *res = malloc(sizeof(double) * q), *uc = malloc(sizeof(double) * q);
    double *c = malloc(sizeof(double) * q), *s = malloc(sizeof(double) * q);
    double *rc = malloc(sizeof(double) * q), *rsn = malloc(sizeof(double) * q);
    for (int i = 0; i < q; i++) { uc[i] = u[i] - 0.5 * (umin + umax); res[i] = y[i] - mean; rc[i] = cos(dg * uc[i]); rsn[i] = sin(dg * uc[i]); }

    for (int k = 0; k < K; k++) {
        double best = -1, bg = dg;
        for (int i = 0; i < q; i++) { c[i] = rc[i]; s[i] = rsn[i]; }
        int nstep = isfinite(gmax / dg) ? (int)fmin(gmax / dg, 4 * GRID) : 0;
        for (int st = 1; st <= nstep; st++) {
            double t = st * dg;
            double cc = 0, cs = 0, ss = 0, cr = 0, sr = 0;
            for (int i = 0; i < q; i++) {
                cc += c[i]*c[i]; cs += c[i]*s[i]; ss += s[i]*s[i]; cr += c[i]*res[i]; sr += s[i]*res[i];
                double nc = c[i]*rc[i] - s[i]*rsn[i];
                s[i] = s[i]*rc[i] + c[i]*rsn[i]; c[i] = nc;
            }
            double det = cc*ss - cs*cs;
            double v = det < 1e-12 ? 0 : ((ss*cr - cs*sr) * cr + (cc*sr - cs*cr) * sr) / det;
            // gamma >= pi/W (= 4 dg): periodo <= 2W, si no la onda es colineal con la constante
            if (st >= 4 && v > best) { best = v; bg = t; }
        }
        double lo = fmax(bg - dg, 4 * dg), hi = bg + dg, phi = 0.6180339887;
        for (int it = 0; it < 40; it++) {
            double m1 = hi - phi * (hi - lo), m2 = lo + phi * (hi - lo);
            if (gain(uc, res, q, m1) > gain(uc, res, q, m2)) hi = m2; else lo = m1;
        }
        g[k] = 0.5 * (lo + hi);

        int n = 2 * (k + 1) + 1;
        double M[(2*KMAX+1)*(2*KMAX+1)], b[2*KMAX+1], col[2*KMAX+1];
        memset(M, 0, sizeof(double) * n * n); memset(b, 0, sizeof(double) * n);
        for (int i = 0; i < q; i++) {
            col[0] = 1;
            for (int j = 0; j <= k; j++) { col[1+2*j] = cos(g[j]*uc[i]); col[2+2*j] = sin(g[j]*uc[i]); }
            for (int r = 0; r < n; r++) {
                b[r] += col[r] * y[i];
                for (int cc2 = 0; cc2 < n; cc2++) M[r*n+cc2] += col[r] * col[cc2];
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
    free(res); free(uc); free(c); free(s); free(rc); free(rsn);
}

typedef struct { double u, rk; } pt;
static int cmppt(const void *a, const void *b) { const pt *x = a, *y = b; return (x->u > y->u) - (x->u < y->u); }

// ordena por clave y construye y_i = (i+1/2) - r (rango_i + 1/2)
static int build(pt *p, int q, double *u, double *y) {
    qsort(p, q, sizeof(pt), cmppt);
    double r = q / fact;
    for (int i = 0; i < q; i++) {
        if (i && p[i].rk == p[i-1].rk) return 0;
        u[i] = p[i].u; y[i] = (i + 0.5) - r * (p[i].rk + 0.5);
    }
    return 1;
}

// amplitud de la onda de frecuencia g ajustada por LS al residuo res
static double ampat(const double *u, const double *res, int q, double g) {
    double cc = 0, cs = 0, ss = 0, cr = 0, sr = 0;
    for (int i = 0; i < q; i++) {
        double c = cos(g * u[i]), s = sin(g * u[i]);
        cc += c*c; cs += c*s; ss += s*s; cr += c*res[i]; sr += s*res[i];
    }
    double det = cc*ss - cs*cs;
    return det < 1e-12 ? 0 : hypot(ss*cr - cs*sr, cc*sr - cs*cr) / det;
}

// test a gamma fija: quita constante + bulto (g0, 2g0, 3g0) y mide la amplitud en
// cada objetivo, y el maximo local en [g-0.5, g+0.5]
#define NT 4
static const double targets[NT] = {5.6, 12.0, 9.0, 15.0};

// centra u en uc y deja en res el residuo sin constante ni bulto (g0, 2g0, 3g0)
static void debump(const double *u, const double *y, int q, double *uc, double *res) {
    double umin = u[0], umax = u[q-1], W = umax - umin, mean = 0;
    for (int i = 0; i < q; i++) { uc[i] = u[i] - 0.5 * (umin + umax); mean += y[i]; }
    mean /= q;
    for (int i = 0; i < q; i++) res[i] = y[i] - mean;
    double g0 = M_PI / W, best = -1;
    for (double t = M_PI / W; t <= 2.0; t += M_PI / (16 * W)) {
        double v = gain(uc, res, q, t);
        if (v > best) { best = v; g0 = t; }
    }
    int n = 7;
    double M[49] = {0}, b[7] = {0}, col[7];
    for (int i = 0; i < q; i++) {
        col[0] = 1;
        for (int h = 1; h <= 3; h++) { col[2*h-1] = cos(h*g0*uc[i]); col[2*h] = sin(h*g0*uc[i]); }
        for (int r = 0; r < n; r++) { b[r] += col[r] * y[i]; for (int c = 0; c < n; c++) M[r*n+c] += col[r] * col[c]; }
    }
    solve(n, M, b);
    for (int i = 0; i < q; i++) {
        double f = b[0];
        for (int h = 1; h <= 3; h++) f += b[2*h-1]*cos(h*g0*uc[i]) + b[2*h]*sin(h*g0*uc[i]);
        res[i] = y[i] - f;
    }
}

static void fixedtest(const double *u, const double *y, int q, double *a, double *gl, double *al) {
    double *uc = malloc(sizeof(double) * q), *res = malloc(sizeof(double) * q);
    debump(u, y, q, uc, res);
    for (int t = 0; t < NT; t++) {
        a[t] = ampat(uc, res, q, targets[t]);
        al[t] = -1;
        for (double g = targets[t] - 0.5; g <= targets[t] + 0.5; g += 0.01) {
            double v = ampat(uc, res, q, g);
            if (v > al[t]) { al[t] = v; gl[t] = g; }
        }
    }
    free(uc); free(res);
}

static void runfixed(void) {
    int nnull = nsol < 5000 ? 100 : 30;
    pt *p = malloc(sizeof(pt) * nsol);
    double *u = malloc(sizeof(double) * nsol), *y = malloc(sizeof(double) * nsol);
    double aS[NT], glS[NT], alS[NT], a[NT], gl[NT], al[NT];
    double m[NT] = {0}, m2[NT] = {0}, ml[NT] = {0}, ml2[NT] = {0};
    mode = 1;
    for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
    build(p, nsol, u, y);
    fixedtest(u, y, nsol, aS, glS, alS);
    unsigned char c[MAXN];
    for (int t = 0; t < nnull; t++) {
        do {
            for (int i = 0; i < nsol; i++) {
                for (int r = 0; r < N; r++) c[r] = r + 1;
                for (int r = N - 1; r > 0; r--) { int j = xr() % (r + 1); unsigned char tt = c[r]; c[r] = c[j]; c[j] = tt; }
                p[i].u = key(c); p[i].rk = rankof(c);
            }
        } while (!build(p, nsol, u, y));
        fixedtest(u, y, nsol, a, gl, al);
        for (int k = 0; k < NT; k++) { m[k] += a[k]; m2[k] += a[k]*a[k]; ml[k] += al[k]; ml2[k] += al[k]*al[k]; }
    }
    printf("\n==== N=%d  Q=%d  sqrtQ=%.1f  nulo=%d\n", N, nsol, sqrt(nsol), nnull);
    printf("   gamma  |  amp(sol)  nulo±sd     z   |  max local: gamma  amp   nulo±sd     z\n");
    for (int k = 0; k < NT; k++) {
        double mu = m[k] / nnull, sd = sqrt(fmax(m2[k] / nnull - mu*mu, 1e-12));
        double mul = ml[k] / nnull, sdl = sqrt(fmax(ml2[k] / nnull - mul*mul, 1e-12));
        printf("  %5.1f%s|  %7.1f  %6.1f±%-5.1f %5.1f  |  %6.2f  %7.1f  %6.1f±%-5.1f %5.1f\n",
               targets[k], k >= 2 ? "c " : "  ", aS[k], mu, sd, (aS[k] - mu) / sd, glS[k], alS[k], mul, sdl, (alS[k] - mul) / sdl);
    }
    fflush(stdout);
    free(p); free(u); free(y);
}

// espectro: amplitud(sol)/media(nulo) en gamma = 1..40, tras quitar el bulto
#define NSPEC 391
static void spectrum(const double *u, const double *y, int q, double *a) {
    double *uc = malloc(sizeof(double) * q), *res = malloc(sizeof(double) * q);
    debump(u, y, q, uc, res);
    for (int s = 0; s < NSPEC; s++) a[s] = ampat(uc, res, q, 1.0 + 0.1 * s);
    free(uc); free(res);
}

static void runspec(void) {
    int nnull = nsol < 5000 ? 40 : 10;
    pt *p = malloc(sizeof(pt) * nsol);
    double *u = malloc(sizeof(double) * nsol), *y = malloc(sizeof(double) * nsol);
    static double sS[NSPEC], sN[NSPEC];
    mode = 1;
    for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
    unsigned char c[MAXN];
    memset(sN, 0, sizeof sN);
    for (int t = -1; t < nnull; t++) {
        if (t >= 0) do {
            for (int i = 0; i < nsol; i++) {
                for (int r = 0; r < N; r++) c[r] = r + 1;
                for (int r = N - 1; r > 0; r--) { int j = xr() % (r + 1); unsigned char tt = c[r]; c[r] = c[j]; c[j] = tt; }
                p[i].u = key(c); p[i].rk = rankof(c);
            }
        } while (!build(p, nsol, u, y));
        else build(p, nsol, u, y);
        static double a[NSPEC];
        spectrum(u, y, nsol, a);
        for (int s = 0; s < NSPEC; s++) { if (t < 0) sS[s] = a[s]; else sN[s] += a[s] / nnull; }
    }
    printf("\n==== N=%d  Q=%d  espectro sol/nulo (gamma: cociente), picos locales con cociente > 3\n", N, nsol);
    for (int s = 1; s < NSPEC - 1; s++) {
        double r0 = sS[s-1] / sN[s-1], r1 = sS[s] / sN[s], r2 = sS[s+1] / sN[s+1];
        if (r1 > 3 && r1 >= r0 && r1 >= r2) printf("  g=%5.1f  amp %8.1f  nulo %6.2f  cociente %6.1f\n", 1.0 + 0.1 * s, sS[s], sN[s], r1);
    }
    double tot = 0; for (int s = 0; s < NSPEC; s++) tot += sS[s] / sN[s];
    printf("  cociente medio en 1..40: %.2f\n", tot / NSPEC);
    fflush(stdout);
    free(p); free(u); free(y);
}

// conjunto sintetico: w[celda] puntos por prefijo de d cifras (redondeo estocastico),
// el resto de cifras es una permutacion al azar de los simbolos que quedan
static int fillset(const double *w, int d, pt *p, int capp) {
    int ncell = 1, q = 0;
    for (int k = 0; k < d; k++) ncell *= N;
    unsigned char c[MAXN];
    for (int cell = 0; cell < ncell; cell++) {
        if (w[cell] <= 0) continue;
        int cnt = (int)floor(w[cell] + (xr() >> 11) * (1.0 / 9007199254740992.0));
        int pre[3], t = cell, ok = 1;
        for (int k = d - 1; k >= 0; k--) { pre[k] = t % N + 1; t /= N; }
        for (int k = 0; k < d; k++) for (int j = 0; j < k; j++) ok &= pre[j] != pre[k];
        if (!ok) continue;
        for (int m = 0; m < cnt && q < capp; m++) {
            int rest[MAXN], nr = 0;
            for (int v = 1; v <= N; v++) { int used2 = 0; for (int k = 0; k < d; k++) used2 |= pre[k] == v; if (!used2) rest[nr++] = v; }
            for (int r = nr - 1; r > 0; r--) { int j = xr() % (r + 1); int tt = rest[r]; rest[r] = rest[j]; rest[j] = tt; }
            for (int k = 0; k < d; k++) c[k] = pre[k];
            for (int k = 0; k < nr; k++) c[d + k] = rest[k];
            p[q].u = key(c); p[q].rk = rankof(c); q++;
        }
    }
    return q;
}

// explica el espectro de lineas: soluciones frente a modelos de prefijo
//   P3 : conteos exactos de soluciones por (c1,c2,c3)
//   P2 : conteos exactos por (c1,c2)
//   MB : marginales f1(c1) f2(c2) por la franja permitida |c1-c2|>1
//   FR : solo la franja, uniforme
#define NMOD 4
#define NFILL 4
static void runprefix(void) {
    static const char *names[NMOD] = {"P3 (c1,c2,c3)", "P2 (c1,c2)", "MB marg x franja", "FR franja sola"};
    mode = 1;
    int capp = 2 * nsol + 1000;
    pt *p = malloc(sizeof(pt) * capp);
    double *u = malloc(sizeof(double) * capp), *y = malloc(sizeof(double) * capp);
    static double sS[NSPEC], sM[NMOD][NSPEC], a[NSPEC];
    for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
    build(p, nsol, u, y);
    spectrum(u, y, nsol, sS);

    int n2 = N * N, n3 = N * N * N;
    double *w3 = calloc(n3, sizeof(double)), *w2 = calloc(n2, sizeof(double));
    double *wb = calloc(n2, sizeof(double)), *wf = calloc(n2, sizeof(double)), f1[MAXN] = {0}, f2[MAXN] = {0};
    for (int i = 0; i < nsol; i++) {
        const unsigned char *s = sols + (size_t)i * N;
        w3[((s[0]-1) * N + s[1]-1) * N + s[2]-1] += 1;
        w2[(s[0]-1) * N + s[1]-1] += 1;
        f1[s[0]-1] += 1; f2[s[1]-1] += 1;
    }
    double zb = 0, zf = 0;
    for (int a1 = 0; a1 < N; a1++) for (int a2 = 0; a2 < N; a2++)
        if (abs(a1 - a2) > 1) { wb[a1*N+a2] = f1[a1] * f2[a2]; zb += wb[a1*N+a2]; wf[a1*N+a2] = 1; zf += 1; }
    for (int k = 0; k < n2; k++) { wb[k] *= nsol / zb; wf[k] *= nsol / zf; }
    const double *ws[NMOD] = {w3, w2, wb, wf};
    const int ds[NMOD] = {3, 2, 2, 2};

    memset(sM, 0, sizeof sM);
    for (int m = 0; m < NMOD; m++)
        for (int f = 0; f < NFILL; f++) {
            int q;
            do q = fillset(ws[m], ds[m], p, capp); while (!build(p, q, u, y));
            spectrum(u, y, q, a);
            for (int s = 0; s < NSPEC; s++) sM[m][s] += a[s] / NFILL;
        }

    // picos de las soluciones en gamma >= 3: maximos locales, los 8 mayores
    int pk[NSPEC], np = 0;
    for (int s = 21; s < NSPEC - 1; s++) if (sS[s] >= sS[s-1] && sS[s] >= sS[s+1]) pk[np++] = s;
    for (int i = 0; i < np; i++) for (int j = i + 1; j < np; j++) if (sS[pk[j]] > sS[pk[i]]) { int t = pk[i]; pk[i] = pk[j]; pk[j] = t; }
    if (np > 8) np = 8;
    for (int i = 0; i < np; i++) for (int j = i + 1; j < np; j++) if (pk[j] < pk[i]) { int t = pk[i]; pk[i] = pk[j]; pk[j] = t; }

    printf("\n==== N=%d  Q=%d   amplitud del modelo / amplitud de las soluciones en sus picos\n", N, nsol);
    printf("  %-18s", "gamma:");
    for (int i = 0; i < np; i++) printf(" %6.1f", 1.0 + 0.1 * pk[i]);
    printf("    corr(log) 3..40\n  %-18s", "sol (amp)");
    for (int i = 0; i < np; i++) printf(" %6.0f", sS[pk[i]]);
    printf("\n");
    for (int m = 0; m < NMOD; m++) {
        printf("  %-18s", names[m]);
        for (int i = 0; i < np; i++) printf(" %6.2f", sM[m][pk[i]] / sS[pk[i]]);
        double mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0; int n = 0;
        for (int s = 20; s < NSPEC; s++) { double x = log(sS[s] + 1e-9), yy = log(sM[m][s] + 1e-9); mx += x; my += yy; n++; }
        mx /= n; my /= n;
        for (int s = 20; s < NSPEC; s++) { double x = log(sS[s] + 1e-9) - mx, yy = log(sM[m][s] + 1e-9) - my; sxx += x*x; syy += yy*yy; sxy += x*yy; }
        printf("    %.3f\n", sxy / sqrt(sxx * syy));
    }
    // picos propios de la franja sola
    // perfil por distancia: n(c1,c1+delta) / modelo MB, sumado sobre c1
    printf("  n/MB por delta=c2-c1:");
    for (int dl = -(N - 1); dl <= N - 1; dl++) {
        double sn = 0, sb = 0;
        for (int a1 = 0; a1 < N; a1++) { int a2 = a1 + dl; if (a2 < 0 || a2 >= N) continue; sn += w2[a1*N+a2]; sb += wb[a1*N+a2]; }
        if (sb > 0) printf(" %+d:%.2f", dl, sn / sb);
    }
    printf("\n");
    double fmx = 0;
    for (int s = 20; s < NSPEC; s++) fmx = fmax(fmx, sM[3][s]);
    printf("  picos de FR (> 30%% del maximo):");
    for (int s = 21; s < NSPEC - 1; s++)
        if (sM[3][s] >= sM[3][s-1] && sM[3][s] >= sM[3][s+1] && sM[3][s] > 0.3 * fmx) printf(" %.1f", 1.0 + 0.1 * s);
    printf("\n");
    fflush(stdout);
    free(w3); free(w2); free(wb); free(wf); free(p); free(u); free(y);
}

// multiescala: nivel k = estructura de la fila k dado el prefijo c1..c_{k-1}.
// Coordenada local t = c_k + c_{k+1}/(N+1) + ... (la cola en base N+1), residuo por
// celda y = (j+1/2) - n_p (rango_cola+1/2)/L!, con L = N-k+1; si las filas >= k fueran
// una permutacion al azar de los simbolos restantes, y seria puro ruido.  Nulo: misma
// celda con la cola barajada (= modelo P_{k-1} visto en su escala).
static double lehmerL(const unsigned char *c, int L) {
    double rk = 0;
    for (int i = 0; i < L; i++) {
        int less = 0;
        for (int j = i + 1; j < L; j++) less += c[j] < c[i];
        rk = rk * (L - i) + less;
    }
    return rk;
}

typedef struct { double t, rk; } tl;
static int cmptl(const void *a, const void *b) { const tl *x = a, *y = b; return (x->t > y->t) - (x->t < y->t); }

// residuo local de todas las celdas para el nivel k; shuffle=1 baraja las colas
static void levelres(int k, int shuffle, double *T, double *Y) {
    int L = N - k + 1;
    double Lf = 1; for (int i = 2; i <= L; i++) Lf *= i;
    tl *cell = malloc(sizeof(tl) * nsol);
    unsigned char tail[MAXN];
    int i0 = 0;
    while (i0 < nsol) {
        int i1 = i0 + 1;   // las soluciones salen en orden lexicografico: celdas contiguas
        while (i1 < nsol && !memcmp(sols + (size_t)i1 * N, sols + (size_t)i0 * N, k - 1)) i1++;
        int n = i1 - i0;
        for (int i = i0; i < i1; i++) {
            memcpy(tail, sols + (size_t)i * N + k - 1, L);
            if (shuffle) for (int r = L - 1; r > 0; r--) { int j = xr() % (r + 1); unsigned char tt = tail[r]; tail[r] = tail[j]; tail[j] = tt; }
            double t = 0;
            for (int j = L - 1; j >= 0; j--) t = t / (N + 1) + tail[j];
            cell[i - i0].t = t; cell[i - i0].rk = lehmerL(tail, L);
        }
        qsort(cell, n, sizeof(tl), cmptl);
        for (int j = 0; j < n; j++) {
            T[i0 + j] = cell[j].t - 0.5 * (N + 2);
            Y[i0 + j] = (j + 0.5) - n * (cell[j].rk + 0.5) / Lf;
        }
        i0 = i1;
    }
    free(cell);
}

#define NSP2 371   // gamma = 3.0 .. 40.0
static void levelspec(const double *T, const double *Y, double *a, double *var) {
    double v = 0;
    for (int i = 0; i < nsol; i++) v += Y[i] * Y[i];
    *var = v;
    for (int s = 0; s < NSP2; s++) a[s] = ampat(T, Y, nsol, 3.0 + 0.1 * s);
}

static void runmulti(void) {
    int nnull = nsol < 20000 ? 10 : 4;
    double *T = malloc(sizeof(double) * nsol), *Y = malloc(sizeof(double) * nsol);
    static double aS[NSP2], aN[NSP2], a[NSP2];
    printf("\n==== N=%d  Q=%d  nulo=%d   (var y espectro: sol / cola barajada)\n", N, nsol, nnull);
    printf("   k  celdas  var    cociente medio  max (gamma)    picos (cociente>3)\n");
    for (int k = 1; k <= N - 2; k++) {
        double vS, vN = 0, v;
        levelres(k, 0, T, Y);
        levelspec(T, Y, aS, &vS);
        memset(aN, 0, sizeof aN);
        for (int t = 0; t < nnull; t++) {
            levelres(k, 1, T, Y);
            levelspec(T, Y, a, &v);
            vN += v / nnull;
            for (int s = 0; s < NSP2; s++) aN[s] += a[s] / nnull;
        }
        int ncell = 1;
        for (int i = 1; i < nsol; i++) ncell += memcmp(sols + (size_t)i * N, sols + (size_t)(i - 1) * N, k - 1) != 0;
        double mr = 0, mx = 0; int smx = 0;
        for (int s = 0; s < NSP2; s++) { double r = aS[s] / aN[s]; mr += r / NSP2; if (r > mx) { mx = r; smx = s; } }
        printf("  %2d  %6d  %5.2f  %8.2f       %6.1f (%4.1f)  ", k, ncell, vS / vN, mr, mx, 3.0 + 0.1 * smx);
        for (int s = 1; s < NSP2 - 1; s++) {
            double r = aS[s] / aN[s];
            if (r > 3 && r >= aS[s-1] / aN[s-1] && r >= aS[s+1] / aN[s+1]) printf(" %.1f", 3.0 + 0.1 * s);
        }
        printf("\n");
        fflush(stdout);
    }
    free(T); free(Y);
}

static int cmpperm(const void *a, const void *b) { return memcmp(a, b, N); }

// conteo predicho conociendo las d primeras cifras: celdas de prefijo exactas, colas al
// azar.  A_d(x_i) = (soluciones en celdas anteriores) + n_celda (rango_cola+1/2)/L!.
// d = 0 es la tendencia r P(x); d = N es el conteo exacto i + 1/2.
static void predicted(int d, double *A) {
    int L = N - d;
    double Lf = 1; for (int i = 2; i <= L; i++) Lf *= i;
    int i0 = 0;
    while (i0 < nsol) {
        int i1 = i0 + 1;
        while (i1 < nsol && !memcmp(sols + (size_t)i1 * N, sols + (size_t)i0 * N, d)) i1++;
        for (int i = i0; i < i1; i++)
            A[i] = i0 + (i1 - i0) * (lehmerL(sols + (size_t)i * N + d, L) + 0.5) / Lf;
        i0 = i1;
    }
}

// reajusta por LS constante + las K primeras ondas de g (el orden del pursuit voraz)
// y devuelve el residuo predicho en f
static void fitwith(const double *u, const double *y, int q, const double *g, int K, double *f, double *coef) {
    double mid = 0.5 * (u[0] + u[q-1]);
    int n = 2 * K + 1;
    double *M = calloc((size_t)n * n, sizeof(double)), *b = calloc(n, sizeof(double)), col[2*KMAX+1];
    for (int i = 0; i < q; i++) {
        col[0] = 1;
        for (int j = 0; j < K; j++) { col[1+2*j] = cos(g[j]*(u[i]-mid)); col[2+2*j] = sin(g[j]*(u[i]-mid)); }
        for (int r = 0; r < n; r++) { b[r] += col[r] * y[i]; for (int c = 0; c < n; c++) M[r*n+c] += col[r] * col[c]; }
    }
    for (int r = 0; r < n; r++) M[r*n+r] += 1e-9;
    solve(n, M, b);
    for (int i = 0; i < q; i++) {
        f[i] = b[0];
        for (int j = 0; j < K; j++) f[i] += b[1+2*j]*cos(g[j]*(u[i]-mid)) + b[2+2*j]*sin(g[j]*(u[i]-mid));
    }
    if (coef) memcpy(coef, b, sizeof(double) * n);
    free(M); free(b);
}

int main(int argc, char **argv) {
    // parametros de la formula con una sola onda: A(x) = Q (P(x)+1/2)/N! + c + R cos(g (u - m) - phi)
    if (argc > 1 && !strcmp(argv[1], "onda1")) {
        int n0 = argc > 2 ? atoi(argv[2]) : 8, n1 = argc > 3 ? atoi(argv[3]) : 14;
        printf("N,Q,m,gamma,periodo,c_sobre_Q,R_sobre_Q,phi,rms_tendencia,rms_1onda\n");
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            fact = 1; for (int i = 2; i <= N; i++) fact *= i;
            mode = 1;
            pt *p = malloc(sizeof(pt) * nsol);
            double *u = malloc(sizeof(double) * nsol), *y = malloc(sizeof(double) * nsol), *f = malloc(sizeof(double) * nsol);
            double R2[1], g[1], amp[1], cf[3], s0 = 0, s1 = 0;
            for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
            build(p, nsol, u, y);
            pursuit(u, y, nsol, 1, R2, g, amp);
            fitwith(u, y, nsol, g, 1, f, cf);
            for (int i = 0; i < nsol; i++) { s0 += y[i] * y[i]; s1 += (y[i] - f[i]) * (y[i] - f[i]); }
            printf("%d,%d,%.4f,%.4f,%.3f,%.4f,%.4f,%.3f,%.4g,%.4g\n", N, nsol, 0.5 * (u[0] + u[nsol-1]), g[0], 2 * M_PI / g[0],
                   cf[0] / nsol, hypot(cf[1], cf[2]) / nsol, atan2(cf[2], cf[1]), sqrt(s0 / nsol), sqrt(s1 / nsol));
            free(p); free(u); free(y); free(f);
        }
        return 0;
    }
    // conteo predicho por tendencia + K ondas buscadas, en cada solucion (para graficar)
    if (argc > 1 && !strcmp(argv[1], "wavecurves")) {
        static const int Ks[4] = {1, 4, 12, 40};
        N = argc > 2 ? atoi(argv[2]) : 13;
        queens(0, 0, 0, 0);
        fact = 1; for (int i = 2; i <= N; i++) fact *= i;
        mode = 1;
        pt *p = malloc(sizeof(pt) * nsol);
        double *u = malloc(sizeof(double) * nsol), *y = malloc(sizeof(double) * nsol);
        double R2[KMAX], g[KMAX], amp[KMAX], *f[4];
        for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
        build(p, nsol, u, y);
        pursuit(u, y, nsol, KMAX, R2, g, amp);
        for (int k = 0; k < 4; k++) { f[k] = malloc(sizeof(double) * nsol); fitwith(u, y, nsol, g, Ks[k], f[k], NULL); }
        printf("x,real,trend,w1,w4,w12,w40\n");
        for (int i = 0; i < nsol; i++) {
            double tr = (i + 0.5) - y[i];   // y = conteo - tendencia
            printf("%.10f,%.1f,%.4f", u[i], i + 0.5, tr);
            for (int k = 0; k < 4; k++) printf(",%.4f", tr + f[k][i]);
            printf("\n");
        }
        return 0;
    }
    // error del conteo con el experimento inicial: tendencia + K ondas buscadas
    if (argc > 1 && !strcmp(argv[1], "waves")) {
        int n0 = argc > 2 ? atoi(argv[2]) : 10, n1 = argc > 3 ? atoi(argv[3]) : 13;
        printf("N,K,param,rms\n");
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            fact = 1; for (int i = 2; i <= N; i++) fact *= i;
            mode = 1;
            pt *p = malloc(sizeof(pt) * nsol);
            double *u = malloc(sizeof(double) * nsol), *y = malloc(sizeof(double) * nsol);
            double R2[KMAX], g[KMAX], amp[KMAX], mean = 0, sst = 0;
            for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
            build(p, nsol, u, y);
            for (int i = 0; i < nsol; i++) mean += y[i];
            mean /= nsol;
            for (int i = 0; i < nsol; i++) sst += (y[i] - mean) * (y[i] - mean);
            printf("%d,0,1,%.6g\n", N, sqrt(sst / nsol));
            pursuit(u, y, nsol, KMAX, R2, g, amp);
            // cada onda: frecuencia + amplitud + fase; mas la constante
            for (int k = 0; k < KMAX; k++) printf("%d,%d,%d,%.6g\n", N, k + 1, 3 * (k + 1) + 1, sqrt(fmax(1 - R2[k], 0) * sst / nsol));
            fflush(stdout);
            free(p); free(u); free(y);
        }
        return 0;
    }
    if (argc > 1 && (!strcmp(argv[1], "curves") || !strcmp(argv[1], "rms"))) {
        int csv = !strcmp(argv[1], "curves");
        int n0 = argc > 2 ? atoi(argv[2]) : 10, n1 = csv ? n0 : (argc > 3 ? atoi(argv[3]) : 14);
        mode = 1;
        if (!csv) printf("N,d,rms,rms_azar\n");
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            double **A = malloc(sizeof(double *) * (N + 1));
            for (int d = 0; d <= N; d++) { A[d] = malloc(sizeof(double) * nsol); predicted(d, A[d]); }
            if (csv) {
                printf("x,real");
                for (int d = 0; d <= N; d++) printf(",d%d", d);
                printf("\n");
                for (int i = 0; i < nsol; i++) {
                    printf("%.10f,%.1f", key(sols + (size_t)i * N), i + 0.5);
                    for (int d = 0; d <= N; d++) printf(",%.4f", A[d][i]);
                    printf("\n");
                }
            } else {
                // nulo: Q permutaciones al azar, misma formula (misma reduccion del espacio)
                int nrep = nsol < 20000 ? 5 : 2;
                double *nul = calloc(N + 1, sizeof(double)), *tmp = malloc(sizeof(double) * nsol);
                unsigned char *keep = sols, *rnd = malloc((size_t)nsol * N);
                for (int t = 0; t < nrep; t++) {
                    for (int i = 0; i < nsol; i++) {
                        unsigned char *c = rnd + (size_t)i * N;
                        for (int r = 0; r < N; r++) c[r] = r + 1;
                        for (int r = N - 1; r > 0; r--) { int j = xr() % (r + 1); unsigned char tt = c[r]; c[r] = c[j]; c[j] = tt; }
                    }
                    qsort(rnd, nsol, N, cmpperm);
                    sols = rnd;
                    for (int d = 0; d <= N; d++) {
                        predicted(d, tmp);
                        double s = 0;
                        for (int i = 0; i < nsol; i++) s += (i + 0.5 - tmp[i]) * (i + 0.5 - tmp[i]);
                        nul[d] += sqrt(s / nsol) / nrep;
                    }
                    sols = keep;
                }
                for (int d = 0; d <= N; d++) {
                    double s = 0;
                    for (int i = 0; i < nsol; i++) s += (i + 0.5 - A[d][i]) * (i + 0.5 - A[d][i]);
                    printf("%d,%d,%.6g,%.6g\n", N, d, sqrt(s / nsol), nul[d]);
                }
                free(nul); free(tmp); free(rnd);
            }
            for (int d = 0; d <= N; d++) free(A[d]);
            free(A);
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "multi")) {
        int n0 = argc > 2 ? atoi(argv[2]) : 10, n1 = argc > 3 ? atoi(argv[3]) : 13;
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            fact = 1; for (int i = 2; i <= N; i++) fact *= i;
            runmulti();
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "prefix")) {
        int n0 = argc > 2 ? atoi(argv[2]) : 10, n1 = argc > 3 ? atoi(argv[3]) : 13;
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            fact = 1; for (int i = 2; i <= N; i++) fact *= i;
            runprefix();
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "spec")) {
        int n0 = argc > 2 ? atoi(argv[2]) : 8, n1 = argc > 3 ? atoi(argv[3]) : 13;
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            fact = 1; for (int i = 2; i <= N; i++) fact *= i;
            runspec();
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "fixed")) {
        int n0 = argc > 2 ? atoi(argv[2]) : 8, n1 = argc > 3 ? atoi(argv[3]) : 14;
        printf("test a gamma fija, base N+1 lineal (c = control)\n");
        for (N = n0; N <= n1; N++) {
            free(sols); sols = NULL; cap = nsol = 0;
            queens(0, 0, 0, 0);
            fact = 1; for (int i = 2; i <= N; i++) fact *= i;
            runfixed();
        }
        return 0;
    }
    int godel = argc > 1 && !strcmp(argv[1], "godel");
    int n0 = argc > 2 ? atoi(argv[2]) : 7, n1 = argc > 3 ? atoi(argv[3]) : 10;
    static const int primes[MAXN] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
    for (int r = 0; r < MAXN; r++) logp[r] = log(primes[r]);

    for (N = n0; N <= n1; N++) {
        free(sols); sols = NULL; cap = nsol = 0;   // el paso de fila cambia con N
        queens(0, 0, 0, 0);
        fact = 1; for (int i = 2; i <= N; i++) fact *= i;
        int K = nsol / 5 < 12 ? nsol / 5 : 12;
        int nnull = nsol < 5000 ? 60 : 20;
        if (godel) {
            gkeys = malloc(sizeof(double) * (size_t)fact); gi = 0;
            memset(used, 0, sizeof used); allperm(0, 0);
            qsort(gkeys, gi, sizeof(double), cmpd);
        }
        printf("\n==== N=%d  Q=%d  nulo=%d\n", N, nsol, nnull);

        for (mode = godel ? 0 : 1; mode <= (godel ? 0 : 2); mode++) {
            pt *p = malloc(sizeof(pt) * nsol);
            double *u = malloc(sizeof(double) * nsol), *y = malloc(sizeof(double) * nsol);
            double R2S[KMAX], gS[KMAX], aS[KMAX], R2[KMAX], g[KMAX], amp[KMAX];
            for (int i = 0; i < nsol; i++) { p[i].u = key(sols + (size_t)i * N); p[i].rk = rankof(sols + (size_t)i * N); }
            build(p, nsol, u, y);
            pursuit(u, y, nsol, K, R2S, gS, aS);

            double m[KMAX] = {0}, m2[KMAX] = {0};
            unsigned char c[MAXN];
            for (int t = 0; t < nnull; t++) {
                do {
                    for (int i = 0; i < nsol; i++) {
                        for (int r = 0; r < N; r++) c[r] = r + 1;
                        for (int r = N - 1; r > 0; r--) { int j = xr() % (r + 1); unsigned char tt = c[r]; c[r] = c[j]; c[j] = tt; }
                        p[i].u = key(c); p[i].rk = rankof(c);
                    }
                } while (!build(p, nsol, u, y));
                pursuit(u, y, nsol, K, R2, g, amp);
                for (int k = 0; k < K; k++) { m[k] += R2[k]; m2[k] += R2[k] * R2[k]; }
            }

            printf("-- %s\n   K   R2(sol)  R2(nulo)±sd     z\n",
                   mode == 0 ? "godel, u = log n" : mode == 1 ? "base N+1, u = x/(N+1)^(N-1)" : "base N+1, u = log x");
            for (int k = 0; k < K; k++) {
                if (!(k < 4 || k == 7 || k == K - 1)) continue;
                double mu = m[k] / nnull, sd = sqrt(fmax(m2[k] / nnull - mu * mu, 1e-12));
                printf("  %2d   %.3f    %.3f±%.3f  %6.1f\n", k + 1, R2S[k], mu, sd, (R2S[k] - mu) / sd);
            }
            printf("   ondas (sol): ");
            for (int k = 0; k < (K < 5 ? K : 5); k++) printf("g=%.2f/amp %.1f/periodo %.3f  ", gS[k], aS[k], 2 * M_PI / gS[k]);
            printf("\n");
            fflush(stdout);
            free(p); free(u); free(y);
        }
        if (godel) free(gkeys);
    }
    return 0;
}
