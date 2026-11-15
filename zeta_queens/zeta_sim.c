// zeta_sim.c — donde caen las 8 simetrias del cuadrado sobre la curva de conteo A(x).
// Las soluciones se ordenan como numeros (lexicografico); para cada simetria g se mide
// a que posicion va cada solucion, y que tramo de la curva hace falta recorrer para ver
// al menos un representante de cada orbita (y asi obtener Q sin conocerlo).
//
// cc -O2 -o zeta_sim zeta_sim.c -lm && ./zeta_sim 8 15
// ./zeta_sim dump 12 > sim_12.csv
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXN 16
#define NG 8
static const char *gname[NG] = {"id", "V", "H", "R180", "T", "R90", "R270", "A"};

static int N, nsol, cap;
static uint64_t *sols;             // 4 bits por cifra: el orden numerico es el lexicografico
static int cur[MAXN];

static void queens(int r, unsigned cols, unsigned d1, unsigned d2) {
    if (r == N) {
        if (nsol == cap) { cap = cap ? 2 * cap : 1024; sols = realloc(sols, sizeof(uint64_t) * cap); }
        uint64_t v = 0;
        for (int i = 0; i < N; i++) v = v << 4 | cur[i];
        sols[nsol++] = v;
        return;
    }
    unsigned free_ = ~(cols | d1 | d2) & ((1u << N) - 1);
    while (free_) {
        unsigned b = free_ & -free_; free_ ^= b;
        cur[r] = __builtin_ctz(b);
        queens(r + 1, cols | b, (d1 | b) << 1, (d2 | b) >> 1);
    }
}

static void unpack(uint64_t v, int *c) { for (int i = N - 1; i >= 0; i--) { c[i] = v & 15; v >>= 4; } }
static uint64_t pack(const int *c) { uint64_t v = 0; for (int i = 0; i < N; i++) v = v << 4 | c[i]; return v; }

// columnas y filas 0..N-1; c[r] = columna de la reina de la fila r
static void apply(int g, const int *c, int *d) {
    int m = N - 1;
    for (int r = 0; r < N; r++)
        switch (g) {
            case 0: d[r] = c[r]; break;
            case 1: d[r] = m - c[r]; break;          // V: espejo de columnas
            case 2: d[r] = c[m - r]; break;          // H: filas al reves
            case 3: d[r] = m - c[m - r]; break;      // R180 = V H
            case 4: d[c[r]] = r; break;              // T: traspuesta (permutacion inversa)
            case 5: d[c[r]] = m - r; break;          // R90 = V T
            case 6: d[m - c[r]] = r; break;          // R270 = H T
            case 7: d[m - c[r]] = m - r; break;      // A: antitraspuesta = R180 T
        }
}

static int find(uint64_t v) {
    int lo = 0, hi = nsol - 1;
    while (lo < hi) { int m = (lo + hi) / 2; if (sols[m] < v) lo = m + 1; else hi = m; }
    return lo;
}

static double pearson(const double *a, const double *b, int n) {
    double ma = 0, mb = 0, sa = 0, sb = 0, sab = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    for (int i = 0; i < n; i++) { sa += (a[i]-ma)*(a[i]-ma); sb += (b[i]-mb)*(b[i]-mb); sab += (a[i]-ma)*(b[i]-mb); }
    return sab / sqrt(sa * sb);
}

static double lehmer(const int *c) {
    double rk = 0;
    for (int i = 0; i < N; i++) {
        int less = 0;
        for (int j = i + 1; j < N; j++) less += c[j] < c[i];
        rk = rk * (N - i) + less;
    }
    return rk;
}

static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

int main(int argc, char **argv) {
    int dump = argc > 1 && !strcmp(argv[1], "dump"), cov = argc > 1 && !strcmp(argv[1], "cov");
    int n0 = atoi(argv[dump || cov ? 2 : 1]), n1 = dump ? n0 : atoi(argv[cov ? 3 : 2]);
    if (cov) printf("N,cifra,frac_curva,frac_Q\n");
    if (dump) { printf("i"); for (int g = 1; g < NG; g++) printf(",%s", gname[g]); printf(",rep\n"); }
    else if (!cov) printf("N,Q,T2,T4,T8,x_ult,frac_curva_ult,frac99,corr: V H R180 T R90 R270 A | nulo H T R90\n");

    for (N = n0; N <= n1; N++) {
        nsol = cap = 0; free(sols); sols = NULL;
        queens(0, 0, 0, 0);          // sale en orden lexicografico
        int *img = malloc(sizeof(int) * (size_t)nsol * NG), c[MAXN], d[MAXN];
        char *isrep = calloc(nsol, 1);
        long orb[9] = {0};
        for (int i = 0; i < nsol; i++) {
            unpack(sols[i], c);
            int mn = i, seen[NG], ns = 0;
            for (int g = 0; g < NG; g++) {
                apply(g, c, d);
                int j = find(pack(d));
                if (sols[j] != pack(d)) { fprintf(stderr, "imagen no es solucion N=%d g=%d\n", N, g); return 1; }
                img[(size_t)i * NG + g] = j;
                if (j < mn) mn = j;
                int nw = 1;
                for (int k = 0; k < ns; k++) nw &= seen[k] != j;
                if (nw) seen[ns++] = j;
            }
            if (mn == i) { isrep[i] = 1; orb[ns]++; }
        }
        if (cov) {
            // recorrer las soluciones con primera cifra <= k: cada una aporta su orbita entera
            for (int k = 1; k <= N; k++) {
                long sum = 0; int upto = 0;
                for (int i = 0; i < nsol; i++) {
                    unpack(sols[i], c);
                    if (c[0] + 1 > k) break;
                    upto = i + 1;
                    if (isrep[i]) {
                        int ns = 0, seen[NG];
                        for (int g = 0; g < NG; g++) {
                            int j = img[(size_t)i * NG + g], nw = 1;
                            for (int q = 0; q < ns; q++) nw &= seen[q] != j;
                            if (nw) seen[ns++] = j;
                        }
                        sum += ns;
                    }
                }
                printf("%d,%d,%.5f,%.6f\n", N, k, (double)upto / nsol, (double)sum / nsol);
            }
        } else if (dump) {
            for (int i = 0; i < nsol; i++) {
                printf("%d", i);
                for (int g = 1; g < NG; g++) printf(",%d", img[(size_t)i * NG + g]);
                printf(",%d\n", isrep[i]);
            }
        } else {
            // representantes: minimo de cada orbita; ultimo y el percentil 99
            long tot = 0, acc = 0, sumsz = 0;
            int last = 0;
            for (int i = 0; i < nsol; i++) if (isrep[i]) { tot++; last = i; }
            int p99 = 0;
            for (int i = 0; i < nsol; i++) if (isrep[i]) { if (++acc * 100 >= tot * 99) { p99 = i; break; } }
            for (int s = 1; s <= 8; s++) sumsz += s * orb[s];
            unpack(sols[last], c);
            double *a = malloc(sizeof(double) * nsol), *b = malloc(sizeof(double) * nsol);
            for (int i = 0; i < nsol; i++) a[i] = i;
            printf("%d,%d,%ld,%ld,%ld,prim=%d,%.4f,%.4f,", N, nsol, orb[2], orb[4], orb[8], c[0] + 1,
                   (last + 1.0) / nsol, (p99 + 1.0) / nsol);
            if (sumsz != nsol) printf("SUMA MAL %ld ", sumsz);
            for (int g = 1; g < NG; g++) {
                for (int i = 0; i < nsol; i++) b[i] = img[(size_t)i * NG + g];
                printf(" %.3f", pearson(a, b, nsol));
            }
            // nulo: las mismas simetrias sobre permutaciones al azar (rango de Lehmer)
            int gs[3] = {2, 4, 5};
            printf(" |");
            int M = 200000;
            double *ra = malloc(sizeof(double) * M), *rb = malloc(sizeof(double) * M);
            for (int t = 0; t < 3; t++) {
                for (int i = 0; i < M; i++) {
                    for (int r = 0; r < N; r++) c[r] = r;
                    for (int r = N - 1; r > 0; r--) { int j = xr() % (r + 1); int tt = c[r]; c[r] = c[j]; c[j] = tt; }
                    apply(gs[t], c, d);
                    ra[i] = lehmer(c); rb[i] = lehmer(d);
                }
                printf(" %.3f", pearson(ra, rb, M));
            }
            printf("\n");
            fflush(stdout);
            free(a); free(b); free(ra); free(rb);
        }
        free(img); free(isrep);
    }
    return 0;
}
