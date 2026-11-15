// zeta_suave.c — en el primer tramo de la curva de conteo (fraccion F de las soluciones)
// cada orbita de simetria puede tener varias soluciones.  Se elige una por orbita con
// distintos criterios y se mide lo regular que queda el trozo de curva: los huecos entre
// puntos consecutivos en u = x / (N+1)^(N-1).
//   primera : la menor de la orbita dentro del tramo
//   azar    : una al azar
//   tv      : menor variacion total  sum |c[r+1]-c[r]|  (la solucion menos brusca)
//   entropia: menor entropia del histograma de saltos |c[r+1]-c[r]|
//   huecos  : busqueda local que minimiza la suma de huecos^2 (mira a los vecinos),
//             medidos en soluciones saltadas: las franjas prohibidas de u no son elegibles
//   ritmo   : de izquierda a derecha, toma una solucion solo si va por detras del ritmo
//             ideal (orbitas/tramo), o si era la ultima candidata de su orbita
//
// cc -O2 -o zeta_suave zeta_suave.c -lm && ./zeta_suave 10 16 0.15
// ./zeta_suave dump 14 0.15 > suave_14.csv
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXN 16
#define NG 8
#define NCRIT 6
static const char *cname[NCRIT] = {"primera", "azar", "tv", "entropia", "huecos", "ritmo"};

static int N, nsol, cap;
static uint64_t *sols;
static int cur[MAXN];

static void queens(int r, unsigned cols, unsigned d1, unsigned d2) {
    if (r == N) {
        if (nsol == cap) { cap = cap ? 2 * cap : 1024; sols = realloc(sols, sizeof(uint64_t) * cap); }
        uint64_t v = 0;
        for (int i = 0; i < N; i++) v = v << 4 | cur[i];
        sols[nsol++] = v;
        return;
    }
    unsigned avail = ~(cols | d1 | d2) & ((1u << N) - 1);
    while (avail) {
        unsigned b = avail & -avail; avail ^= b;
        cur[r] = __builtin_ctz(b);
        queens(r + 1, cols | b, (d1 | b) << 1, (d2 | b) >> 1);
    }
}

static void unpack(uint64_t v, int *c) { for (int i = N - 1; i >= 0; i--) { c[i] = v & 15; v >>= 4; } }
static uint64_t pack(const int *c) { uint64_t v = 0; for (int i = 0; i < N; i++) v = v << 4 | c[i]; return v; }

static void apply(int g, const int *c, int *d) {
    int m = N - 1;
    for (int r = 0; r < N; r++)
        switch (g) {
            case 0: d[r] = c[r]; break;
            case 1: d[r] = m - c[r]; break;
            case 2: d[r] = c[m - r]; break;
            case 3: d[r] = m - c[m - r]; break;
            case 4: d[c[r]] = r; break;
            case 5: d[c[r]] = m - r; break;
            case 6: d[m - c[r]] = r; break;
            case 7: d[m - c[r]] = m - r; break;
        }
}

static double uof(uint64_t v) {
    int c[MAXN]; unpack(v, c);
    double x = 0;
    for (int r = N - 1; r >= 0; r--) x = x / (N + 1) + c[r] + 1;
    return x;
}

static double tv(const int *c) { double s = 0; for (int r = 0; r + 1 < N; r++) s += abs(c[r + 1] - c[r]); return s; }

static double hjump(const int *c) {
    int h[MAXN] = {0};
    for (int r = 0; r + 1 < N; r++) h[abs(c[r + 1] - c[r])]++;
    double e = 0, n = N - 1;
    for (int i = 0; i < MAXN; i++) if (h[i]) e -= h[i] / n * log(h[i] / n);
    return e;
}

typedef struct { uint64_t canon; int idx; } item;
static int cmpitem(const void *a, const void *b) {
    const item *x = a, *y = b;
    if (x->canon != y->canon) return x->canon < y->canon ? -1 : 1;
    return x->idx - y->idx;
}

static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

// metricas de los huecos entre puntos en u (on[] marca los elegidos, ordenados por indice)
static void metrics(const double *u, const char *on, int w, double *m) {
    double prev = -1, s = 0, s2 = 0, mx = 0, h = 0; int n = 0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < w; i++) if (on[i]) {
        if (prev >= 0) { double g = u[i] - prev; s += g; s2 += g * g; if (g > mx) mx = g; n++; }
        prev = u[i];
    }
    int cnt = 0;
    prev = -1;
    for (int i = 0; i < w; i++) if (on[i]) {
        if (prev >= 0) { double g = (u[i] - prev) / s; if (g > 0) h -= g * log(g); }
        prev = u[i];
        double y = cnt++;
        sx += u[i]; sy += y; sxx += u[i] * u[i]; sxy += u[i] * y;
    }
    double mean = s / n, v = s2 / n - mean * mean;
    m[0] = sqrt(v) / mean;                      // CV de los huecos (azar de Poisson = 1)
    m[1] = mx / mean;                           // hueco mayor / hueco medio
    m[2] = h / log(n);                          // entropia normalizada (1 = todos iguales)
    double N1 = cnt, b = (N1 * sxy - sx * sy) / (N1 * sxx - sx * sx);
    double a = (sy - b * sx) / N1, ss = 0, c0 = 0;
    for (int i = 0; i < w; i++) if (on[i]) { double e = c0 - (a + b * u[i]); ss += e * e; c0 += 1; }
    m[3] = sqrt(ss / N1) / N1;                  // RMS frente a recta, en fraccion del total
}

// huecos medidos en soluciones saltadas, y error de la curva adelgazada frente a la real
static void metrics2(const char *on, int w, double *m) {
    int prev = -1, n = 0, cnt = 0, mx = 0;
    double s = 0, s2 = 0, err = 0, tot = 0;
    for (int i = 0; i < w; i++) tot += on[i];
    double scale = w / tot;
    for (int i = 0; i < w; i++) {
        if (on[i]) {
            if (prev >= 0) { int g = i - prev; s += g; s2 += (double)g * g; if (g > mx) mx = g; n++; }
            prev = i; cnt++;
        }
        double e = cnt * scale - (i + 1);
        err += e * e;
    }
    double mean = s / n;
    m[0] = sqrt(s2 / n - mean * mean) / mean;
    m[1] = mx;
    m[2] = sqrt(err / w);
    m[3] = mean;
}

int main(int argc, char **argv) {
    int dump = argc > 1 && !strcmp(argv[1], "dump");
    int n0 = atoi(argv[dump ? 2 : 1]), n1 = dump ? n0 : atoi(argv[2]);
    double F = atof(argv[3]);
    if (dump) printf("crit,u,i\n");


    for (N = n0; N <= n1; N++) {
        nsol = cap = 0; free(sols); sols = NULL;
        queens(0, 0, 0, 0);
        int w = (int)ceil(F * nsol);
        double *u = malloc(sizeof(double) * w);
        item *it = malloc(sizeof(item) * w);
        int *oz = malloc(sizeof(int) * w);
        for (int i = 0; i < w; i++) {
            int c[MAXN], d[MAXN], ns = 0;
            uint64_t mn = ~0ULL, seen[NG];
            unpack(sols[i], c);
            for (int g = 0; g < NG; g++) {
                apply(g, c, d);
                uint64_t v = pack(d);
                if (v < mn) mn = v;
                int nw = 1;
                for (int k = 0; k < ns; k++) nw &= seen[k] != v;
                if (nw) seen[ns++] = v;
            }
            u[i] = uof(sols[i]); it[i].canon = mn; it[i].idx = i; oz[i] = ns;
        }
        qsort(it, w, sizeof(item), cmpitem);
        int norb = 0, *start = malloc(sizeof(int) * (w + 1));
        long cover = 0, m1 = 0, m2 = 0;
        for (int i = 0; i < w; i++)
            if (i == 0 || it[i].canon != it[i - 1].canon) {
                start[norb++] = i;
                cover += oz[it[i].idx];
            }
        start[norb] = w;
        for (int o = 0; o < norb; o++) { if (start[o + 1] - start[o] == 1) m1++; else m2++; }
        if (!dump) printf("\nN=%d Q=%d tramo=%d orbitas=%d cobertura=%.3f una_candidata=%.3f varias=%.3f\n",
                          N, nsol, w, norb, (double)cover / nsol, (double)m1 / norb, (double)m2 / norb);

        char *on = malloc(w), *done = calloc(norb, 1);
        int *orbof = malloc(sizeof(int) * w), *left = malloc(sizeof(int) * norb);
        for (int o = 0; o < norb; o++) {
            left[o] = start[o + 1] - start[o];
            for (int j = start[o]; j < start[o + 1]; j++) orbof[it[j].idx] = o;
        }
        int *pick = malloc(sizeof(int) * norb);
        for (int cr = 0; cr < NCRIT; cr++) {
            memset(on, 0, w);
            for (int o = 0; o < norb; o++) {
                int a = start[o], b = start[o + 1], best = it[a].idx;
                if (cr == 1 || cr == 4) best = it[a + xr() % (b - a)].idx;
                if (cr == 2 || cr == 3) {
                    double bv = 1e18;
                    for (int j = a; j < b; j++) {
                        int c[MAXN];
                        unpack(sols[it[j].idx], c);
                        double v = cr == 2 ? tv(c) : hjump(c);
                        if (v < bv - 1e-12) { bv = v; best = it[j].idx; }
                    }
                }
                pick[o] = best; on[best] = 1;
            }
            if (cr == 4) {
                // busqueda local: mover la eleccion de cada orbita si baja la suma de huecos^2
                for (int sweep = 0; sweep < 30; sweep++) {
                    long moves = 0;
                    for (int o = 0; o < norb; o++) {
                        int a = start[o], b = start[o + 1];
                        if (b - a < 2) continue;
                        int i = pick[o], bestj = i;
                        double bestd = -1e-15;
                        on[i] = 0;
                        int l = i - 1, r = i + 1;
                        while (l >= 0 && !on[l]) l--;
                        while (r < w && !on[r]) r++;
                        double ul = l >= 0 ? (double)l : 0.0, ur = r < w ? (double)r : (double)(w - 1);
                        double base = ((double)i - ul) * ((double)i - ul) + (ur - (double)i) * (ur - (double)i) - (ur - ul) * (ur - ul);
                        for (int j = a; j < b; j++) {
                            int t = it[j].idx;
                            if (t == i) continue;
                            int lt = t - 1, rt = t + 1;
                            while (lt >= 0 && !on[lt]) lt--;
                            while (rt < w && !on[rt]) rt++;
                            double al = lt >= 0 ? (double)lt : 0.0, ar = rt < w ? (double)rt : (double)(w - 1);
                            double add = ((double)t - al) * ((double)t - al) + (ar - (double)t) * (ar - (double)t) - (ar - al) * (ar - al);
                            if (add - base < bestd) { bestd = add - base; bestj = t; }
                        }
                        on[bestj] = 1; pick[o] = bestj;
                        if (bestj != i) moves++;
                    }
                    if (!moves) break;
                }
            }
            if (cr == 5) {
                memset(on, 0, w);
                int cnt = 0;
                double rate = (double)norb / w;
                for (int i = 0; i < w; i++) {
                    int o = orbof[i];
                    if (done[o] == 2) continue;
                    done[o] = --left[o] == 0 ? 1 : 0;
                    if (done[o] == 1 || cnt < rate * (i + 1)) { on[i] = 1; cnt++; done[o] = 2; }
                }
                for (int o = 0; o < norb; o++) { done[o] = 0; left[o] = start[o + 1] - start[o]; }
            }
            if (dump) {
                for (int i = 0; i < w; i++) if (on[i]) printf("%s,%.8f,%d\n", cname[cr], u[i], i);
            } else {
                double m[4], k[4];
                metrics(u, on, w, m);
                metrics2(on, w, k);
                printf("  %-9s en u: CV=%.2f max=%.0f | en soluciones: hueco medio=%.2f CV=%.3f max=%.0f error_curva=%.2f\n",
                       cname[cr], m[0], m[1], k[3], k[0], k[1], k[2]);
            }
        }
        if (dump) for (int i = 0; i < w; i++) printf("todas,%.8f,%d\n", u[i], i);
        fflush(stdout);
        free(u); free(it); free(oz); free(start); free(on); free(pick); free(done); free(orbof); free(left);
    }
    return 0;
}
