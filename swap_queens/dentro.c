/* dentro.c — ¿qué comparten las soluciones de una isla de nivel 2?

   Isla: no toroidales unidas por cambiar 2 filas o menos, o por giro constante,
   una por grupo de simetría. Los miembros se alinean recorriendo la isla: cada uno
   se coloca en la simetría y el giro de su vecino. Por isla:
     arbol   aristas - nodos + 1 (0 = árbol puro)
     nucleo  filas con la misma reina en todos los miembros alineados
     W       pares de costura: se atacan en el toro y no en el tablero (0 = toroidal)
     X       reinas que mover hasta la toroidal más cercana
     eje     parte de los miembros a X <= 4 de una misma toroidal, y su nº de rectas

   Uso: dentro N
   cc -O3 -mcpu=native -pthread -Wall dentro.c -o dentro */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 16
#define HILOS 8
typedef struct { uint8_t x[MAXN]; } Perm;
typedef struct { uint32_t a, b; } Arista;

static int N;
static Perm *g, *tor, *rep, *pal;
static uint8_t (*im)[8][MAXN];
static int ng, cg, nt, ct, nr, sig_i;
static Arista *E[HILOS];
static uint64_t nE[HILOS];
static int inv[MAXN];

static void pon(Perm **v, int *n, int *cap, const uint8_t *t) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *v = realloc(*v, *cap * sizeof(Perm)); }
    memcpy((*v)[*n].x, t, N);
    (*n)++;
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, N); }

static int toroidal(const uint8_t *s) {
    uint32_t d = 0, e = 0, todo = (1u << N) - 1;
    for (int r = 0; r < N; r++) {
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    return d == todo && e == todo;
}

static void reinas(int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) {
        if (toroidal(s)) pon(&tor, &nt, &ct, s); else pon(&g, &ng, &cg, s);
        return;
    }
    uint32_t lib = ((1u << N) - 1) & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

static void imagen(const uint8_t *s, int e, uint8_t *t) {
    uint8_t a[MAXN];
    if (e & 4) for (int r = 0; r < N; r++) a[s[r]] = r;
    else memcpy(a, s, N);
    for (int r = 0; r < N; r++) {
        int f = (e & 2) ? N - 1 - r : r;
        t[f] = (e & 1) ? N - 1 - a[r] : a[r];
    }
}

/* -1 si no se une; si se une, deja en P la imagen alineada con A */
static int une(const uint8_t *A, int u, uint8_t *P) {
    for (int e = 0; e < 8; e++) {
        const uint8_t *V = im[u][e];
        int c[MAXN] = {0}, mx = 0, mc = 0;
        for (int r = 0; r < N; r++) {
            int v = V[r] - A[r];
            if (v < 0) v += N;
            if (++c[v] > mx) { mx = c[v]; mc = v; }
        }
        if (N - c[0] <= 2) { if (P) memcpy(P, V, N); return 1; }
        if (mx == N) {
            if (P) for (int r = 0; r < N; r++) P[r] = (V[r] + N - mc) % N;
            return 1;
        }
    }
    return -1;
}

static void *obrero(void *arg) {
    int yo = (int)(intptr_t)arg;
    uint64_t cap = 1 << 16, n = 0;
    Arista *b = malloc(cap * sizeof(Arista));
    for (;;) {
        int i = __sync_fetch_and_add(&sig_i, 1);
        if (i >= nr) break;
        for (int j = i + 1; j < nr; j++) {
            if (une(rep[i].x, j, NULL) < 0) continue;
            if (n == cap) { cap *= 2; b = realloc(b, cap * sizeof(Arista)); }
            b[n].a = i; b[n].b = j; n++;
        }
    }
    E[yo] = b; nE[yo] = n;
    return NULL;
}

static int costura(const uint8_t *s) {
    int w = 0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            int d = abs((int)s[i] - s[j]);
            if (d == N - (j - i)) w++;
        }
    return w;
}

static int distancia(const uint8_t *a, const uint8_t *b) {
    int e = 0;
    for (int r = 0; r < N; r++) e += a[r] == b[r];
    return N - e;
}

static int es_recta(const uint8_t *t) {
    int a = (t[1] - t[0] + N) % N;
    for (int r = 0; r < N; r++)
        if (t[r] != (t[0] + a * r) % N) return 0;
    return 1;
}

static int dos_rectas(const uint8_t *f) {
    static const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int p = 0; p < 3; p++) {
        int i = P[p][0], j = P[p][1];
        int a = (f[j] - f[i] + N) * inv[j - i] % N, c = (f[i] + (N - a) * i) % N;
        int res[MAXN], nr2 = 0, ok = 1;
        for (int r = 0; r < N; r++)
            if (f[r] != (a * r + c) % N) res[nr2++] = r;
        if (nr2 < 2) return 1;
        int u = res[0], v = res[1];
        int b = (f[v] - f[u] + N) * inv[v - u] % N, d = (f[u] + (N - b) * u) % N;
        for (int k = 2; k < nr2; k++)
            if (f[res[k]] != (b * res[k] + d) % N) { ok = 0; break; }
        if (ok) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: dentro N\n"); return 1; }
    N = atoi(argv[1]);
    for (int a = 1; a < N; a++)
        for (int b = 1; b < N; b++)
            if (a * b % N == 1) inv[a] = b;
    uint8_t s[MAXN], t[MAXN];
    reinas(0, s, 0, 0, 0);
    qsort(g, ng, sizeof(Perm), cmp);
    rep = malloc(ng * sizeof(Perm));
    pal = malloc(ng * sizeof(Perm));
    for (int i = 0; i < ng; i++) {
        int minimo = 1;
        for (int e = 1; e < 8; e++) {
            imagen(g[i].x, e, t);
            if (memcmp(t, g[i].x, N) < 0) { minimo = 0; break; }
        }
        if (minimo) rep[nr++] = g[i];
    }
    im = malloc(nr * sizeof(*im));
    for (int i = 0; i < nr; i++)
        for (int e = 0; e < 8; e++) imagen(rep[i].x, e, im[i][e]);
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, obrero, (void *)i);
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    uint64_t ar = 0;
    for (int h = 0; h < HILOS; h++) ar += nE[h];
    uint32_t *ini = calloc(nr + 2, 4), *ady = malloc((2 * ar + 1) * 4);
    for (int h = 0; h < HILOS; h++)
        for (uint64_t e = 0; e < nE[h]; e++) { ini[E[h][e].a + 1]++; ini[E[h][e].b + 1]++; }
    for (int i = 0; i < nr; i++) ini[i + 1] += ini[i];
    uint32_t *pos = malloc((nr + 1) * 4);
    memcpy(pos, ini, (nr + 1) * 4);
    for (int h = 0; h < HILOS; h++)
        for (uint64_t e = 0; e < nE[h]; e++) {
            ady[pos[E[h][e].a]++] = E[h][e].b;
            ady[pos[E[h][e].b]++] = E[h][e].a;
        }
    int *W = malloc(nr * 4), *X = malloc(nr * 4);
    for (int i = 0; i < nr; i++) {
        W[i] = costura(rep[i].x);
        X[i] = N;
        for (int j = 0; j < nt; j++) {
            int d = distancia(rep[i].x, tor[j].x);
            if (d < X[i]) X[i] = d;
        }
    }
    double sw = 0, sx = 0;
    for (int i = 0; i < nr; i++) { sw += W[i]; sx += X[i]; }
    printf("N=%d grupos=%d toroidales=%d aristas=%llu media_W=%.2f media_X=%.2f\n", N, nr, nt,
           (unsigned long long)ar, sw / nr, sx / nr);
    enum { TRAMOS = 6 };
    static const int lim[TRAMOS + 1] = {1, 2, 3, 5, 10, 30, 1 << 30};
    double tw[TRAMOS] = {0}, tx[TRAMOS] = {0}, tc[TRAMOS] = {0}, ta[TRAMOS] = {0}, tn[TRAMOS] = {0}, te[TRAMOS] = {0};
    int ti[TRAMOS] = {0};
    uint8_t *visto = calloc(nr, 1);
    int *cola = malloc(nr * 4);
    for (int i = 0; i < nr; i++) {
        if (visto[i]) continue;
        int qi = 0, qf = 0;
        cola[qf++] = i; visto[i] = 1;
        memcpy(pal[i].x, rep[i].x, N);
        uint64_t grad = 0;
        while (qi < qf) {
            int v = cola[qi++];
            grad += ini[v + 1] - ini[v];
            for (uint32_t a = ini[v]; a < ini[v + 1]; a++) {
                int u = ady[a];
                if (visto[u]) continue;
                visto[u] = 1;
                une(pal[v].x, u, pal[u].x);
                cola[qf++] = u;
            }
        }
        int n = qf, nucleo = 0, mW = 0, mX = 0, bucles = (int)(grad / 2) - n + 1;
        for (int r = 0; r < N; r++) {
            int ig = 1;
            for (int k = 1; k < n; k++)
                if (pal[cola[k]].x[r] != pal[i].x[r]) { ig = 0; break; }
            nucleo += ig;
        }
        for (int k = 0; k < n; k++) { mW += W[cola[k]]; mX += X[cola[k]]; }
        int mejor = 0, mt = -1;
        if (n >= 3)
            for (int j = 0; j < nt; j++) {
                int c = 0;
                for (int k = 0; k < n; k++)
                    if (distancia(pal[cola[k]].x, tor[j].x) <= 4) c++;
                if (c > mejor) { mejor = c; mt = j; }
            }
        int tr = 0;
        while (n > lim[tr + 1] - 1 && tr < TRAMOS - 1) tr++;
        ti[tr]++; tn[tr] += n; tw[tr] += mW; tx[tr] += mX; tc[tr] += nucleo; ta[tr] += bucles;
        te[tr] += n >= 3 ? (double)mejor / n : 0;
        if (n >= 20) {
            printf("isla n=%d bucles=%d nucleo=%d W=%.1f X=%.1f eje=%.0f%% rectas=%s\n", n, bucles, nucleo,
                   (double)mW / n, (double)mX / n, 100.0 * mejor / n,
                   mt < 0 ? "-" : es_recta(tor[mt].x) ? "1" : dos_rectas(tor[mt].x) ? "2" : "3+");
        }
    }
    printf("tamano islas nucleo_medio bucles_medios W X eje\n");
    for (int t2 = 0; t2 < TRAMOS; t2++) {
        if (!ti[t2]) continue;
        printf("%d-%d %d %.1f %.1f %.2f %.2f %.0f%%\n", lim[t2], lim[t2 + 1] - 1, ti[t2], tc[t2] / ti[t2],
               ta[t2] / ti[t2], tw[t2] / tn[t2], tx[t2] / tn[t2], 100.0 * te[t2] / ti[t2]);
    }
    return 0;
}
