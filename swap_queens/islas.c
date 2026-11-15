/* islas.c — islas de soluciones por niveles de reglas.

   Nodos: soluciones no toroidales, una por grupo de simetría del tablero (8).
   Arista de nivel k, familia A: dos soluciones difieren en k filas o menos, o son
   giro constante una de otra. Familia B: lo mismo tras un giro constante libre.
   Para cada k: islas, tamaño de la mayor, aisladas y profundidad de la mayor
   (BFS desde su nodo de más vecinos = niveles del árbol).

   Uso: islas N control
     control 0 real, 1 permutaciones al azar en igual número
   cc -O3 -mcpu=native -pthread -Wall islas.c -o islas */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 16
#define HILOS 8
#define TOPE 8
typedef struct { uint8_t x[MAXN]; } Perm;
typedef struct { uint32_t a, b; uint8_t d, dr; } Arista;

static int N, control;
static Perm *g, *rep;
static uint8_t (*im)[8][MAXN];
static int ng, cg, nr;
static int sig_i;
static Arista *E[HILOS];
static uint64_t nE[HILOS];
static uint64_t sem = 0x9E3779B97F4A7C15ull;

static uint64_t xs(void) {
    sem ^= sem << 13; sem ^= sem >> 7; sem ^= sem << 17;
    return sem;
}

static void pon(const uint8_t *t) {
    if (ng == cg) { cg = cg ? 2 * cg : 4096; g = realloc(g, cg * sizeof(Perm)); }
    memcpy(g[ng].x, t, N);
    ng++;
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
    if (r == N) { if (!toroidal(s)) pon(s); return; }
    uint32_t lib = ((1u << N) - 1) & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

/* las 8 simetrías: e&1 espejo columnas, e&2 espejo filas, e&4 trasponer */
static void imagen(const uint8_t *s, int e, uint8_t *t) {
    uint8_t a[MAXN];
    if (e & 4) for (int r = 0; r < N; r++) a[s[r]] = r;
    else memcpy(a, s, N);
    for (int r = 0; r < N; r++) {
        int f = (e & 2) ? N - 1 - r : r;
        t[f] = (e & 1) ? N - 1 - a[r] : a[r];
    }
}

static void *obrero(void *arg) {
    int yo = (int)(intptr_t)arg;
    uint64_t cap = 1 << 20, n = 0;
    Arista *b = malloc(cap * sizeof(Arista));
    for (;;) {
        int i = __sync_fetch_and_add(&sig_i, 1);
        if (i >= nr) break;
        const uint8_t *U = rep[i].x;
        for (int j = i + 1; j < nr; j++) {
            int bd = N, bdr = N;
            for (int e = 0; e < 8; e++) {
                const uint8_t *V = im[j][e];
                int c[MAXN] = {0}, mx = 0;
                for (int r = 0; r < N; r++) {
                    int v = V[r] - U[r];
                    if (v < 0) v += N;
                    if (++c[v] > mx) mx = c[v];
                }
                if (N - c[0] < bd) bd = N - c[0];
                if (N - mx < bdr) bdr = N - mx;
            }
            if (bdr > TOPE && bd > TOPE) continue;
            if (n == cap) { cap *= 2; b = realloc(b, cap * sizeof(Arista)); }
            b[n].a = i; b[n].b = j; b[n].d = bd; b[n].dr = bdr; n++;
        }
    }
    E[yo] = b; nE[yo] = n;
    return NULL;
}

static int *pad;
static int raiz(int x) {
    while (pad[x] != x) { pad[x] = pad[pad[x]]; x = pad[x]; }
    return x;
}

/* familia 0: bd <= k o giro exacto. familia 1: bdr <= k */
static inline int entra(const Arista *a, int fam, int k) {
    return fam ? a->dr <= k : (a->d <= k || a->dr == 0);
}

static void nivel(int fam, int k) {
    for (int i = 0; i < nr; i++) pad[i] = i;
    uint64_t ar = 0;
    int *gr = calloc(nr + 1, 4);
    for (int h = 0; h < HILOS; h++)
        for (uint64_t e = 0; e < nE[h]; e++)
            if (entra(&E[h][e], fam, k)) {
                int a = raiz(E[h][e].a), b = raiz(E[h][e].b);
                if (a != b) pad[a] = b;
                gr[E[h][e].a]++; gr[E[h][e].b]++;
                ar++;
            }
    int *tam = calloc(nr, 4), islas = 0, solas = 0, may = 0, rm = -1;
    for (int i = 0; i < nr; i++) tam[raiz(i)]++;
    for (int i = 0; i < nr; i++)
        if (tam[i]) {
            islas++;
            if (tam[i] == 1) solas++;
            if (tam[i] > may) { may = tam[i]; rm = i; }
        }
    uint32_t *ini = calloc(nr + 2, 4), *ady = malloc((2 * ar + 1) * 4);
    for (int i = 0; i < nr; i++) ini[i + 1] = ini[i] + gr[i];
    uint32_t *pos = malloc((nr + 1) * 4);
    memcpy(pos, ini, (nr + 1) * 4);
    for (int h = 0; h < HILOS; h++)
        for (uint64_t e = 0; e < nE[h]; e++)
            if (entra(&E[h][e], fam, k)) {
                ady[pos[E[h][e].a]++] = E[h][e].b;
                ady[pos[E[h][e].b]++] = E[h][e].a;
            }
    int origen = -1, mg = -1;
    for (int i = 0; i < nr; i++)
        if (raiz(i) == rm && gr[i] > mg) { mg = gr[i]; origen = i; }
    int *dist = malloc(nr * 4), *cola = malloc(nr * 4), qi = 0, qf = 0, prof = 0;
    memset(dist, -1, nr * 4);
    dist[origen] = 0; cola[qf++] = origen;
    while (qi < qf) {
        int v = cola[qi++];
        if (dist[v] > prof) prof = dist[v];
        for (uint32_t a = ini[v]; a < ini[v + 1]; a++)
            if (dist[ady[a]] < 0) { dist[ady[a]] = dist[v] + 1; cola[qf++] = ady[a]; }
    }
    printf("%s k=%d aristas=%llu islas=%d aisladas=%d mayor=%d (%.1f%%) prof=%d\n", fam ? "B" : "A", k,
           (unsigned long long)ar, islas, solas, may, 100.0 * may / nr, prof);
    free(gr); free(tam); free(ini); free(ady); free(pos); free(dist); free(cola);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "uso: islas N control\n"); return 1; }
    N = atoi(argv[1]); control = atoi(argv[2]);
    uint8_t s[MAXN], t[MAXN];
    reinas(0, s, 0, 0, 0);
    qsort(g, ng, sizeof(Perm), cmp);
    rep = malloc(ng * sizeof(Perm));
    for (int i = 0; i < ng; i++) {
        int minimo = 1;
        for (int e = 1; e < 8; e++) {
            imagen(g[i].x, e, t);
            if (memcmp(t, g[i].x, N) < 0) { minimo = 0; break; }
        }
        if (minimo) rep[nr++] = g[i];
    }
    if (control)
        for (int i = 0; i < nr; i++) {
            for (int r = 0; r < N; r++) rep[i].x[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs() % (r + 1), v = rep[i].x[r];
                rep[i].x[r] = rep[i].x[j]; rep[i].x[j] = v;
            }
        }
    im = malloc(nr * sizeof(*im));
    for (int i = 0; i < nr; i++)
        for (int e = 0; e < 8; e++) imagen(rep[i].x, e, im[i][e]);
    pad = malloc(nr * 4);
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, obrero, (void *)i);
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    printf("N=%d control=%d grupos=%d\n", N, control, nr);
    for (int fam = 0; fam < 2; fam++)
        for (int k = 0; k <= TOPE; k++) nivel(fam, k);
    return 0;
}
