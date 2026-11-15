/* saltos.c — ¿hay listas de rotaciones por fila que hagan saltar de una solución
   a otra mejor que el resto?

   Conjunto G: todas las soluciones menos las toroidales de una y dos espirales.
   Lista d: cada fila r gira d[r]. Salto: U y U + d están en G.
   Una lista al azar casi nunca acierta (Q²/N^N por lista), así que se recorren
   todos los pares {U,W}: d = W - U son todas las listas que aciertan alguna vez.
   d y -d son el mismo salto al revés y se cuentan juntas.

   Uso: saltos N control cubos
     control 0 real, 1 permutaciones al azar, tantas como en G
     cubos: pasadas en que se reparte el conteo para que quepa en memoria
   Salida: nº de listas por nº de saltos; para las de 3 o más, filas fuera de la
   mejor recta a·r + c, y las mejores listas.
   cc -O3 -mcpu=native -pthread -Wall saltos.c -o saltos */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 16
#define HILOS 8
typedef struct { uint8_t x[MAXN]; } Perm;

static int N, control, cubos;
static Perm *g, *ex;
static int ng, nex, cg, cex;
static uint64_t Z[MAXN][MAXN];
static uint64_t sem = 0x9E3779B97F4A7C15ull;
static int sig_i, cubo;
static uint64_t **buf, *nbuf;

typedef struct { uint64_t h; int k; } Lista;
static Lista *rep;
static uint64_t nrep, caprep;
static uint8_t *filtro;
#define FBITS 30
static pthread_mutex_t cerrojo = PTHREAD_MUTEX_INITIALIZER;

enum { TOPK = 64, TOPS = 12 };
typedef struct { int k; uint8_t d[MAXN]; } Mejor;
static Mejor mejores[TOPS];
static uint64_t fuera[TOPK + 1][MAXN + 1];
static uint8_t *visto;

static uint64_t xs(void) {
    sem ^= sem << 13; sem ^= sem >> 7; sem ^= sem << 17;
    return sem;
}

static void pon(Perm **v, int *n, int *cap, const uint8_t *t) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *v = realloc(*v, *cap * sizeof(Perm)); }
    memcpy((*v)[*n].x, t, N);
    (*n)++;
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, N); }

static int toroidal(const uint8_t *s) {
    uint32_t c = 0, d = 0, e = 0, todo = (1u << N) - 1;
    for (int r = 0; r < N; r++) {
        c |= 1u << s[r];
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    return c == todo && d == todo && e == todo;
}

static int es_recta(const uint8_t *t) {
    int a = (t[1] - t[0] + N) % N;
    for (int r = 0; r < N; r++)
        if (t[r] != (t[0] + a * r) % N) return 0;
    return 1;
}

static void dos(int r, int a, int c, int b, int d, uint8_t *t,
                uint32_t col, uint32_t dg, uint32_t ad) {
    if (r == N) { if (!es_recta(t) && toroidal(t)) pon(&ex, &nex, &cex, t); return; }
    int op[2] = {(a * r + c) % N, (b * r + d) % N};
    for (int k = 0; k < 2; k++) {
        int v = op[k];
        if (k == 1 && op[1] == op[0]) break;
        uint32_t m1 = 1u << v, m2 = 1u << ((v - r + N) % N), m3 = 1u << ((v + r) % N);
        if ((col & m1) || (dg & m2) || (ad & m3)) continue;
        t[r] = v;
        dos(r + 1, a, c, b, d, t, col | m1, dg | m2, ad | m3);
    }
}

/* las que se excluyen: una y dos rectas que además sean toroidales */
static void excluidas(void) {
    uint8_t t[MAXN];
    for (int a = 0; a < N; a++)
        for (int c = 0; c < N; c++) {
            for (int r = 0; r < N; r++) t[r] = (a * r + c) % N;
            if (toroidal(t)) pon(&ex, &nex, &cex, t);
        }
    for (int a = 0; a < N; a++)
        for (int b = a + 1; b < N; b++)
            for (int c = 0; c < N; c++)
                for (int d = 0; d < N; d++)
                    dos(0, a, c, b, d, t, 0, 0, 0);
    qsort(ex, nex, sizeof(Perm), cmp);
}

static void reinas(int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) {
        if (!toroidal(s) || !bsearch(s, ex, nex, sizeof(Perm), cmp)) pon(&g, &ng, &cg, s);
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

/* lista W - U en forma canónica: primer valor no nulo en la mitad baja */
static inline uint64_t lista(const uint8_t *U, const uint8_t *W, uint8_t *d) {
    uint64_t h = 0;
    int neg = -1;
    for (int r = 0; r < N; r++) {
        int v = W[r] - U[r];
        if (v < 0) v += N;
        if (neg < 0 && v) neg = 2 * v > N;
        d[r] = v;
    }
    if (neg == 1)
        for (int r = 0; r < N; r++) d[r] = d[r] ? N - d[r] : 0;
    for (int r = 0; r < N; r++) h ^= Z[r][d[r]];
    return h;
}

static void *cuenta(void *arg) {
    int yo = (int)(intptr_t)arg;
    uint64_t *b = NULL, n = 0, cap = 0;
    uint8_t d[MAXN];
    for (;;) {
        int i = __sync_fetch_and_add(&sig_i, 1);
        if (i >= ng) break;
        for (int j = i + 1; j < ng; j++) {
            uint64_t h = lista(g[i].x, g[j].x, d);
            if ((int)(h % cubos) != cubo) continue;
            if (n == cap) { cap = cap ? 2 * cap : 1 << 20; b = realloc(b, cap * 8); }
            b[n++] = h;
        }
    }
    buf[yo] = b; nbuf[yo] = n;
    return NULL;
}

static void radix(uint64_t *a, uint64_t n, uint64_t *t) {
    for (int s = 0; s < 64; s += 11) {
        uint64_t c[2049] = {0};
        for (uint64_t i = 0; i < n; i++) c[((a[i] >> s) & 2047) + 1]++;
        for (int i = 0; i < 2048; i++) c[i + 1] += c[i];
        for (uint64_t i = 0; i < n; i++) t[c[(a[i] >> s) & 2047]++] = a[i];
        uint64_t *x = a; a = t; t = x;
    }
    /* seis pasadas: el orden queda en t al volver */
    memcpy(t, a, n * 8);
}

static int cmpl(const void *a, const void *b) {
    uint64_t x = ((const Lista *)a)->h, y = ((const Lista *)b)->h;
    return x < y ? -1 : x > y;
}

static int fuera_de_recta(const uint8_t *d) {
    int mejor = 0;
    for (int a = 0; a < N; a++) {
        int cu[MAXN] = {0};
        for (int r = 0; r < N; r++) {
            int c = (d[r] + (N - a) * r) % N;
            if (++cu[c] > mejor) mejor = cu[c];
        }
    }
    return N - mejor;
}

static void *clasifica(void *arg) {
    uint8_t d[MAXN];
    for (;;) {
        int i = __sync_fetch_and_add(&sig_i, 1);
        if (i >= ng) break;
        for (int j = i + 1; j < ng; j++) {
            uint64_t h = lista(g[i].x, g[j].x, d);
            uint64_t b = h >> (64 - FBITS);
            if (!((filtro[b >> 3] >> (b & 7)) & 1)) continue;
            Lista q = {h, 0}, *p = bsearch(&q, rep, nrep, sizeof(Lista), cmpl);
            if (!p) continue;
            uint64_t idx = p - rep;
            pthread_mutex_lock(&cerrojo);
            if (!visto[idx]) {
                visto[idx] = 1;
                int kk = p->k < TOPK ? p->k : TOPK;
                fuera[kk][fuera_de_recta(d)]++;
                for (int t = 0; t < TOPS; t++)
                    if (p->k > mejores[t].k) {
                        memmove(&mejores[t + 1], &mejores[t], (TOPS - t - 1) * sizeof(Mejor));
                        mejores[t].k = p->k;
                        memcpy(mejores[t].d, d, N);
                        break;
                    }
            }
            pthread_mutex_unlock(&cerrojo);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: saltos N control cubos\n"); return 1; }
    N = atoi(argv[1]); control = atoi(argv[2]); cubos = atoi(argv[3]);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) Z[r][c] = xs();
    excluidas();
    uint8_t s[MAXN];
    reinas(0, s, 0, 0, 0);
    if (control)
        for (int i = 0; i < ng; i++) {
            for (int r = 0; r < N; r++) g[i].x[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs() % (r + 1), v = g[i].x[r];
                g[i].x[r] = g[i].x[j]; g[i].x[j] = v;
            }
        }
    fprintf(stderr, "N=%d excluidas=%d G=%d\n", N, nex, ng);
    buf = calloc(HILOS, sizeof(uint64_t *));
    nbuf = calloc(HILOS, 8);
    static uint64_t hist[1 << 16];
    uint64_t listas = 0;
    filtro = calloc(1ull << (FBITS - 3), 1);
    for (cubo = 0; cubo < cubos; cubo++) {
        sig_i = 0;
        pthread_t th[HILOS];
        for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, cuenta, (void *)i);
        for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
        uint64_t n = 0;
        for (int i = 0; i < HILOS; i++) n += nbuf[i];
        uint64_t *a = malloc((n + 1) * 8), *t = malloc((n + 1) * 8), k = 0;
        for (int i = 0; i < HILOS; i++) {
            memcpy(a + k, buf[i], nbuf[i] * 8);
            k += nbuf[i];
            free(buf[i]);
        }
        radix(a, n, t);
        for (uint64_t x = 0, y; x < n; x = y) {
            for (y = x; y < n && t[y] == t[x]; y++);
            uint64_t m = y - x;
            listas++;
            hist[m < 65535 ? m : 65535]++;
            if (m >= 3) {
                if (nrep == caprep) {
                    caprep = caprep ? 2 * caprep : 1 << 16;
                    rep = realloc(rep, caprep * sizeof(Lista));
                }
                rep[nrep].h = t[x]; rep[nrep].k = m; nrep++;
                uint64_t b = t[x] >> (64 - FBITS);
                filtro[b >> 3] |= 1 << (b & 7);
            }
        }
        free(a); free(t);
        fprintf(stderr, "cubo %d/%d pares=%llu\n", cubo + 1, cubos, (unsigned long long)n);
    }
    qsort(rep, nrep, sizeof(Lista), cmpl);
    visto = calloc(nrep + 1, 1);
    sig_i = 0;
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, clasifica, (void *)i);
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    printf("N=%d control=%d G=%d listas=%llu\nsaltos_por_lista", N, control, ng, (unsigned long long)listas);
    for (int i = 1; i < 1 << 16; i++)
        if (hist[i]) printf(" %d:%llu", i, (unsigned long long)hist[i]);
    printf("\n");
    for (int k = 3; k <= TOPK; k++) {
        uint64_t tot = 0;
        for (int x = 0; x <= N; x++) tot += fuera[k][x];
        if (!tot) continue;
        printf("saltos=%d%s filas_fuera_de_recta", k, k == TOPK ? "+" : "");
        for (int x = 0; x <= N; x++)
            if (fuera[k][x]) printf(" %d:%llu", x, (unsigned long long)fuera[k][x]);
        printf("\n");
    }
    for (int t = 0; t < TOPS && mejores[t].k; t++) {
        printf("mejor saltos=%d fuera=%d lista=", mejores[t].k, fuera_de_recta(mejores[t].d));
        for (int r = 0; r < N; r++) printf("%d ", mejores[t].d[r]);
        printf("\n");
    }
    return 0;
}
