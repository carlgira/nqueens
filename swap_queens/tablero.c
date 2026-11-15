/* tablero.c — ¿un conjunto de soluciones no toroidales puede tomar forma de
   rectas mod N tras permutar libremente las N² celdas del tablero?

   Una permutación p de celdas con p(S_i) = recta_i existe sii el conjunto se
   solapa como las rectas. Las rectas de una pendiente parten el tablero en N
   piezas: una "clase" = N soluciones disjuntas = cuadrado latino L(r,c) = pieza.
   Dos clases se cortan como dos pendientes sii son ortogonales: cada pareja de
   piezas comparte 1 celda. Entonces (L1,L2) ya es la permutación p.
   Tres clases: además L3, escrito en esas coordenadas, debe ser Z_N.

   Uso: tablero N modo [tope]
     modo 0: no toroidales. modo 1: tantas permutaciones al azar como no toroidales.
     modo 2: todas las soluciones (control: aparecen las clases de rectas).
   tope = máximo de clases guardadas para los pasos de ortogonalidad.
   cc -O3 -mcpu=native -pthread -Wall tablero.c -o tablero */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 19
typedef struct { uint64_t m[3]; } Mask; /* hasta 192 celdas */
typedef struct { uint8_t x[MAXN]; Mask m; } Sol;

static int N, modo;
static Sol *pool;
static int np, cap;
static uint8_t *clases; /* tope × N², L(r,c) */
static uint64_t ncl, tope, guardadas;
static uint64_t semilla = 0x9E3779B97F4A7C15ull;

static uint64_t xs(void) {
    semilla ^= semilla << 13; semilla ^= semilla >> 7; semilla ^= semilla << 17;
    return semilla;
}

static void mete(const uint8_t *s) {
    if (np == cap) { cap = cap ? 2 * cap : 4096; pool = realloc(pool, cap * sizeof(Sol)); }
    Sol *o = &pool[np++];
    memset(o, 0, sizeof(Sol));
    for (int r = 0; r < N; r++) {
        o->x[r] = s[r];
        int b = r * N + s[r];
        o->m.m[b >> 6] |= 1ull << (b & 63);
    }
}

static int toroidal(const uint8_t *s) {
    uint32_t d = 0, e = 0, todo = (1u << N) - 1;
    for (int r = 0; r < N; r++) {
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    return d == todo && e == todo;
}

static void reinas(int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) { if (modo == 2 || !toroidal(s)) mete(s); return; }
    uint32_t lib = ((1u << N) - 1) & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

/* cobertura exacta: siempre la celda libre con menos candidatas; varios hilos
   reparten las candidatas de la primera celda */
static pthread_mutex_t cerrojo = PTHREAD_MUTEX_INITIALIZER;
static int primero[1 << 20], nprimero, sig_primero, c0;

typedef struct { int *lis[MAXN + 1]; int elegida[MAXN]; } Pila;

static void guarda(Pila *P) {
    pthread_mutex_lock(&cerrojo);
    ncl++;
    if (guardadas < tope) {
        uint8_t *L = clases + guardadas++ * N * N;
        for (int k = 0; k < N; k++)
            for (int r = 0; r < N; r++) L[r * N + pool[P->elegida[k]].x[r]] = k;
    }
    pthread_mutex_unlock(&cerrojo);
}

static void cubre(Pila *P, int prof, int n, Mask libres) {
    if (prof == N) { guarda(P); return; }
    int cuenta[MAXN * MAXN];
    memset(cuenta, 0, sizeof(int) * N * N);
    const int *lis = P->lis[prof];
    for (int i = 0; i < n; i++) {
        const Sol *o = &pool[lis[i]];
        for (int r = 0; r < N; r++) cuenta[r * N + o->x[r]]++;
    }
    int mejor = -1, mc = 1 << 30;
    for (int b = 0; b < N * N; b++)
        if ((libres.m[b >> 6] >> (b & 63)) & 1) {
            if (!cuenta[b]) return;
            if (cuenta[b] < mc) { mc = cuenta[b]; mejor = b; }
        }
    int fila = mejor / N, col = mejor % N;
    int *dest = P->lis[prof + 1];
    for (int i = 0; i < n; i++) {
        const Sol *e = &pool[lis[i]];
        if (e->x[fila] != col) continue;
        int k = 0;
        for (int j = 0; j < n; j++) {
            const Mask *o = &pool[lis[j]].m;
            if (!((o->m[0] & e->m.m[0]) | (o->m[1] & e->m.m[1]) | (o->m[2] & e->m.m[2])))
                dest[k++] = lis[j];
        }
        P->elegida[prof] = lis[i];
        Mask l2 = libres;
        for (int w = 0; w < 3; w++) l2.m[w] &= ~e->m.m[w];
        cubre(P, prof + 1, k, l2);
    }
}

static void *obrero(void *arg) {
    Pila P;
    for (int d = 0; d <= N; d++) P.lis[d] = malloc(np * sizeof(int));
    Mask todo = {{0, 0, 0}};
    for (int b = 0; b < N * N; b++) todo.m[b >> 6] |= 1ull << (b & 63);
    for (;;) {
        int t = __sync_fetch_and_add(&sig_primero, 1);
        if (t >= nprimero) break;
        const Sol *e = &pool[primero[t]];
        int k = 0;
        for (int j = 0; j < np; j++) {
            const Mask *o = &pool[j].m;
            if (!((o->m[0] & e->m.m[0]) | (o->m[1] & e->m.m[1]) | (o->m[2] & e->m.m[2])))
                P.lis[1][k++] = j;
        }
        P.elegida[0] = primero[t];
        Mask l2 = todo;
        for (int w = 0; w < 3; w++) l2.m[w] &= ~e->m.m[w];
        cubre(&P, 1, k, l2);
        if (t % 100 == 0) { fprintf(stderr, "%d/%d clases=%llu\n", t, nprimero, (unsigned long long)ncl); }
    }
    return NULL;
}

static int ortogonales(const uint8_t *A, const uint8_t *B) {
    uint32_t visto[MAXN] = {0};
    for (int i = 0; i < N * N; i++) {
        uint32_t b = 1u << B[i];
        if (visto[A[i]] & b) return 0;
        visto[A[i]] |= b;
    }
    return 1;
}

/* C en coordenadas (u,v) = (A,B); normalizado, ¿es la tabla de Z_N? */
static int es_ciclico(const uint8_t *A, const uint8_t *B, const uint8_t *C) {
    int8_t T[MAXN][MAXN], fil[MAXN], col[MAXN], g[MAXN][MAXN];
    for (int i = 0; i < N * N; i++) T[A[i]][B[i]] = C[i];
    for (int v = 0; v < N; v++) col[T[0][v]] = v;     /* símbolo -> v tal que (0,v) da símbolo */
    for (int u = 0; u < N; u++) fil[T[u][0]] = u;
    int8_t sim0[MAXN];
    (void)sim0;
    /* lazo: x*y = símbolo de (fil[x'],col[y']) con identidad en T[0][0] */
    int8_t ident_v[MAXN], ident_u[MAXN];
    for (int s = 0; s < N; s++) { ident_v[s] = col[s]; ident_u[s] = fil[s]; }
    for (int x = 0; x < N; x++)
        for (int y = 0; y < N; y++) g[x][y] = T[(int)ident_u[x]][(int)ident_v[y]];
    for (int x = 0; x < N; x++)
        for (int y = 0; y < N; y++)
            for (int z = 0; z < N; z++)
                if (g[(int)g[x][y]][z] != g[x][(int)g[y][z]]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "uso: tablero N modo [tope]\n"); return 1; }
    N = atoi(argv[1]); modo = atoi(argv[2]);
    tope = argc > 3 ? strtoull(argv[3], 0, 10) : 200000;
    uint8_t s[MAXN];
    reinas(0, s, 0, 0, 0);
    if (modo == 1) {
        int n = np;
        np = 0;
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < N; r++) s[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs() % (r + 1), v = s[r]; s[r] = s[j]; s[j] = v;
            }
            mete(s);
        }
    }
    clases = malloc(tope * N * N);
    for (int i = 0; i < np; i++) if (pool[i].x[0] == c0) primero[nprimero++] = i;
    printf("N=%d modo=%d piezas=%d\n", N, modo, np);
    fflush(stdout);
    pthread_t th[8];
    for (int i = 0; i < 8; i++) pthread_create(&th[i], NULL, obrero, NULL);
    for (int i = 0; i < 8; i++) pthread_join(th[i], NULL);
    printf("clases=%llu guardadas=%llu\n", (unsigned long long)ncl, (unsigned long long)guardadas);
    fflush(stdout);
    /* grafo de ortogonalidad entre las clases guardadas */
    uint64_t pares = 0, trios = 0, trios_z = 0;
    int G = guardadas;
    int *ady = malloc(sizeof(int) * (G + 1));
    for (int i = 0; i < G; i++) {
        const uint8_t *A = clases + (uint64_t)i * N * N;
        int na = 0;
        for (int j = i + 1; j < G; j++)
            if (ortogonales(A, clases + (uint64_t)j * N * N)) ady[na++] = j;
        pares += na;
        for (int a = 0; a < na; a++)
            for (int b = a + 1; b < na; b++) {
                const uint8_t *B = clases + (uint64_t)ady[a] * N * N;
                const uint8_t *C = clases + (uint64_t)ady[b] * N * N;
                if (ortogonales(B, C)) {
                    trios++;
                    if (es_ciclico(A, B, C)) trios_z++;
                }
            }
    }
    printf("pares_ortogonales=%llu trios=%llu trios_ciclicos=%llu\n",
           (unsigned long long)pares, (unsigned long long)trios, (unsigned long long)trios_z);
    return 0;
}
