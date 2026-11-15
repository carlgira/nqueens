/* cerca.c — ¿cuántas parejas de celdas hay que intercambiar para llevar una
   solución normal de N reinas a una toroidal de una o dos espirales?

   Cada pareja (reina, celda vacía) mueve una reina, así que para dos
   permutaciones S y T:  X = nº de filas con S[r] != T[r].

   Objetivos: 1 espiral = T(r) = a·r + c mod N;
              2 espirales = cada fila en una de dos rectas, sin caber en una.

   Uso: cerca N modo muestra
     modo 0: soluciones reales contra espirales (muestra = 1 de cada k)
     modo 1: nulo, permutaciones al azar contra espirales (muestra = cuántas)
     modo 2: nulo, soluciones reales contra espirales con filas y columnas
             barajadas una vez (muestra = 1 de cada k)
   Salida CSV: origen,objetivo,X,cuenta
   cc -O3 -mcpu=native -pthread cerca.c -o cerca */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 32 /* bytes por permutación, relleno para comparar en bloque */
typedef struct { uint8_t x[W]; } Perm;

static int N, modo;
static uint64_t muestra;
static Perm *obj;      /* primero las de 1 espiral, luego las de 2 */
static int n1, n2, nobj;
static uint64_t *hist; /* [origen 0..2][objetivo 0..2][X 0..N] */
static uint8_t *mejor; /* por objetivo, X mínimo desde una no toroidal */
static Perm *mejor_de;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static int tarea_sig;
static int cap, tam;

#define H(o, t, x) hist[((o) * 3 + (t)) * (N + 1) + (x)]

static uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

static void pon(const uint8_t *t) {
    if (tam == cap) { cap = cap ? 2 * cap : 4096; obj = realloc(obj, cap * sizeof(Perm)); }
    memset(&obj[tam], 0xFF, W);
    memcpy(obj[tam].x, t, N);
    tam++;
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, W); }

static int es_recta(const uint8_t *t) {
    int a = (t[1] - t[0] + N) % N;
    for (int r = 0; r < N; r++)
        if (t[r] != (t[0] + a * r) % N) return 0;
    return 1;
}

static void dos(int r, int a, int c, int b, int d, uint8_t *t,
                uint32_t col, uint32_t dg, uint32_t ad) {
    if (r == N) { if (!es_recta(t)) pon(t); return; }
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

static void genera(void) {
    uint8_t t[W];
    for (int a = 2; a < N - 1; a++)
        for (int c = 0; c < N; c++) {
            for (int r = 0; r < N; r++) t[r] = (a * r + c) % N;
            pon(t);
        }
    n1 = tam;
    for (int a = 0; a < N; a++)
        for (int b = a + 1; b < N; b++)
            for (int c = 0; c < N; c++)
                for (int d = 0; d < N; d++)
                    dos(0, a, c, b, d, t, 0, 0, 0);
    qsort(obj + n1, tam - n1, sizeof(Perm), cmp);
    int k = n1;
    for (int i = n1; i < tam; i++)
        if (i == n1 || memcmp(&obj[i], &obj[i - 1], W)) obj[k++] = obj[i];
    n2 = k - n1;
    nobj = k;
}

static void baraja_objetivos(void) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    int pf[W], pc[W];
    for (int i = 0; i < N; i++) pf[i] = pc[i] = i;
    for (int i = N - 1; i > 0; i--) {
        int j = xs(&s) % (i + 1), v = pf[i]; pf[i] = pf[j]; pf[j] = v;
        j = xs(&s) % (i + 1); v = pc[i]; pc[i] = pc[j]; pc[j] = v;
    }
    for (int i = 0; i < nobj; i++) {
        Perm o = obj[i];
        for (int r = 0; r < N; r++) obj[i].x[pf[r]] = pc[o.x[r]];
    }
}

static inline int iguales(const uint8_t *restrict a, const uint8_t *restrict b) {
    uint8_t e = 0;
    for (int j = 0; j < W; j++) e += a[j] == b[j];
    return e;
}

static int toroidal(const uint8_t *s) {
    uint32_t d = 0, e = 0;
    for (int r = 0; r < N; r++) {
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    uint32_t todo = (N == 32) ? ~0u : (1u << N) - 1;
    return d == todo && e == todo;
}

typedef struct { uint64_t *h; uint8_t *m; Perm *de; uint64_t s; } Local;

static void procesa(Local *L, const uint8_t *s, int origen) {
    uint8_t q[W] __attribute__((aligned(32)));
    memset(q, 0xFE, W);
    memcpy(q, s, N);
    int b1 = 0, b2 = 0;
    for (int i = 0; i < n1; i++) {
        int e = iguales(q, obj[i].x);
        if (e > b1) b1 = e;
        if (origen == 0 && N - e < L->m[i]) { L->m[i] = N - e; memcpy(L->de[i].x, q, W); }
    }
    for (int i = n1; i < nobj; i++) {
        int e = iguales(q, obj[i].x);
        if (e > b2) b2 = e;
        if (origen == 0 && N - e < L->m[i]) { L->m[i] = N - e; memcpy(L->de[i].x, q, W); }
    }
    int b = b1 > b2 ? b1 : b2;
    uint64_t *h = L->h;
    h[(origen * 3 + 0) * (N + 1) + N - b1]++;
    h[(origen * 3 + 1) * (N + 1) + N - b2]++;
    h[(origen * 3 + 2) * (N + 1) + N - b]++;
}

static void reinas(Local *L, int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) {
        if (muestra > 1 && xs(&L->s) % muestra) return;
        procesa(L, s, toroidal(s) ? 1 : 0);
        return;
    }
    uint32_t todo = (1u << N) - 1, lib = todo & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(L, r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

static void *obrero(void *arg) {
    Local L;
    L.h = calloc(9 * (N + 1), sizeof(uint64_t));
    L.m = malloc(nobj);
    L.de = calloc(nobj, sizeof(Perm));
    memset(L.m, 255, nobj);
    L.s = 88172645463325252ull + 0x9E3779B97F4A7C15ull * (uint64_t)(intptr_t)arg;
    uint8_t s[W];
    if (modo == 1) {
        for (uint64_t i = 0; i < muestra / 8; i++) {
            for (int r = 0; r < N; r++) s[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs(&L.s) % (r + 1), v = s[r]; s[r] = s[j]; s[j] = v;
            }
            procesa(&L, s, 2);
        }
    } else {
        for (;;) {
            int t = __sync_fetch_and_add(&tarea_sig, 1);
            if (t >= N * N) break;
            int c0 = t / N, c1 = t % N;
            if (abs(c0 - c1) < 2) continue;
            uint32_t m0 = 1u << c0, m1 = 1u << c1;
            s[0] = c0; s[1] = c1;
            reinas(&L, 2, s, m0 | m1, ((m0 << 1) | m1) << 1, ((m0 >> 1) | m1) >> 1);
        }
    }
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < 9 * (N + 1); i++) hist[i] += L.h[i];
    for (int i = 0; i < nobj; i++)
        if (L.m[i] < mejor[i]) { mejor[i] = L.m[i]; mejor_de[i] = L.de[i]; }
    pthread_mutex_unlock(&mutex);
    free(L.h); free(L.m); free(L.de);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: cerca N modo muestra\n"); return 1; }
    N = atoi(argv[1]); modo = atoi(argv[2]); muestra = strtoull(argv[3], 0, 10);
    genera();
    fprintf(stderr, "N=%d una=%d dos=%d (dos/N²=%g)\n", N, n1, n2, (double)n2 / (N * N));
    if (modo == 2) baraja_objetivos();
    hist = calloc(9 * (N + 1), sizeof(uint64_t));
    mejor = malloc(nobj); memset(mejor, 255, nobj);
    mejor_de = calloc(nobj, sizeof(Perm));
    enum { HILOS = 8 };
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, obrero, (void *)(i + 1));
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    static const char *orig[3] = {"normal", "toroidal", "azar"};
    static const char *dest[3] = {"una", "dos", "cualquiera"};
    for (int o = 0; o < 3; o++)
        for (int t = 0; t < 3; t++)
            for (int x = 0; x <= N; x++)
                if (H(o, t, x)) printf("%s,%s,%d,%llu\n", orig[o], dest[t], x, (unsigned long long)H(o, t, x));
    if (modo == 0) {
        /* cuántas espirales tienen una no toroidal a distancia X */
        for (int t = 0; t < 2; t++) {
            uint64_t c[64] = {0};
            int ini = t ? n1 : 0, fin = t ? nobj : n1, m = 255, im = -1;
            for (int i = ini; i < fin; i++) {
                if (mejor[i] != 255) c[mejor[i]]++;
                if (mejor[i] < m) { m = mejor[i]; im = i; }
            }
            for (int x = 0; x <= N; x++)
                if (c[x]) printf("porobjetivo,%s,%d,%llu\n", dest[t], x, (unsigned long long)c[x]);
            if (im >= 0) {
                printf("ejemplo,%s,%d,", dest[t], m);
                for (int r = 0; r < N; r++) printf("%d ", mejor_de[im].x[r]);
                printf("->");
                for (int r = 0; r < N; r++) printf(" %d", obj[im].x[r]);
                printf("\n");
            }
        }
    }
    return 0;
}
