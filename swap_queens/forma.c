/* forma.c — ¿cuántas soluciones de un conjunto pueden tomar a la vez la forma de
   espirales tras permutar libremente las N² celdas?

   Dadas S_1..S_k y destinos T_1..T_k existe p con p(S_i) = T_i sii para cada
   vector de pertenencia el átomo de celdas tiene el mismo tamaño en origen y
   destino. Se mantienen los átomos y cada pareja nueva (S,T) se admite solo si
   corta cada átomo igual en los dos lados.

   Búsqueda voraz al azar con reinicios: da cotas inferiores, no el máximo.

   Uso: forma N origen destino reinicios muestra
     origen  0: no toroidales. 1: permutaciones al azar, tantas como no toroidales.
             2: rectas (control positivo: la forma es exacta).
     destino 1: una espiral. 2: una o dos espirales.
     muestra: soluciones de origen que se prueban en cada reinicio.
   cc -O3 -mcpu=native -pthread -Wall forma.c -o forma */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 19
#define C (MAXN * MAXN)
#define HILOS 8
typedef struct { uint8_t x[MAXN]; } Perm;

static int N, orig, dest, reinicios, muestra;
static Perm *fue, *obj;
static int nfue, nobj, n1, capf, capo;
static uint64_t hist[4096], suma;
static int mejor;
static int sig_r;
static pthread_mutex_t cerrojo = PTHREAD_MUTEX_INITIALIZER;

static void pon(Perm **v, int *n, int *cap, const uint8_t *t) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *v = realloc(*v, *cap * sizeof(Perm)); }
    memcpy((*v)[*n].x, t, N);
    (*n)++;
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
    if (r == N) { if (!toroidal(s)) pon(&fue, &nfue, &capf, s); return; }
    uint32_t lib = ((1u << N) - 1) & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, N); }

static int es_recta(const uint8_t *t) {
    int a = (t[1] - t[0] + N) % N;
    for (int r = 0; r < N; r++)
        if (t[r] != (t[0] + a * r) % N) return 0;
    return 1;
}

static void dos(int r, int a, int c, int b, int d, uint8_t *t,
                uint32_t col, uint32_t dg, uint32_t ad) {
    if (r == N) { if (!es_recta(t)) pon(&obj, &nobj, &capo, t); return; }
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

static void objetivos(void) {
    uint8_t t[MAXN];
    for (int a = 2; a < N - 1; a++)
        for (int c = 0; c < N; c++) {
            for (int r = 0; r < N; r++) t[r] = (a * r + c) % N;
            pon(&obj, &nobj, &capo, t);
        }
    n1 = nobj;
    if (dest == 2) {
        for (int a = 0; a < N; a++)
            for (int b = a + 1; b < N; b++)
                for (int c = 0; c < N; c++)
                    for (int d = 0; d < N; d++)
                        dos(0, a, c, b, d, t, 0, 0, 0);
        qsort(obj + n1, nobj - n1, sizeof(Perm), cmp);
        int k = n1;
        for (int i = n1; i < nobj; i++)
            if (i == n1 || memcmp(&obj[i], &obj[i - 1], N)) obj[k++] = obj[i];
        nobj = k;
    }
}

static uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

typedef struct {
    uint16_t src[C], dst[C];
    int tam[C + 1], na;
    int cs[C + 1], ct[C + 1], nuevo[C + 1];
} Atomos;

static int compatible(Atomos *A, const uint8_t *s, const uint8_t *t) {
    int ok = 1;
    for (int r = 0; r < N; r++) {
        A->cs[A->src[r * N + s[r]]]++;
        A->ct[A->dst[r * N + t[r]]]++;
    }
    for (int r = 0; r < N; r++) {
        int i = A->src[r * N + s[r]];
        if (A->cs[i] != A->ct[i]) { ok = 0; break; }
    }
    if (ok)
        for (int r = 0; r < N; r++) {
            int i = A->dst[r * N + t[r]];
            if (A->cs[i] != A->ct[i]) { ok = 0; break; }
        }
    for (int r = 0; r < N; r++) {
        A->cs[A->src[r * N + s[r]]] = 0;
        A->ct[A->dst[r * N + t[r]]] = 0;
    }
    return ok;
}

static void fija(Atomos *A, const uint8_t *s, const uint8_t *t) {
    for (int r = 0; r < N; r++) A->cs[A->src[r * N + s[r]]]++;
    for (int r = 0; r < N; r++) {
        int i = A->src[r * N + s[r]];
        if (A->cs[i] && A->cs[i] < A->tam[i] && !A->nuevo[i]) A->nuevo[i] = A->na++;
    }
    for (int r = 0; r < N; r++) {
        int i = A->src[r * N + s[r]], n = A->nuevo[i];
        if (n) { A->src[r * N + s[r]] = n; A->tam[n]++; A->tam[i]--; }
    }
    for (int r = 0; r < N; r++) {
        int i = A->dst[r * N + t[r]];
        if (A->nuevo[i]) A->dst[r * N + t[r]] = A->nuevo[i];
    }
    memset(A->cs, 0, sizeof(A->cs));
    memset(A->nuevo, 0, sizeof(A->nuevo));
}

static void *obrero(void *arg) {
    uint64_t sem = 88172645463325252ull + 0x9E3779B97F4A7C15ull * (uint64_t)(intptr_t)arg;
    Atomos *A = malloc(sizeof(Atomos));
    int *pf = malloc(nfue * sizeof(int)), *po = malloc(nobj * sizeof(int));
    for (int i = 0; i < nfue; i++) pf[i] = i;
    for (int i = 0; i < nobj; i++) po[i] = i;
    int lim = muestra < nfue ? muestra : nfue;
    while (__sync_fetch_and_add(&sig_r, 1) < reinicios) {
        memset(A, 0, sizeof(Atomos));
        A->na = 1;
        A->tam[0] = N * N;
        for (int i = 0; i < lim; i++) {
            int j = i + xs(&sem) % (nfue - i), v = pf[i];
            pf[i] = pf[j]; pf[j] = v;
        }
        int k = 0;
        for (int i = 0; i < lim; i++) {
            const uint8_t *s = fue[pf[i]].x;
            int ini = xs(&sem) % nobj;
            for (int q = 0; q < nobj; q++) {
                int w = q + ini; if (w >= nobj) w -= nobj;
                const uint8_t *t = obj[po[w]].x;
                if (compatible(A, s, t)) { fija(A, s, t); k++; break; }
            }
        }
        pthread_mutex_lock(&cerrojo);
        hist[k]++; suma += k;
        if (k > mejor) mejor = k;
        pthread_mutex_unlock(&cerrojo);
        for (int i = nobj - 1; i > 0; i--) {
            int j = xs(&sem) % (i + 1), v = po[i];
            po[i] = po[j]; po[j] = v;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "uso: forma N origen destino reinicios muestra\n"); return 1; }
    N = atoi(argv[1]); orig = atoi(argv[2]); dest = atoi(argv[3]);
    reinicios = atoi(argv[4]); muestra = atoi(argv[5]);
    objetivos();
    uint8_t s[MAXN];
    reinas(0, s, 0, 0, 0);
    if (orig == 1) {
        uint64_t sem = 0x2545F4914F6CDD1Dull;
        for (int i = 0; i < nfue; i++) {
            for (int r = 0; r < N; r++) fue[i].x[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs(&sem) % (r + 1), v = fue[i].x[r];
                fue[i].x[r] = fue[i].x[j]; fue[i].x[j] = v;
            }
        }
    } else if (orig == 2) {
        nfue = 0;
        for (int i = 0; i < n1; i++) pon(&fue, &nfue, &capf, obj[i].x);
    }
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, obrero, (void *)(i + 1));
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    printf("N=%d origen=%d destino=%d piezas=%d objetivos=%d mejor=%d media=%.2f\n",
           N, orig, dest, nfue, nobj, mejor, (double)suma / reinicios);
    return 0;
}
