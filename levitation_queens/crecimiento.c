/* Niveles de fusion desde las toroidales de un tramo (igual que crecimiento.py --correr, en C y con hilos).
   Uso: ./crecimiento N [hasta] [hilos]     -> imprime el acumulado por nivel
   Compilar: cc -O3 -o crecimiento crecimiento.c -lpthread                                                */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

typedef unsigned __int128 K;
typedef struct Node { K key; struct Node *next; } Node;
typedef struct { K key; uint64_t d1, d2; uint32_t cm; } Half;

#define NB (1u << 26)
#define NL 8192
static int N, NT = 8;
static Node **heads;
static pthread_mutex_t locks[NL];
static K *sol; static size_t H, cap;
static uint32_t *gates; static int ngates;
static atomic_int nextgate;

typedef struct { K *nw; size_t n, cap; Node *chunk; int used; Half *op, *cl; } Th;

static inline uint32_t hsh(K k) {
    uint64_t x = (uint64_t)k ^ ((uint64_t)(k >> 64) * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 29; x *= 0x94D049BB133111EBULL; x ^= x >> 32;
    return (uint32_t)x & (NB - 1);
}
static int present(K k) {                       /* lectura sin cerrojo: las cadenas solo crecen por la cabeza */
    for (Node *p = heads[hsh(k)]; p; p = p->next) if (p->key == k) return 1;
    return 0;
}
static int insert(Th *t, K k) {
    uint32_t h = hsh(k); pthread_mutex_t *m = &locks[h & (NL - 1)];
    pthread_mutex_lock(m);
    for (Node *p = heads[h]; p; p = p->next) if (p->key == k) { pthread_mutex_unlock(m); return 0; }
    if (!t->chunk || t->used == 65536) { t->chunk = malloc(65536 * sizeof(Node)); t->used = 0; }
    Node *nd = &t->chunk[t->used++]; nd->key = k; nd->next = heads[h];
    __atomic_store_n(&heads[h], nd, __ATOMIC_RELEASE);
    pthread_mutex_unlock(m);
    if (t->n == t->cap) { t->cap = t->cap ? 2 * t->cap : 4096; t->nw = realloc(t->nw, t->cap * sizeof(K)); }
    t->nw[t->n++] = k;
    return 1;
}
static K pack(const int *p) { K k = 0; for (int i = 0; i < N; i++) k |= (K)p[i] << (5 * i); return k; }
static void add_sym(Th *t, K k) {
    int p[32], q[32], m = N - 1;
    for (int i = 0; i < N; i++) p[i] = (int)(k >> (5 * i)) & 31;
    for (int g = 0; g < 8; g++) {
        for (int i = 0; i < N; i++) {
            int j = p[i], i2, j2;
            switch (g) {
                case 0: i2 = i; j2 = j; break;        case 1: i2 = j; j2 = m - i; break;
                case 2: i2 = m - i; j2 = m - j; break; case 3: i2 = m - j; j2 = i; break;
                case 4: i2 = i; j2 = m - j; break;     case 5: i2 = m - i; j2 = j; break;
                case 6: i2 = j; j2 = i; break;         default: i2 = m - j; j2 = m - i; break;
            }
            q[i2] = j2;
        }
        insert(t, pack(q));
    }
}
static int cmp_cl(const void *a, const void *b) {
    const Half *x = a, *y = b;
    if (x->cm != y->cm) return x->cm < y->cm ? -1 : 1;
    return x->key < y->key ? -1 : x->key > y->key;
}
static int cmp_op(const void *a, const void *b) {
    const Half *x = a, *y = b;
    return x->key < y->key ? -1 : x->key > y->key;
}
static void *work(void *arg) {
    Th *t = arg; uint32_t full = (1u << N) - 1;
    for (;;) {
        int gi = atomic_fetch_add(&nextgate, 1);
        if (gi >= ngates) break;
        uint32_t G = gates[gi];
        for (size_t s = 0; s < H; s++) {
            K k = sol[s]; Half o = {0, 0, 0, 0}, c = {0, 0, 0, 0};
            for (int i = 0; i < N; i++) {
                int v = (int)(k >> (5 * i)) & 31;
                Half *h = (G >> i & 1) ? &o : &c;
                h->key |= (K)v << (5 * i); h->cm |= 1u << v; h->d1 |= 1ULL << (i + v); h->d2 |= 1ULL << (i - v + N);
            }
            t->op[s] = o; t->cl[s] = c;
        }
        qsort(t->op, H, sizeof(Half), cmp_op); qsort(t->cl, H, sizeof(Half), cmp_cl);
        size_t no = 0, nc = 0;
        for (size_t s = 0; s < H; s++) if (!s || t->op[s].key != t->op[s - 1].key) t->op[no++] = t->op[s];
        for (size_t s = 0; s < H; s++) if (!s || t->cl[s].key != t->cl[s - 1].key || t->cl[s].cm != t->cl[s - 1].cm) t->cl[nc++] = t->cl[s];
        for (size_t a = 0; a < no; a++) {
            Half *o = &t->op[a]; uint32_t want = full ^ o->cm;
            size_t lo = 0, hi = nc;
            while (lo < hi) { size_t mid = (lo + hi) / 2; if (t->cl[mid].cm < want) lo = mid + 1; else hi = mid; }
            for (; lo < nc && t->cl[lo].cm == want; lo++) {
                Half *c = &t->cl[lo];
                if ((o->d1 & c->d1) || (o->d2 & c->d2)) continue;
                K k = o->key | c->key;
                if (!present(k)) add_sym(t, k);
            }
        }
    }
    return NULL;
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); size_t hasta = argc > 2 ? strtoull(argv[2], 0, 10) : 1000000; if (argc > 3) NT = atoi(argv[3]);
    heads = calloc(NB, sizeof(Node *));
    for (int i = 0; i < NL; i++) pthread_mutex_init(&locks[i], NULL);
    /* puertas de onda: filas con (u*i - x) mod N < w, 2 <= w <= N-2; una por par {G, complemento} */
    gates = malloc(sizeof(uint32_t) * N * N * N); uint32_t full = (1u << N) - 1;
    for (int u = 1; u <= N / 2; u++) for (int x = 0; x < N; x++) for (int w = 2; w < N - 1; w++) {
        uint32_t G = 0;
        for (int i = 0; i < N; i++) if (((u * i - x) % N + N) % N < w) G |= 1u << i;
        uint32_t c = full ^ G; if (c < G) G = c;
        int dup = 0; for (int j = 0; j < ngates; j++) if (gates[j] == G) { dup = 1; break; }
        if (!dup) gates[ngates++] = G;
    }
    Th *th = calloc(NT, sizeof(Th));
    int p[32];
    for (int a = 2; a < N - 1; a++) for (int c = 0; c < N; c++) { for (int i = 0; i < N; i++) p[i] = (a * i + c) % N; insert(&th[0], pack(p)); }
    time_t t0 = time(NULL);
    printf("N=%d, %d puertas, %d hilos\n", N, ngates, NT);
    for (int lev = 0;; lev++) {
        size_t add = 0; for (int i = 0; i < NT; i++) add += th[i].n;
        if (H + add > cap) { cap = 2 * (H + add); sol = realloc(sol, cap * sizeof(K)); }
        for (int i = 0; i < NT; i++) { memcpy(sol + H, th[i].nw, th[i].n * sizeof(K)); H += th[i].n; th[i].n = 0; }
        printf("nivel %d: %zu (+%zu) %lds\n", lev, H, add, (long)(time(NULL) - t0)); fflush(stdout);
        if (!add || H >= hasta) break;
        pthread_t tid[64]; atomic_store(&nextgate, 0);
        for (int i = 0; i < NT; i++) { th[i].op = malloc(H * sizeof(Half)); th[i].cl = malloc(H * sizeof(Half)); pthread_create(&tid[i], NULL, work, &th[i]); }
        for (int i = 0; i < NT; i++) { pthread_join(tid[i], NULL); free(th[i].op); free(th[i].cl); }
    }
    return 0;
}
