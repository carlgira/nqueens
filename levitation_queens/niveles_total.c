/* Niveles de fusion desde las toroidales de un tramo, HASTA EL FINAL, con punto de guardado.
   Mismo proceso que crecimiento.c (puertas de onda, 8 simetrias por nivel), pero calculado al reves:
   se enumeran todas las soluciones y, por nivel y puerta, una solucion aun no alcanzada entra si sus
   dos mitades ya existen entre las alcanzadas. Coste por nivel ~ Q * puertas, sin probar pares.

   Uso: ./niveles_total N [hilos]
   Guarda tras cada nivel: data/niveles_total_N{N}.lev (nivel de cada solucion, orden por clave)
                           data/niveles_total_N{N}.txt (acumulado por nivel)
   Al terminar:            data/niveles_total_N{N}_nunca.txt (soluciones no alcanzadas)
   Si existe el .lev, continua desde ahi.
   Compilar: cc -O3 -o niveles_total niveles_total.c -lpthread                                          */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

typedef unsigned __int128 K;
static int N, NT = 5, LEV, MODE;            /* MODE 0: tablas con las conocidas; 1: tablas con las no alcanzadas */
static K *U; static size_t Q, cap;
static signed char *lev;
static uint32_t gates[20000]; static int ngates;
static atomic_int nextgate;
#define MARK (1ULL << 63)

typedef struct { uint64_t *t; size_t cap, n, alloc; } Set;
static inline uint64_t mix(uint64_t x) { x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 29; x *= 0x94D049BB133111EBULL; x ^= x >> 32; return x; }
static void set_reset(Set *s) {
    s->cap = 4096; s->n = 0;
    if (s->alloc < s->cap) { free(s->t); s->t = malloc(s->cap * 8); s->alloc = s->cap; }
    memset(s->t, 0, s->cap * 8);
}
static uint64_t *set_find(Set *s, uint64_t v) {          /* v guardado como v+1; devuelve hueco (vacio u ocupado por v) */
    size_t m = s->cap - 1, h = mix(v) & m;
    while (s->t[h] && (s->t[h] & ~MARK) != v + 1) h = (h + 1) & m;
    return &s->t[h];
}
static void set_add(Set *s, uint64_t v) {
    uint64_t *p = set_find(s, v);
    if (*p) return;
    *p = v + 1;
    if (++s->n * 2 > s->cap) {
        uint64_t *old = s->t; size_t oc = s->cap;
        s->cap *= 2; s->t = calloc(s->cap, 8); s->alloc = s->cap;
        for (size_t i = 0; i < oc; i++) if (old[i]) *set_find(s, (old[i] & ~MARK) - 1) = old[i];
        free(old);
    }
}
static inline uint64_t enc(K k, const int *rows, int nr) {
    uint64_t e = 0;
    for (int j = 0; j < nr; j++) e = e * N + ((uint64_t)(k >> (5 * rows[j])) & 31);
    return e;
}
static K pack(const int *p) { K k = 0; for (int i = 0; i < N; i++) k |= (K)p[i] << (5 * i); return k; }
static size_t find(K k) {
    size_t lo = 0, hi = Q;
    while (lo < hi) { size_t mid = (lo + hi) / 2; if (U[mid] < k) lo = mid + 1; else hi = mid; }
    return lo;
}
static void mark_sym(size_t idx) {
    int p[32], q[32], m = N - 1; K k = U[idx];
    for (int i = 0; i < N; i++) p[i] = (int)(k >> (5 * i)) & 31;
    for (int g = 0; g < 8; g++) {
        for (int i = 0; i < N; i++) {
            int j = p[i], i2, j2;
            switch (g) {
                case 0: i2 = i; j2 = j; break;         case 1: i2 = j; j2 = m - i; break;
                case 2: i2 = m - i; j2 = m - j; break; case 3: i2 = m - j; j2 = i; break;
                case 4: i2 = i; j2 = m - j; break;     case 5: i2 = m - i; j2 = j; break;
                case 6: i2 = j; j2 = i; break;         default: i2 = m - j; j2 = m - i; break;
            }
            q[i2] = j2;
        }
        size_t j = find(pack(q));
        if (lev[j] < 0) lev[j] = (signed char)LEV;
    }
}
static void *work(void *arg) {
    Set A = {0}, B = {0};
    int ra[32], rb[32];
    for (;;) {
        int gi = atomic_fetch_add(&nextgate, 1);
        if (gi >= ngates) break;
        uint32_t G = gates[gi]; int na = 0, nb = 0;
        for (int i = 0; i < N; i++) if (G >> i & 1) ra[na++] = i; else rb[nb++] = i;
        set_reset(&A); set_reset(&B);
        if (MODE == 0) {
            for (size_t i = 0; i < Q; i++) { signed char l = lev[i]; if (l >= 0 && l < LEV) { set_add(&A, enc(U[i], ra, na)); set_add(&B, enc(U[i], rb, nb)); } }
            for (size_t i = 0; i < Q; i++) if (lev[i] < 0) {
                if (!*set_find(&A, enc(U[i], ra, na))) continue;
                if (!*set_find(&B, enc(U[i], rb, nb))) continue;
                mark_sym(i);
            }
        } else {
            for (size_t i = 0; i < Q; i++) { signed char l = lev[i]; if (l < 0 || l == LEV) { set_add(&A, enc(U[i], ra, na)); set_add(&B, enc(U[i], rb, nb)); } }
            for (size_t i = 0; i < Q; i++) { signed char l = lev[i]; if (l >= 0 && l < LEV) {
                uint64_t *p = set_find(&A, enc(U[i], ra, na)); if (*p) *p |= MARK;
                p = set_find(&B, enc(U[i], rb, nb)); if (*p) *p |= MARK; } }
            for (size_t i = 0; i < Q; i++) if (lev[i] < 0) {
                if (!(*set_find(&A, enc(U[i], ra, na)) & MARK)) continue;
                if (!(*set_find(&B, enc(U[i], rb, nb)) & MARK)) continue;
                mark_sym(i);
            }
        }
    }
    free(A.t); free(B.t);
    return NULL;
}
static int perm[32];
static void rec(int r, uint32_t cols, uint64_t d1, uint64_t d2) {
    if (r == N) { if (Q == cap) { cap = cap ? 2 * cap : 1 << 20; U = realloc(U, cap * sizeof(K)); } U[Q++] = pack(perm); return; }
    for (int c = 0; c < N; c++) {
        if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue;
        perm[r] = c; rec(r + 1, cols | 1u << c, d1 | 1ULL << (r + c), d2 | 1ULL << (r - c + N));
    }
}
static int cmpK(const void *a, const void *b) { K x = *(const K *)a, y = *(const K *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    N = atoi(argv[1]); if (argc > 2) NT = atoi(argv[2]);
    char f_lev[256], f_tmp[256], f_txt[256], f_no[256];
    snprintf(f_lev, 256, "data/niveles_total_N%d.lev", N); snprintf(f_tmp, 256, "data/niveles_total_N%d.lev.tmp", N);
    snprintf(f_txt, 256, "data/niveles_total_N%d.txt", N); snprintf(f_no, 256, "data/niveles_total_N%d_nunca.txt", N);
    time_t t0 = time(NULL);
    rec(0, 0, 0, 0); qsort(U, Q, sizeof(K), cmpK);
    uint32_t full = (1u << N) - 1;
    for (int u = 1; u <= N / 2; u++) for (int x = 0; x < N; x++) for (int w = 2; w < N - 1; w++) {
        uint32_t G = 0;
        for (int i = 0; i < N; i++) if (((u * i - x) % N + N) % N < w) G |= 1u << i;
        if (__builtin_popcount(G) * 2 > N) G = full ^ G;           /* lado A = el pequeno */
        int dup = 0; for (int j = 0; j < ngates; j++) if (gates[j] == G) { dup = 1; break; }
        if (!dup) gates[ngates++] = G;
    }
    lev = malloc(Q); memset(lev, -1, Q);
    FILE *f = fopen(f_lev, "rb"); int start = 0;
    if (f) { if (fread(lev, 1, Q, f) != Q) { fprintf(stderr, "lev incompleto\n"); return 1; } fclose(f);
             for (size_t i = 0; i < Q; i++) if (lev[i] > start) start = lev[i]; }
    else { int p[32]; for (int a = 2; a < N - 1; a++) for (int c = 0; c < N; c++) { for (int i = 0; i < N; i++) p[i] = (a * i + c) % N; lev[find(pack(p))] = 0; } }
    size_t H = 0; for (size_t i = 0; i < Q; i++) H += lev[i] >= 0;
    printf("N=%d: Q=%zu, %d puertas, %d hilos, enumerado en %lds; %s nivel %d con %zu\n", N, Q, ngates, NT, (long)(time(NULL) - t0),
           start ? "continua desde" : "empieza en", start, H); fflush(stdout);
    for (LEV = start + 1;; LEV++) {
        MODE = (H > Q - H);
        pthread_t tid[64]; atomic_store(&nextgate, 0);
        for (int i = 0; i < NT; i++) pthread_create(&tid[i], NULL, work, NULL);
        for (int i = 0; i < NT; i++) pthread_join(tid[i], NULL);
        size_t H2 = 0; for (size_t i = 0; i < Q; i++) H2 += lev[i] >= 0;
        printf("nivel %d: %zu (+%zu) %.3f%% %lds\n", LEV, H2, H2 - H, 100.0 * H2 / Q, (long)(time(NULL) - t0)); fflush(stdout);
        if (H2 == H) break;
        f = fopen(f_tmp, "wb"); fwrite(lev, 1, Q, f); fclose(f); rename(f_tmp, f_lev);
        f = fopen(f_txt, "a"); fprintf(f, "%d %zu %zu\n", LEV, H2, H2 - H); fclose(f);
        H = H2;
    }
    f = fopen(f_no, "w");
    for (size_t i = 0; i < Q; i++) if (lev[i] < 0) { for (int r = 0; r < N; r++) fprintf(f, "%d ", (int)(U[i] >> (5 * r)) & 31); fprintf(f, "\n"); }
    fclose(f);
    printf("terminado: %zu alcanzadas, %zu nunca\n", H, Q - H);
    return 0;
}
