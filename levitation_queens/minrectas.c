/* Minimo numero de rectas toroidales (pendiente 2..p-2) que cubren las reinas de cada solucion.
   Recorre TODAS las soluciones de N=p. Para m<=3 guarda las pendientes de una cobertura minima.
   Uso: ./minrectas p [hilos]  -> data/minrectas_p{p}.txt                                          */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>

static int P, NT = 8, TORO = 0;
#define B1(r, c) (1ULL << (TORO ? ((r) + (c)) % P : (r) + (c)))
#define B2(r, c) (1ULL << (TORO ? ((r) - (c) + P) % P : (r) - (c) + P))
static atomic_int nextcol;
static FILE *DUMP; static int DUMPMAX; static pthread_mutex_t dmx = PTHREAD_MUTEX_INITIALIZER;
typedef struct { long hist[40]; long *pat; } Th;     /* pat[m][a1][a2][a3] (pendientes ordenadas) */

typedef struct { int mask[400], slope[400], n; int through[32][40], nth[32]; } Lines;
static __thread int best, bestsl[32], cur[32];
static int popc(int x) { return __builtin_popcount(x); }
static void dfs(const Lines *L, int covered, int cnt, int full, int maxsz) {
    if (covered == full) { if (cnt < best) { best = cnt; memcpy(bestsl, cur, sizeof(int) * cnt); } return; }
    int left = popc(full & ~covered);
    if (cnt + (left + maxsz - 1) / maxsz >= best) return;
    int r = __builtin_ctz(full & ~covered);
    /* probar primero las rectas que cubren mas filas nuevas */
    int order[40], gain[40], k = L->nth[r];
    for (int j = 0; j < k; j++) { order[j] = L->through[r][j]; gain[j] = popc(L->mask[order[j]] & ~covered); }
    for (int x = 1; x < k; x++) for (int y = x; y > 0 && gain[y] > gain[y - 1]; y--) {
        int t = gain[y]; gain[y] = gain[y - 1]; gain[y - 1] = t; t = order[y]; order[y] = order[y - 1]; order[y - 1] = t; }
    for (int j = 0; j < k; j++) { cur[cnt] = L->slope[order[j]]; dfs(L, covered | L->mask[order[j]], cnt + 1, full, maxsz); }
}
static void solve(Th *t, const int *s) {
    Lines L; L.n = 0; memset(L.nth, 0, sizeof L.nth);
    int maxsz = 1;
    for (int a = 2; a <= P - 2; a++) {
        int bym[64]; memset(bym, 0, sizeof bym);
        for (int i = 0; i < P; i++) bym[((s[i] - a * i) % P + P) % P] |= 1 << i;
        for (int c = 0; c < P; c++) if (bym[c]) {
            L.mask[L.n] = bym[c]; L.slope[L.n] = a;
            if (popc(bym[c]) > maxsz) maxsz = popc(bym[c]);
            for (int i = 0; i < P; i++) if (bym[c] >> i & 1) L.through[i][L.nth[i]++] = L.n;
            L.n++;
        }
    }
    best = P + 1;
    dfs(&L, 0, 0, (1 << P) - 1, maxsz);
    t->hist[best]++;
    if (DUMP && best <= DUMPMAX) {
        pthread_mutex_lock(&dmx); fprintf(DUMP, "%d", best);
        for (int i = 0; i < P; i++) fprintf(DUMP, " %d", s[i]);
        fprintf(DUMP, "\n"); pthread_mutex_unlock(&dmx);
    }
    if (best <= 3) {
        int a[3] = {0, 0, 0}; for (int j = 0; j < best; j++) a[j] = bestsl[j];
        for (int x = 0; x < 3; x++) for (int y = x + 1; y < 3; y++) if (a[y] < a[x]) { int z = a[x]; a[x] = a[y]; a[y] = z; }
        t->pat[((best * P + a[0]) * P + a[1]) * P + a[2]]++;
    }
}
static __thread Th *TT;
static int perm[32];
static void rec(int r, uint32_t cols, uint64_t d1, uint64_t d2, int *pm) {
    if (r == P) { solve(TT, pm); return; }
    for (int c = 0; c < P; c++) {
        if (cols >> c & 1 || d1 & B1(r, c) || d2 & B2(r, c)) continue;
        pm[r] = c; rec(r + 1, cols | 1u << c, d1 | B1(r, c), d2 | B2(r, c), pm);
    }
}
/* best/bestsl/cur son globales: cada hilo necesita los suyos */
static void *work(void *arg) {
    TT = arg; int pm[32];
    for (;;) {
        int c = atomic_fetch_add(&nextcol, 1);
        if (c >= P * P) break;
        int c0 = c / P, c1 = c % P;
        if (c0 == c1 || (B1(0, c0) & B1(1, c1)) || (B2(0, c0) & B2(1, c1))) continue;
        pm[0] = c0; pm[1] = c1;
        rec(2, (1u << c0) | (1u << c1), B1(0, c0) | B1(1, c1), B2(0, c0) | B2(1, c1), pm);
    }
    return NULL;
}
int main(int argc, char **argv) {
    P = atoi(argv[1]); if (argc > 2) NT = atoi(argv[2]);
    if (getenv("TORO")) TORO = 1;
    if (argc > 3) {                        /* modelo nulo: n permutaciones al azar (no soluciones) */
        long n = atol(argv[3]); Th t; memset(&t, 0, sizeof t); t.pat = calloc(4 * P * P * P, sizeof(long));
        int pm[32]; srand(1);
        for (long k = 0; k < n; k++) {
            for (int i = 0; i < P; i++) pm[i] = i;
            for (int i = P - 1; i > 0; i--) { int j = rand() % (i + 1), z = pm[i]; pm[i] = pm[j]; pm[j] = z; }
            solve(&t, pm);
        }
        printf("azar p=%d:", P); for (int m = 1; m <= P; m++) if (t.hist[m]) printf("  %d:%ld", m, t.hist[m]); printf("\n");
        return 0;
    }
    if (getenv("DUMPMAX")) { char fn[128]; DUMPMAX = atoi(getenv("DUMPMAX")); snprintf(fn, 128, TORO ? "data/clases_toro_p%d.txt" : "data/clases_p%d.txt", P); DUMP = fopen(fn, "w"); }
    Th *th = calloc(NT, sizeof(Th)); pthread_t tid[64];
    for (int i = 0; i < NT; i++) { th[i].pat = calloc(4 * P * P * P, sizeof(long)); pthread_create(&tid[i], NULL, work, &th[i]); }
    for (int i = 0; i < NT; i++) pthread_join(tid[i], NULL);
    char fn[128]; snprintf(fn, 128, TORO ? "data/minrectas_toro_p%d.txt" : "data/minrectas_p%d.txt", P); FILE *f = fopen(fn, "w");
    long tot = 0;
    for (int m = 1; m <= P; m++) { long h = 0; for (int i = 0; i < NT; i++) h += th[i].hist[m]; tot += h; fprintf(f, "H %d %ld\n", m, h); printf("%d rectas: %ld\n", m, h); }
    printf("total %ld\n", tot);
    for (int x = 0; x < 4 * P * P * P; x++) { long h = 0; for (int i = 0; i < NT; i++) h += th[i].pat[x]; if (h) fprintf(f, "P %d %d %d %d %ld\n", x / (P * P * P), x / (P * P) % P, x / P % P, x % P, h); }
    fclose(f);
    if (DUMP) fclose(DUMP);
    return 0;
}
