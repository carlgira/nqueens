/* Variante de minrectas.c solo para toroidales, ~p veces mas rapida: fija T(0)=0 (las toroidales
   son invariantes por traslacion de columnas) y multiplica el recuento por p.
   Uso: ./minrectas_toro p [hilos]      -> data/minrectas_toro0_p{p}.txt (histograma, ya multiplicado por p)
   Entorno: DUMPMAX=k  vuelca las de clase <= k con T(0)=0 a data/clases_toro0_p{p}.txt
            CAP=k      no busca coberturas de mas de k rectas: las de clase > k se cuentan juntas en k+1 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>

static int P, NT = 8, CAP = 0;
static atomic_int nextjob;
static FILE *DUMP; static int DUMPMAX; static pthread_mutex_t dmx = PTHREAD_MUTEX_INITIALIZER;
typedef struct { long hist[40]; } Th;
typedef struct { int mask[800], n; int through[32][40], nth[32]; } Lines;
static __thread int best;
static int popc(int x) { return __builtin_popcount(x); }
static void dfs(const Lines *L, int covered, int cnt, int full, int maxsz) {
    if (covered == full) { if (cnt < best) best = cnt; return; }
    int left = popc(full & ~covered);
    if (cnt + (left + maxsz - 1) / maxsz >= best) return;
    int r = __builtin_ctz(full & ~covered);
    int order[40], gain[40], k = L->nth[r];
    for (int j = 0; j < k; j++) { order[j] = L->through[r][j]; gain[j] = popc(L->mask[order[j]] & ~covered); }
    for (int x = 1; x < k; x++) for (int y = x; y > 0 && gain[y] > gain[y - 1]; y--) {
        int t = gain[y]; gain[y] = gain[y - 1]; gain[y - 1] = t; t = order[y]; order[y] = order[y - 1]; order[y - 1] = t; }
    for (int j = 0; j < k; j++) dfs(L, covered | L->mask[order[j]], cnt + 1, full, maxsz);
}
static void solve(Th *t, const int *s) {
    Lines L; L.n = 0; memset(L.nth, 0, sizeof L.nth);
    int maxsz = 1;
    for (int a = 2; a <= P - 2; a++) {
        int bym[64]; memset(bym, 0, sizeof bym);
        for (int i = 0; i < P; i++) bym[((s[i] - a * i) % P + P) % P] |= 1 << i;
        for (int c = 0; c < P; c++) if (bym[c]) {
            L.mask[L.n] = bym[c];
            if (popc(bym[c]) > maxsz) maxsz = popc(bym[c]);
            for (int i = 0; i < P; i++) if (bym[c] >> i & 1) L.through[i][L.nth[i]++] = L.n;
            L.n++;
        }
    }
    best = CAP ? CAP + 1 : P + 1;
    dfs(&L, 0, 0, (1 << P) - 1, maxsz);
    t->hist[best]++;
    if (DUMP && best <= DUMPMAX) {
        pthread_mutex_lock(&dmx); fprintf(DUMP, "%d", best);
        for (int i = 0; i < P; i++) fprintf(DUMP, " %d", s[i]);
        fprintf(DUMP, "\n"); pthread_mutex_unlock(&dmx);
    }
}
static __thread Th *TT;
#define B1(r, c) (1u << (((r) + (c)) % P))
#define B2(r, c) (1u << (((r) - (c) + P) % P))
static void rec(int r, uint32_t cols, uint32_t d1, uint32_t d2, int *pm) {
    if (r == P) { solve(TT, pm); return; }
    for (int c = 0; c < P; c++) {
        if (cols >> c & 1 || d1 & B1(r, c) || d2 & B2(r, c)) continue;
        pm[r] = c; rec(r + 1, cols | 1u << c, d1 | B1(r, c), d2 | B2(r, c), pm);
    }
}
static void *work(void *arg) {
    TT = arg; int pm[32];
    for (;;) {
        int j = atomic_fetch_add(&nextjob, 1);
        if (j >= P * P) break;
        int c[3] = {0, j / P, j % P}; uint32_t cols = 0, d1 = 0, d2 = 0; int ok = 1;
        for (int r = 0; r < 3 && ok; r++) {
            if (cols >> c[r] & 1 || d1 & B1(r, c[r]) || d2 & B2(r, c[r])) ok = 0;
            cols |= 1u << c[r]; d1 |= B1(r, c[r]); d2 |= B2(r, c[r]); pm[r] = c[r];
        }
        if (ok) rec(3, cols, d1, d2, pm);
    }
    return NULL;
}
int main(int argc, char **argv) {
    P = atoi(argv[1]); if (argc > 2) NT = atoi(argv[2]);
    if (getenv("CAP")) CAP = atoi(getenv("CAP"));
    if (getenv("DUMPMAX")) { char fn[128]; DUMPMAX = atoi(getenv("DUMPMAX")); snprintf(fn, 128, "data/clases_toro0_p%d.txt", P); DUMP = fopen(fn, "w"); }
    Th *th = calloc(NT, sizeof(Th)); pthread_t tid[64];
    for (int i = 0; i < NT; i++) pthread_create(&tid[i], NULL, work, &th[i]);
    for (int i = 0; i < NT; i++) pthread_join(tid[i], NULL);
    char fn[128]; snprintf(fn, 128, "data/minrectas_toro0_p%d.txt", P); FILE *f = fopen(fn, CAP ? "a" : "w");
    long tot = 0;
    if (CAP) fprintf(f, "# minrectas_toro CAP=%d (la ultima fila agrupa clase > %d)\n", CAP, CAP);
    for (int m = 1; m <= P; m++) { long h = 0; for (int i = 0; i < NT; i++) h += th[i].hist[m]; h *= P; tot += h; if (h) { fprintf(f, "H %d %ld\n", m, h); printf("%d rectas: %ld\n", m, h); } }
    printf("total %ld\n", tot);
    fclose(f); if (DUMP) fclose(DUMP);
    return 0;
}
