/*
 * rachas_mrv.c — Igual que rachas.c (S, S+1, ..., S+R-1 todas soluciones), pero en cada paso
 * coloca la fila con MENOS columnas posibles (heurística MRV) en vez de ir de arriba abajo.
 * Para N grandes la diferencia es enorme: los choques al envolver se detectan antes.
 *
 * Máscara de columnas prohibidas de la fila r por la copia t: en la copia la columna es
 * ct = c + t (mod N); d1[t] marca r+ct, e2[t] marca ct-r+N-1. Se rota la máscara t posiciones.
 *
 * Máscaras de 64 bits: vale hasta N=32 (las diagonales r+ct llegan a 2N-2).
 * env SEMILLA=k: elige las columnas en orden aleatorio (para reinicios).
 * uso: ./rachas_mrv N R [limite_s=60] [max_ejemplos=3] [fijas "r:c,..."] [contar=0: parar en el primero]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int N, R, S[64], maxex, contar; static long cnt, nodes; static double lim; static clock_t t0; static int to, done_;
typedef unsigned long long u64;
static u64 d1[64], e2[64]; static u64 full;
static int placed[64];
static unsigned long long semilla; /* env SEMILLA=k (k>0): orden de columnas al azar */
static inline u64 rng(void) { semilla ^= semilla << 13; semilla ^= semilla >> 7; semilla ^= semilla << 17; return semilla; }

static inline u64 rot(u64 m, int t) { return t ? ((m >> t) | (m << (N - t))) & full : m; }
static inline u64 cand(int r, u64 cols) {
    u64 bad = cols;
    for (int t = 0; t < R; t++) { u64 m = ((d1[t] >> r) | (e2[t] >> (N - 1 - r))) & full; bad |= rot(m, t); }
    return ~bad & full;
}
static void put(int r, int c) { for (int t = 0; t < R; t++) { int ct = c + t; if (ct >= N) ct -= N; d1[t] ^= 1ull << (r + ct); e2[t] ^= 1ull << (ct - r + N - 1); } }

static void dfs(int k, u64 cols) {
    if (to || done_) return;
    if ((++nodes & 0xffff) == 0 && (double)(clock() - t0) / CLOCKS_PER_SEC > lim) { to = 1; return; }
    if (k == N) {
        if (cnt < maxex) { printf("  ["); for (int i = 0; i < N; i++) printf("%d%s", S[i], i < N - 1 ? "," : ""); printf("]\n"); fflush(stdout); }
        cnt++; if (!contar) done_ = 1; return;
    }
    int br = -1, bn = 99; u64 bm = 0;
    for (int r = 0; r < N; r++) { if (placed[r]) continue; u64 m = cand(r, cols); int n = __builtin_popcountll(m); if (n < bn) { bn = n; br = r; bm = m; if (!n) return; } }
    placed[br] = 1;
    while (bm) { int c;
        if (semilla) { int n = __builtin_popcountll(bm), k = (int)(rng() % n); u64 x = bm; while (k--) x &= x - 1; c = __builtin_ctzll(x); bm &= ~(1ull << c); }
        else { c = __builtin_ctzll(bm); bm &= bm - 1; } put(br, c); S[br] = c; dfs(k + 1, cols | 1ull << c); put(br, c); if (to || done_) break; }
    placed[br] = 0;
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); R = atoi(argv[2]); lim = argc > 3 ? atof(argv[3]) : 60; maxex = argc > 4 ? atoi(argv[4]) : 3;
    contar = argc > 6 ? atoi(argv[6]) : 0; full = N >= 64 ? ~0ull : (1ull << N) - 1;
    int k = 0; u64 cols = 0;
    if (argc > 5 && argv[5][0]) { char *p = argv[5]; while (*p) { int r, c, n; if (sscanf(p, "%d:%d%n", &r, &c, &n) != 2) break; if (r < 0) r += N;
        if (!((cand(r, cols) >> c) & 1)) { printf("esquina incompatible\n"); return 0; } put(r, c); S[r] = c; placed[r] = 1; cols |= 1ull << c; k++; p += n; if (*p == ',') p++; } }
    if (getenv("SEMILLA")) semilla = 0x9E3779B97F4A7C15ull * (unsigned long long)atoll(getenv("SEMILLA"));
    t0 = clock(); dfs(k, cols);
    printf("N=%d R=%d: %ld encontrados%s%s  (%.1fs, nodos %ld)\n", N, R, cnt, contar ? "" : " (para en el primero)", to ? " [tiempo agotado]" : (cnt || contar ? "" : " [no existe: busqueda completa]"), (double)(clock() - t0) / CLOCKS_PER_SEC, nodes);
    return 0;
}
