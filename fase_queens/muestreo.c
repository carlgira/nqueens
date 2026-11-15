/*
 * muestreo.c — Muchas condiciones iniciales, todas saliendo de una solución.
 *
 * Planetas sin interacción, uno por órbita. Rejilla fina: posiciones y velocidades en
 * unidades de 1/MM de sector. Foto k: columna_r = floor((th_r + k v_r) mod (N*MM) / MM).
 * Para cada tablero de partida S y cada muestra: th_r = S_r*MM + azar(MM), v_r = azar(N*MM).
 * El bucle tiene periodo p = N*MM / gcd(N*MM, v_0..v_{N-1}).
 *
 * Firma del bucle:
 *   - grupo: soluciones distintas que aparecen en el bucle;
 *   - huecos: distancias (cíclicas) entre fotos-solución consecutivas;
 *   - clase de huecos: 'igual' (todos iguales), 'bloques' (secuencia periódica más corta),
 *     'otra';
 *   - velocidades: número de valores distintos; si son 2, si se separan medio tablero.
 * Se guardan sin repetir los bucles con >= GMIN soluciones (firma canónica por rotación).
 *
 * uso: ./muestreo N conjunto MM muestras_por_inicio semilla [GMIN=4]
 *   salida: datos/muestreo_N<N>_<conj>_MM<MM>.txt  y resumen por pantalla
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int N, Q, cap; static int *sol; static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) { if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return; }
    for (int c = 0; c < N; c++) { if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue; cur[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N)); }
}
static unsigned long long rs = 12345;
static inline unsigned long long nx(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline int ri(int n) { return (int)((nx() >> 11) % (unsigned long long)n); }
static void randperm(int *p) { for (int r = 0; r < N; r++) p[r] = r; for (int r = N - 1; r > 0; r--) { int j = ri(r + 1); int t = p[r]; p[r] = p[j]; p[j] = t; } }
static int have(const int *g, int n) { for (int k = 0; k < n; k++) if (!memcmp(g, sol + (long)k * N, sizeof(int) * N)) return 1; return 0; }
static void push(const int *g) { if (Q == cap) { cap = cap ? cap * 2 : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, g, sizeof(int) * N); Q++; }
static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }
static int cmpi(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

/* conjunto de firmas ya vistas (hash de 64 bits) */
#define HB 24
static uint64_t *seen; static long nseen;
static int insert(uint64_t h) {
    if (!h) h = 1; uint64_t i = h >> (64 - HB);
    while (seen[i]) { if (seen[i] == h) return 0; i = (i + 1) & ((1u << HB) - 1); }
    if (nseen > (1l << (HB - 1))) return 0;
    seen[i] = h; nseen++; return 1;
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); int set = atoi(argv[2]); int MM = atoi(argv[3]); long S = atol(argv[4]);
    rs = strtoull(argv[5], 0, 10) * 0x9E3779B97F4A7C15ULL | 1; int GMIN = argc > 6 ? atoi(argv[6]) : 4;
    bt(0, 0, 0, 0); int want = Q;
    unsigned long long keep = rs;
    if (set > 0) { rs ^= (unsigned long long)set * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32]; while (Q < want) { randperm(p); if (!have(p, Q)) push(p); } }
    else if (set < 0) {
        rs = 12345 ^ ((unsigned long long)(-set) * 0x9E3779B97F4A7C15ULL); Q = 0; int p[32], g[32], q[32];
        while (Q < want) { randperm(p); for (int e = 0; e < 8; e++) { memcpy(q, p, sizeof q); if (e & 4) for (int r = 0; r < N; r++) q[p[r]] = r; for (int r = 0; r < N; r++) g[r] = q[(e & 2) ? N - 1 - r : r]; if (e & 1) for (int r = 0; r < N; r++) g[r] = N - 1 - g[r]; if (!have(g, Q)) push(g); } }
    }
    rs = keep;
    char tag[32]; if (set == 0) snprintf(tag, sizeof tag, "sol"); else if (set > 0) snprintf(tag, sizeof tag, "azar%d", set); else snprintf(tag, sizeof tag, "sim%d", -set);
    long tot = 1; for (int r = 0; r < N; r++) tot *= N;
    int *idx = malloc(sizeof(int) * tot); for (long i = 0; i < tot; i++) idx[i] = -1;
    for (int i = 0; i < Q; i++) { long k = 0; for (int r = 0; r < N; r++) k = k * N + sol[(long)i * N + r]; idx[k] = i; }
    mkdir("datos", 0755);
    char fn[160]; snprintf(fn, sizeof fn, "datos/tableros_N%d_%s.txt", N, tag);
    FILE *ft = fopen(fn, "w"); for (int i = 0; i < Q; i++) for (int r = 0; r < N; r++) fprintf(ft, "%d%c", sol[(long)i * N + r], r < N - 1 ? ' ' : '\n'); fclose(ft);
    snprintf(fn, sizeof fn, "datos/muestreo_N%d_%s_MM%d.txt", N, tag, MM);
    FILE *fo = fopen(fn, "w");
    fprintf(fo, "# n_sol | clase_huecos | huecos | n_vel | medio_tablero | periodo | grupo | th | v\n");
    seen = calloc(1u << HB, sizeof(uint64_t));
    int NM = N * MM;
    long hist[256] = {0}, cls[3][64] = {{0}}, uniq = 0, total = 0;
    int th[32], v[32], fot[512], g[512];
    for (int i = 0; i < Q; i++) {
        const int *B = sol + (long)i * N;
        for (long s = 0; s < S; s++) {
            int gg = NM;
            for (int r = 0; r < N; r++) { th[r] = B[r] * MM + ri(MM); v[r] = ri(NM); gg = gcd(gg, v[r]); }
            int p = NM / gg; int nf = 0, n = 0;
            for (int k = 0; k < p; k++) {
                long key = 0; for (int r = 0; r < N; r++) key = key * N + ((th[r] + k * v[r]) % NM) / MM;
                int id = idx[key]; if (id < 0) continue;
                fot[nf] = k; g[nf] = id; nf++;
            }
            /* soluciones distintas */
            int gs[512]; memcpy(gs, g, sizeof(int) * nf); qsort(gs, nf, sizeof(int), cmpi);
            for (int a = 0; a < nf; a++) if (!a || gs[a] != gs[a - 1]) gs[n++] = gs[a];
            total++; if (n < 256) hist[n]++;
            if (n < GMIN) continue;
            /* huecos cíclicos entre fotos-solución */
            int gap[512]; for (int a = 0; a < nf; a++) gap[a] = (a + 1 < nf ? fot[a + 1] : fot[0] + p) - fot[a];
            int c = 2;
            int alleq = 1; for (int a = 1; a < nf; a++) if (gap[a] != gap[0]) alleq = 0;
            if (alleq) c = 0;
            else { for (int q = 1; q < nf; q++) if (nf % q == 0) { int ok = 1; for (int a = 0; a < nf && ok; a++) if (gap[a] != gap[a % q]) ok = 0; if (ok) { c = 1; break; } } }
            if (n < 64) cls[c][n]++;
            /* velocidades */
            int vals[32], nv = 0; for (int r = 0; r < N; r++) { int dup = 0; for (int a = 0; a < nv; a++) if (vals[a] == v[r]) dup = 1; if (!dup) vals[nv++] = v[r]; }
            int half = (nv == 2 && ((vals[0] - vals[1] + NM) % NM) == NM / 2);
            /* firma canónica: grupo + huecos rotados al mínimo lexicográfico */
            int best = 0; for (int q = 1; q < nf; q++) { for (int a = 0; a < nf; a++) { int x = gap[(q + a) % nf], y = gap[(best + a) % nf]; if (x != y) { if (x < y) best = q; break; } } }
            uint64_t h = 1469598103934665603ULL;
            for (int a = 0; a < n; a++) { h ^= (uint64_t)gs[a] + 7; h *= 1099511628211ULL; }
            for (int a = 0; a < nf; a++) { h ^= (uint64_t)gap[(best + a) % nf] + 1000; h *= 1099511628211ULL; }
            if (!insert(h)) continue;
            uniq++;
            static const char *cn[3] = {"igual", "bloques", "otra"};
            fprintf(fo, "%d | %s |", n, cn[c]); for (int a = 0; a < nf; a++) fprintf(fo, " %d", gap[(best + a) % nf]);
            fprintf(fo, " | %d | %d | %d |", nv, half, p); for (int a = 0; a < n; a++) fprintf(fo, " %d", gs[a]);
            fprintf(fo, " | "); for (int r = 0; r < N; r++) fprintf(fo, "%d%c", th[r], r < N - 1 ? ',' : ' ');
            fprintf(fo, "| "); for (int r = 0; r < N; r++) fprintf(fo, "%d%c", v[r], r < N - 1 ? ',' : '\n');
        }
    }
    fclose(fo);
    printf("N=%d %s MM=%d: %ld bucles probados, firmas distintas con >=%d soluciones: %ld\n", N, tag, MM, total, GMIN, uniq);
    printf("  bucles por numero de soluciones distintas:"); for (int k = 1; k < 256; k++) if (hist[k]) printf(" %d:%ld", k, hist[k]); printf("\n");
    printf("  con >=%d soluciones, por clase de huecos (igual / bloques / otra):\n", GMIN);
    for (int k = GMIN; k < 64; k++) if (cls[0][k] + cls[1][k] + cls[2][k]) printf("    %2d soluciones: %ld / %ld / %ld\n", k, cls[0][k], cls[1][k], cls[2][k]);
    return 0;
}
