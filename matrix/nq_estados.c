// Poda exacta: dos prefijos con el mismo (columnas, diagonales aún en tablero) tienen las mismas continuaciones.
// BFS por filas fusionando estados iguales; la multiplicidad reproduce Q(N). Uso: ./nq_estados N
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
typedef struct { uint64_t k, v; } Sl;
static Sl *T; static size_t cap, used;
static void tinit(size_t c) { cap = c; T = calloc(cap, sizeof(Sl)); used = 0; }
static uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; return x; }
static void put(uint64_t k, uint64_t v);
static void grow(void) { Sl *o = T; size_t oc = cap; tinit(oc * 2); for (size_t i = 0; i < oc; i++) if (o[i].v) put(o[i].k, o[i].v); free(o); }
static void put(uint64_t k, uint64_t v) { if (used * 10 > cap * 7) grow(); size_t i = mix(k) & (cap - 1); while (T[i].v && T[i].k != k) i = (i + 1) & (cap - 1); if (!T[i].v) { T[i].k = k; used++; } T[i].v += v; }
int main(int ac, char **av) { int N = atoi(av[1]); uint32_t F = (1u << N) - 1; struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    Sl *cur = malloc(sizeof(Sl)); cur[0].k = 0; cur[0].v = 1; size_t n = 1; uint64_t plain = 1, dist = 1; size_t peak = 1;
    for (int r = 0; r < N; r++) { tinit(1 << 20);
        for (size_t i = 0; i < n; i++) { uint64_t s = cur[i].k, m = cur[i].v; uint32_t c = s & 0xFFFFF, x = (s >> 20) & 0xFFFFF, y = (s >> 40) & 0xFFFFF, v = ~(c | x | y) & F;
            while (v) { uint32_t q = v & -v; v ^= q; uint64_t k = (uint64_t)(c | q) | (uint64_t)(((x | q) << 1) & F) << 20 | (uint64_t)((y | q) >> 1) << 40; put(k, m); plain += m; } }
        free(cur); cur = malloc(used * sizeof(Sl)); size_t j = 0; for (size_t i = 0; i < cap; i++) if (T[i].v) cur[j++] = T[i]; n = j; free(T); dist += n; if (n > peak) peak = n;
        }
    uint64_t q = 0; for (size_t i = 0; i < n; i++) q += cur[i].v; clock_gettime(CLOCK_MONOTONIC, &b);
    printf("N=%d Q=%llu nodos=%llu estados=%llu (%.1f%%, x%.1f) pico=%zu tiempo=%.2fs\n", N, q, plain, dist, 100.0 * dist / plain, (double)plain / dist, peak, b.tv_sec - a.tv_sec + 1e-9 * (b.tv_nsec - a.tv_nsec));
}
