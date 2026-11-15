/* Filtros baratos para encontrar los vecinos válidos (swap / ciclo3 / doble) de una solución de N-reinas.
   Uso: ./nq_filter N M seed    (M soluciones al azar)
   - fuerza bruta: prueba los C(n,2)+2C(n,3)+3C(n,4) candidatos con la comprobación de diagonales original (referencia)
   - filtro A (parejas): las reinas nuevas no se atacan entre sí
   - filtro B (celdas): cada celda nueva solo puede estar atacada en diagonal por filas que también se mueven
   - generador: usa la tabla de atacantes para construir solo candidatos plausibles; se verifica que da exactamente los mismos vecinos */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int N;
static uint8_t *S; static size_t CNT;

static void gen(int r, uint32_t cols, uint32_t d1, uint32_t d2, uint8_t *q, size_t *pos) {
    uint32_t full = (1u << N) - 1;
    if (r == N) { memcpy(S + *pos * 16, q, N); (*pos)++; return; }
    uint32_t av = ~(cols | d1 | d2) & full;
    while (av) {
        uint32_t bit = av & -av; av ^= bit;
        q[r] = (uint8_t)__builtin_ctz(bit);
        gen(r + 1, cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1, q, pos);
    }
}

static uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static int q[16], d1o[16], d2o[16], occ1[32], occ2[32];
static uint32_t att[16][16], need[16][16];

static inline uint64_t key(int k, const int *rows, const int *nc) {      // rows en cualquier orden -> clave canónica
    int idx[4] = {0, 1, 2, 3};
    for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) if (rows[idx[b]] < rows[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    uint64_t h = 1469598103934665603ULL;
    for (int a = 0; a < k; a++) h = (h ^ (uint64_t)(rows[idx[a]] * 17 + nc[idx[a]] + 1)) * 1099511628211ULL;
    return mix(h);
}

static inline int internal_ok(int k, const int *rows, const int *nc) {
    for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++)
        if (rows[a] - nc[a] == rows[b] - nc[b] || rows[a] + nc[a] == rows[b] + nc[b]) return 0;
    return 1;
}
static inline int cells_ok(int k, const int *rows, const int *nc, uint32_t R) {
    for (int a = 0; a < k; a++) if (att[rows[a]][nc[a]] & ~R) return 0;
    return 1;
}
/* referencia: la comprobación original con máscaras de diagonales */
static inline int valid_ref(const int *rows, const int *nc, int k) {
    uint32_t m1 = 0, m2 = 0;
    for (int r = 0; r < N; r++) { m1 |= 1u << d1o[r]; m2 |= 1u << d2o[r]; }
    uint32_t r1 = m1, r2 = m2, a1 = 0, a2 = 0;
    for (int t = 0; t < k; t++) { r1 &= ~(1u << d1o[rows[t]]); r2 &= ~(1u << d2o[rows[t]]); }
    for (int t = 0; t < k; t++) {
        uint32_t b1 = 1u << (rows[t] - nc[t] + N - 1), b2 = 1u << (rows[t] + nc[t]);
        if (((r1 | a1) & b1) || ((r2 | a2) & b2)) return 0;
        a1 |= b1; a2 |= b2;
    }
    return 1;
}

typedef struct { double tot, pA, pB, valid, gen; } Ty;
static double fc[3], fi[3], fdup, frev;
static uint64_t *SH, *SH2; static size_t nsh = 0, shcap = 0;
static void rec_shape(int type, int k, const int *rows, const int *nc) {
    int idx[4] = {0, 1, 2, 3};
    for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) if (rows[idx[b]] < rows[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    uint64_t g = type, d = 0;
    for (int a = 0; a < k; a++) { g = g * 16 + (rows[idx[a]] - rows[idx[0]]); d = d * 32 + (nc[idx[a]] - q[rows[idx[a]]] + 16); }
    if (nsh == shcap) { shcap = shcap ? shcap * 2 : 1 << 20; SH = realloc(SH, shcap * 8); SH2 = realloc(SH2, shcap * 8); }
    SH[nsh] = g; SH2[nsh] = (g << 24) ^ mix(d) >> 8 ^ (g * 0x9E3779B97F4A7C15ULL); nsh++;
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 8;
    int M = argc > 2 ? atoi(argv[2]) : 1000;
    rs ^= (uint64_t)(argc > 3 ? atoi(argv[3]) : 1) * 0x9E3779B97F4A7C15ULL;
    size_t cap = 15000000; S = malloc(cap * 16); CNT = 0; uint8_t qq[16];
    gen(0, 0, 0, 0, qq, &CNT);
    Ty T[3]; memset(T, 0, sizeof T);
    long long mism = 0; double edgeE[3] = {0, 0, 0};
    for (int s = 0; s < M; s++) {
        size_t id = rnd() % CNT;
        for (int i = 0; i < N; i++) q[i] = S[id * 16 + i];
        memset(occ1, -1, sizeof occ1); memset(occ2, -1, sizeof occ2);
        for (int r = 0; r < N; r++) { d1o[r] = r - q[r] + N - 1; d2o[r] = r + q[r]; occ1[d1o[r]] = r; occ2[d2o[r]] = r; }
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) {
            uint32_t m = 0; int o = occ1[r - c + N - 1]; if (o >= 0 && o != r) m |= 1u << o;
            o = occ2[r + c]; if (o >= 0 && o != r) m |= 1u << o; att[r][c] = m;
        }
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (i != j) {
            need[i][j] = att[i][q[j]];
            int e = __builtin_popcount(need[i][j] & ~(1u << j)); edgeE[e]++;
        }
        /* ---- fuerza bruta + etapas ---- */
        uint64_t hb[3] = {0, 0, 0}; long long nb[3] = {0, 0, 0};
        int rows[4], nc[4];
#define TEST(type, k, R) do { \
            T[type].tot++; int a_ = internal_ok(k, rows, nc), b_ = cells_ok(k, rows, nc, R); \
            T[type].pA += a_; T[type].pB += b_; \
            if (valid_ref(rows, nc, k)) { T[type].valid++; nb[type]++; rec_shape(type, k, rows, nc); hb[type] ^= key(k, rows, nc); \
                if (!(a_ && b_)) mism++; } else if (a_ && b_) mism++; } while (0)
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
            rows[0] = i; rows[1] = j; nc[0] = q[j]; nc[1] = q[i]; TEST(0, 2, (1u << i) | (1u << j));
        }
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) for (int k = j + 1; k < N; k++) {
            uint32_t R = (1u << i) | (1u << j) | (1u << k);
            rows[0] = i; rows[1] = j; rows[2] = k; nc[0] = q[j]; nc[1] = q[k]; nc[2] = q[i]; TEST(1, 3, R);
            nc[0] = q[k]; nc[1] = q[i]; nc[2] = q[j]; TEST(1, 3, R);
        }
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) for (int k = j + 1; k < N; k++) for (int l = k + 1; l < N; l++) {
            uint32_t R = (1u << i) | (1u << j) | (1u << k) | (1u << l);
            rows[0] = i; rows[1] = j; rows[2] = k; rows[3] = l;
            nc[0] = q[j]; nc[1] = q[i]; nc[2] = q[l]; nc[3] = q[k]; TEST(2, 4, R);
            nc[0] = q[k]; nc[1] = q[l]; nc[2] = q[i]; nc[3] = q[j]; TEST(2, 4, R);
            nc[0] = q[l]; nc[1] = q[k]; nc[2] = q[j]; nc[3] = q[i]; TEST(2, 4, R);
        }
        /* ---- generador guiado por la tabla de atacantes ---- */
        uint64_t hg[3] = {0, 0, 0}; long long ng[3] = {0, 0, 0}; double gw[3] = {0, 0, 0};
        uint32_t ext[16][16]; int hsok[16][16];
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
            ext[i][j] = (need[i][j] | need[j][i]) & ~((1u << i) | (1u << j));
            hsok[i][j] = !(i - q[j] == j - q[i] || i + q[j] == j + q[i]);
        }
        /* aristas dirigidas vivas: a->b con extras e(a,b) = need & ~bit(b) de 0 o 1 filas */
        int out1[16][16], no1[16]; uint32_t ex1[16][16]; int al0[256][2], nal0 = 0;
        memset(no1, 0, sizeof no1);
        for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) if (a != b) {
            uint32_t e = need[a][b] & ~(1u << b); int pc = __builtin_popcount(e);
            if (pc == 0) { al0[nal0][0] = a; al0[nal0][1] = b; nal0++; }
            if (pc <= 1) { out1[a][no1[a]] = b; ex1[a][no1[a]] = e; no1[a]++; }
        }
        /* swaps: ambas aristas a->b y b->a con 0 extras */
        for (int t = 0; t < nal0; t++) {
            int i = al0[t][0], j = al0[t][1]; gw[0]++;
            if (i < j && ext[i][j] != 0) frev++;
            if (i < j && ext[i][j] == 0) { rows[0] = i; rows[1] = j; nc[0] = q[j]; nc[1] = q[i]; ng[0]++; hg[0] ^= key(2, rows, nc); }
        }
        /* buckets de half-swaps por ext */
        int L0[128][2], n0 = 0, Lx[16][128][2], nx[16], Lxy[128][3], nxy = 0;
        memset(nx, 0, sizeof nx);
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) if (hsok[i][j]) {
            uint32_t e = ext[i][j]; int pc = __builtin_popcount(e);
            if (pc == 0) { L0[n0][0] = i; L0[n0][1] = j; n0++; }
            else if (pc == 1) { int x = __builtin_ctz(e); Lx[x][nx[x]][0] = i; Lx[x][nx[x]][1] = j; nx[x]++; }
            else { Lxy[nxy][0] = i; Lxy[nxy][1] = j; Lxy[nxy][2] = (int)e; nxy++; }
        }
#define TRYDS(i, j, k, l) do { gw[2]++; \
            uint32_t R_ = (1u << (i)) | (1u << (j)) | (1u << (k)) | (1u << (l)); \
            rows[0] = (i); rows[1] = (j); rows[2] = (k); rows[3] = (l); nc[0] = q[j]; nc[1] = q[i]; nc[2] = q[l]; nc[3] = q[k]; \
            if ((i) * N + (j) >= (k) * N + (l)) fdup++; else if (!cells_ok(4, rows, nc, R_)) fc[2]++; else if (!internal_ok(4, rows, nc)) fi[2]++; else { ng[2]++; hg[2] ^= key(4, rows, nc); } } while (0)
        /* P=(i,j) con ext(P) de <=2 filas; Q debe salir de los cubos con ext(Q) incluido en {i,j} y contener ext(P) */
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) if (hsok[i][j] && __builtin_popcount(ext[i][j]) <= 2) {
            uint32_t e = ext[i][j], ij = (1u << i) | (1u << j);
#define CAND(k, l) do { int k_ = (k), l_ = (l); if (k_ != i && k_ != j && l_ != i && l_ != j && (e & ~((1u << k_) | (1u << l_))) == 0) TRYDS(i, j, k_, l_); } while (0)
            for (int t = 0; t < n0; t++) CAND(L0[t][0], L0[t][1]);
            for (int u = 0; u < 2; u++) { int x = u ? j : i; for (int t = 0; t < nx[x]; t++) CAND(Lx[x][t][0], Lx[x][t][1]); }
            for (int t = 0; t < nxy; t++) if ((uint32_t)Lxy[t][2] == ij) CAND(Lxy[t][0], Lxy[t][1]);
        }
        /* ciclos de 3: a minimo, a->b->c->a (nueva col de a = q[b], ...) */
#define TRYC3(a, b, c) do { gw[1]++; uint32_t R_ = (1u << (a)) | (1u << (b)) | (1u << (c)); \
            rows[0] = (a); rows[1] = (b); rows[2] = (c); nc[0] = q[b]; nc[1] = q[c]; nc[2] = q[a]; \
            if (!cells_ok(3, rows, nc, R_)) fc[1]++; else if (!internal_ok(3, rows, nc)) fi[1]++; else { ng[1]++; hg[1] ^= key(3, rows, nc); } } while (0)
        for (int a = 0; a < N; a++) for (int t = 0; t < no1[a]; t++) {                 // a->b viva, a mínimo del ciclo
            int b = out1[a][t]; if (b < a) continue; uint32_t e = ex1[a][t];
            if (e) { int c = __builtin_ctz(e); if (c > a) TRYC3(a, b, c); }            // el extra obliga a c
            else for (int u = 0; u < no1[b]; u++) {                                     // b->c viva con extra incluido en {a}
                int c = out1[b][u]; if (c <= a || c == b) continue;
                if (ex1[b][u] & ~(1u << a)) continue;
                TRYC3(a, b, c);
            }
        }
        /* la otra orientación: a->c->b->a con b<c se cubre arriba al iterar (a,b) y c cualquiera > a, así que
           los ciclos a->x->y->a con x>y salen cuando la arista de salida es a->x con x>a: ya incluida */
        for (int t = 0; t < 3; t++) { gw[t] += 0; if (hb[t] != hg[t] || nb[t] != ng[t]) mism++; T[t].gen += gw[t]; }
    }
    const char *nm[3] = {"swap", "ciclo3", "doble"};
    double tt = 0, gg = 0, vv = 0;
    printf("N=%d  soluciones=%zu  muestra=%d   (desacuerdos filtro/generador vs referencia: %lld)\n", N, CNT, M, mism);
    for (int t = 0; t < 3; t++) {
        printf("  %-7s candidatos=%8.1f  pasan A(parejas)=%7.1f (%5.1f%%)  pasan B(celdas)=%7.1f (%5.2f%%)  validos=%5.2f   generador examina=%8.1f (%5.2f%% del total)\n",
               nm[t], T[t].tot / M, T[t].pA / M, 100 * T[t].pA / T[t].tot, T[t].pB / M, 100 * T[t].pB / T[t].tot, T[t].valid / M, T[t].gen / M, 100 * T[t].gen / T[t].tot);
        tt += T[t].tot; gg += T[t].gen; vv += T[t].valid;
    }
    printf("  TOTAL   candidatos=%8.1f  validos=%5.2f  generador examina=%8.1f  (%5.2f%%, factor %.0fx menos)   aristas dirigidas i->j con 0/1/2 extras: %.1f/%.1f/%.1f de %d\n",
           tt / M, vv / M, gg / M, 100 * gg / tt, tt / gg, edgeE[0] / M, edgeE[1] / M, edgeE[2] / M, N * (N - 1));
    printf("  por que fallan los examinados (media/solucion): swap: arista inversa no viva=%.1f | ciclo3: celda mal=%.1f, parejas mal=%.1f | doble: celda mal=%.1f, parejas mal=%.1f, duplicados=%.1f\n",
           frev / M, fc[1] / M, fi[1] / M, fc[2] / M, fi[2] / M, fdup / M);
    if (argc > 4) {
        const char *nm2[3] = {"swap", "ciclo3", "doble"};
        for (int pass = 0; pass < 2; pass++) {
            uint64_t *A = pass ? SH2 : SH; size_t n = nsh;
            int cmp(const void *x, const void *y) { uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y; return a < b ? -1 : a > b; }
            qsort(A, n, 8, cmp);
            size_t *cnt = malloc(n * sizeof(size_t)), nd = 0; uint64_t *rep = malloc(n * 8);
            for (size_t i = 0; i < n;) { size_t j = i; while (j < n && A[j] == A[i]) j++; rep[nd] = A[i]; cnt[nd++] = j - i; i = j; }
            size_t srt[10] = {0}; // top-10 counts
            for (size_t i = 0; i < nd; i++) { size_t c = cnt[i]; for (int t = 0; t < 10; t++) if (c > srt[t]) { size_t tmp = srt[t]; srt[t] = c; c = tmp; } }
            size_t top = 0; for (int t = 0; t < 10; t++) top += srt[t];
            size_t once = 0; for (size_t i = 0; i < nd; i++) if (cnt[i] == 1) once++;
            printf("  FORMAS (%s) movimientos validos=%zu  formas distintas=%zu  top10 cubre %.1f%%  formas vistas 1 sola vez=%zu (%.1f%% de los movimientos)  mayor forma=%.2f%%\n",
                   pass ? "huecos de filas + desplazamientos de columna" : "solo huecos de filas", n, nd, 100.0 * top / n, once, 100.0 * once / n, 100.0 * srt[0] / n);
            free(cnt); free(rep);
        }
    }
    return 0;
}
