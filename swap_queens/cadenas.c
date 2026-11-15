/* cadenas.c — saltos por listas de rotaciones sin toroidales ni simetrías, y
   caminos largos de solución en solución.

   G: soluciones no toroidales. Grupo S = {identidad, espejo de columnas, espejo de
   filas, giro 180°}: actúa sobre pares {U,W} y sobre listas d (d, -d, d al revés,
   -d al revés). Cada par se cuenta una vez por órbita y cada lista por su forma
   canónica; así un acierto es un salto de verdad distinto.

   Clases de lista: constante; rampa a·r + c (a != 0); casi (1 a 4 filas fuera de
   la mejor recta); global (5 o más).

   Cadena: U, U+d, U+2d, ... todas en G, exacta para toda d.
   Camino: grafo con arista si la lista aparece en K o más órbitas y tiene al menos
   minfuera filas fuera de recta. Rotaciones de Pósa con reinicios: cota inferior.

   Uso: cadenas N control cubos K minfuera reinicios
     control 0 real, 1 permutaciones al azar cerradas por S, del mismo tamaño
   cc -O3 -mcpu=native -pthread -Wall cadenas.c -o cadenas */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 16
#define HILOS 8
#define SETBITS 24
#define FBITS 30
typedef struct { uint8_t x[MAXN]; } Perm;
typedef struct { uint64_t h; int k; int fuera; } Lista;

static int N, control, cubos, K, minfuera, reinicios;
static Perm *g;
static int ng, cg;
static int (*img)[4];
static uint64_t Z[MAXN][MAXN], Zs[MAXN][MAXN];
static uint64_t *conjunto;
static uint64_t sem = 0x9E3779B97F4A7C15ull;
static int sig_i, cubo, fase;
static uint64_t **buf, *nbuf;
static Lista *rep;
static uint64_t nrep, caprep;
static uint8_t *filtro, *visto;
static pthread_mutex_t cerrojo = PTHREAD_MUTEX_INITIALIZER;

enum { CLASES = 4, TOPK = 64, TOPS = 6 };
static const char *nom[CLASES] = {"constante", "rampa", "casi", "global"};
typedef struct { int k; uint8_t d[MAXN]; } Mejor;
static Mejor mejores[CLASES][TOPS];
static uint64_t porclase[CLASES][TOPK + 1];
static uint64_t cad[CLASES][MAXN + 2];
static int cadmax[CLASES];
static uint8_t cadU[CLASES][MAXN], cadD[CLASES][MAXN];

static uint32_t *ea, *eb;
static uint64_t nar, capar;

static uint64_t xs(void) {
    sem ^= sem << 13; sem ^= sem >> 7; sem ^= sem << 17;
    return sem;
}

static void pon(const uint8_t *t) {
    if (ng == cg) { cg = cg ? 2 * cg : 4096; g = realloc(g, cg * sizeof(Perm)); }
    memset(&g[ng], 0, sizeof(Perm));
    memcpy(g[ng].x, t, N);
    ng++;
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, N); }

static int toroidal(const uint8_t *s) {
    uint32_t d = 0, e = 0, todo = (1u << N) - 1;
    for (int r = 0; r < N; r++) {
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    return d == todo && e == todo;
}

static void reinas(int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) { if (!toroidal(s)) pon(s); return; }
    uint32_t lib = ((1u << N) - 1) & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

static void imagen(const uint8_t *s, int e, uint8_t *t) {
    for (int r = 0; r < N; r++) {
        int f = (e & 2) ? N - 1 - r : r;
        t[f] = (e & 1) ? N - 1 - s[r] : s[r];
    }
}

static inline uint64_t hsol(const uint8_t *s) {
    uint64_t h = 0;
    for (int r = 0; r < N; r++) h ^= Zs[r][s[r]];
    return h | 1;
}

static void mete_conjunto(uint64_t h) {
    uint64_t m = (1ull << SETBITS) - 1, i = h >> 20 & m;
    while (conjunto[i]) i = (i + 1) & m;
    conjunto[i] = h;
}

static inline int esta(uint64_t h) {
    uint64_t m = (1ull << SETBITS) - 1, i = h >> 20 & m;
    while (conjunto[i]) {
        if (conjunto[i] == h) return 1;
        i = (i + 1) & m;
    }
    return 0;
}

static inline uint64_t hlista(const uint8_t *d) {
    uint64_t a = 0, b = 0, c = 0, e = 0;
    for (int r = 0; r < N; r++) {
        int v = d[r], n = v ? N - v : 0, q = N - 1 - r;
        a ^= Z[r][v]; b ^= Z[r][n]; c ^= Z[q][v]; e ^= Z[q][n];
    }
    uint64_t m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    if (e < m) m = e;
    return m;
}

static inline int par_canonico(int i, int j) {
    for (int e = 1; e < 4; e++) {
        int a = img[i][e], b = img[j][e];
        if (a > b) { int t = a; a = b; b = t; }
        if (a < i || (a == i && b < j)) return 0;
    }
    return 1;
}

static int fuera_de_recta(const uint8_t *d, int *pend) {
    int mejor = 0, pa = 0;
    for (int a = 0; a < N; a++) {
        int cu[MAXN] = {0}, m = 0;
        for (int r = 0; r < N; r++) {
            int c = (d[r] + (N - a) * r) % N;
            if (++cu[c] > m) m = cu[c];
        }
        if (m > mejor) { mejor = m; pa = a; }
    }
    *pend = pa;
    return N - mejor;
}

static int clase(const uint8_t *d) {
    int a, f = fuera_de_recta(d, &a);
    if (f == 0) return a == 0 ? 0 : 1;
    return f <= 4 ? 2 : 3;
}

static void anda(const uint8_t *U, const uint8_t *d) {
    uint8_t p[MAXN], q[MAXN];
    int L = 1;
    memcpy(p, U, N);
    for (int r = 0; r < N; r++) { q[r] = U[r] + N - d[r]; if (q[r] >= N) q[r] -= N; }
    int atras = esta(hsol(q));
    for (;;) {
        for (int r = 0; r < N; r++) { p[r] += d[r]; if (p[r] >= N) p[r] -= N; }
        if (!memcmp(p, U, N)) { atras = 0; break; }
        if (!esta(hsol(p))) break;
        L++;
    }
    if (atras || L < 3) return;
    int c = clase(d);
    pthread_mutex_lock(&cerrojo);
    cad[c][L]++;
    if (L > cadmax[c]) {
        cadmax[c] = L;
        memcpy(cadU[c], U, N);
        memcpy(cadD[c], d, N);
    }
    pthread_mutex_unlock(&cerrojo);
}

static void *pasada(void *arg) {
    int yo = (int)(intptr_t)arg;
    uint64_t *b = NULL, n = 0, cap = 0;
    uint8_t d[MAXN], e[MAXN];
    for (;;) {
        int i = __sync_fetch_and_add(&sig_i, 1);
        if (i >= ng) break;
        const uint8_t *U = g[i].x;
        for (int j = i + 1; j < ng; j++) {
            const uint8_t *W = g[j].x;
            if (fase == 0) {
                if (!par_canonico(i, j)) continue;
                for (int r = 0; r < N; r++) { int v = W[r] - U[r]; d[r] = v < 0 ? v + N : v; }
                uint64_t h = hlista(d);
                if ((int)(h % cubos) == cubo) {
                    if (n == cap) { cap = cap ? 2 * cap : 1 << 20; b = realloc(b, cap * 8); }
                    b[n++] = h;
                }
                if (cubo == 0) {
                    anda(U, d);
                    for (int r = 0; r < N; r++) e[r] = d[r] ? N - d[r] : 0;
                    anda(W, e);
                }
            } else {
                for (int r = 0; r < N; r++) { int v = W[r] - U[r]; d[r] = v < 0 ? v + N : v; }
                uint64_t h = hlista(d), f = h >> (64 - FBITS);
                if (!((filtro[f >> 3] >> (f & 7)) & 1)) continue;
                uint64_t lo = 0, hi = nrep;
                while (lo < hi) {
                    uint64_t m = (lo + hi) / 2;
                    if (rep[m].h < h) lo = m + 1; else hi = m;
                }
                if (lo == nrep || rep[lo].h != h) continue;
                pthread_mutex_lock(&cerrojo);
                Lista *p = &rep[lo];
                if (!visto[lo]) {
                    visto[lo] = 1;
                    int a, c = clase(d);
                    p->fuera = fuera_de_recta(d, &a) + (c == 0 || c == 1 ? 0 : 0);
                    porclase[c][p->k < TOPK ? p->k : TOPK]++;
                    for (int t = 0; t < TOPS; t++)
                        if (p->k > mejores[c][t].k) {
                            memmove(&mejores[c][t + 1], &mejores[c][t], (TOPS - t - 1) * sizeof(Mejor));
                            mejores[c][t].k = p->k;
                            memcpy(mejores[c][t].d, d, N);
                            break;
                        }
                }
                if (p->k >= K && p->fuera >= minfuera) {
                    if (nar == capar) {
                        capar = capar ? 2 * capar : 1 << 20;
                        ea = realloc(ea, capar * 4); eb = realloc(eb, capar * 4);
                    }
                    ea[nar] = i; eb[nar] = j; nar++;
                }
                pthread_mutex_unlock(&cerrojo);
            }
        }
    }
    if (fase == 0) { buf[yo] = b; nbuf[yo] = n; }
    return NULL;
}

static void radix(uint64_t *a, uint64_t n, uint64_t *t) {
    for (int s = 0; s < 64; s += 11) {
        uint64_t c[2049] = {0};
        for (uint64_t i = 0; i < n; i++) c[((a[i] >> s) & 2047) + 1]++;
        for (int i = 0; i < 2048; i++) c[i + 1] += c[i];
        for (uint64_t i = 0; i < n; i++) t[c[(a[i] >> s) & 2047]++] = a[i];
        uint64_t *x = a; a = t; t = x;
    }
    memcpy(t, a, n * 8);
}

static int cmpl(const void *a, const void *b) {
    uint64_t x = ((const Lista *)a)->h, y = ((const Lista *)b)->h;
    return x < y ? -1 : x > y;
}

static void lanza(void) {
    pthread_t th[HILOS];
    sig_i = 0;
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, pasada, (void *)i);
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
}

/* camino por extensión y rotaciones */
static void camino(void) {
    uint32_t *ini = calloc(ng + 2, 4), *ady = malloc(2 * nar * 4 + 4);
    for (uint64_t e = 0; e < nar; e++) { ini[ea[e] + 1]++; ini[eb[e] + 1]++; }
    for (int i = 0; i < ng; i++) ini[i + 1] += ini[i];
    uint32_t *pos = malloc((ng + 1) * 4);
    memcpy(pos, ini, (ng + 1) * 4);
    for (uint64_t e = 0; e < nar; e++) { ady[pos[ea[e]]++] = eb[e]; ady[pos[eb[e]]++] = ea[e]; }
    int *comp = malloc(ng * 4), *pila = malloc(ng * 4), nc = 0, mayor = 0, cmay = -1, activos = 0;
    memset(comp, -1, ng * 4);
    for (int i = 0; i < ng; i++) {
        if (ini[i + 1] > ini[i]) activos++;
        if (comp[i] >= 0 || ini[i + 1] == ini[i]) continue;
        int n = 0, tam = 0;
        pila[n++] = i; comp[i] = nc;
        while (n) {
            int v = pila[--n]; tam++;
            for (uint32_t a = ini[v]; a < ini[v + 1]; a++)
                if (comp[ady[a]] < 0) { comp[ady[a]] = nc; pila[n++] = ady[a]; }
        }
        if (tam > mayor) { mayor = tam; cmay = nc; }
        nc++;
    }
    printf("grafo K=%d minfuera=%d nodos=%d aristas=%llu componentes=%d mayor=%d\n", K, minfuera,
           activos, (unsigned long long)nar, nc, mayor);
    if (cmay < 0) return;
    int *nodos = malloc(mayor * 4), nn = 0;
    for (int i = 0; i < ng; i++) if (comp[i] == cmay) nodos[nn++] = i;
    uint8_t *usado = malloc(ng);
    int *lib = malloc(ng * 4), *cam = malloc(ng * 4), *mejor = malloc(ng * 4), bl = 0;
    for (int re = 0; re < reinicios; re++) {
        /* rotaciones de Pósa: atascado en v, con vecino u dentro del camino en la
           posición p, se invierte el tramo p+1..fin y el final pasa a ser otro */
        memset(lib, -1, ng * 4);
        int L = 0, v = nodos[xs() % nn];
        cam[L] = v; lib[v] = L++;
        int pasos = 40 * nn < 400000 ? 40 * nn : 400000;
        for (int p = 0; p < pasos && L < nn; p++) {
            v = cam[L - 1];
            uint32_t g0 = ini[v + 1] - ini[v], ini0 = xs() % g0;
            int ext = -1;
            for (uint32_t q = 0; q < g0; q++) {
                int u = ady[ini[v] + (ini0 + q) % g0];
                if (lib[u] < 0) { ext = u; break; }
            }
            if (ext >= 0) { cam[L] = ext; lib[ext] = L++; continue; }
            int u = ady[ini[v] + ini0], i0 = lib[u] + 1, i1 = L - 1;
            while (i0 < i1) {
                int a = cam[i0], b = cam[i1];
                cam[i0] = b; lib[b] = i0; cam[i1] = a; lib[a] = i1;
                i0++; i1--;
            }
        }
        if (L > bl) { bl = L; memcpy(mejor, cam, L * 4); }
    }
    printf("camino_posa longitud=%d de %d en la mayor componente\n", bl, mayor);
    int ini_l = bl < 12 ? bl : 12;
    for (int i = 0; i + 1 < ini_l; i++) {
        printf("paso %d lista=", i);
        for (int r = 0; r < N; r++) printf("%d ", (g[mejor[i + 1]].x[r] - g[mejor[i]].x[r] + N) % N);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "uso: cadenas N control cubos K minfuera reinicios\n"); return 1; }
    N = atoi(argv[1]); control = atoi(argv[2]); cubos = atoi(argv[3]);
    K = atoi(argv[4]); minfuera = atoi(argv[5]); reinicios = atoi(argv[6]);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) { Z[r][c] = xs(); Zs[r][c] = xs(); }
    uint8_t s[MAXN], t[MAXN];
    reinas(0, s, 0, 0, 0);
    if (control) {
        int total = ng;
        ng = 0;
        while (ng < total) {
            for (int r = 0; r < N; r++) s[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs() % (r + 1), v = s[r]; s[r] = s[j]; s[j] = v;
            }
            for (int e = 0; e < 4 && ng < total + 3; e++) { imagen(s, e, t); pon(t); }
        }
    }
    qsort(g, ng, sizeof(Perm), cmp);
    int k = 0;
    for (int i = 0; i < ng; i++)
        if (!i || memcmp(&g[i], &g[i - 1], N)) g[k++] = g[i];
    ng = k;
    img = malloc(ng * sizeof(*img));
    conjunto = calloc(1ull << SETBITS, 8);
    for (int i = 0; i < ng; i++) {
        mete_conjunto(hsol(g[i].x));
        for (int e = 0; e < 4; e++) {
            imagen(g[i].x, e, t);
            Perm *p = bsearch(t, g, ng, sizeof(Perm), cmp);
            img[i][e] = p - g;
        }
    }
    fprintf(stderr, "N=%d G=%d\n", N, ng);
    buf = calloc(HILOS, sizeof(uint64_t *));
    nbuf = calloc(HILOS, 8);
    filtro = calloc(1ull << (FBITS - 3), 1);
    static uint64_t hist[1 << 16];
    uint64_t listas = 0, pares = 0;
    for (cubo = 0; cubo < cubos; cubo++) {
        lanza();
        uint64_t n = 0, c = 0;
        for (int i = 0; i < HILOS; i++) n += nbuf[i];
        uint64_t *a = malloc((n + 1) * 8), *b = malloc((n + 1) * 8);
        for (int i = 0; i < HILOS; i++) { memcpy(a + c, buf[i], nbuf[i] * 8); c += nbuf[i]; free(buf[i]); }
        radix(a, n, b);
        pares += n;
        for (uint64_t x = 0, y; x < n; x = y) {
            for (y = x; y < n && b[y] == b[x]; y++);
            uint64_t m = y - x;
            listas++;
            hist[m < 65535 ? m : 65535]++;
            if (m >= 2) {
                if (nrep == caprep) { caprep = caprep ? 2 * caprep : 1 << 16; rep = realloc(rep, caprep * sizeof(Lista)); }
                rep[nrep].h = b[x]; rep[nrep].k = m; rep[nrep].fuera = -1; nrep++;
                uint64_t f = b[x] >> (64 - FBITS);
                filtro[f >> 3] |= 1 << (f & 7);
            }
        }
        free(a); free(b);
        fprintf(stderr, "cubo %d/%d\n", cubo + 1, cubos);
    }
    qsort(rep, nrep, sizeof(Lista), cmpl);
    visto = calloc(nrep + 1, 1);
    fase = 1;
    lanza();
    printf("N=%d control=%d G=%d pares_sin_simetria=%llu listas=%llu\nsaltos_por_lista", N, control, ng,
           (unsigned long long)pares, (unsigned long long)listas);
    for (int i = 1; i < 1 << 16; i++)
        if (hist[i]) printf(" %d:%llu", i, (unsigned long long)hist[i]);
    printf("\n");
    for (int c = 0; c < CLASES; c++) {
        printf("clase %s aciertos", nom[c]);
        for (int m = 2; m <= TOPK; m++)
            if (porclase[c][m]) printf(" %d:%llu", m, (unsigned long long)porclase[c][m]);
        printf("\n");
        for (int t = 0; t < TOPS && mejores[c][t].k; t++) {
            printf("  mejor saltos=%d lista=", mejores[c][t].k);
            for (int r = 0; r < N; r++) printf("%d ", mejores[c][t].d[r]);
            printf("\n");
        }
    }
    for (int c = 0; c < CLASES; c++) {
        printf("cadena %s", nom[c]);
        for (int L = 3; L <= MAXN + 1; L++)
            if (cad[c][L]) printf(" %d:%llu", L, (unsigned long long)cad[c][L]);
        printf("\n");
        if (cadmax[c]) {
            printf("  mas_larga L=%d U=", cadmax[c]);
            for (int r = 0; r < N; r++) printf("%d ", cadU[c][r]);
            printf(" d=");
            for (int r = 0; r < N; r++) printf("%d ", cadD[c][r]);
            printf("\n");
        }
    }
    camino();
    return 0;
}
