/* familias.c — islas que son copias unas de otras: misma forma, mismo tamaño y
   mismas transformaciones.

   Islas de nivel 2 como en dentro.c. Cada arista lleva una etiqueta:
     giro c, con c y -c iguales;
     vector: intercambio de dos reinas, con (filas, columnas) de separación sin orden;
     lugar:  el mismo rectángulo de cuatro celdas en el tablero, salvo simetría.
   Huella de isla: refinamiento de colores (Weisfeiler-Lehman) sobre la isla etiquetada.
   Familia: islas con igual huella. Solo cuentan islas de 3 o más miembros.
   Nulo: las mismas islas con las etiquetas barajadas entre todas las aristas.

   Uso: familias N
   cc -O3 -mcpu=native -pthread -Wall familias.c -o familias */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 16
#define HILOS 8
#define BARAJAS 20
typedef struct { uint8_t x[MAXN]; } Perm;
typedef struct { uint32_t a, b; int v, l; } Arista;
typedef struct { uint64_t h; int n, i; } Huella;

static int N;
static Perm *g, *rep;
static uint8_t (*im)[8][MAXN];
static int ng, cg, nr, sig_i;
static Arista *E[HILOS], *A;
static uint64_t nE[HILOS], na;
static uint32_t *ini, *ady_v, *ady_e;
static int *isla, ni;
static uint64_t sem = 0x9E3779B97F4A7C15ull;

static uint64_t xs(void) {
    sem ^= sem << 13; sem ^= sem >> 7; sem ^= sem << 17;
    return sem;
}

static uint64_t mezcla(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static void pon(const uint8_t *t) {
    if (ng == cg) { cg = cg ? 2 * cg : 4096; g = realloc(g, cg * sizeof(Perm)); }
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
    uint8_t a[MAXN];
    if (e & 4) for (int r = 0; r < N; r++) a[s[r]] = r;
    else memcpy(a, s, N);
    for (int r = 0; r < N; r++) {
        int f = (e & 2) ? N - 1 - r : r;
        t[f] = (e & 1) ? N - 1 - a[r] : a[r];
    }
}

/* rectángulo (r1,r2)x(c1,c2) salvo las 8 simetrías */
static int lugar(int r1, int r2, int c1, int c2) {
    if (c1 > c2) { int t = c1; c1 = c2; c2 = t; }
    int mejor = 1 << 30;
    for (int e = 0; e < 8; e++) {
        int a = r1, b = r2, c = c1, d = c2;
        if (e & 4) { int t = a; a = c; c = t; t = b; b = d; d = t; }
        if (e & 1) { c = N - 1 - c; d = N - 1 - d; }
        if (e & 2) { a = N - 1 - a; b = N - 1 - b; }
        if (a > b) { int t = a; a = b; b = t; }
        if (c > d) { int t = c; c = d; d = t; }
        int k = ((a * 16 + b) * 16 + c) * 16 + d;
        if (k < mejor) mejor = k;
    }
    return mejor;
}

/* 0 si no se une; si se une, las dos etiquetas más pequeñas */
static int une(const uint8_t *U, int j, int *ev, int *el) {
    int hay = 0, bv = 1 << 30, bl = 1 << 30;
    for (int e = 0; e < 8; e++) {
        const uint8_t *V = im[j][e];
        int c[MAXN] = {0}, mx = 0, mc = 0, df[MAXN], nd = 0;
        for (int r = 0; r < N; r++) {
            int v = V[r] - U[r];
            if (v < 0) v += N;
            if (++c[v] > mx) { mx = c[v]; mc = v; }
            if (v && nd < 3) df[nd++] = r;
        }
        if (mx == N) {
            int q = mc < N - mc ? mc : N - mc, k = 100000 + q;
            if (k < bv) bv = k;
            if (k < bl) bl = k;
            hay = 1;
        } else if (N - c[0] == 2) {
            int a = df[0], b = df[1], dr = b - a, dc = abs((int)U[a] - U[b]);
            int k = dr < dc ? dr * 32 + dc : dc * 32 + dr, l = 200000 + lugar(a, b, U[a], U[b]);
            if (k < bv) bv = k;
            if (l < bl) bl = l;
            hay = 1;
        }
    }
    *ev = bv; *el = bl;
    return hay;
}

static void *obrero(void *arg) {
    int yo = (int)(intptr_t)arg;
    uint64_t cap = 1 << 16, n = 0;
    Arista *b = malloc(cap * sizeof(Arista));
    for (;;) {
        int i = __sync_fetch_and_add(&sig_i, 1);
        if (i >= nr) break;
        for (int j = i + 1; j < nr; j++) {
            int v, l;
            if (!une(rep[i].x, j, &v, &l)) continue;
            if (n == cap) { cap *= 2; b = realloc(b, cap * sizeof(Arista)); }
            b[n].a = i; b[n].b = j; b[n].v = v; b[n].l = l; n++;
        }
    }
    E[yo] = b; nE[yo] = n;
    return NULL;
}

static int cmph(const void *a, const void *b) {
    const Huella *x = a, *y = b;
    if (x->h != y->h) return x->h < y->h ? -1 : 1;
    return x->i - y->i;
}

/* tipo: 0 solo forma, 1 vector, 2 lugar */
static int huellas(int tipo, const int *et, Huella *H) {
    static uint64_t *col, *nue;
    if (!col) { col = malloc(nr * 8); nue = malloc(nr * 8); }
    int *cnt = calloc(ni, 4);
    for (int v = 0; v < nr; v++) cnt[isla[v]]++;
    for (int v = 0; v < nr; v++) col[v] = 1;
    for (int ronda = 0; ronda < 8; ronda++) {
        for (int v = 0; v < nr; v++) {
            uint64_t s = 0;
            for (uint32_t a = ini[v]; a < ini[v + 1]; a++) {
                int l = tipo == 0 ? 0 : et[ady_e[a]];
                s += mezcla(col[ady_v[a]] * 0x9E3779B97F4A7C15ull + (uint64_t)l * 0xD6E8FEB86659FD93ull + 1);
            }
            nue[v] = mezcla(col[v] + 3 * s);
        }
        memcpy(col, nue, nr * 8);
    }
    uint64_t *s = calloc(ni, 8);
    for (int v = 0; v < nr; v++) s[isla[v]] += mezcla(col[v]);
    int n = 0;
    if (tipo && getenv("SOLOINTERC")) {
        /* fuera las islas hechas solo de giros */
        uint8_t *hay = calloc(ni, 1);
        for (uint64_t e = 0; e < na; e++)
            if (et[e] < 100000 || et[e] >= 200000) hay[isla[A[e].a]] = 1;
        for (int i = 0; i < ni; i++) if (!hay[i]) cnt[i] = 0;
        free(hay);
    }
    for (int i = 0; i < ni; i++)
        if (cnt[i] >= 3) { H[n].h = s[i] ^ mezcla(cnt[i]); H[n].n = cnt[i]; H[n].i = i; n++; }
    qsort(H, n, sizeof(Huella), cmph);
    free(cnt); free(s);
    return n;
}

typedef struct { int fam, enrep, mayorfam, mayorrep; } Res;

static Res resume(const Huella *H, int n) {
    Res R = {0, 0, 0, 0};
    for (int a = 0, b; a < n; a = b) {
        for (b = a; b < n && H[b].h == H[a].h; b++);
        R.fam++;
        if (b - a >= 2) {
            R.enrep += b - a;
            if (b - a > R.mayorfam) R.mayorfam = b - a;
            if (H[a].n > R.mayorrep) R.mayorrep = H[a].n;
        }
    }
    return R;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: familias N\n"); return 1; }
    N = atoi(argv[1]);
    uint8_t s[MAXN], t[MAXN];
    reinas(0, s, 0, 0, 0);
    qsort(g, ng, sizeof(Perm), cmp);
    rep = malloc(ng * sizeof(Perm));
    for (int i = 0; i < ng; i++) {
        int minimo = 1;
        for (int e = 1; e < 8; e++) {
            imagen(g[i].x, e, t);
            if (memcmp(t, g[i].x, N) < 0) { minimo = 0; break; }
        }
        if (minimo) rep[nr++] = g[i];
    }
    im = malloc(nr * sizeof(*im));
    for (int i = 0; i < nr; i++)
        for (int e = 0; e < 8; e++) imagen(rep[i].x, e, im[i][e]);
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, obrero, (void *)i);
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    for (int h = 0; h < HILOS; h++) na += nE[h];
    A = malloc(na * sizeof(Arista));
    uint64_t k = 0;
    for (int h = 0; h < HILOS; h++) { memcpy(A + k, E[h], nE[h] * sizeof(Arista)); k += nE[h]; }
    ini = calloc(nr + 2, 4);
    ady_v = malloc((2 * na + 1) * 4); ady_e = malloc((2 * na + 1) * 4);
    for (uint64_t e = 0; e < na; e++) { ini[A[e].a + 1]++; ini[A[e].b + 1]++; }
    for (int i = 0; i < nr; i++) ini[i + 1] += ini[i];
    uint32_t *pos = malloc((nr + 1) * 4);
    memcpy(pos, ini, (nr + 1) * 4);
    for (uint64_t e = 0; e < na; e++) {
        ady_v[pos[A[e].a]] = A[e].b; ady_e[pos[A[e].a]++] = e;
        ady_v[pos[A[e].b]] = A[e].a; ady_e[pos[A[e].b]++] = e;
    }
    isla = malloc(nr * 4);
    memset(isla, -1, nr * 4);
    int *cola = malloc(nr * 4);
    for (int i = 0; i < nr; i++) {
        if (isla[i] >= 0) continue;
        int q = 0, f = 0;
        cola[f++] = i; isla[i] = ni;
        while (q < f) {
            int v = cola[q++];
            for (uint32_t a = ini[v]; a < ini[v + 1]; a++)
                if (isla[ady_v[a]] < 0) { isla[ady_v[a]] = ni; cola[f++] = ady_v[a]; }
        }
        ni++;
    }
    int *ev = malloc(na * 4), *el = malloc(na * 4);
    for (uint64_t e = 0; e < na; e++) { ev[e] = A[e].v; el[e] = A[e].l; }
    Huella *H = malloc((ni + 1) * sizeof(Huella));
    printf("N=%d grupos=%d aristas=%llu islas=%d\n", N, nr, (unsigned long long)na, ni);
    static const char *nom[3] = {"forma", "vector", "lugar"};
    for (int tipo = 0; tipo < 3; tipo++) {
        const int *et = tipo == 2 ? el : ev;
        int n = huellas(tipo, et, H);
        Res R = resume(H, n);
        double sf = 0, sr = 0, sm = 0, sg = 0;
        if (tipo) {
            int *ba = malloc(na * 4);
            memcpy(ba, et, na * 4);
            for (int b = 0; b < BARAJAS; b++) {
                for (uint64_t e = na - 1; e > 0; e--) {
                    uint64_t j = xs() % (e + 1);
                    int x = ba[e]; ba[e] = ba[j]; ba[j] = x;
                }
                Res Q = resume(H, huellas(tipo, ba, H));
                sf += Q.fam; sr += Q.enrep; sm += Q.mayorfam; sg += Q.mayorrep;
            }
            free(ba);
            n = huellas(tipo, et, H);
        }
        printf("%s islas3+=%d familias=%d en_repetidas=%d mayor_familia=%d mayor_isla_repetida=%d", nom[tipo], n,
               R.fam, R.enrep, R.mayorfam, R.mayorrep);
        if (tipo) printf(" | barajado familias=%.1f en_repetidas=%.1f mayor_familia=%.1f mayor_isla=%.1f",
                         sf / BARAJAS, sr / BARAJAS, sm / BARAJAS, sg / BARAJAS);
        printf("\n");
        if (tipo == 0) continue;
        /* las familias de mayor tamaño de isla */
        int mostradas = 0;
        for (int obj = 1 << 20; obj >= 3 && mostradas < 6; obj--) {
            for (int a = 0, b; a < n && mostradas < 6; a = b) {
                for (b = a; b < n && H[b].h == H[a].h; b++);
                if (b - a < 2 || H[a].n != obj) continue;
                printf("  familia isla=%d copias=%d etiquetas:", obj, b - a);
                int i0 = H[a].i, vistos = 0;
                for (uint64_t e = 0; e < na && vistos < 12; e++)
                    if (isla[A[e].a] == i0) {
                        int l = et[e];
                        if (l >= 200000) printf(" L%d", l - 200000);
                        else if (l >= 100000) printf(" g%d", l - 100000);
                        else printf(" (%d,%d)", l / 32, l % 32);
                        vistos++;
                    }
                printf("\n");
                mostradas++;
            }
            if (obj > nr) obj = nr + 1;
        }
    }
    return 0;
}
