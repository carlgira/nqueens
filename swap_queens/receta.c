/* receta.c — ¿un mismo cambio de celdas lleva de espirales a espirales?

   F = toroidales de una y dos espirales.
   Para cada par {U,V} de F:
     X = nº de reinas movidas, por tipos (1-1, 1-2, 2-2).
     molde = U xor V, las 2X celdas que cambian. Un molde es una receta fija
       sobre el tablero: varios pares con el mismo molde = la receta se repite.
     reflejo = V sale de U reflejando c -> s-c solo en un conjunto S de filas.
       La receta (s,S) intercambia |S|(N-1)/2 parejas de celdas por fila reflejada.

   Uso: receta N falso
     falso=0: F real.  falso=1: permutaciones al azar con los mismos tamaños.
     falso=2: aplica cada receta centrada (S = -S, s = 0) a toda F.
   cc -O3 -mcpu=native -Wall receta.c -o receta */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 32
typedef struct { uint8_t x[W]; } Perm;

static int N, n1, n2, nobj, cap, tam;
static Perm *obj;
static uint64_t Z[W][W];

static uint64_t s0 = 0x9E3779B97F4A7C15ull;
static uint64_t xs(void) {
    s0 ^= s0 << 13; s0 ^= s0 >> 7; s0 ^= s0 << 17;
    return s0;
}

static void pon(const uint8_t *t) {
    if (tam == cap) { cap = cap ? 2 * cap : 4096; obj = realloc(obj, cap * sizeof(Perm)); }
    memset(&obj[tam], 0xFF, W);
    memcpy(obj[tam].x, t, N);
    tam++;
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, W); }

static int es_recta(const uint8_t *t) {
    int a = (t[1] - t[0] + N) % N;
    for (int r = 0; r < N; r++)
        if (t[r] != (t[0] + a * r) % N) return 0;
    return 1;
}

static void dos(int r, int a, int c, int b, int d, uint8_t *t,
                uint32_t col, uint32_t dg, uint32_t ad) {
    if (r == N) { if (!es_recta(t)) pon(t); return; }
    int op[2] = {(a * r + c) % N, (b * r + d) % N};
    for (int k = 0; k < 2; k++) {
        int v = op[k];
        if (k == 1 && op[1] == op[0]) break;
        uint32_t m1 = 1u << v, m2 = 1u << ((v - r + N) % N), m3 = 1u << ((v + r) % N);
        if ((col & m1) || (dg & m2) || (ad & m3)) continue;
        t[r] = v;
        dos(r + 1, a, c, b, d, t, col | m1, dg | m2, ad | m3);
    }
}

static void genera(void) {
    uint8_t t[W];
    for (int a = 2; a < N - 1; a++)
        for (int c = 0; c < N; c++) {
            for (int r = 0; r < N; r++) t[r] = (a * r + c) % N;
            pon(t);
        }
    n1 = tam;
    for (int a = 0; a < N; a++)
        for (int b = a + 1; b < N; b++)
            for (int c = 0; c < N; c++)
                for (int d = 0; d < N; d++)
                    dos(0, a, c, b, d, t, 0, 0, 0);
    qsort(obj + n1, tam - n1, sizeof(Perm), cmp);
    int k = n1;
    for (int i = n1; i < tam; i++)
        if (i == n1 || memcmp(&obj[i], &obj[i - 1], W)) obj[k++] = obj[i];
    n2 = k - n1;
    nobj = k;
}

static void falsifica(void) {
    for (int i = 0; i < nobj; i++) {
        for (int r = 0; r < N; r++) obj[i].x[r] = r;
        for (int r = N - 1; r > 0; r--) {
            int j = xs() % (r + 1), v = obj[i].x[r];
            obj[i].x[r] = obj[i].x[j]; obj[i].x[j] = v;
        }
    }
}

static Perm *busca(const uint8_t *t) {
    Perm q;
    memset(&q, 0xFF, W);
    memcpy(q.x, t, N);
    return bsearch(&q, obj, nobj, sizeof(Perm), cmp);
}

/* centro.  Receta: reflejar c -> -c en las filas S = -S (centro en fila 0).
   Se aplica a todas las de F; cuenta las que cambian y caen de nuevo en F. */
static void centro(void) {
    int m = (N - 1) / 2;
    uint64_t rec[W + 1] = {0}, buen[W + 1] = {0}, tot[W + 1] = {0}, mx[W + 1] = {0};
    uint32_t tm[W + 1] = {0};
    uint64_t d11[W + 1] = {0}, d12[W + 1] = {0}, d22[W + 1] = {0};
    uint8_t t[W];
    for (uint32_t k = 1; k < 1u << m; k++) {
        uint8_t en[W] = {0};
        int nf = 0;
        for (int b = 0; b < m; b++)
            if ((k >> b) & 1) { en[b + 1] = en[N - b - 1] = 1; nf += 2; }
        uint64_t c = 0, a11 = 0, a12 = 0, a22 = 0;
        for (int i = 0; i < nobj; i++) {
            memcpy(t, obj[i].x, N);
            int mov = 0;
            for (int r = 0; r < N; r++)
                if (en[r] && t[r]) { t[r] = N - t[r]; mov = 1; }
            if (!mov) continue;
            Perm *p = busca(t);
            if (!p) continue;
            int ui = es_recta(obj[i].x), uj = es_recta(p->x);
            c++;
            if (ui && uj) a11++;
            else if (ui || uj) a12++;
            else a22++;
        }
        rec[nf]++;
        if (c) { buen[nf]++; tot[nf] += c; }
        if (c > mx[nf]) { mx[nf] = c; tm[nf] = k; d11[nf] = a11; d12[nf] = a12; d22[nf] = a22; }
    }
    for (int f = 2; f < N; f += 2) {
        printf("centro,filas=%d,parejas_celdas=%d,recetas=%llu,utiles=%llu,mejor=%llu de %d (1-1 %llu, 1-2 %llu, 2-2 %llu),filas:",
               f, f * m, (unsigned long long)rec[f], (unsigned long long)buen[f],
               (unsigned long long)mx[f], nobj, (unsigned long long)d11[f],
               (unsigned long long)d12[f], (unsigned long long)d22[f]);
        for (int b = 0; b < m; b++)
            if ((tm[f] >> b) & 1) printf(" +-%d", b + 1);
        printf("\n");
    }
}

static int cmp64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void imprime(int i) {
    printf("%s[", i < n1 ? "una" : "dos");
    for (int r = 0; r < N; r++) printf(r ? " %d" : "%d", obj[i].x[r]);
    printf("]");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "uso: receta N falso\n"); return 1; }
    N = atoi(argv[1]);
    int falso = atoi(argv[2]);
    genera();
    if (falso == 2) {
        qsort(obj, nobj, sizeof(Perm), cmp);
        centro();
        return 0;
    }
    if (falso) falsifica();
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) Z[r][c] = xs() >> 6; /* 58 bits de clave + X arriba */
    uint64_t np = (uint64_t)nobj * (nobj - 1) / 2, kp = 0;
    uint64_t *clave = malloc(np * sizeof(uint64_t));
    size_t cr = 1 << 16, nr = 0;
    uint64_t *refl = malloc(cr * sizeof(uint64_t));
    uint64_t esp[3][W + 1] = {{0}};
    for (int i = 0; i < nobj; i++) {
        const uint8_t *u = obj[i].x;
        for (int j = i + 1; j < nobj; j++) {
            const uint8_t *v = obj[j].x;
            uint64_t k = 0;
            int X = 0, s = -1, ok = 1;
            uint32_t S = 0;
            for (int r = 0; r < N; r++) {
                if (u[r] == v[r]) continue;
                X++;
                k ^= Z[r][u[r]] ^ Z[r][v[r]];
                S |= 1u << r;
                int t = (u[r] + v[r]) % N;
                if (s < 0) s = t; else if (t != s) ok = 0;
            }
            esp[(i >= n1) + (j >= n1)][X]++;
            clave[kp++] = ((uint64_t)X << 58) | k;
            if (ok) {
                if (nr == cr) { cr *= 2; refl = realloc(refl, cr * sizeof(uint64_t)); }
                refl[nr++] = ((uint64_t)S << 8) | s;
            }
        }
    }
    printf("#N=%d una=%d dos=%d pares=%llu\n", N, n1, n2, (unsigned long long)np);
    static const char *ti[3] = {"1-1", "1-2", "2-2"};
    for (int t = 0; t < 3; t++)
        for (int x = 0; x <= N; x++)
            if (esp[t][x]) printf("espectro,%s,%d,%llu\n", ti[t], x, (unsigned long long)esp[t][x]);
    qsort(clave, np, sizeof(uint64_t), cmp64);
    uint64_t moldes[W + 1] = {0}, rep[W + 1] = {0}, maxm[W + 1] = {0}, top[W + 1] = {0};
    for (uint64_t a = 0, b; a < np; a = b) {
        for (b = a; b < np && clave[b] == clave[a]; b++);
        int X = clave[a] >> 58;
        uint64_t m = b - a;
        moldes[X]++;
        if (m > 1) rep[X]++;
        if (m > maxm[X]) { maxm[X] = m; top[X] = clave[a]; }
    }
    for (int x = 0; x <= N; x++)
        if (moldes[x]) printf("molde,X=%d,moldes=%llu,repetidos=%llu,maxpares=%llu\n", x,
                              (unsigned long long)moldes[x], (unsigned long long)rep[x],
                              (unsigned long long)maxm[x]);
    /* ejemplos: los pares del molde más repetido de cada X con repetición */
    for (int i = 0; i < nobj; i++)
        for (int j = i + 1; j < nobj; j++) {
            const uint8_t *u = obj[i].x, *v = obj[j].x;
            uint64_t k = 0;
            int X = 0;
            for (int r = 0; r < N; r++)
                if (u[r] != v[r]) { X++; k ^= Z[r][u[r]] ^ Z[r][v[r]]; }
            if (maxm[X] > 1 && (((uint64_t)X << 58) | k) == top[X]) {
                static int vistos[W + 1];
                if (vistos[X]++ < 2) {
                    printf("ejemplo,X=%d,m=%llu ", X, (unsigned long long)maxm[X]);
                    imprime(i); printf(" <-> "); imprime(j);
                    printf("\n");
                }
            }
        }
    qsort(refl, nr, sizeof(uint64_t), cmp64);
    uint64_t rk[W + 1] = {0}, rp[W + 1] = {0}, rm[W + 1] = {0}, rt[W + 1] = {0};
    for (size_t a = 0, b; a < nr; a = b) {
        for (b = a; b < nr && refl[b] == refl[a]; b++);
        int k = __builtin_popcount((uint32_t)(refl[a] >> 8));
        rk[k]++; rp[k] += b - a;
        if (b - a > rm[k]) { rm[k] = b - a; rt[k] = refl[a]; }
    }
    for (int k = 0; k <= N; k++)
        if (rk[k]) {
            printf("reflejo,filas=%d,recetas=%llu,pares=%llu,maxpares=%llu,s=%d,S=",
                   k, (unsigned long long)rk[k], (unsigned long long)rp[k],
                   (unsigned long long)rm[k], (int)(rt[k] & 255));
            for (int r = 0; r < N; r++)
                if ((rt[k] >> (8 + r)) & 1) printf("%d ", r);
            printf("\n");
        }
    return 0;
}
