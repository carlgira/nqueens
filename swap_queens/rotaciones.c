/* rotaciones.c — ¿hay listas de rotaciones por fila que lleven más soluciones
   del grupo 2 al grupo 1?

   Lista d: cada fila r gira d[r]. Acierto: U + d es exactamente una solución del
   grupo 1 (una o dos espirales). Sumar una constante a d no cambia nada, porque el
   grupo 1 es cerrado por giros, así que d se normaliza con d[0] = 0.

   No hay muestreo de listas: para cada U del grupo 2 y cada V del grupo 1 con
   V[0] = 0, la lista d = V - U acierta con U. Agrupando todos los pares salen los
   aciertos exactos de todas las listas que aciertan al menos una vez.
   Las U que son giro entero unas de otras cuentan como una.

   Uso: rotaciones N grupo2 control diezmo
     grupo2  0 no toroidales, 1 todo lo que no es grupo 1
     control 0 real, 1 U al azar, 2 V al azar
   cc -O3 -mcpu=native -Wall rotaciones.c -o rotaciones */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 19
typedef struct { uint8_t x[MAXN]; } Perm;
typedef struct { uint64_t d, u; } Reg;

static int N, grupo2, control, diezmo;
static Perm *g1, *g2, *v0;
static int ng1, n1, ng2, nv0, c1, c2, cv;
static uint64_t Zd[MAXN][MAXN], Zu[MAXN][MAXN];
static uint64_t sem = 0x9E3779B97F4A7C15ull;

static uint64_t xs(void) {
    sem ^= sem << 13; sem ^= sem >> 7; sem ^= sem << 17;
    return sem;
}

static void pon(Perm **v, int *n, int *cap, const uint8_t *t) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *v = realloc(*v, *cap * sizeof(Perm)); }
    memcpy((*v)[*n].x, t, N);
    (*n)++;
}

static int cmp(const void *a, const void *b) { return memcmp(a, b, N); }

static int es_recta(const uint8_t *t) {
    int a = (t[1] - t[0] + N) % N;
    for (int r = 0; r < N; r++)
        if (t[r] != (t[0] + a * r) % N) return 0;
    return 1;
}

static void dos(int r, int a, int c, int b, int d, uint8_t *t,
                uint32_t col, uint32_t dg, uint32_t ad) {
    if (r == N) { if (!es_recta(t)) pon(&g1, &ng1, &c1, t); return; }
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

static void grupo1(void) {
    uint8_t t[MAXN];
    for (int a = 2; a < N - 1; a++)
        for (int c = 0; c < N; c++) {
            for (int r = 0; r < N; r++) t[r] = (a * r + c) % N;
            pon(&g1, &ng1, &c1, t);
        }
    n1 = ng1;
    for (int a = 0; a < N; a++)
        for (int b = a + 1; b < N; b++)
            for (int c = 0; c < N; c++)
                for (int d = 0; d < N; d++)
                    dos(0, a, c, b, d, t, 0, 0, 0);
    qsort(g1, ng1, sizeof(Perm), cmp);
    /* las rectas quedan mezcladas: se quitan duplicados y es_recta da el tipo */
    int k = 0;
    for (int i = 0; i < ng1; i++)
        if (!i || memcmp(&g1[i], &g1[i - 1], N)) g1[k++] = g1[i];
    ng1 = k;
}

static int toroidal(const uint8_t *s) {
    uint32_t d = 0, e = 0, todo = (1u << N) - 1;
    for (int r = 0; r < N; r++) {
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    return d == todo && e == todo;
}

static void reinas(int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) {
        int t = toroidal(s);
        if (t && (!grupo2 || bsearch(s, g1, ng1, sizeof(Perm), cmp))) return;
        if (diezmo == 1 || xs() % diezmo == 0) pon(&g2, &ng2, &c2, s);
        return;
    }
    uint32_t lib = ((1u << N) - 1) & ~(col | dl | dr);
    while (lib) {
        int v = __builtin_ctz(lib);
        uint32_t m = 1u << v;
        lib &= lib - 1;
        s[r] = v;
        reinas(r + 1, s, col | m, (dl | m) << 1, (dr | m) >> 1);
    }
}

static void azar(Perm *p) {
    for (int r = 0; r < N; r++) p->x[r] = r;
    for (int r = N - 1; r > 0; r--) {
        int j = xs() % (r + 1), v = p->x[r];
        p->x[r] = p->x[j]; p->x[j] = v;
    }
}

static int cmpr(const void *a, const void *b) {
    const Reg *x = a, *y = b;
    if (x->d != y->d) return x->d < y->d ? -1 : 1;
    return x->u < y->u ? -1 : x->u > y->u;
}

typedef struct { uint64_t d; int k, una; } Top;

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "uso: rotaciones N grupo2 control diezmo\n"); return 1; }
    N = atoi(argv[1]); grupo2 = atoi(argv[2]); control = atoi(argv[3]); diezmo = atoi(argv[4]);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) { Zd[r][c] = xs(); Zu[r][c] = xs() & ~1ull; }
    grupo1();
    uint8_t s[MAXN];
    reinas(0, s, 0, 0, 0);
    if (control == 1) for (int i = 0; i < ng2; i++) azar(&g2[i]);
    for (int i = 0; i < ng1; i++) if (g1[i].x[0] == 0) pon(&v0, &nv0, &cv, g1[i].x);
    uint8_t *es_una = calloc(nv0, 1);
    for (int i = 0; i < nv0; i++) es_una[i] = es_recta(v0[i].x);
    if (control == 2)
        for (int i = 0; i < nv0; i++) {
            azar(&v0[i]);
            int c = v0[i].x[0];
            for (int r = 0; r < N; r++) v0[i].x[r] = (v0[i].x[r] - c + N) % N;
        }
    uint64_t np = (uint64_t)ng2 * nv0, k = 0;
    fprintf(stderr, "N=%d grupo1=%d grupo2=%d V0=%d pares=%llu\n", N, ng1, ng2, nv0, (unsigned long long)np);
    Reg *R = malloc(np * sizeof(Reg));
    for (int i = 0; i < ng2; i++) {
        const uint8_t *U = g2[i].x;
        uint64_t uh = 0;
        int u0 = U[0];
        for (int r = 0; r < N; r++) uh ^= Zu[r][(U[r] - u0 + N) % N];
        for (int j = 0; j < nv0; j++) {
            const uint8_t *V = v0[j].x;
            uint64_t dh = 0;
            for (int r = 0; r < N; r++) dh ^= Zd[r][(V[r] - U[r] + u0 + N) % N];
            R[k].d = dh;
            R[k].u = uh | es_una[j];
            k++;
        }
    }
    qsort(R, np, sizeof(Reg), cmpr);
    enum { TOPS = 8 };
    Top top[TOPS];
    memset(top, 0, sizeof(top));
    uint64_t hist[4096] = {0}, listas = 0;
    int maxk = 0;
    for (uint64_t a = 0, b; a < np; a = b) {
        int kk = 0, una = 0;
        for (b = a; b < np && R[b].d == R[a].d; b++)
            if (b == a || (R[b].u | 1) != (R[b - 1].u | 1)) { kk++; una += R[b].u & 1; }
        listas++;
        if (kk < 4096) hist[kk]++;
        if (kk > maxk) maxk = kk;
        for (int t = 0; t < TOPS; t++)
            if (kk > top[t].k) {
                memmove(&top[t + 1], &top[t], (TOPS - t - 1) * sizeof(Top));
                top[t].d = R[a].d; top[t].k = kk; top[t].una = una;
                break;
            }
    }
    printf("N=%d grupo2=%d control=%d U=%d V0=%d listas=%llu max=%d\n", N, grupo2, control,
           ng2, nv0, (unsigned long long)listas, maxk);
    printf("hist");
    for (int i = 1; i < 4096; i++)
        if (hist[i]) printf(" %d:%llu", i, (unsigned long long)hist[i]);
    printf("\n");
    /* cuántas filas se apartan del giro más frecuente, en las listas repetidas */
    uint64_t nrep = 0;
    Reg *rep = malloc((listas + 1) * sizeof(Reg));
    for (uint64_t a = 0, b; a < np; a = b) {
        int kk = 0;
        for (b = a; b < np && R[b].d == R[a].d; b++)
            if (b == a || (R[b].u | 1) != (R[b - 1].u | 1)) kk++;
        if (kk >= 2) { rep[nrep].d = R[a].d; rep[nrep].u = kk; nrep++; }
    }
    static uint64_t apartadas[17][MAXN + 1];
    for (int i = 0; i < ng2; i++) {
        const uint8_t *U = g2[i].x;
        int u0 = U[0];
        for (int j = 0; j < nv0; j++) {
            const uint8_t *V = v0[j].x;
            uint64_t dh = 0;
            for (int r = 0; r < N; r++) dh ^= Zd[r][(V[r] - U[r] + u0 + N) % N];
            Reg q = {dh, 0}, *p = NULL;
            uint64_t lo = 0, hi = nrep;
            while (lo < hi) {
                uint64_t m = (lo + hi) / 2;
                if (rep[m].d < dh) lo = m + 1; else hi = m;
            }
            if (lo < nrep && rep[lo].d == q.d) p = &rep[lo];
            if (!p || p->u >> 32) continue;
            int cu[MAXN] = {0}, mx = 0;
            for (int r = 0; r < N; r++) {
                int c = ++cu[(V[r] - U[r] + u0 + N) % N];
                if (c > mx) mx = c;
            }
            int kk = (int)p->u < 16 ? (int)p->u : 16;
            apartadas[kk][N - mx]++;
            p->u |= 1ull << 32;
            if (kk >= 4 && N - mx >= 7) {
                printf("global aciertos=%d apartadas=%d lista=", kk, N - mx);
                for (int r = 0; r < N; r++) printf("%d ", (V[r] - U[r] + u0 + N) % N);
                printf("\n");
            }
        }
    }
    for (int kk = 2; kk <= 16; kk++) {
        uint64_t tot = 0;
        for (int x = 0; x <= N; x++) tot += apartadas[kk][x];
        if (!tot) continue;
        printf("aciertos=%d filas_apartadas", kk);
        for (int x = 0; x <= N; x++)
            if (apartadas[kk][x]) printf(" %d:%llu", x, (unsigned long long)apartadas[kk][x]);
        printf("\n");
    }
    /* la lista de cada una de las mejores y un ejemplo */
    for (int t = 0; t < TOPS && top[t].k > 1; t++) {
        int visto = 0;
        for (int i = 0; i < ng2 && !visto; i++) {
            const uint8_t *U = g2[i].x;
            int u0 = U[0];
            for (int j = 0; j < nv0; j++) {
                const uint8_t *V = v0[j].x;
                uint64_t dh = 0;
                for (int r = 0; r < N; r++) dh ^= Zd[r][(V[r] - U[r] + u0 + N) % N];
                if (dh == top[t].d) {
                    printf("top aciertos=%d a_una=%d lista=", top[t].k, top[t].una);
                    for (int r = 0; r < N; r++) printf("%d ", (V[r] - U[r] + u0 + N) % N);
                    printf("\n");
                    visto = 1;
                    break;
                }
            }
        }
    }
    return 0;
}
