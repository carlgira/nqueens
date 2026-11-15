/* figura.c — tablero nuevo tal que S se convierte en T; ¿qué otras soluciones
   U forman en él la figura de una o dos rectas mod N?

   Grupo 1: toroidales de una y dos espirales. Grupo 2: el resto.
   Se eligen T y S al azar y se construye el tablero:
     giro:  cada fila r gira T[r]-S[r] columnas, sin elecciones.
     azar:  cada fila recibe una permutación al azar de columnas con S[r] -> T[r].
   Toda U pasa a f(r) y se mira si f cabe en una o dos rectas mod N, sea o no solución.

   Uso: figura N tablero origenU origenT intentos diezmo
     tablero 0 giro, 1 azar
     origenU 0 no toroidales, 1 permutaciones al azar, 2 toroidales de tres o más
     origenT 0 grupo 1, 1 permutaciones al azar
     diezmo: de U se guarda 1 de cada diezmo
   Salida: por tipo de T (una, dos, azar) y clase de acierto, cuenta por tamaño del
   lado menor del reparto. cc -O3 -mcpu=native -pthread -Wall figura.c -o figura */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 19
#define HILOS 8
typedef struct { uint8_t x[MAXN]; } Perm;

static int N, tablero, origenU, origenT, intentos, diezmo;
static Perm *g1, *g2, *gt3, *pu;
static int n1, ng1, ng2, ngt3, npu;
static int cg1, cg2, cgt3, inv[MAXN];
static int sig_i;
static uint64_t sem_g = 0x2545F4914F6CDD1Dull;
static pthread_mutex_t cerrojo = PTHREAD_MUTEX_INITIALIZER;

enum { TIPOS = 3, CLASES = 4 };
/* clase 0 pendientes cualesquiera, 1 pendientes de espiral (no 0, ±1),
   2 además figura de solución toroidal, 3 U es S girada entera */
static uint64_t H[TIPOS][CLASES][MAXN + 1], ntipo[TIPOS], conalguno[TIPOS];
static uint64_t linea[TIPOS][MAXN + 1];

static void pon(Perm **v, int *n, int *cap, const uint8_t *t) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *v = realloc(*v, *cap * sizeof(Perm)); }
    memcpy((*v)[*n].x, t, N);
    (*n)++;
}

static uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

static int toroidal(const uint8_t *s) {
    uint32_t c = 0, d = 0, e = 0, todo = (1u << N) - 1;
    for (int r = 0; r < N; r++) {
        c |= 1u << s[r];
        d |= 1u << ((s[r] - r + N) % N);
        e |= 1u << ((s[r] + r) % N);
    }
    return c == todo && d == todo && e == todo;
}

/* mejor reparto en dos rectas: devuelve lado menor (0 = una recta), -1 si no cabe.
   *pa = 1 si hay reparto con ambas pendientes de espiral */
static inline int pend_ok(int a) { return a != 0 && a != 1 && a != N - 1; }

static int cabe(const uint8_t *f, int *pa) {
    static const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    int mejor = -1;
    *pa = 0;
    for (int p = 0; p < 3; p++) {
        int i = P[p][0], j = P[p][1];
        int a = (f[j] - f[i] + N) * inv[j - i] % N;
        int c = (f[i] + (N - a) * i) % N;
        int resto[MAXN], nr = 0;
        for (int r = 0; r < N; r++)
            if (f[r] != (a * r + c) % N) resto[nr++] = r;
        if (nr == 0) { *pa = pend_ok(a); return 0; }
        int b = 2, ok = 1; /* recta sin fijar si solo queda una fila */
        if (nr >= 2) {
            int u = resto[0], v = resto[1];
            b = (f[v] - f[u] + N) * inv[(v - u + N) % N] % N;
            int d = (f[u] + (N - b) * u) % N;
            for (int k = 2; k < nr; k++)
                if (f[resto[k]] != (b * resto[k] + d) % N) { ok = 0; break; }
        }
        if (!ok) continue;
        int menor = nr < N - nr ? nr : N - nr;
        int bien = pend_ok(a) && pend_ok(b);
        if (menor > mejor || (menor == mejor && bien && !*pa)) { mejor = menor; *pa = bien; }
    }
    return mejor;
}

static int mejor_recta(const uint8_t *f) {
    int b = 0;
    for (int a = 0; a < N; a++) {
        int cu[MAXN] = {0};
        for (int r = 0; r < N; r++) {
            int c = (f[r] + (N - a) * r) % N;
            if (++cu[c] > b) b = cu[c];
        }
    }
    return b;
}

static void reinas(int r, uint8_t *s, uint32_t col, uint32_t dl, uint32_t dr) {
    if (r == N) {
        if (toroidal(s)) pon(&gt3, &ngt3, &cgt3, s);
        else if (diezmo == 1 || xs(&sem_g) % diezmo == 0) pon(&g2, &ng2, &cg2, s);
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

static int es_grupo1(const uint8_t *s) {
    int pa, m = cabe(s, &pa);
    return m >= 0 && pa;
}

static void *obrero(void *arg) {
    uint64_t sem = 88172645463325252ull + 0x9E3779B97F4A7C15ull * (uint64_t)(intptr_t)arg;
    uint64_t L[TIPOS][CLASES][MAXN + 1] = {{{0}}}, lin[TIPOS][MAXN + 1] = {{0}};
    uint64_t nt[TIPOS] = {0}, ca[TIPOS] = {0};
    uint8_t F[MAXN][MAXN], f[MAXN], tr[MAXN];
    while (__sync_fetch_and_add(&sig_i, 1) < intentos) {
        const uint8_t *S = g2[xs(&sem) % ng2].x, *T;
        int tipo;
        if (origenT == 0) {
            int i = xs(&sem) % ng1;
            T = g1[i].x;
            tipo = i < n1 ? 0 : 1;
        } else {
            for (int r = 0; r < N; r++) tr[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs(&sem) % (r + 1), v = tr[r]; tr[r] = tr[j]; tr[j] = v;
            }
            T = tr;
            tipo = 2;
        }
        for (int r = 0; r < N; r++) {
            int d = (T[r] - S[r] + N) % N;
            if (tablero == 0) {
                for (int c = 0; c < N; c++) F[r][c] = (c + d) % N;
            } else {
                uint8_t p[MAXN];
                for (int c = 0; c < N; c++) p[c] = c;
                for (int c = N - 1; c > 0; c--) {
                    int j = xs(&sem) % (c + 1), v = p[c]; p[c] = p[j]; p[j] = v;
                }
                int k = 0;
                while (p[k] != T[r]) k++;
                p[k] = p[S[r]]; p[S[r]] = T[r];
                memcpy(F[r], p, N);
            }
        }
        nt[tipo]++;
        int algun = 0;
        for (int i = 0; i < npu; i++) {
            const uint8_t *U = pu[i].x;
            if (!memcmp(U, S, N)) continue;
            int cte = 1, d0 = (U[0] - S[0] + N) % N;
            for (int r = 0; r < N; r++) {
                f[r] = F[r][U[r]];
                if ((U[r] - S[r] + N) % N != d0) cte = 0;
            }
            if (i % 4096 == 0) lin[tipo][mejor_recta(f)]++;
            int pa, m = cabe(f, &pa);
            if (m < 0) continue;
            if (tablero == 0 && cte) { L[tipo][3][m]++; continue; }
            L[tipo][0][m]++;
            if (getenv("EJEMPLO") && pa && m >= 4) {
                pthread_mutex_lock(&cerrojo);
                printf("EJ T=%s menor=%d valida=%d\n", tipo ? "dos" : "una", m, toroidal(f));
                const uint8_t *q[4] = {S, T, U, f};
                for (int w = 0; w < 4; w++) {
                    for (int r = 0; r < N; r++) printf("%d,", q[w][r]);
                    printf("\n");
                }
                pthread_mutex_unlock(&cerrojo);
            }
            if (pa) {
                L[tipo][1][m]++;
                if (toroidal(f)) L[tipo][2][m]++;
            }
            if (m >= 3) algun = 1;
        }
        ca[tipo] += algun;
    }
    pthread_mutex_lock(&cerrojo);
    for (int t = 0; t < TIPOS; t++) {
        ntipo[t] += nt[t]; conalguno[t] += ca[t];
        for (int k = 0; k <= N; k++) {
            linea[t][k] += lin[t][k];
            for (int c = 0; c < CLASES; c++) H[t][c][k] += L[t][c][k];
        }
    }
    pthread_mutex_unlock(&cerrojo);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "uso: figura N tablero origenU origenT intentos diezmo\n"); return 1; }
    N = atoi(argv[1]); tablero = atoi(argv[2]); origenU = atoi(argv[3]);
    origenT = atoi(argv[4]); intentos = atoi(argv[5]); diezmo = atoi(argv[6]);
    for (int a = 1; a < N; a++)
        for (int b = 1; b < N; b++)
            if (a * b % N == 1) inv[a] = b;
    uint8_t s[MAXN];
    reinas(0, s, 0, 0, 0);
    /* grupo 1: separar de las toroidales las que caben en dos rectas */
    Perm *resto = NULL; int nr = 0, cr = 0, cu = 0;
    for (int i = 0; i < ngt3; i++) {
        int pa, m = cabe(gt3[i].x, &pa);
        if (m == 0) pon(&g1, &ng1, &cg1, gt3[i].x);
    }
    n1 = ng1;
    for (int i = 0; i < ngt3; i++) {
        int pa, m = cabe(gt3[i].x, &pa);
        if (m == 0) continue;
        if (es_grupo1(gt3[i].x)) pon(&g1, &ng1, &cg1, gt3[i].x);
        else pon(&resto, &nr, &cr, gt3[i].x);
    }
    (void)cu;
    gt3 = resto; ngt3 = nr;
    fprintf(stderr, "N=%d una=%d dos=%d toroidales3+=%d grupo2=%d\n", N, n1, ng1 - n1, ngt3, ng2);
    if (origenU == 0) { pu = g2; npu = ng2; }
    else if (origenU == 2) { pu = gt3; npu = ngt3; }
    else {
        npu = ng2;
        pu = malloc(npu * sizeof(Perm));
        uint64_t sem = 0x123456789ABCDEFull;
        for (int i = 0; i < npu; i++) {
            for (int r = 0; r < N; r++) pu[i].x[r] = r;
            for (int r = N - 1; r > 0; r--) {
                int j = xs(&sem) % (r + 1), v = pu[i].x[r];
                pu[i].x[r] = pu[i].x[j]; pu[i].x[j] = v;
            }
        }
    }
    pthread_t th[HILOS];
    for (intptr_t i = 0; i < HILOS; i++) pthread_create(&th[i], NULL, obrero, (void *)(i + 1));
    for (int i = 0; i < HILOS; i++) pthread_join(th[i], NULL);
    static const char *ti[TIPOS] = {"una", "dos", "azar"};
    static const char *cl[CLASES] = {"cualquiera", "pend_espiral", "toroidal", "S_girada"};
    printf("N=%d tablero=%d origenU=%d origenT=%d U=%d\n", N, tablero, origenU, origenT, npu);
    for (int t = 0; t < TIPOS; t++) {
        if (!ntipo[t]) continue;
        printf("T=%s intentos=%llu con_acierto_menor>=3=%llu (%.4f)\n", ti[t],
               (unsigned long long)ntipo[t], (unsigned long long)conalguno[t],
               (double)conalguno[t] / ntipo[t]);
        for (int c = 0; c < CLASES; c++) {
            printf("  %-13s", cl[c]);
            for (int k = 0; k <= N; k++)
                if (H[t][c][k]) printf(" %d:%llu", k, (unsigned long long)H[t][c][k]);
            printf("\n");
        }
        printf("  mejor_recta  ");
        for (int k = 0; k <= N; k++)
            if (linea[t][k]) printf(" %d:%llu", k, (unsigned long long)linea[t][k]);
        printf("\n");
    }
    return 0;
}
