/*
 * reparto.c — Reparto voraz de todas las soluciones en grupos recorridos por una onda
 * con aceleración común (misma fuerza para todas las reinas del grupo). Basado en parabolas.c.
 *
 * Cada reina r: fase th_r(k) = beta_r + v_r k + a_r k^2/2   (reales), lectura floor(th) mod N.
 * Es una rueda empujada por una fuerza constante, con 3 números reales por reina.
 *
 * Ojo: con 3 parámetros cualquier terna es factible (3 puntos, 3 incógnitas). La información
 * empieza en la cuarta solución. Por eso aquí se buscan cadenas completas.
 *
 * Factibilidad exacta: eliminando beta, (v,a) es factible sii para todo par i<j
 *     x_j - x_i - 1 < v (j-i) + a (j^2-i^2)/2 < x_j - x_i + 1,
 * con x_k la posición sin envolver (x_k = c_k + m_k N). Es un polígono convexo en (v,a), que se
 * recorta al añadir cada punto. Cada elección de vueltas m_k es una rama.
 * Dominio: v en [0,N), a en [-N,N)  (periodos de la lectura mod N).
 *
 * Modo comun=1: además, la MISMA aceleración a para todas las reinas (velocidades libres).
 *
 * Salida: histograma de cadenas maximales, la más larga, y datos/parabolas_N<N>_<conj>.txt
 * con todas las cadenas de longitud >= minguardar.
 *
 * uso: ./parabolas N conjunto(0 sol, s>0 azar, s<0 sim) [comun=0] [limite_s=300] [minguardar=6]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define MAXV 48
#define MAXB 8
#define MAXK 64
typedef struct { int n; double v[MAXV], a[MAXV]; long x[MAXK]; } Br;
typedef struct { int nb; Br b[MAXB]; } Set;

static int N, Q, cap, COMUN;
static int *sol;
static const double EPS = 1e-9;
static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) {
        if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); }
        memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return;
    }
    for (int c = 0; c < N; c++) {
        if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue;
        cur[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N));
    }
}
static unsigned long long rs = 12345;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static void randperm(int *p) { for (int r = 0; r < N; r++) p[r] = r; for (int r = N - 1; r > 0; r--) { int j = (int)(ur() * (r + 1)); int t = p[r]; p[r] = p[j]; p[j] = t; } }
static int have(const int *g, int n) { for (int k = 0; k < n; k++) if (!memcmp(g, sol + (long)k * N, sizeof(int) * N)) return 1; return 0; }
static void push(const int *g) { if (Q == cap) { cap = cap ? cap * 2 : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, g, sizeof(int) * N); Q++; }

/* recorta polígono con  p*v + q*a <= c  */
static int cut(Br *B, double p, double q, double c) {
    double nv[MAXV], na[MAXV]; int n = 0;
    for (int i = 0; i < B->n; i++) {
        int j = (i + 1) % B->n;
        double fi = p * B->v[i] + q * B->a[i] - c, fj = p * B->v[j] + q * B->a[j] - c;
        if (fi <= 0) { if (n < MAXV) { nv[n] = B->v[i]; na[n] = B->a[i]; n++; } }
        if ((fi <= 0) != (fj <= 0)) {
            double t = fi / (fi - fj);
            if (n < MAXV) { nv[n] = B->v[i] + t * (B->v[j] - B->v[i]); na[n] = B->a[i] + t * (B->a[j] - B->a[i]); n++; }
        }
    }
    B->n = n; memcpy(B->v, nv, sizeof(double) * n); memcpy(B->a, na, sizeof(double) * n);
    if (n < 3) return 0;
    double s = 0; for (int i = 0; i < n; i++) { int j = (i + 1) % n; s += B->v[i] * B->a[j] - B->v[j] * B->a[i]; }
    return fabs(s) > 1e-12;
}

/* añade el punto k con columna c a todas las ramas de S -> T */
static int addpt(const Set *S, Set *T, int k, int c) {
    T->nb = 0;
    for (int b = 0; b < S->nb; b++) {
        const Br *B = &S->b[b];
        double fmin = 1e18, fmax = -1e18;
        for (int i = 0; i < B->n; i++) { double f = B->v[i] * k + B->a[i] * k * k / 2.0; if (f < fmin) fmin = f; if (f > fmax) fmax = f; }
        /* x_k - x_0 en (fmin-1, fmax+1) */
        double lo = B->x[0] + fmin - 1, hi = B->x[0] + fmax + 1;
        long m0 = (long)floor((lo - c) / N), m1 = (long)ceil((hi - c) / N);
        for (long m = m0; m <= m1; m++) {
            long xk = c + m * N;
            if (xk <= lo - 1e-12 || xk >= hi + 1e-12) continue;
            if (T->nb >= MAXB) return T->nb > 0;
            Br R = *B; R.x[k] = xk; int ok = 1;
            for (int i = 0; i < k && ok; i++) {
                double p = k - i, q = (double)(k * k - i * i) / 2.0, d = (double)(xk - R.x[i]);
                ok = cut(&R, p, q, d + 1 - EPS) && cut(&R, -p, -q, -(d - 1 + EPS));
            }
            if (ok) T->b[T->nb++] = R;
        }
    }
    return T->nb > 0;
}

/* ¿existe una aceleración común? rango de a de cada rama: [amin,amax]; unión por reina, intersección */
static int common_ok(Set *st) {
    double cand[MAXB * 32 * 2]; int nc = 0;
    double lo[32][MAXB], hi[32][MAXB]; int nb[32];
    for (int r = 0; r < N; r++) {
        nb[r] = st[r].nb;
        for (int b = 0; b < st[r].nb; b++) {
            double mn = 1e18, mx = -1e18; const Br *B = &st[r].b[b];
            for (int i = 0; i < B->n; i++) { if (B->a[i] < mn) mn = B->a[i]; if (B->a[i] > mx) mx = B->a[i]; }
            lo[r][b] = mn; hi[r][b] = mx; cand[nc++] = mn + 1e-9;
        }
    }
    for (int t = 0; t < nc; t++) {
        double a = cand[t]; int good = 1;
        for (int r = 0; r < N && good; r++) { int any = 0; for (int b = 0; b < nb[r]; b++) if (a >= lo[r][b] && a <= hi[r][b]) any = 1; if (!any) good = 0; }
        if (good) return 1;
    }
    return 0;
    /* nota: proyectar a sobre cada polígono es exacto para "existe v" porque v es libre por reina */
}


static Set st[MAXK][32];
static int chain[MAXK], K, best, bestchain[MAXK];
static char *used; static char *gone;
static long nodes; static double tlimit; static clock_t t0; static int timed_out;

static void dfs(void) {
    if (timed_out) return;
    if ((++nodes & 0x3f) == 0 && (double)(clock() - t0) / CLOCKS_PER_SEC > tlimit) { timed_out = 1; return; }
    int ext = 0;
    if (K < MAXK) for (int j = 0; j < Q && !timed_out; j++) {
        if (used[j] || gone[j]) continue;
        const int *T = sol + (long)j * N; int ok = 1;
        for (int r = 0; r < N && ok; r++) ok = addpt(&st[K - 1][r], &st[K][r], K, T[r]);
        if (ok && COMUN) ok = common_ok(st[K]);
        if (!ok) continue;
        ext = 1; chain[K++] = j; used[j] = 1; dfs(); used[j] = 0; K--;
    }
    if (!ext && K > best) { best = K; memcpy(bestchain, chain, sizeof(int) * K); }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); int set = atoi(argv[2]); COMUN = 1;
    tlimit = argc > 3 ? atof(argv[3]) : 120;
    bt(0, 0, 0, 0); int want = Q;
    if (set > 0) { rs ^= (unsigned long long)set * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32]; while (Q < want) { randperm(p); if (!have(p, Q)) push(p); } }
    else if (set < 0) {
        rs ^= (unsigned long long)(-set) * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32], g[32], q[32];
        while (Q < want) { randperm(p); for (int e = 0; e < 8; e++) { memcpy(q, p, sizeof q); if (e & 4) for (int r = 0; r < N; r++) q[p[r]] = r; for (int r = 0; r < N; r++) g[r] = q[(e & 2) ? N - 1 - r : r]; if (e & 1) for (int r = 0; r < N; r++) g[r] = N - 1 - g[r]; if (!have(g, Q)) push(g); } }
    }
    char tag[32]; if (set == 0) snprintf(tag, sizeof tag, "sol"); else if (set > 0) snprintf(tag, sizeof tag, "azar%d", set); else snprintf(tag, sizeof tag, "sim%d", -set);
    mkdir("datos", 0755);
    char fn[160]; snprintf(fn, sizeof fn, "datos/tableros_N%d_%s.txt", N, tag);
    FILE *ft = fopen(fn, "w"); for (int i = 0; i < Q; i++) for (int r = 0; r < N; r++) fprintf(ft, "%d%c", sol[(long)i * N + r], r < N - 1 ? ' ' : '\n'); fclose(ft);
    snprintf(fn, sizeof fn, "datos/reparto_N%d_%s.txt", N, tag);
    FILE *fo = fopen(fn, "w"); fprintf(fo, "# grupos del reparto voraz (indices en datos/tableros_N%d_%s.txt)\n", N, tag);
    used = calloc(Q, 1); gone = calloc(Q, 1);
    int left = Q, ng = 0, sizes[4096], anytimeout = 0;
    while (left > 0) {
        best = 0; t0 = clock(); timed_out = 0;
        for (int i = 0; i < Q && !timed_out; i++) {
            if (gone[i]) continue;
            for (int r = 0; r < N; r++) { Set *S = &st[0][r]; S->nb = 1; Br *B = &S->b[0]; B->n = 4; B->v[0] = 0; B->a[0] = -N; B->v[1] = N; B->a[1] = -N; B->v[2] = N; B->a[2] = N; B->v[3] = 0; B->a[3] = N; B->x[0] = sol[(long)i * N + r]; }
            chain[0] = i; used[i] = 1; K = 1; dfs(); used[i] = 0;
        }
        if (timed_out) anytimeout = 1;
        for (int k = 0; k < best; k++) { gone[bestchain[k]] = 1; fprintf(fo, "%d%c", bestchain[k], k < best - 1 ? ' ' : '\n'); }
        sizes[ng++] = best; left -= best;
    }
    fclose(fo);
    printf("N=%d %s Q=%d: %d grupos%s  tamanos:", N, tag, Q, ng, anytimeout ? " (alguna busqueda cortada)" : "");
    for (int g = 0; g < ng; g++) printf(" %d", sizes[g]);
    printf("\n");
    return 0;
}
