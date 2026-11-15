// zeta_cob.c — cuanta Q se ve recorriendo las soluciones con primera cifra <= k.
// No guarda soluciones: para cada una mide m, la distancia a la esquina mas cercana de
// sus cuatro reinas de borde (m = la menor primera cifra entre sus 8 imagenes).  Asi
//   tramo de curva(k) = #{c1 <= k} / Q      parte de Q(k) = #{m <= k} / Q
// Paralelo con pthreads; usa el espejo V (m no cambia) para recorrer media primera fila.
//
// cc -O2 -pthread -o zeta_cob zeta_cob.c && ./zeta_cob 8 18
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define MAXN 24

static int N, ntask;
static int tasks[MAXN * MAXN][2];
static int nexttask;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long H[MAXN + 2], F[MAXN + 2];   // H[m], F[c1]

typedef struct { unsigned long long h[MAXN + 2]; int c1; } ctx;

static inline int imin(int a, int b) { return a < b ? a : b; }

// inv0, invl: fila de la reina en la columna 0 y en la ultima; cl: columna de la ultima fila
static void rec(ctx *t, int r, unsigned cols, unsigned d1, unsigned d2, int inv0, int invl) {
    unsigned avail = ~(cols | d1 | d2) & ((1u << N) - 1);
    if (r == N - 1) {
        if (!avail) return;
        int cl = __builtin_ctz(avail);
        if (cl == 0) inv0 = r;
        if (cl == N - 1) invl = r;
        int m = imin(t->c1, N - 1 - t->c1);
        m = imin(m, imin(cl, N - 1 - cl));
        m = imin(m, imin(inv0, N - 1 - inv0));
        m = imin(m, imin(invl, N - 1 - invl));
        t->h[m + 1]++;
        return;
    }
    while (avail) {
        unsigned b = avail & -avail;
        avail ^= b;
        rec(t, r + 1, cols | b, (d1 | b) << 1, (d2 | b) >> 1,
            b == 1u ? r : inv0, b == (1u << (N - 1)) ? r : invl);
    }
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&lock);
        int i = nexttask++;
        pthread_mutex_unlock(&lock);
        if (i >= ntask) return NULL;
        int a = tasks[i][0], c = tasks[i][1];
        unsigned ba = 1u << a, bc = 1u << c;
        ctx t;
        memset(&t, 0, sizeof t);
        t.c1 = a;
        rec(&t, 2, ba | bc, ((ba << 1) | bc) << 1, ((ba >> 1) | bc) >> 1,
            a == 0 ? 0 : (c == 0 ? 1 : -1000), a == N - 1 ? 0 : (c == N - 1 ? 1 : -1000));
        // el peso 2 cuenta tambien el espejo; la columna central solo cuenta una vez
        int w = (2 * a == N - 1) ? 1 : 2;
        unsigned long long sum = 0;
        for (int m = 1; m <= N; m++) sum += t.h[m];
        pthread_mutex_lock(&lock);
        for (int m = 1; m <= N; m++) H[m] += w * t.h[m];
        F[a + 1] += sum;
        if (w == 2) F[N - a] += sum;
        pthread_mutex_unlock(&lock);
    }
}

int main(int argc, char **argv) {
    int n0 = atoi(argv[1]), n1 = atoi(argv[2]), nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    printf("N,Q,k,n_c1,h_m\n");
    for (N = n0; N <= n1; N++) {
        ntask = nexttask = 0;
        memset(H, 0, sizeof H); memset(F, 0, sizeof F);
        for (int a = 0; a < (N + 1) / 2; a++)
            for (int c = 0; c < N; c++)
                if (c != a && abs(c - a) != 1) { tasks[ntask][0] = a; tasks[ntask][1] = c; ntask++; }
        pthread_t th[64];
        for (int i = 0; i < nt && i < 64; i++) pthread_create(&th[i], NULL, worker, NULL);
        for (int i = 0; i < nt && i < 64; i++) pthread_join(th[i], NULL);
        unsigned long long Q = 0;
        for (int k = 1; k <= N; k++) Q += F[k];
        for (int k = 1; k <= N; k++) printf("%d,%llu,%d,%llu,%llu\n", N, Q, k, F[k], H[k]);
        fflush(stdout);
        fprintf(stderr, "N=%d Q=%llu\n", N, Q);
    }
    return 0;
}
