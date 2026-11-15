/* shape_atlas/atlas.c — atlas de formas epicíclicas de tableros de N-reinas.
 *
 *   E0 (fila→frecuencia):  z(t) = Σ_r exp(i((r+1)t + 2π c_r/N))
 *   E1 (columna→frecuencia): w(t) = Σ_r exp(i((c_r+1)t + 2π r/N))
 *   E2 (diagonales→frecuencia, fase 0): bandas r+c+1 | c−r+3N−1 | c+4N−1
 *
 * Uso:  ./atlas facts            > out/facts.txt
 *       ./atlas stats N          > out/stats_N.txt   (+ out/feat_N.csv)
 *       ./atlas gallery          > gallery/data.js
 *
 * Compilar: cc -O2 -o atlas atlas.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#define MAXN 16
#define MAXK 64
#define MS 4099               /* muestras por curva (primo: ningún t_j cae en 2πk/N) */
typedef double complex cplx;
static const double TAU = 6.283185307179586;

/* ------------------------------------------------------------------ RNG */
static unsigned long long rng_s = 0x9E3779B97F4A7C15ULL;
static unsigned long long rnd(void){ rng_s ^= rng_s<<13; rng_s ^= rng_s>>7; rng_s ^= rng_s<<17; return rng_s; }
static int rndi(int n){ return (int)(rnd()%(unsigned long long)n); }

/* ------------------------------------------------------------ polinomio */
typedef struct { int K; int f[MAXK]; cplx a[MAXK]; } Poly;
static void padd(Poly*p,int f,cplx a){
    for(int k=0;k<p->K;k++) if(p->f[k]==f){ p->a[k]+=a; return; }
    p->f[p->K]=f; p->a[p->K]=a; p->K++;
}
static void enc0(const int*c,int N,Poly*p){ p->K=0; for(int r=0;r<N;r++) padd(p,r+1,cexp(I*TAU*c[r]/N)); }
static void enc1(const int*c,int N,Poly*p){ p->K=0; for(int r=0;r<N;r++) padd(p,c[r]+1,cexp(I*TAU*r/N)); }
static void enc2(const int*c,int N,Poly*p){ p->K=0; for(int r=0;r<N;r++){
    padd(p,r+c[r]+1,1); padd(p,c[r]-r+3*N-1,1); padd(p,c[r]+4*N-1,1); } }
static cplx zval(const Poly*p,double t){ cplx s=0; for(int k=0;k<p->K;k++) s+=p->a[k]*cexp(I*p->f[k]*t); return s; }
static cplx dval(const Poly*p,double t){ cplx s=0; for(int k=0;k<p->K;k++) s+=I*p->f[k]*p->a[k]*cexp(I*p->f[k]*t); return s; }

static cplx Z[MS],D1[MS],D2[MS];
static void sample(const Poly*p){
    cplx w[MAXK],rot[MAXK]; double dt=TAU/MS;
    for(int k=0;k<p->K;k++){ w[k]=p->a[k]; rot[k]=cexp(I*p->f[k]*dt); }
    for(int j=0;j<MS;j++){
        cplx z=0,d1=0,d2=0;
        for(int k=0;k<p->K;k++){ double f=p->f[k]; z+=w[k]; d1+=I*f*w[k]; d2+=-f*f*w[k]; w[k]*=rot[k]; }
        Z[j]=z; D1[j]=d1; D2[j]=d2;
        if((j&511)==511) for(int k=0;k<p->K;k++) w[k]=p->a[k]*cexp(I*p->f[k]*dt*(j+1));
    }
}
/* mínimo de |z|² (deriv=0) o |z'|² (deriv=1) en [a,b], sección áurea */
static double gmin(const Poly*p,double a,double b,int deriv,double*tb){
    const double g=0.6180339887498949; double x1=b-g*(b-a), x2=a+g*(b-a);
    cplx v1=deriv?dval(p,x1):zval(p,x1), v2=deriv?dval(p,x2):zval(p,x2);
    double f1=creal(v1*conj(v1)), f2=creal(v2*conj(v2));
    for(int it=0;it<90;it++){
        if(f1<f2){ b=x2; x2=x1; f2=f1; x1=b-g*(b-a); cplx v=deriv?dval(p,x1):zval(p,x1); f1=creal(v*conj(v)); }
        else     { a=x1; x1=x2; f1=f2; x2=a+g*(b-a); cplx v=deriv?dval(p,x2):zval(p,x2); f2=creal(v*conj(v)); }
    }
    if(tb) *tb=0.5*(a+b);
    return sqrt(f1<f2?f1:f2);
}

/* --------------------------------------------------- autointersecciones */
#define GG 64
#define ORIG_EXCL 0.02   /* cruces en el origen se cuentan aparte: C(nzero,2) */
static int cellcnt[GG*GG+1], cellstart[GG*GG+1], *items=NULL; static int itemcap=0;
static int seg_cross(const cplx*z,int n,int i,int j,cplx*P){
    cplx p=z[i], r=z[(i+1)%n]-p, q=z[j], s=z[(j+1)%n]-q;
    double den=creal(r)*cimag(s)-cimag(r)*creal(s);
    if(fabs(den)<1e-300) return 0;
    cplx qp=q-p;
    double t=(creal(qp)*cimag(s)-cimag(qp)*creal(s))/den;
    double u=(creal(qp)*cimag(r)-cimag(qp)*creal(r))/den;
    if(t>=0&&t<1&&u>=0&&u<1){ *P=p+t*r; return 1; }
    return 0;
}
static int count_cross(const cplx*z,int n){
    double x0=1e300,x1=-1e300,y0=1e300,y1=-1e300;
    for(int j=0;j<n;j++){ double x=creal(z[j]),y=cimag(z[j]); if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; }
    double cw=(x1-x0)/GG+1e-12, ch=(y1-y0)/GG+1e-12;
    #define CX(x) ({int _c=(int)floor(((x)-x0)/cw); _c<0?0:(_c>=GG?GG-1:_c);})
    #define CY(y) ({int _c=(int)floor(((y)-y0)/ch); _c<0?0:(_c>=GG?GG-1:_c);})
    memset(cellcnt,0,sizeof cellcnt);
    int tot=0;
    for(int pass=0;pass<2;pass++){
        if(pass==1){
            if(tot>itemcap){ itemcap=tot*2; items=realloc(items,sizeof(int)*itemcap); }
            int acc=0; for(int c=0;c<GG*GG;c++){ cellstart[c]=acc; acc+=cellcnt[c]; cellcnt[c]=0; }
        }
        for(int i=0;i<n;i++){
            cplx a=z[i], b=z[(i+1)%n];
            int ax=CX(fmin(creal(a),creal(b))), bx=CX(fmax(creal(a),creal(b)));
            int ay=CY(fmin(cimag(a),cimag(b))), by=CY(fmax(cimag(a),cimag(b)));
            for(int cx=ax;cx<=bx;cx++) for(int cy=ay;cy<=by;cy++){
                int c=cy*GG+cx;
                if(pass==0){ cellcnt[c]++; tot++; }
                else items[cellstart[c]+cellcnt[c]++]=i;
            }
        }
    }
    int cnt=0;
    for(int c=0;c<GG*GG;c++){
        int *L=items+cellstart[c], m=cellcnt[c];
        for(int a=0;a<m;a++) for(int b=a+1;b<m;b++){
            int i=L[a], j=L[b]; if(i>j){int t=i;i=j;j=t;}
            if(j==i+1 || (i==0&&j==n-1) || i==j) continue;
            cplx P;
            if(seg_cross(z,n,i,j,&P)){ if(cabs(P)>ORIG_EXCL && CY(cimag(P))*GG+CX(creal(P))==c) cnt++; }
        }
    }
    return cnt;
}

/* -------------------------------------------------------------- rasgos */
#define NF 19
static const char*FN[NF]={"L","area","rmin","rmax","msr","m4","ani","cen","tabs","minspeed","zt1",
                          "nzero","nzlat","wind","turn","ncross","nlobes","ninfl","ncusp"};
static const char*FDESC[NF]={
 "longitud de arco","área algebraica ½∮(x dy−y dx)","|z| mínimo (refinado)","|z| máximo",
 "⟨|z|²⟩ en t uniforme","⟨|z|⁴⟩ en t uniforme","anisotropía del trazo (inercia, peso ds)",
 "|centroide del trazo|/rmax","curvatura absoluta total (vueltas)","velocidad mínima |z'|",
 "|z(2π/N)| (punto marcado)","pasos exactos por el origen","pasos por el origen en t=2πk/N",
 "número de giro alrededor del origen (−99 si pasa por él)","número de rotación de la tangente",
 "autointersecciones","lóbulos (máximos locales de |z|)","inflexiones","cúspides (z'=0)"};
typedef struct { double v[NF]; } Feat;
static double R2[MS],S2[MS],CR[MS];

static void features(const Poly*p,int N,int lattice,Feat*F){
    sample(p); double dt=TAU/MS;
    double L=0,A=0,msr=0,m4=0,rmax=0,tabs=0,minsp=1e300; cplx sw=0; double ws=0;
    for(int j=0;j<MS;j++){
        R2[j]=creal(Z[j]*conj(Z[j])); S2[j]=creal(D1[j]*conj(D1[j])); CR[j]=cimag(conj(D1[j])*D2[j]);
        double sp=sqrt(S2[j]); L+=sp*dt; A+=0.5*cimag(conj(Z[j])*D1[j])*dt;
        msr+=R2[j]; m4+=R2[j]*R2[j]; if(R2[j]>rmax) rmax=R2[j];
        if(S2[j]>1e-30) tabs+=fabs(CR[j])/S2[j]*dt; if(sp<minsp) minsp=sp;
        sw+=sp*Z[j]; ws+=sp;
    }
    rmax=sqrt(rmax); msr/=MS; m4/=MS; tabs/=TAU;
    cplx cen=sw/ws, I2=0; double J2=0;
    for(int j=0;j<MS;j++){ double sp=sqrt(S2[j]); cplx d=Z[j]-cen; I2+=sp*d*d; J2+=sp*creal(d*conj(d)); }
    double ani=cabs(I2)/J2;
    double rmin=1e300; int nzero=0,nlobes=0,ninfl=0,ncusp=0;
    for(int j=0;j<MS;j++){
        int jm=(j+MS-1)%MS, jp=(j+1)%MS; double t=j*dt;
        if(R2[j]<=R2[jm]&&R2[j]<R2[jp]){ double v=gmin(p,t-dt,t+dt,0,NULL); if(v<rmin) rmin=v; if(v<1e-6) nzero++; }
        if(R2[j]>R2[jm]&&R2[j]>=R2[jp]) nlobes++;
        if((CR[j]>0)!=(CR[jp]>0)) ninfl++;
        if(S2[j]<=S2[jm]&&S2[j]<S2[jp]){ double v=gmin(p,t-dt,t+dt,1,NULL); if(v<minsp) minsp=v; if(v<1e-6) ncusp++; }
    }
    double wind=-99, turn=-99;
    if(rmin>1e-6){ double s=0; for(int j=0;j<MS;j++) s+=carg(Z[(j+1)%MS]/Z[j]); wind=round(s/TAU); }
    if(minsp>1e-6){ double s=0; for(int j=0;j<MS;j++) s+=carg(D1[(j+1)%MS]/D1[j]); turn=round(s/TAU); }
    int nzlat=0; double zt1=0;
    if(lattice){ for(int k=0;k<N;k++) if(cabs(zval(p,TAU*k/N))<1e-9) nzlat++; zt1=cabs(zval(p,TAU/N)); }
    int ncross=count_cross(Z,MS)+nzero*(nzero-1)/2;
    double v[NF]={L,A,rmin,rmax,msr,m4,ani,cabs(cen)/rmax,tabs,minsp,zt1,nzero,nzlat,wind,turn,ncross,nlobes,ninfl,ncusp};
    memcpy(F->v,v,sizeof v);
}

/* ------------------------------------------------------------ tableros */
static int conflicts(const int*c,int N){
    int k=0; for(int i=0;i<N;i++) for(int j=i+1;j<N;j++) if(c[i]==c[j]||abs(c[i]-c[j])==j-i) k++; return k;
}
static int is_perm(const int*c,int N){ int m=0; for(int r=0;r<N;r++) m|=1<<c[r]; return m==(1<<N)-1; }
static long long bkey(const int*c,int N){ long long k=0; for(int r=0;r<N;r++) k=k*N+c[r]; return k; }
static void shear(const int*c,int N,int a,int b,int*o){ for(int r=0;r<N;r++) o[r]=((c[r]+a*(r+1)+b)%N+N)%N; }
static long long keyG(const int*c,int N,int withb,int withmirror){
    long long best=-1; int o[MAXN],m[MAXN];
    for(int s=0;s<=withmirror;s++){
        for(int r=0;r<N;r++) m[r]=s?(N-c[r])%N:c[r];
        for(int a=0;a<N;a++) for(int b=0;b<(withb?N:1);b++){ shear(m,N,a,b,o); long long k=bkey(o,N); if(best<0||k<best) best=k; }
    }
    return best;
}
/* simetrías D4 sobre permutaciones */
static void d4(const int*c,int N,int s,int*o){
    for(int r=0;r<N;r++){
        int x=r,y=c[r],X,Y;
        switch(s){ case 0:X=x;Y=y;break; case 1:X=y;Y=N-1-x;break; case 2:X=N-1-x;Y=N-1-y;break; case 3:X=N-1-y;Y=x;break;
                   case 4:X=x;Y=N-1-y;break; case 5:X=N-1-x;Y=y;break; case 6:X=y;Y=x;break; default:X=N-1-y;Y=N-1-x;break; }
        o[X]=Y;
    }
}
static long long canon(const int*c,int N){ long long b=-1; int o[MAXN]; for(int s=0;s<8;s++){ d4(c,N,s,o); long long k=bkey(o,N); if(b<0||k<b) b=k; } return b; }

static int *SOL=NULL, nsol=0;
static void sols_rec(int N,int r,int*c,unsigned cols,unsigned d1,unsigned d2){
    if(r==N){ SOL=realloc(SOL,sizeof(int)*N*(nsol+1)); memcpy(SOL+N*nsol,c,sizeof(int)*N); nsol++; return; }
    for(int x=0;x<N;x++){ if(cols>>x&1) continue; if(d1>>(r+x)&1) continue; if(d2>>(r-x+N)&1) continue;
        c[r]=x; sols_rec(N,r+1,c,cols|1u<<x,d1|1u<<(r+x),d2|1u<<(r-x+N)); }
}
static void solve(int N){ int c[MAXN]; free(SOL); SOL=NULL; nsol=0; sols_rec(N,0,c,0,0,0); }
static int *PER=NULL; static int nper=0;
static void perm_rec(int N,int r,int*c,unsigned used){
    if(r==N){ memcpy(PER+N*nper,c,sizeof(int)*N); nper++; return; }
    for(int x=0;x<N;x++) if(!(used>>x&1)){ c[r]=x; perm_rec(N,r+1,c,used|1u<<x); }
}
static void allperms(int N){ long long f=1; for(int i=2;i<=N;i++) f*=i; PER=realloc(PER,sizeof(int)*N*f); nper=0; int c[MAXN]; perm_rec(N,0,c,0); }
static void randperm(int N,int*c){ for(int i=0;i<N;i++) c[i]=i; for(int i=N-1;i>0;i--){ int j=rndi(i+1),t=c[i]; c[i]=c[j]; c[j]=t; } }

/* ============================================================== FACTS */
/* (a) vía vectores de multiplicidad: z(0)=Σ m_j ω^j */
static long long fact_[20];
static int comp_m[MAXN]; static long long nvan_comp, nvan_boards, nperm_like;
static void comp_rec(int N,int j,int left){
    if(j==N-1){ comp_m[j]=left; cplx s=0; for(int k=0;k<N;k++) s+=comp_m[k]*cexp(I*TAU*k/N);
        if(cabs(s)<1e-9){ nvan_comp++; long long b=fact_[N]; for(int k=0;k<N;k++) b/=fact_[comp_m[k]]; nvan_boards+=b;
            int allone=1; for(int k=0;k<N;k++) if(comp_m[k]!=1) allone=0; nperm_like+=allone; }
        return; }
    for(int v=0;v<=left;v++){ comp_m[j]=v; comp_rec(N,j+1,left-v); }
}
/* (b) fuerza bruta sobre todos los N^N tableros */
static cplx W[MAXN];
static long long bb_zero3, bb_tor, bb_mis, bb_zero0, bb_perm, bb_mis0;
static void brute_rec(int N,int r,int*c,cplx s0,cplx sp,cplx sm){
    if(r==N){
        int z0=cabs(s0)<1e-9, z3=z0&&cabs(sp)<1e-9&&cabs(sm)<1e-9;
        int pm=is_perm(c,N), tor=0;
        if(pm){ unsigned a=0,b=0; for(int i=0;i<N;i++){ a|=1u<<((c[i]+i)%N); b|=1u<<(((c[i]-i)%N+N)%N);} tor=(a==(1u<<N)-1)&&(b==(1u<<N)-1); }
        bb_zero0+=z0; bb_perm+=pm; bb_mis0+=(z0!=pm); bb_zero3+=z3; bb_tor+=tor; bb_mis+=(z3!=tor);
        return;
    }
    for(int x=0;x<N;x++){ c[r]=x; brute_rec(N,r+1,c,s0+W[x],sp+W[(x+r+1)%N],sm+W[((x-r-1)%N+N)%N]); }
}

static void do_facts(void){
    fact_[0]=1; for(int i=1;i<20;i++) fact_[i]=fact_[i-1]*i;
    printf("== (a) z(0)=0  <=>  una reina por columna ? ==\n");
    printf("z(0)=Σ_r ω^{c_r} solo depende de las multiplicidades m_j de cada columna.\n");
    printf("Se enumeran TODOS los vectores (m_0..m_{N-1}) con Σm=N y se mira cuáles anulan Σ m_j ω^j.\n\n");
    printf(" N  primo  vectores_anulan  tableros_con_z(0)=0     N!(perms)   ratio   iff?\n");
    for(int N=3;N<=12;N++){
        nvan_comp=nvan_boards=nperm_like=0; comp_rec(N,0,N);
        int pr=1; for(int d=2;d<N;d++) if(N%d==0) pr=0;
        printf("%2d   %s   %10lld   %18lld  %12lld  %7.3f   %s\n",N,pr?"sí":"no",nvan_comp,nvan_boards,fact_[N],
               (double)nvan_boards/fact_[N], nvan_boards==fact_[N]?"SÍ":"NO");
    }
    printf("\nComprobación directa (fuerza bruta sobre los N^N tableros) del punto (a) y del (b):\n");
    printf("(b) toroidal  <=>  z(0)=z(2π/N)=z(-2π/N)=0\n\n");
    printf(" N     N^N       z(0)=0     perms   discrep(a)   z(0,±2π/N)=0  toroidales  discrep(b)\n");
    for(int N=4;N<=9;N++){
        for(int k=0;k<N;k++) W[k]=cexp(I*TAU*k/N);
        bb_zero3=bb_tor=bb_mis=bb_zero0=bb_perm=bb_mis0=0; int c[MAXN];
        brute_rec(N,0,c,0,0,0);
        long long NN=1; for(int i=0;i<N;i++) NN*=N;
        printf("%2d %10lld %10lld %9lld %10lld %14lld %10lld %10lld\n",N,NN,bb_zero0,bb_perm,bb_mis0,bb_zero3,bb_tor,bb_mis);
        fflush(stdout);
    }
    /* N=11: z(0)=0 ⇒ permutación está probado arriba por los vectores de multiplicidad; basta recorrer 11! */
    for(int N=10;N<=11;N++){
        for(int k=0;k<N;k++) W[k]=cexp(I*TAU*k/N);
        allperms(N); long long z3=0,tor=0,mis=0;
        for(int i=0;i<nper;i++){ int*c=PER+N*i; cplx sp=0,sm=0; unsigned a=0,b=0;
            for(int r=0;r<N;r++){ sp+=W[(c[r]+r+1)%N]; sm+=W[((c[r]-r-1)%N+N)%N]; a|=1u<<((c[r]+r)%N); b|=1u<<(((c[r]-r)%N+N)%N); }
            int zz=cabs(sp)<1e-9&&cabs(sm)<1e-9, tt=(a==(1u<<N)-1)&&(b==(1u<<N)-1); z3+=zz; tor+=tt; mis+=(zz!=tt); }
        printf("%2d  (sobre las %d permutaciones)  z(±2π/N)=0: %lld   toroidales: %lld   discrep: %lld\n",N,nper,z3,tor,mis);
        fflush(stdout);
    }
    free(PER); PER=NULL;

    printf("\n== (c) solución lineal c_r = a·r mod N  →  ¿flor de N−1 pétalos? ==\n");
    printf("Fórmula: z_a(t) = e^{it}·(1−e^{iNt})/(1−e^{i(t+2πa/N)}),  |z_a| = |sin(Nt/2)/sin((t+2πa/N)/2)|\n");
    printf("y z_a(t) = e^{−2πia/N}·z_0(t+2πa/N): TODAS las a dan la misma curva girada (también a=0,1,N−1, que no son solución).\n\n");
    printf(" N  a  solución?  toroidal?  pasos_origen  lóbulos  autoint  |z|max  dif_vs_a=0\n");
    int Ns[3]={7,8,11};
    for(int q=0;q<3;q++){ int N=Ns[q];
        for(int a=0;a<N;a++){
            int c[MAXN],c0[MAXN]; for(int r=0;r<N;r++){ c[r]=(a*r)%N; c0[r]=0; }
            Poly p,p0; enc0(c,N,&p); enc0(c0,N,&p0); Feat F; features(&p,N,1,&F);
            double th=TAU*a/N, dif=0;
            for(int j=0;j<997;j++){ double t=TAU*j/997; double d=cabs(zval(&p,t)-cexp(-I*th)*zval(&p0,t+th)); if(d>dif) dif=d; }
            int tor=0; if(is_perm(c,N)){ unsigned A=0,B=0; for(int r=0;r<N;r++){A|=1u<<((c[r]+r)%N); B|=1u<<(((c[r]-r)%N+N)%N);} tor=(A==(1u<<N)-1)&&(B==(1u<<N)-1); }
            printf("%2d %2d   %s        %s      %4.0f        %3.0f    %4.0f   %6.3f   %.1e\n",N,a,
                   (is_perm(c,N)&&conflicts(c,N)==0)?"sí":"no",tor?"sí":"no",F.v[11],F.v[16],F.v[15],F.v[3],dif);
        }
        printf("\n");
    }
    printf("== Área (Parseval) en E0 ==\n");
    { double mx=0; for(int it=0;it<2000;it++){ int N=8,c[MAXN]; for(int r=0;r<N;r++) c[r]=rndi(N); Poly p; enc0(c,N,&p); Feat F; features(&p,N,1,&F);
        double d=fabs(F.v[1]-M_PI*N*(N+1)/2.0); if(d>mx) mx=d; }
      printf("N=8, 2000 tableros al azar: max|área − π·N(N+1)/2| = %.2e   (π·36 = %.4f)\n",mx,M_PI*36); }
}

/* ============================================================== STATS */
static double r7(double v){ if(v==0||!isfinite(v)) return v; double e=pow(10,6-floor(log10(fabs(v)))); return round(v*e)/e; }
static double auc(const double*x0,const int*lab,int n){
    double *x=malloc(sizeof(double)*n); for(int i=0;i<n;i++) x[i]=r7(x0[i]);
    /* P(x_pos > x_neg) + ½P(=) */
    int *idx=malloc(sizeof(int)*n); for(int i=0;i<n;i++) idx[i]=i;
    /* sort by x */
    for(int gap=n/2;gap>0;gap/=2) for(int i=gap;i<n;i++){ int t=idx[i],j=i; while(j>=gap&&x[idx[j-gap]]>x[t]){ idx[j]=idx[j-gap]; j-=gap; } idx[j]=t; }
    double rs=0; long long np=0,nn=0;
    for(int i=0;i<n;){ int j=i; while(j<n&&x[idx[j]]==x[idx[i]]) j++; double rk=0.5*(i+1+j);
        for(int k=i;k<j;k++) if(lab[idx[k]]) { rs+=rk; np++; } else nn++; i=j; }
    free(idx); free(x);
    if(!np||!nn) return NAN;
    return (rs-np*(np+1)/2.0)/((double)np*nn);
}
/* Fisher LDA con validación cruzada 5-fold → AUC */
static double lda_cv(double*X,const int*lab,int n,int D){
    double *proj=malloc(sizeof(double)*n); int *fold=malloc(sizeof(int)*n);
    for(int i=0;i<n;i++) fold[i]=rndi(5);
    for(int f=0;f<5;f++){
        double m1[NF]={0},m0[NF]={0},S[NF][NF]={{0}}; int n1=0,n0=0;
        for(int i=0;i<n;i++) if(fold[i]!=f){ double*x=X+i*D; if(lab[i]){n1++; for(int d=0;d<D;d++) m1[d]+=x[d];} else {n0++; for(int d=0;d<D;d++) m0[d]+=x[d];} }
        for(int d=0;d<D;d++){ m1[d]/=n1; m0[d]/=n0; }
        for(int i=0;i<n;i++) if(fold[i]!=f){ double*x=X+i*D,*m=lab[i]?m1:m0; for(int a=0;a<D;a++) for(int b=0;b<D;b++) S[a][b]+=(x[a]-m[a])*(x[b]-m[b]); }
        double A[NF][NF+1]; double tr=0; for(int a=0;a<D;a++) tr+=S[a][a];
        for(int a=0;a<D;a++){ for(int b=0;b<D;b++) A[a][b]=S[a][b]+(a==b?1e-6*tr/D+1e-12:0); A[a][D]=m1[a]-m0[a]; }
        for(int col=0;col<D;col++){ int pv=col; for(int r=col+1;r<D;r++) if(fabs(A[r][col])>fabs(A[pv][col])) pv=r;
            for(int k=0;k<=D;k++){ double t=A[col][k]; A[col][k]=A[pv][k]; A[pv][k]=t; }
            for(int r=0;r<D;r++) if(r!=col){ double q=A[r][col]/A[col][col]; for(int k=col;k<=D;k++) A[r][k]-=q*A[col][k]; } }
        double w[NF]; for(int a=0;a<D;a++) w[a]=A[a][D]/A[a][a];
        for(int i=0;i<n;i++) if(fold[i]==f){ double s=0; for(int d=0;d<D;d++) s+=w[d]*X[i*D+d]; proj[i]=s; }
    }
    double r=auc(proj,lab,n); free(proj); free(fold); return r;
}

typedef struct { int c[MAXN]; int conf, perm, cls; Feat f[3]; } Rec;
/* cls: 0 solución, 1 casi (1 conflicto, perm), 2 resto perms, 3 fila al azar */
static const char*CLSN[4]={"sol","casi","perm","azar"};

static void describe(Rec*R,int n,int N,FILE*csv){
    if(csv){ fprintf(csv,"board,cls,conf"); for(int e=0;e<3;e++) for(int k=0;k<NF;k++) fprintf(csv,",E%d_%s",e,FN[k]); fprintf(csv,"\n");
        for(int i=0;i<n;i++){ for(int r=0;r<N;r++) fputc('0'+R[i].c[r],csv); fprintf(csv,",%s,%d",CLSN[R[i].cls],R[i].conf);
            for(int e=0;e<3;e++) for(int k=0;k<NF;k++) fprintf(csv,",%.6g",R[i].f[e].v[k]); fprintf(csv,"\n"); } }
    int cnt[4]={0}; for(int i=0;i<n;i++) cnt[R[i].cls]++;
    printf("muestras: sol=%d casi=%d perm(≥2 conflictos)=%d azar=%d\n\n",cnt[0],cnt[1],cnt[2],cnt[3]);
    double *x=malloc(sizeof(double)*n); int *lab=malloc(sizeof(int)*n), *lab2=malloc(sizeof(int)*n);
    double *xs=malloc(sizeof(double)*n);
    for(int e=0;e<3;e++){
        printf("--- codificación E%d ---\n",e);
        printf("%-9s %10s %10s %10s %10s  %8s %8s  %s\n","rasgo","sol","casi","perm","azar","AUC_s/p","AUC_s/c","rango_sol (frac. no-sol perms dentro)");
        for(int k=0;k<NF;k++){
            double s[4]={0}; double mn=1e300,mx=-1e300;
            for(int i=0;i<n;i++){ s[R[i].cls]+=R[i].f[e].v[k]; if(R[i].cls==0){ if(R[i].f[e].v[k]<mn) mn=R[i].f[e].v[k]; if(R[i].f[e].v[k]>mx) mx=R[i].f[e].v[k]; } }
            int m=0,m2=0,inr=0,np=0;
            for(int i=0;i<n;i++){ if(R[i].cls<=2){ x[m]=R[i].f[e].v[k]; lab[m]=R[i].cls==0; m++;
                    if(R[i].cls>0){ np++; if(R[i].f[e].v[k]>=mn-1e-9&&R[i].f[e].v[k]<=mx+1e-9) inr++; } }
                if(R[i].cls<=1){ xs[m2]=R[i].f[e].v[k]; lab2[m2]=R[i].cls==0; m2++; } }
            printf("%-9s %10.4g %10.4g %10.4g %10.4g  %8.3f %8.3f  [%.4g, %.4g]  %.3f\n",FN[k],s[0]/fmax(cnt[0],1),s[1]/fmax(cnt[1],1),s[2]/fmax(cnt[2],1),s[3]/fmax(cnt[3],1),
                   auc(x,lab,m),auc(xs,lab2,m2),mn,mx,(double)inr/np);
        }
        /* LDA con todos los rasgos no constantes (excluye zt1/nzlat: puntos marcados) */
        int m=0; int use[NF],D=0;
        for(int k=0;k<NF;k++){ if(k==10||k==12) continue; double a=1e300,b=-1e300; for(int i=0;i<n;i++) if(R[i].cls<=2){ double v=R[i].f[e].v[k]; if(v<a)a=v; if(v>b)b=v; } if(b-a>1e-7) use[D++]=k; }
        double *X=malloc(sizeof(double)*n*D);
        for(int i=0;i<n;i++) if(R[i].cls<=2){ for(int d=0;d<D;d++) X[m*D+d]=R[i].f[e].v[use[d]]; lab[m]=R[i].cls==0; m++; }
        printf("LDA (5-fold CV) con %d rasgos geométricos puros: AUC sol vs perms = %.3f\n\n",D,lda_cv(X,lab,m,D));
        free(X);
    }
    free(x); free(lab); free(lab2); free(xs);
}

static void orbit_report(int N){
    /* Para cada solución: su órbita bajo G = {c_r → c_r + a(r+1) + b} (curva igual salvo giro y origen de t),
       bajo S = {a} solo (curva idéntica, misma orientación), y G+espejo. */
    printf("== Órbitas de forma (N=%d, %d soluciones) ==\n",N,nsol);
    const char*nm[3]={"S: cizalla a (misma curva, misma orientación)","G: cizalla a + giro b (misma forma salvo rotación)","G+espejo (forma salvo isometría)"};
    for(int g=0;g<3;g++){
        int withb=g>=1, mir=g==2;
        /* conjunto de claves de órbitas con solución */
        long long *keys=malloc(sizeof(long long)*nsol); for(int i=0;i<nsol;i++) keys[i]=keyG(SOL+N*i,N,withb,mir);
        /* recorrer cada órbita distinta: generar miembros */
        long long boards=0,perms=0,sols=0,orbs=0; int maxsol=0; double minfrac=1;
        for(int i=0;i<nsol;i++){ int dup=0; for(int j=0;j<i;j++) if(keys[j]==keys[i]) dup=1; if(dup) continue; orbs++;
            int o[MAXN],m[MAXN]; long long seen[2*MAXN*MAXN]; int ns=0, op=0, os=0;
            for(int s=0;s<=mir;s++){ for(int r=0;r<N;r++) m[r]=s?(N-SOL[N*i+r])%N:SOL[N*i+r];
                for(int a=0;a<N;a++) for(int b=0;b<(withb?N:1);b++){ shear(m,N,a,b,o); long long k=bkey(o,N); int d=0; for(int q=0;q<ns;q++) if(seen[q]==k) d=1; if(d) continue; seen[ns++]=k;
                    if(is_perm(o,N)){ op++; if(conflicts(o,N)==0) os++; } } }
            boards+=ns; perms+=op; sols+=os; if(os>maxsol) maxsol=os; if((double)os/op<minfrac) minfrac=(double)os/op; }
        printf("%s\n   órbitas con solución: %lld | tableros en ellas: %lld | permutaciones: %lld | soluciones: %lld\n",nm[g],orbs,boards,perms,sols);
        printf("   ⇒ techo de precisión de CUALQUIER clasificador que solo vea la forma: %.4f (con recall 1, solo permutaciones)\n",(double)sols/perms);
        printf("   máx. soluciones en una misma órbita: %d ; peor fracción sol/perms en una órbita: %.3f\n",maxsol,minfrac);
        free(keys);
    }
    printf("\n");
}

static void do_stats(int N){
    solve(N);
    int nrand=5000, nsamp_perm=(N<=8)?0:40000;
    Rec *R=NULL; int n=0, cap=0;
    #define PUSH(cc,cl) do{ if(n==cap){cap=cap?cap*2:4096; R=realloc(R,sizeof(Rec)*cap);} memcpy(R[n].c,cc,sizeof(int)*N); R[n].cls=cl; n++; }while(0)
    if(N<=8){ allperms(N); for(int i=0;i<nper;i++){ int*c=PER+N*i; int k=conflicts(c,N); PUSH(c,k==0?0:(k==1?1:2)); } }
    else {
        for(int i=0;i<nsol;i++) PUSH(SOL+N*i,0);
        for(int i=0;i<nsamp_perm;i++){ int c[MAXN]; randperm(N,c); int k=conflicts(c,N); if(k==0) continue; PUSH(c,k==1?1:2); }
        /* casi-soluciones extra: mover una reina de una solución intercambiando dos filas */
        for(int i=0;i<4000;i++){ int c[MAXN]; memcpy(c,SOL+N*rndi(nsol),sizeof(int)*N); int a=rndi(N),b=rndi(N),t=c[a]; c[a]=c[b]; c[b]=t; if(conflicts(c,N)==1) PUSH(c,1); }
    }
    for(int i=0;i<nrand;i++){ int c[MAXN]; for(int r=0;r<N;r++) c[r]=rndi(N); PUSH(c,3); }
    for(int i=0;i<n;i++){ R[i].conf=conflicts(R[i].c,N); R[i].perm=is_perm(R[i].c,N);
        Poly p; enc0(R[i].c,N,&p); features(&p,N,1,&R[i].f[0]); enc1(R[i].c,N,&p); features(&p,N,1,&R[i].f[1]); enc2(R[i].c,N,&p); features(&p,N,0,&R[i].f[2]); }
    printf("##### N=%d  (%s)\n",N,N<=8?"todas las permutaciones":"todas las soluciones + 40000 permutaciones al azar + casi-soluciones por intercambio");
    printf("Rasgos:\n"); for(int k=0;k<NF;k++) printf("  %-9s %s\n",FN[k],FDESC[k]); printf("\n");
    char fn[64]; sprintf(fn,"out/feat_N%d.csv",N); FILE*csv=fopen(fn,"w");
    describe(R,n,N,csv); if(csv) fclose(csv);
    orbit_report(N);
    { int h[2*MAXN+1]={0}; for(int i=0;i<nsol;i++){ int s=0; for(int r=0;r<N;r++) s+=((r+SOL[N*i+r])%2)?-1:1; h[s+MAXN]++; }
      printf("Balance de color de las soluciones (#casillas pares − #impares): "); for(int k=0;k<=2*MAXN;k++) if(h[k]) printf(" %+d:%d",k-MAXN,h[k]); printf("\n");
      printf("   (en E2 y N par, para toda permutación z(π) = 2·Σ_r(−1)^{r+c_r+1}: la curva pasa por el origen en t=π ⇔ balance 0)\n"); }
    /* comprobación: rasgos idénticos dentro de una órbita G */
    { double mx=0; int dk[NF]={0}; for(int t=0;t<50;t++){ int*c=SOL+N*rndi(nsol); int o[MAXN]; shear(c,N,1+rndi(N-1),rndi(N),o); Poly p; Feat a,b;
        enc0(c,N,&p); features(&p,N,1,&a); enc0(o,N,&p); features(&p,N,1,&b);
        for(int k=0;k<NF;k++){ if(k==10||k==12) continue; double d=fabs(a.v[k]-b.v[k])/(1+fabs(a.v[k])); if(d>1e-6) dk[k]++; if(d>mx) mx=d; } }
      printf("Control: diferencia relativa máx. de rasgos geométricos entre solución y su cizalla+giro (50 casos): %.2e\n",mx);
      for(int k=0;k<NF;k++) if(dk[k]) printf("   rasgo %s difiere en %d/50 casos\n",FN[k],dk[k]); }
    /* E2: testigo exacto por área */
    { double A0=0; int ok=1; double mind=1e300; Poly p; enc2(SOL,N,&p); Feat F; features(&p,N,0,&F); A0=F.v[1];
      for(int i=0;i<n;i++){ if(!R[i].perm) continue; double d=R[i].f[2].v[1]-A0; if(R[i].conf==0 && fabs(d)>1e-6) ok=0; if(R[i].conf>0 && d<mind) mind=d; }
      printf("E2: área de toda solución = %.6f (= π·%.1f); %s; mínimo exceso en permutaciones con conflicto = %.4f (= π·%.3f)\n",
             A0,A0/M_PI, ok?"idéntica en todas":"¡NO idéntica!", mind, mind/M_PI); }
    free(R);
}

/* ============================================================ GALLERY */
static void pts(FILE*o,const Poly*p,int np,double sc){
    fputc('[',o); for(int j=0;j<np;j++){ cplx z=zval(p,TAU*j/np); fprintf(o,"%s%d,%d",j?",":"",(int)lround(creal(z)*sc),(int)lround(cimag(z)*sc)); } fputc(']',o);
}
static void item(FILE*o,const int*c,int N,int full,int fund){
    Poly p; Feat F0,F1,F2;
    fprintf(o,"{b:\""); for(int r=0;r<N;r++) fprintf(o,"%x",c[r]); fprintf(o,"\",conf:%d,perm:%d,cls:%d,",conflicts(c,N),is_perm(c,N),fund);
    enc0(c,N,&p); features(&p,N,1,&F0); fprintf(o,"e0:"); pts(o,&p,240,100); fprintf(o,",f0:[");
    for(int k=0;k<NF;k++) fprintf(o,"%s%.4g",k?",":"",F0.v[k]); fprintf(o,"]");
    if(full){
        enc1(c,N,&p); features(&p,N,1,&F1); fprintf(o,",e1:"); pts(o,&p,240,100); fprintf(o,",f1:["); for(int k=0;k<NF;k++) fprintf(o,"%s%.4g",k?",":"",F1.v[k]); fprintf(o,"]");
        enc2(c,N,&p); features(&p,N,0,&F2); fprintf(o,",e2:"); pts(o,&p,420,30); fprintf(o,",f2:["); for(int k=0;k<NF;k++) fprintf(o,"%s%.4g",k?",":"",F2.v[k]); fprintf(o,"]");
    }
    fprintf(o,"}");
}
static void do_gallery(void){
    FILE*o=stdout; int N=8; solve(N);
    long long cls[200]; int ncls=0, fund[200];
    for(int i=0;i<nsol;i++){ long long k=canon(SOL+N*i,N); int f=-1; for(int j=0;j<ncls;j++) if(cls[j]==k) f=j; if(f<0){ f=ncls; cls[ncls++]=k; } fund[i]=f; }
    fprintf(o,"window.ATLAS={N:8,fn:["); for(int k=0;k<NF;k++) fprintf(o,"%s\"%s\"",k?",":"",FN[k]); fprintf(o,"],fd:[");
    for(int k=0;k<NF;k++) fprintf(o,"%s\"%s\"",k?",":"",FDESC[k]); fprintf(o,"],\n");
    fprintf(o,"sol:[\n"); for(int i=0;i<nsol;i++){ if(i) fprintf(o,",\n"); item(o,SOL+N*i,N,1,fund[i]); } fprintf(o,"],\n");
    allperms(N);
    int *near=NULL,nn=0,*oth=NULL,no=0;
    for(int i=0;i<nper;i++){ int k=conflicts(PER+N*i,N); if(k==1){ near=realloc(near,sizeof(int)*(nn+1)); near[nn++]=i; } else if(k>=2){ oth=realloc(oth,sizeof(int)*(no+1)); oth[no++]=i; } }
    fprintf(o,"casi:[\n"); for(int i=0;i<92;i++){ if(i) fprintf(o,",\n"); item(o,PER+N*near[rndi(nn)],N,1,-1); } fprintf(o,"],\n");
    fprintf(o,"perm:[\n"); for(int i=0;i<92;i++){ if(i) fprintf(o,",\n"); item(o,PER+N*oth[rndi(no)],N,1,-1); } fprintf(o,"],\n");
    fprintf(o,"azar:[\n"); for(int i=0;i<92;i++){ int c[MAXN]; for(int r=0;r<N;r++) c[r]=rndi(N); if(i) fprintf(o,",\n"); item(o,c,N,1,-1); } fprintf(o,"],\n");
    /* gemelos: misma forma, distinto estatus */
    fprintf(o,"twins:[\n");
    for(int f=0;f<ncls;f++){ int i=0; while(fund[i]!=f) i++; int*c=SOL+N*i, o1[MAXN], o2[MAXN];
        shear(c,N,0,1,o1);                 /* giro: c+1 mod N */
        int best=1; for(int a=1;a<N;a++){ shear(c,N,a,0,o2); if(is_perm(o2,N)){ best=a; break; } } shear(c,N,best,0,o2);
        if(f) fprintf(o,",\n"); fprintf(o,"["); item(o,c,N,0,f); fprintf(o,","); item(o,o1,N,0,-1); fprintf(o,","); item(o,o2,N,0,-1); fprintf(o,"]"); }
    fprintf(o,"],\n");
    /* flores lineales N=7,11 y toroidales no lineales N=13 */
    fprintf(o,"flores:{");
    int Ns[2]={7,11};
    for(int q=0;q<2;q++){ int n=Ns[q]; fprintf(o,"%s%d:[",q?",":"",n);
        for(int a=0;a<n;a++){ int c[MAXN]; for(int r=0;r<n;r++) c[r]=(a*r)%n; if(a) fprintf(o,","); item(o,c,n,0,a); } fprintf(o,"]"); }
    fprintf(o,"},\n");
    /* N=13 toroidales: buscar no lineales */
    { int n=13; solve(n); fprintf(o,"tor13:[");
      int shown=0, lin=0, tot=0;
      for(int i=0;i<nsol && shown<12;i++){ int*c=SOL+n*i; unsigned A=0,B=0; for(int r=0;r<n;r++){A|=1u<<((c[r]+r)%n); B|=1u<<(((c[r]-r)%n+n)%n);} if(A!=(1u<<n)-1||B!=(1u<<n)-1) continue;
          int isl=1; int d=(c[1]-c[0]+n)%n; for(int r=1;r<n;r++) if((c[r]-c[r-1]+n)%n!=d) isl=0; tot++; if(isl){ lin++; if(lin>2) continue; }
          if(shown) fprintf(o,","); item(o,c,n,0,isl); shown++; }
      fprintf(o,"]\n};\n"); }
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"uso: atlas facts|stats N|gallery\n"); return 1; }
    if(!strcmp(argv[1],"facts")) do_facts();
    else if(!strcmp(argv[1],"stats")) do_stats(argc>2?atoi(argv[2]):8);
    else if(!strcmp(argv[1],"gallery")) do_gallery();
    else if(!strcmp(argv[1],"joint")) { int joint_main(int); return joint_main(atoi(argv[2])); }
    else if(!strcmp(argv[1],"check2")) { int check2_main(void); return check2_main(); }
    else if(!strcmp(argv[1],"check")) { int check_main(void); return check_main(); }
    return 0;
}
/* depuración: comparar rasgos de una solución y su cizalla */
int check_main(void){
    int N=8; solve(N);
    for(int t=0;t<200;t++){ int*c=SOL+N*rndi(nsol); int o[MAXN]; int a=1+rndi(N-1),b=rndi(N); shear(c,N,a,b,o); Poly p; Feat A,B;
        enc0(c,N,&p); features(&p,N,1,&A); enc0(o,N,&p); features(&p,N,1,&B);

        for(int k=0;k<NF;k++) if(k!=10&&k!=12&&fabs(A.v[k]-B.v[k])>1e-4*(1+fabs(A.v[k]))) printf("  %s %.8g %.8g\n",FN[k],A.v[k],B.v[k]); }
    return 0;
}
int check2_main(void){
    int N=8; solve(N);
    for(int t=0;t<200;t++){ int*c=SOL+N*rndi(nsol); int o[MAXN]; int a=1+rndi(N-1),b=rndi(N); shear(c,N,a,b,o); Poly p; Feat A,B;
        enc0(c,N,&p); features(&p,N,1,&A); enc0(o,N,&p); features(&p,N,1,&B);
        if(A.v[15]==B.v[15]) continue;
        printf("board "); for(int r=0;r<N;r++) printf("%d",c[r]); printf("  ncross %g vs %g  nzero %g\n",A.v[15],B.v[15],A.v[11]);
        enc0(c,N,&p); double dt=TAU/MS; sample(&p);
        for(int j=0;j<MS;j++){ double r0=cabs(Z[j]); int jm=(j+MS-1)%MS,jp=(j+1)%MS; if(r0<=cabs(Z[jm])&&r0<cabs(Z[jp])){ double tb; double v=gmin(&p,j*dt-dt,j*dt+dt,0,&tb); if(v<0.3){ cplx d=dval(&p,tb); printf("   min |z|=%.2e at t=%.6f (t*N/2pi=%.4f) dir=%.4f speed=%.3f\n",v,tb,tb*N/TAU,carg(d),cabs(d)); } } }
        break; }
    return 0;
}
/* Forma conjunta (E0, E1): ¿cuántas permutaciones comparten con una solución
   la forma fila→frecuencia Y la forma columna→frecuencia a la vez? */
static void inv(const int*c,int N,int*o){ for(int r=0;r<N;r++) o[c[r]]=r; }
int joint_main(int N){
    solve(N);
    { int mir=0,m[MAXN]; for(int i=0;i<nsol;i++){ for(int r=0;r<N;r++) m[r]=(N-SOL[N*i+r])%N; if(keyG(m,N,1,0)==keyG(SOL+N*i,N,1,0)) mir++; }
      printf("N=%d: soluciones cuya forma E0 tiene simetría especular: %d/%d\n",N,mir,nsol); }
    for(int withb=0;withb<=1;withb++){
        long long tot_p=0,tot_s=0; int amb=0;
        long long *seen=malloc(sizeof(long long)*nsol*N*N); int ns=0;
        for(int i=0;i<nsol;i++){
            int *s=SOL+N*i, si[MAXN]; inv(s,N,si); long long k1=keyG(si,N,withb,0);
            int o[MAXN],oi[MAXN],op=0;
            for(int a=0;a<N;a++) for(int b=0;b<(withb?N:1);b++){ shear(s,N,a,b,o); if(!is_perm(o,N)) continue; inv(o,N,oi);
                if(keyG(oi,N,withb,0)!=k1) continue;
                long long k=bkey(o,N); int d=0; for(int q=0;q<ns;q++) if(seen[q]==k) d=1; if(!d){ seen[ns++]=k; tot_p++; if(conflicts(o,N)==0) tot_s++; }
                if(conflicts(o,N)) op++; }
            if(op) amb++;
        }
        printf("N=%d %s: permutaciones con la misma forma E0 Y E1 que alguna solución: %lld, de ellas soluciones %lld; soluciones con gemelo no-solución: %d/%d\n",
               N, withb?"(salvo giros independientes)":"(curvas idénticas, misma orientación)", tot_p, tot_s, amb, nsol);
        free(seen);
    }
    return 0;
}
