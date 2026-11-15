/* muestra.c — K soluciones distintas de N-reinas al azar (backtracking con orden aleatorio).
   uso: ./muestra N K semilla > archivo */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int N,c[64],col[64],d1[128],d2[128],found; static unsigned long long rs;
static int ri(int n){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (int)((rs>>11)%n);}
static void bt(int r){ if(found) return; if(r==N){found=1;return;}
  int ord[64]; for(int i=0;i<N;i++)ord[i]=i; for(int i=N-1;i>0;i--){int j=ri(i+1);int t=ord[i];ord[i]=ord[j];ord[j]=t;}
  for(int k=0;k<N&&!found;k++){int x=ord[k]; if(col[x]||d1[r+x]||d2[r-x+N])continue; col[x]=d1[r+x]=d2[r-x+N]=1;c[r]=x;bt(r+1); col[x]=d1[r+x]=d2[r-x+N]=0;}}
int main(int a,char**v){N=atoi(v[1]);int K=atoi(v[2]);rs=strtoull(v[3],0,10)*0x9E3779B97F4A7C15ULL|1;
 int *S=malloc(sizeof(int)*N*K),n=0;
 while(n<K){found=0;memset(col,0,sizeof col);memset(d1,0,sizeof d1);memset(d2,0,sizeof d2);bt(0);
  int dup=0;for(int i=0;i<n&&!dup;i++)if(!memcmp(S+i*N,c,sizeof(int)*N))dup=1; if(dup)continue; memcpy(S+n*N,c,sizeof(int)*N);n++;
  for(int r=0;r<N;r++)printf("%d%c",c[r],r<N-1?' ':'\n');}
}
