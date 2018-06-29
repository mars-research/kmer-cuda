
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <stdio.h>

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti = N + 1; /* mti==N+1 means mt[N] is not initialized */

						/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
	mt[0] = s & 0xffffffffUL;
	for (mti = 1; mti<N; mti++) {
		mt[mti] =
			(1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
		/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
		/* In the previous versions, MSBs of the seed affect   */
		/* only MSBs of the array mt[].                        */
		/* 2002/01/09 modified by Makoto Matsumoto             */
		mt[mti] &= 0xffffffffUL;
		/* for >32 bit machines */
	}
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
	int i, j, k;
	init_genrand(19650218UL);
	i = 1; j = 0;
	k = (N>key_length ? N : key_length);
	for (; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL))
			+ init_key[j] + j; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++; j++;
		if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
		if (j >= key_length) j = 0;
	}
	for (k = N - 1; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL))
			- i; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++;
		if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
	}

	mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
	unsigned long y;
	static unsigned long mag01[2] = { 0x0UL, MATRIX_A };
	/* mag01[x] = x * MATRIX_A  for x=0,1 */

	if (mti >= N) { /* generate N words at one time */
		int kk;

		if (mti == N + 1)   /* if init_genrand() has not been called, */
			init_genrand(5489UL); /* a default initial seed is used */

		for (kk = 0; kk<N - M; kk++) {
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		for (; kk<N - 1; kk++) {
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

		mti = 0;
	}

	y = mt[mti++];

	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

#define BIG_CONSTANT(x) (x##LLU)
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
struct KMer {
	char arr[50];
	char padding[14];
};
struct HTEntry {
	union {
		KMer kmer;
		struct {
			char padding[56];
			unsigned int count;
			unsigned int offset;
		};
	};
};


#define LOG_SZ 24
#define TBL_SZ 1llu<<LOG_SZ
#define LOG_BATCH_SZ 24
#define BATCH_SZ 1llu<<LOG_BATCH_SZ
__device__   unsigned long long fibonacci_mix(unsigned long long hash) {
	return (hash * 11400714819323198485llu) >> (64 - LOG_SZ);
}

__device__   unsigned long long hash_kmer(KMer * km) {
	unsigned long long * characs = (unsigned long long *)km->arr;
	unsigned long long hash = fibonacci_mix(characs[0]) ^ fibonacci_mix(characs[1]) ^ fibonacci_mix(characs[2]) ^ fibonacci_mix(characs[3]) ^ fibonacci_mix(characs[4]) ^ fibonacci_mix(characs[5]) ^ fibonacci_mix(characs[6]) ^ fibonacci_mix(characs[7]);
	return (hash);

}
#define min(a,b) a<b?a:b
#define max(a,b) a>b?a:b
__device__  bool kmer_cmp(KMer * o1, KMer * o2) {
	unsigned long long *a1 = (unsigned long long *)o1->arr, *a2 = (unsigned long long *)o2->arr;
	if (*a1 != *a2) return false;
	*a1++;
	*a2++;
	if (*a1 != *a2) return false;
	*a1++;
	*a2++;
	if (*a1 != *a2) return false;
	*a1++;
	*a2++;
	if (*a1 != *a2) return false;
	*a1++;
	*a2++;
	if (*a1 != *a2) return false;
	*a1++;
	*a2++;
	if (*a1 != *a2) return false;
	*a1++;
	*a2++;
	if (*a1 != *a2) return false;
	return true;
}
__global__ void histoKernel(KMer* ptr, HTEntry * tbl) {
	KMer * ind = &(ptr[(threadIdx.x*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x)]);
	auto index = hash_kmer(ind);
	auto orig = index;
	bool bval = false;
	//atomicAdd(counter, index);
	while ((bval = (atomicInc(&(tbl[index].count), 0) != 0)) && !kmer_cmp(&(tbl[index].kmer), ind)) {
		index += orig*orig;
		index %= TBL_SZ;
		if (index == orig) { break; }
	}
	if (!bval) atomicAdd(&(tbl[index].count), 1);
	else { tbl[index].kmer = *ind; }
}
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("%s\n",cudaGetErrorString(x)); \
    return 0;}} while(0)
// Helper function for using CUDA to add vectors in parallel.
HTEntry * histoCuda(char * filename)
{
	KMer *dev_data, *my_data;
	HTEntry * tbl = 0;


	dim3 griddims(1 << ((LOG_BATCH_SZ - 7) / 2), (1 << ((LOG_BATCH_SZ - 7))) / (1 << ((LOG_BATCH_SZ - 7) / 2)));
	//printf("sizes: %d, %d\n", 1 << ((LOG_BATCH_SZ - 6) / 2), (1 << ((LOG_BATCH_SZ - 6))) / (1 << ((LOG_BATCH_SZ - 6) / 2)));
	cudaError_t cudaStatus;
	cudaSetDevice(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaMalloc((void**)&dev_data, sizeof(KMer) *BATCH_SZ);
	my_data = (KMer *)malloc(sizeof(KMer) * BATCH_SZ);

	memset((void*)my_data, 0, sizeof(KMer) * BATCH_SZ);
	unsigned long * a = (unsigned long *)my_data;
	for (unsigned long i = 0; i < sizeof(KMer)*(BATCH_SZ) / 4; i++) {
		a[i] = i*i;
	}
	cudaMemcpy(dev_data, my_data, sizeof(KMer) * BATCH_SZ, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {

		printf("cm0 fail %s\n", cudaGetErrorString(cudaStatus));

	}
	cudaStatus = cudaMalloc((void**)&tbl, sizeof(HTEntry)*TBL_SZ);
	cudaMemset((void*)tbl, 0, sizeof(HTEntry) * TBL_SZ);
	if (cudaStatus != cudaSuccess) {

		printf("cm01 fail %s\n", cudaGetErrorString(cudaStatus));

	}
	//printf("addr: %x\n", dev_data);
	//printf("addr: %x\n", tbl);
	CUDA_CALL(cudaDeviceSynchronize());
	unsigned long long t1, t2, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&t1);
	histoKernel << <griddims, 128 >> > (dev_data, tbl);

	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	QueryPerformanceCounter((LARGE_INTEGER*)&t2);
	double xr = t2 - t1;
	double y = freq;
	double z = xr / y;
	printf("done with CUDA, time taken, freq: %lf, %lf\n", z, y);
	cudaFree(dev_data);
	cudaFree(tbl);

	return tbl;
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	unsigned long long t1, t2, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&t1);
	// Add vectors in parallel.
	auto x = histoCuda("");
	QueryPerformanceCounter((LARGE_INTEGER*)&t2);
	double xr = t2 - t1;
	double y = freq;
	double z = xr / y;
	printf("done, time taken, freq: %lf, %lf\n", z, y);
	while (1) continue;
	return 0;
}
