
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
	char arr[25];
	char padding[7];
};
struct HTEntry {
	union {
		KMer sto;
		struct {
			unsigned char padding[26];
			unsigned short dist;
			unsigned int count;
			unsigned long long h2;
			

		};
	};
};


#define LOG_SZ 24
#define TBL_SZ (1llu<<LOG_SZ)
#define LOG_BATCH_SZ 24
#define BATCH_SZ (1llu<<LOG_BATCH_SZ)

#define BIG_PRIME_1 50120398274937569llu 
#define BIG_PRIME_2 46774526020399987llu
#define BIG_PRIME_3 6731754961944698579llu 

struct hash128 {
	unsigned long long h1, h2;
};
__device__ __forceinline__ unsigned long long rotl64(unsigned long long x, unsigned long long r)
{
	return (x << r) | (x >> (64 - r));
}
#define ROTL64(x,y)	rotl64(x,y)
#define getblock(p, i) (p[i])
__device__ __forceinline__ unsigned long long fmix64(unsigned long long k)
{
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xff51afd7ed558ccd);
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
	k ^= k >> 33;

	return k;
}
__device__ __forceinline__ void MurmurHash3_x64_128(const void * key,
	const unsigned int seed, void * out)
{

	const unsigned char * data = (const unsigned char*)key;
	const int nblocks = 25 / 16;
	int i;

	unsigned long long h1 = seed;
	unsigned long long h2 = seed;

	unsigned long long c1 = (0x87c37b91114253d5llu);
	unsigned long long c2 = (0x4cf5ad432745937fllu);

	//----------
	// body

	const unsigned long long * blocks = (const unsigned long long *)(data);


	{
		unsigned long long k1 = blocks[0];
		unsigned long long k2 = blocks[1];

		k1 *= c1; k1 = ROTL64(k1, 31); k1 *= c2; h1 ^= k1;

		h1 = ROTL64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

		k2 *= c2; k2 = ROTL64(k2, 33); k2 *= c1; h2 ^= k2;

		h2 = ROTL64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
	}
	//----------
	// tail

	const unsigned char * tail = (const unsigned char*)(data + nblocks * 16);

	unsigned long long k1 = 0;
	unsigned long long k2 = 0;

	{
		k2 ^= (unsigned long long)(tail[9]) << 8;
		k2 ^= (unsigned long long)(tail[8]) << 0;
		k2 *= c2; k2 = ROTL64(k2, 33); k2 *= c1; h2 ^= k2;

		k1 ^= (unsigned long long)(tail[7]) << 56;
		k1 ^= (unsigned long long)(tail[6]) << 48;
		k1 ^= (unsigned long long)(tail[5]) << 40;
		k1 ^= (unsigned long long)(tail[4]) << 32;
		k1 ^= (unsigned long long)(tail[3]) << 24;
		k1 ^= (unsigned long long)(tail[2]) << 16;
		k1 ^= (unsigned long long)(tail[1]) << 8;
		k1 ^= (unsigned long long)(tail[0]) << 0;
		k1 *= c1; k1 = ROTL64(k1, 31); k1 *= c2; h1 ^= k1;
	};

	//----------
	// finalization

	h1 ^= 25; h2 ^= 25;

	h1 += h2;
	h2 += h1;

	h1 = fmix64(h1);
	h2 = fmix64(h2);

	h1 += h2;
	h2 += h1;

	((unsigned long long*)out)[0] = h1;
	((unsigned long long*)out)[1] = h2;
}

#define min(a,b) a<b?a:b
#define max(a,b) a>b?a:b
// generate random data
#define RATIO 2llu
KMer * generate_data() {
	auto my_data = (KMer *)malloc(sizeof(KMer) * BATCH_SZ);
	memset((void*)my_data, 0, sizeof(KMer) * BATCH_SZ);
	auto sz = BATCH_SZ / RATIO;
	for (unsigned long i = 0; i < sz* sizeof(KMer) / sizeof(unsigned long); i++) {
			((unsigned long *)((my_data)))[i] = i*i;
	}
	for (auto i = 1; i < RATIO;i++) {
		memcpy(&(my_data[0]), &(my_data[sz*i]), sz * sizeof(KMer));
	}
	return my_data;
}
__device__ __forceinline__ bool kmer_cmp(KMer * o1, KMer * o2) {
	for (auto  i = 0; i < 3; i++)if (((unsigned long long *)o1->arr)[i] != ((unsigned long long *)o2->arr)[i]) return false;
	return o1->arr[24]==o2->arr[24];
}
__global__ void histoKernel(KMer* ptr, HTEntry * tbl, unsigned long long * res) {
	KMer * ind = &(ptr[(threadIdx.x*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x)]);
	unsigned long long hashes[2];
	MurmurHash3_x64_128(ind->arr, BIG_PRIME_3, &hashes);
	auto index = hashes[0]>>(64-LOG_SZ);
	atomicAdd(res,1);
	auto h2 = hashes[1];
	auto orig = index;
	auto bval  = atomicCAS(&(tbl[index].count), 0ul, 1ul);
	do {
		while (!(bval == 0 || tbl[index].h2 == h2)) {

			index += 1;
			index %= TBL_SZ;
			if (index == orig) { return; }
			bval = atomicCAS(&(tbl[index].count), 0ul, 1ul);
		}
		
	} while (!bval&&!kmer_cmp(ind, &(tbl[index].sto)));
	if (bval==0) {  tbl[index].h2 = h2;  memcpy(tbl[index].sto.arr, ind, 25); }
	else { atomicAdd(res, -1); atomicAdd(&(tbl[index].count), 1); }
	
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
	my_data = generate_data();

	cudaMemcpy(dev_data, my_data, sizeof(KMer) * BATCH_SZ, cudaMemcpyHostToDevice);
	cudaStatus = cudaMalloc((void**)&tbl, sizeof(HTEntry)*TBL_SZ);
	cudaMemset(tbl, 0, sizeof(HTEntry) * TBL_SZ);
	unsigned long long * temp;
	cudaMalloc((void**)&temp, 8);
	cudaMemset(temp, 0, 8);
	cudaMemset(dev_data, 0, sizeof(KMer) * BATCH_SZ);
	if (cudaStatus != cudaSuccess) {

		printf("cm01 fail %s\n", cudaGetErrorString(cudaStatus));

	}
	//printf("addr: %x\n", dev_data);
	//printf("addr: %x\n", tbl);
	CUDA_CALL(cudaDeviceSynchronize());
	unsigned long long t1, t2, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&t1);
	histoKernel <<<griddims, 128 >>> (dev_data, tbl, temp);

	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	QueryPerformanceCounter((LARGE_INTEGER*)&t2);
	
	unsigned long long col;
	cudaMemcpy(&col, temp, 8, cudaMemcpyDeviceToHost);
	double xr = t2 - t1;
	double y = freq;
	double z = xr / y;
	printf("done with CUDA, time taken, freq, collisions: %lf, %lf, %llu\n", z, y,col);
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
