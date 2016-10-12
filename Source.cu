/*

InfiniTAM 7

simplifies MemoryBlock to use managed memory instead of explicit synchronization

All cpu functionality is not thread-safe.
Compiles with and without CUDA (make it a C++ file to compile as c++).

For vs13/15 and CUDA 7.0, sm50

*/


// Custom headers
#include <paul.h>




#if !defined(_WIN64) || _MSC_VER < 1800
#error Compile for windows x64, tested with visual studio 2013 and up. Cuda __managed__ memory requires 64 bits.
#endif


#ifdef __CUDACC__

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 500
#error Always use the latest cuda arch. Old versions dont support any amount of thread blocks being submitted at once.
#endif

// CUDA Kernel launch and error reporting framework

dim3 _lastLaunch_gridDim, _lastLaunch_blockDim;


// actual implementation, providing direct error reporting and 
// (for now forced) synchronization right after launch
#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, ...) {\
    cudaSafeCall(cudaGetLastError());\
    _lastLaunch_gridDim = dim3(gridDim); _lastLaunch_blockDim = dim3(blockDim);\
    auto t0 = clock(); \
    kernelFunction << <gridDim, blockDim >>>  (__VA_ARGS__);\
    cudaDeviceSynchronize();\
    printf("%s finished in %f s\n",#kernelFunction,(double)(clock()-t0)/CLOCKS_PER_SEC);\
    \
    cudaCheckLaunch(#kernelFunction "<<<" #gridDim ", " #blockDim ">>>(args omitted)" /*cannot get __VA_ARGS__ to stringify */ , __FILE__, __LINE__);\
    \
    cudaSafeCall(cudaDeviceSynchronize()); /* TODO synchronizing greatly alters the execution logic */\
    cudaSafeCall(cudaGetLastError());\
}






CPU_FUNCTION(void, reportCudaError, (cudaError err, const char * const expr, const char * const file, const int line), "") {

    const char* e = cudaGetErrorString(err);
    if (!e) e = "! cudaGetErrorString returned 0 !";

    char s[10000]; // TODO uses lots of stack - consider global - but this is not nested so... // but not using even more stack might help
    sprintf_s(s, "\n%s(%i) : cudaSafeCall(%s)\nRuntime API error : %s.\n",
        file,
        line,
        expr,
        e);
    puts(s);
    if (err == cudaError::cudaErrorLaunchFailure) {
        printf("NOTE maybe this error signifies an illegal memory access (memcpy(0,0,4) et.al)  or failed assertion, try the CUDA debugger\n\n"
            );
    }

    if (err == cudaError::cudaErrorInvalidConfiguration) {
        printf("configuration was (%d,%d,%d), (%d,%d,%d)\n",
            xyz(_lastLaunch_gridDim),
            xyz(_lastLaunch_blockDim)
            );
    }

    if (err == cudaError::cudaErrorIllegalInstruction) {
        puts("maybe the illegal instruction was asm(trap;) of a failed assertion?");
    }

}
CPU_FUNCTION(void, cudaCheckLaunch, (const char* const launchCommand, const char * const file, const int line), "") {
    auto err = cudaGetLastError();
    if (err == cudaSuccess) return;
    reportCudaError(err, launchCommand, file, line);
}

// cudaSafeCall wrapper


// Implementation detail:
// cudaSafeCall is an expression that evaluates to 
// 0 when err is cudaSuccess (0), such that cudaSafeCall(cudaSafeCall(cudaSuccess)) will not block
// this is important because we might have legacy code that explicitly does 
// cudaSafeCall(cudaDeviceSynchronize());
// but we extended cudaDeviceSynchronize to include this already, giving
// cudaSafeCall(cudaSafeCall(cudaDeviceSynchronize()))

// it debug-breaks and returns 
CPU_FUNCTION(bool, cudaSafeCallImpl, (cudaError err, const char * const expr, const char * const file, const int line), "");

// If err is cudaSuccess, cudaSafeCallImpl will return true, early-out of || will make DebugBreak not evaluated.
// The final expression will be 0.
// Otherwise we evaluate debug break, which returns true as well and then return 0.
#define cudaSafeCall(err) \
    !(cudaSafeCallImpl((cudaError)(err), #err, __FILE__, __LINE__) || ([]() {fatalError("CUDA error in cudaSafeCall"); return true;})() )



// Automatically wrap some common cuda functions in cudaSafeCall
#ifdef __CUDACC__ // hack to hide these from intellisense
#define cudaDeviceSynchronize(...) cudaSafeCall(cudaDeviceSynchronize(__VA_ARGS__))
#define cudaMalloc(...) cudaSafeCall(cudaMalloc(__VA_ARGS__))
#define cudaMemcpy(...) cudaSafeCall(cudaMemcpy(__VA_ARGS__))
#define cudaMemset(...) cudaSafeCall(cudaMemset(__VA_ARGS__))
#define cudaMemcpyAsync(...) cudaSafeCall(cudaMemcpyAsync(__VA_ARGS__))
#define cudaFree(...) cudaSafeCall(cudaFree(__VA_ARGS__))
#define cudaMallocManaged(...) cudaSafeCall(cudaMallocManaged(__VA_ARGS__))
#endif



/// \returns true if err is cudaSuccess
/// Fills errmsg in UNIT_TESTING build.
CPU_FUNCTION(bool, cudaSafeCallImpl, (cudaError err, const char * const expr, const char * const file, const int line), "")
{
    if (cudaSuccess == err) return true;

    cudaGetLastError(); // Reset error flag

    reportCudaError(err, expr, file, line);

    //flushStd();

    return false;
}


#endif


// HACK to make intellisense shut up about illegal C++ 
//#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, arguments, ...) ((void)0)



// Device agnostic atomics


FUNCTION(void, _atomicAdd,(_Inout_ unsigned int* target, const unsigned int x), "") {
#if GPU_CODE
    atomicAdd(target, x);
#else
    *target += x;
    // TODO this needs to change when CPU multithreading is to be used
#endif
}


// memcmp for device
FUNCTION(
    int, _memcmp, (void const * const _s1, void const * const _s2, size_t n),
    "like memcmp") {
#if GPU_CODE
    // http://opensource.apple.com//source/tcl/tcl-3.1/tcl/compat/memcmp.c
    auto s1 = (unsigned char const * const)_s1;
    auto s2 = (unsigned char const * const)_s2;

    unsigned char u1, u2;

    for (; n--; s1++, s2++) {
        u1 = *(unsigned char *)s1;
        u2 = *(unsigned char *)s2;
        if (u1 != u2) {
            return (u1 - u2);
        }
    }
    return 0;

#else
    return memcmp(_s1, _s2, n);
#endif
}



















FUNCTION(
	void,
	assert_restricted,
	(void const* const a, void const* const b),
	"") {
	assert(a != b, "Pointers where assumed to be different (__restrict'ed even for their lifetime), but where the same (initially). Undefined behaviour would result: %p %p", a, b);
}
























// Logging/debugging
/* dprintf:
a printing macro that can potentially be disabled completely, eliminating any performance impact.
Because of this, it's parameters shall never have side-effects.
*/

GLOBAL(
	int,
	dprintEnabled,
	false,
	"if true, dprintf writes to stdout, otherwise dprintf does nothing"
	"It would be more efficient to compile with dprintf defined to nothing of course"
	"Default: true"
	);

#ifdef __CUDA_ARCH__
#define dprintf(formatstr, ...) {if (dprintEnabled) printf("CUDA " formatstr, __VA_ARGS__);}
#else
#define dprintf(formatstr, ...) {if (dprintEnabled) printf(formatstr, __VA_ARGS__);}
#endif

// Purity: Has side effects, depends on environment.
FUNCTION(void,
	print,
	(_In_z_ const char* const x),
	"prints a string to stdout") {
	printf("print: %s\n", x);
}

// Purity: Has side effects, depends on environment.
FUNCTION(
	void,
	printd,
	(_In_reads_(n) const int* v, size_t n),
	"dprints a vector of integers, space separated and newline terminated"
	) {
	while (n--) dprintf("%d ", *v++); dprintf("\n");
}

// Purity: Has side effects, depends on environment.
FUNCTION(
	void,
	printd,
	(_In_reads_(n) const unsigned int* v, size_t n),
	"dprints a vector of integers, space separated and newline terminated"
	) {
	while (n--) dprintf("%u ", *v++); dprintf("\n");
}

/*
Implementation note:
size_t n is not const because the implementation modifies it

changes to n are not visible outside anyways:
-> the last const spec is always an implementation detail, not a promise to the caller, but can indicate conceptual thinking
*/
// Purity: Has side effects, depends on environment.
FUNCTION(
	void,
	printv,
	(_In_reads_(n) const float* v, size_t n),
	"dprints a vector of doubles, space separated and newline terminated"
	) {
	while (n--) dprintf("%f ", *v++); dprintf("\n");
}






// Declaration of ostream << CLASSNAME operator, for stringification of objects
#define OSTREAM(CLASSNAME) friend std::ostream& operator<<(std::ostream& os, const CLASSNAME& o)


































// Serialization infrastructure

template<typename T>
CPU_FUNCTION(void, binwrite, (ofstream& f, const T* const x), "") {
	auto p = f.tellp(); // DEBUG
	f.write((char*)x, sizeof(T));
	assert(sizeof(T) == f.tellp() - p);
}

template<typename T>
CPU_FUNCTION(T, binread, (ifstream& f), "") {
	T x;
	f.read((char*)&x, sizeof(T));
	return x;
}

template<typename T>
CPU_FUNCTION(void, binread, (ifstream& f, T* const x), "") {
	f.read((char*)x, sizeof(T));
}

#define SERIALIZE_VERSION(x) static const int serialize_version = x
#define SERIALIZE_WRITE_VERSION(file) bin(file, serialize_version)
#define SERIALIZE_READ_VERSION(file) {const int sv = bin<int>(file); \
assert(sv == serialize_version\
, "Serialized version in file, '%d', does not match expected version '%d'"\
, sv, serialize_version);}

template<typename T>
CPU_FUNCTION(void, bin, (ofstream& f, const T& x), "") {
	binwrite(f, &x);
}
template<typename T>
CPU_FUNCTION(void, bin, (ifstream& f, T& x), "") {
	binread(f, &x);
}
template<typename T>
CPU_FUNCTION(T, bin, (ifstream& f), "") {
	return binread<T>(f);
}

CPU_FUNCTION(ofstream, binopen_write, (string fn), "") {
	return ofstream(fn, ios::binary);
}

CPU_FUNCTION(ifstream, binopen_read, (string fn), "") {
	return ifstream(fn, ios::binary);
}



































#ifdef __CUDACC__

FUNCTION(unsigned int, toLinearId, (const dim3 dim, const uint3 id)
	, "3d to 1d coordinate conversion (think 3-digit mixed base number, where dim is the bases and id the digits)"
	"The digits in id may not be larger than dim", PURITY_PURE) {
	assert(id.x < dim.x);
	assert(id.y < dim.y);
	assert(id.z < dim.z); // actually, the highest digit (or all digits) could be allowed to be anything, but then the representation would not be unique
	return dim.x * dim.y * id.z + dim.x * id.y + id.x;
}

FUNCTION(unsigned int, toLinearId2D, (const dim3 dim, const uint3 id), "", PURITY_PURE) {
	assert(1 == dim.z);
	return toLinearId(dim, id);
}

CUDA_FUNCTION(unsigned int, linear_threadIdx, (), "", PURITY_ENVIRONMENT_DEPENDENT) {
	return toLinearId(blockDim, threadIdx);
}

CUDA_FUNCTION(unsigned int, linear_blockIdx, (), "", PURITY_ENVIRONMENT_DEPENDENT) {
	return toLinearId(gridDim, blockIdx);
}

// PCFunction[unsigned int, volume, {{dim3, d}}, "C110", Pure, d.x*d.y*d.z.w]
FUNCTION(unsigned int, volume, (dim3 d), "C110. Undefined where the product would overflow", PURITY_PURE) {
	return d.x*d.y*d.z;
}


// Universal thread identifier
FUNCTION(unsigned int, linear_global_threadId, (), "", PURITY_ENVIRONMENT_DEPENDENT) {
#if GPU_CODE
	return linear_blockIdx() * volume(blockDim) + linear_threadIdx();
#else
	return 0; // TODO support CPU multithreading (multi-blocking - one thread is really one block, there is no cross cooperation ideally to ensure lock-freeness and best performance)
#endif
}


FUNCTION(dim3, getGridSize, (dim3 taskSize, dim3 blockSize),
	"Given the desired blockSize (threads per block) and total amount of tasks, compute a sufficient grid size"
	"Note that some blocks will not be completely occupied. You need to add manual checks in the kernels"
	"Undefined for blockSize = 0 in some component and some large values where the used operations overflow."
	, PURITY_PURE)
{
	assert(0 != blockSize.x && 0 != blockSize.y && 0 != blockSize.z);
	return dim3(
		(taskSize.x + blockSize.x - 1) / blockSize.x,
		(taskSize.y + blockSize.y - 1) / blockSize.y,
		(taskSize.z + blockSize.z - 1) / blockSize.z);
}

#endif


























_Must_inspect_result_
FUNCTION(bool, approximatelyEqual, (float x, float y), "whether x and y differ by at most 1e-5f. Undefined for infinite values.", PURITY_PURE) {
	return abs(assertFinite(x) - assertFinite(y)) < 1e-5f;
}




FUNCTION(void, assertApproxEqual, (const float a, const float b, const unsigned int considered_initial_bits = 20, const float tooSmallThreshold = 0.0001f), "crude relative float equality test", PURITY_PURE) {
	if (abs(a) <= tooSmallThreshold && abs(b) <= tooSmallThreshold) return;

	assert(considered_initial_bits > 8 + 1); // should consider at least sign and full exponent // TODO exponent may be off by 1 and the values still very close...
	assert(considered_initial_bits <= 32);

	const unsigned int ai = *(unsigned int*)&a;
	const unsigned int bi = *(unsigned int*)&b;
	const auto ait = ai >> (32 - considered_initial_bits);
	const auto bit = bi >> (32 - considered_initial_bits);

	assert(ait == bit, "%f != %f, %x != %x, %x != %x, considering %d bits",
		a, b, ai, bi, ait, bit, considered_initial_bits
		);
}

FUNCTION(void, assertApproxEqual, (
	
	_In_reads_(m) float const* const a, 
	_In_reads_(m) float const* const b,
	const unsigned int m,
	const unsigned int considered_initial_bits = 20
	
	), "", PURITY_PURE) {
	DO(i,m) {
		assertApproxEqual(a[i], b[i], considered_initial_bits);
	}
}




























template<typename T>
FUNCTION(
	bool,
	assertEachInRange,
	(
		_In_reads_(len) const T* v,
		size_t len,
		const T min,
		const T max
		),
	"true if each element of v (of length len) is >= min, <= max, undefined otherwise"
	, PURITY_PURE) {
	assert(v);
	while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
		assert(min <= *v && *v <= max);
		v++;
	}
	return true;
}

template<typename T>
FUNCTION(
	bool,
	assertLessEqual,
	(
		_In_reads_(len) const T* v,
		_In_ size_t len, // not const for implementation
		const T max
		),
	"v <= max"
	, PURITY_PURE) {
	assert(v);
	while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
		assert(*v <= max);
		v++;
	}
	return true;
}

template<typename T>
FUNCTION(
	bool,
	assertLess,
	(
		_In_reads_(len) const T* v,
		size_t len,
		const T max
		),
	"v < max"
	, PURITY_PURE) {
	assert(v);
	while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
		assert(*v < max, "%d %d", *v, max);
		v++;
	}
	return true;
}

template<typename T>
CPU_FUNCTION(
	void,
	assertLess,
	(
		_In_ const vector<T>& v,
		const T max
		),
	"v < max"
	, PURITY_PURE) {
	assertLess(v.data(), v.size(), max);
}


























FUNCTION(
	void,

	axpyWithReindexing,

	(
		_Inout_updates_(targetLength) float* __restrict const targetBase,
		const unsigned int targetLength,
		float const a,

		_In_reads_(targetIndicesAndAddedValuesLength) const float* __restrict const addedValues,
		_In_reads_(targetIndicesAndAddedValuesLength) const unsigned int* __restrict const targetIndices,
		const unsigned int targetIndicesAndAddedValuesLength
		),
	"targetBase[[targetIndices]] += a * addedValues"
	""
	"Repeated indices are not supported, so addedValues cannot be longer than the target."
	"Note that not necessarily all of target is updated (_Inout_updates_, not _Inout_updates_all_)."
	"targetBase[0..targetLength-1], addedValues[0..targetIndicesAndAddedValuesLength-1] and targetIndices[0..targetIndicesAndAddedValuesLength-1] must all point to valid pairwise different (hence __restrict) memory locations which are left unchanged during the execution of this function"

	, PURITY_OUTPUT_POINTERS) {
	assert_restricted(targetBase, addedValues);
	assert_restricted(targetIndices, addedValues);
	assert_restricted(addedValues, targetBase);
	assert(targetLength); // targetLength - 1 overflows otherwise
	assertFinite(a);
	assert(targetIndicesAndAddedValuesLength <= targetLength);
	//dprintf("axpyWithReindexing %f %d %d\n", a, targetLength, targetIndicesAndAddedValuesLength);
	//
	//dprintf("target before:\n"); printv(targetBase, targetLength);
	//dprintf("targetIndices:\n"); printd(targetIndices, targetIndicesAndAddedValuesLength);
	//dprintf("addedValues:\n"); printv(addedValues, targetIndicesAndAddedValuesLength);

	assertLess(targetIndices, targetIndicesAndAddedValuesLength, targetLength);

	DO(j, targetIndicesAndAddedValuesLength)
		assertFinite(targetBase[targetIndices[j]] += addedValues[j] * a);

	//dprintf("target after:\n"); printv(targetBase, targetLength);
}

TEST(axpyWithReindexing1) {
	float target[] = { 0.f, 0.f };
	float addedValues[] = { 1.f, 2.f };
	unsigned int targetIndices[] = { 1, 0 };

	axpyWithReindexing(target, 2, 1.f, addedValues, targetIndices, 2);

	assert(target[0] == 2.f);
	assert(target[1] == 1.f);
}

FUNCTION(void, extract, (
	_Out_writes_all_(sourceIndicesAndTargetLength) float* __restrict const
	target,

	_In_reads_(sourceLength) const float* __restrict const
	source,
	const unsigned int
	sourceLength,

	_In_reads_(sourceIndicesAndTargetLength) const unsigned int* __restrict const
	sourceIndices,

	const unsigned int
	sourceIndicesAndTargetLength
	),
	"target = source[[sourceIndices]]. "
	""
	"Note that all of target is initialized (_Out_writes_all_)."
	"All sourceIndices must be < sourceLength"
	"target, source and sourceIndices must be different and nonoverlapping"
	, PURITY_OUTPUT_POINTERS) {
	assert_restricted(target, source);
	assert_restricted(target, sourceIndices);
	assert_restricted(source, sourceIndices);
	assertLess(sourceIndices, sourceIndicesAndTargetLength, sourceLength);

	DO(i, sourceIndicesAndTargetLength)
		target[i] = source[sourceIndices[i]];
}

TEST(extract1) {
	float target[2];
	float source[] = { 1.f, 2.f };
	unsigned int sourceIndices[] = { 1, 0 };

	extract(target, source, 2, sourceIndices, 2);

	assert(target[0] == 2.f);
	assert(target[1] == 1.f);
}





























/*
CSPARSE
A Concise Sparse Matrix Package in C

http://people.sc.fsu.edu/~jburkardt/c_src/csparse/csparse.html

CSparse Version 1.2.0 Copyright (c) Timothy A. Davis, 2006

reduced to only the things needed for sparse conjugate gradient method

by Paul Frischknecht, August 2016

and for running on CUDA, with a user-supplied memory-pool

modified & used without permission
*/


/* --- primary CSparse routines and data structures ------------------------- */
struct cs    /* matrix in compressed-column or triplet form . must be aligned on 8 bytes */
{
	unsigned int nzmax;	/* maximum number of entries allocated for triplet. Actual number of entries for compressed col. > 0 */
	unsigned int m;	    /* number of rows > 0 */

	unsigned int n;	    /* number of columns  > 0 */
	int nz;	    /* # of used entries (of x) in triplet matrix, NZ_COMPRESSED_COLUMN_INDICATOR for compressed-col, >= 0 otherwise --> must be signed TODO this will reduce the range that makes sense for nzmax.*/

				// Note: this order preserves 8-byte pointer (64 bit) alignment, DO NOT CHANGE
				// all pointers are always valid
	unsigned int *p;	    /* column pointers (size n+1) or col indices (size nzmax) */

	unsigned int *i;	    /* row indices, size nzmax */

	float *x;	/* finite numerical values, size nzmax. if cs_is_compressed_col, nzmax entries are used, if cs_is_triplet, nz many are used (use cs_x_used_entries()) */

};

FUNCTION(bool, cs_is_triplet, (const cs * const A), "whether A is a triplet matrix. Undefined if A does not point to a valid cs instance.", PURITY_PURE) {
	assert(A);
	return A->nz >= 0;
}

const int NZ_COMPRESSED_COLUMN_INDICATOR = -1;

FUNCTION(bool, cs_is_compressed_col, (const cs * const A), "whether A is a crompressed-column form matrix", PURITY_PURE) {
	assert(A);
	assert(A->m >= 1 && A->n >= 1);
	return A->nz == NZ_COMPRESSED_COLUMN_INDICATOR;
}


FUNCTION(unsigned int, cs_x_used_entries, (const cs * const A), "how many entries of the x vector are actually used", PURITY_PURE) {
	assert(cs_is_triplet(A) != /*xor*/ cs_is_compressed_col(A));
	return cs_is_triplet(A) ? A->nz : A->nzmax;
}


// hacky arbitrary memory-management by passing
// reduces memory_size and increases memoryPool on use
#define MEMPOOL _Inout_ char*& memoryPool, /*using int to detect over-deallocation */ _Inout_ int& memory_size // the function taking these modifies memoryPool to point to the remaining free memory
#define MEMPOOLPARAM memoryPool, memory_size 


FUNCTION(char*, cs_malloc_, (MEMPOOL, unsigned int sz /* use unsigned int?*/),
	"allocate new stuff. can only allocate multiples of 8 bytes to preserve alignment of pointers in cs. Use nextEven to round up when allocating 4 byte stuff (e.g. int)") {
	// assert(sz < INT_MAX) // TODO to be 100% correct, we'd have to check that memory_size + sz doesn't overflow
	assert((size_t)memory_size >= sz, "%d %d", memory_size, sz);
	assert(aligned(memoryPool, 8));
	assert(sz, "tried to allocate 0 bytes");
	assert(divisible(sz, 8));
	auto out = memoryPool;
	memoryPool += sz;
	memory_size -= (int)sz; // TODO possible loss of data...
	return out;
}

#define cs_malloc(varname, sz) {(varname) = (decltype(varname))cs_malloc_(MEMPOOLPARAM, (sz));}

FUNCTION(void, cs_free_, (char*& memoryPool, int& memory_size, unsigned int sz), "free the last allocated thing of given size") {
	// assert(sz < INT_MAX) // TODO to be 100% correct, we'd have to check that memory_size + sz doesn't overflow
	assert(divisible(sz, 8));
	assert(aligned(memoryPool, 8));
	memoryPool -= sz;
	memory_size += (int)sz;
	assert(memory_size >= 0);
}

#define cs_free(sz) {cs_free_(MEMPOOLPARAM, (sz));}


FUNCTION(unsigned int, cs_spalloc_size, (const unsigned int m, const unsigned int n, const unsigned int nzmax, bool triplet),
	"amount of bytes a sparse matrix with the given characteristics will occupy", PURITY_PURE) {
	DBG_UNREFERENCED_PARAMETER(m); // independent of these
	DBG_UNREFERENCED_PARAMETER(n);
	return sizeof(cs) + nextEven(triplet ? nzmax : n + 1) * sizeof(int) + nextEven(nzmax) *  (sizeof(int) + sizeof(float));
}


FUNCTION(cs *, cs_spalloc, (const unsigned int m, const unsigned int n, const unsigned int nzmax, bool triplet, MEMPOOL),
	"allocates a sparse matrix using memory starting at memoryPool,"
	"uses exactly"
	"sizeof(cs) + cs_spalloc_size(m, n, nzmax, triplet) BYTES"
	"of the pool"
	)
{
	assert(nzmax <= m * n); // cannot have more than a full matrix
	char* initial_memoryPool = memoryPool;
	assert(nzmax > 0);

	cs* A; cs_malloc(A, sizeof(cs));    /* allocate the cs struct */

	A->m = m;				    /* define dimensions and nzmax */
	A->n = n;
	A->nzmax = nzmax;
	A->nz = triplet ? 0 : NZ_COMPRESSED_COLUMN_INDICATOR;		    /* allocate triplet or comp.col */

																	// Allocate too much to preserve alignment
	cs_malloc(A->p, nextEven(triplet ? nzmax : n + 1) * sizeof(int));
	cs_malloc(A->i, nextEven(nzmax) * sizeof(int));
	cs_malloc(A->x, nextEven(nzmax) * sizeof(float));

	assert(memoryPool == initial_memoryPool + cs_spalloc_size(m, n, nzmax, triplet));
	return A;
}


FUNCTION(unsigned int, cs_cumsum, (
	_Inout_updates_all_(n + 1) unsigned int *p,
	_Inout_updates_all_(n) unsigned int *c,
	_In_ const unsigned int n
	),
	"p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c "
	, PURITY_OUTPUT_POINTERS)
{
	assert(p && c); /* check inputs */
	unsigned int i, nz = 0;
	for (i = 0; i < n; i++)
	{
		p[i] = nz;
		nz += c[i];
		c[i] = p[i];
	}
	p[n] = nz;
	return (nz);		    /* return sum (c [0..n-1]) */
}

FUNCTION(unsigned int*, allocZeroedIntegers, (_In_ const int n, MEMPOOL), "Allocate n integers set to 0. Implements calloc(n, sizeof(int)). n must be even") {
	assert(divisible(n, 2));
	unsigned int* w;
	cs_malloc(w, n * sizeof(unsigned int));
	memset(w, 0, n*sizeof(unsigned int)); // w = (int*)cs_calloc(n, sizeof(int)); /* get workspace */
	return w;
}

// alloc/free a list of integers w, initialized to 0
#define allocTemporaryW(count) unsigned int wsz = nextEven((count)); unsigned int* w = allocZeroedIntegers(wsz, MEMPOOLPARAM); 
#define freeTemporaryW() cs_free(wsz * sizeof(unsigned int)); 


FUNCTION(cs *, cs_transpose, (const cs * const A, MEMPOOL),
	"C = A'"
	""
	"memoryPool must be big enough to contain the following:"
	"cs_spalloc_size(n, m, Ap[n], 0) --location of output"
	"nextEven(m)*sizeof(int) --temporary")
{
	assert(A && cs_is_compressed_col(A));

	const unsigned int m = A->m;
	const unsigned int n = A->n;
	unsigned int const * const Ai = A->i;
	unsigned int const * const Ap = A->p;
	float const * const Ax = A->x;

	cs *C; C = cs_spalloc(n, m, Ap[n], 0, MEMPOOLPARAM); /* allocate result */

	allocTemporaryW(m); /* get workspace */

	unsigned int* const Cp = C->p; unsigned int* const Ci = C->i; float* const Cx = C->x;
	assert(Cp && Ci && Cx);

	for (unsigned int p = 0; p < Ap[n]; p++) w[Ai[p]]++;	   /* row counts */
	cs_cumsum(Cp, w, m);				   /* row pointers */
	for (unsigned int j = 0; j < n; j++)
	{
		for (unsigned int p = Ap[j]; p < Ap[j + 1]; p++)
		{
			int q;
			Ci[q = w[Ai[p]]++] = j;	/* place A(i,j) as entry C(j,i) */
			Cx[q] = Ax[p];
		}
	}

	freeTemporaryW();

	return C;	/* success; free w and return C */
}

FUNCTION(cs *, cs_triplet, (const cs * const T, MEMPOOL),
	"C = compressed-column form of a triplet matrix T"
	""
	"memoryPool must be big enough to contain the following"
	"cs_spalloc_size(m, n, nz, 0) --location of output"
	"nextEven(n)* sizeof(int) --temporary")
{
	assert(T && cs_is_triplet(T));/* check inputs */

	const int m = T->m;
	const int n = T->n;
	unsigned int const * const Ti = T->i;
	unsigned int const * const Tj = T->p;
	float const * const Tx = T->x;
	const auto nz = T->nz;

	assert(m > 0 && n > 0);
	cs *C; C = cs_spalloc(m, n, nz, 0, memoryPool, memory_size);		/* allocate result */

	allocTemporaryW(n); /* get workspace */

	unsigned int* const Cp = C->p; unsigned int* const Ci = C->i; float* const Cx = C->x;
	assert(Cp && Ci && Cx);

	for (int k = 0; k < nz; k++) w[Tj[k]]++;		/* column counts */
	cs_cumsum(Cp, w, n);				/* column pointers */
	for (int k = 0; k < nz; k++)
	{
		int p;
		Ci[p = w[Tj[k]]++] = Ti[k];    /* A(i,j) is the pth entry in C */
		Cx[p] = Tx[k];
	}

	freeTemporaryW();

	return C;	    /* success; free w and return C */
}

FUNCTION(void, cs_entry, (cs * const T, const unsigned int i, const unsigned int j, const float x),
	"add an entry to a triplet matrix; assertion failure if there's no space for this or the matrix has wrong dimensions")
{
	assert(cs_is_triplet(T));
	assert(i < T->m && j < T->n); // cannot enlarge matrix
	assert(T->nzmax < INT_MAX, "nzmax overflows int, cannot compare with nz");
	assert(T->nz < (int)T->nzmax); // cannot enlarge matrix
	assert(T->x);
	assertFinite(x);

	T->x[T->nz] = x;
	T->i[T->nz] = i;
	T->p[T->nz++] = j;
}

CPU_FUNCTION(void, cs_dump, (const cs * const A, FILE* f), "Similar to cs_print. dump a sparse matrix in the format:\n"
	"height width\n"
	"i j x\n"
	"etc. Note that this format does not allow reconstructing whether the matrix was in triplet or compressed column form,"
	"but the cc-form indices will increase more regularly than in the triplet case."
	, PURITY_ENVIRONMENT_DEPENDENT | PURITY_OUTPUT_POINTERS) {
	assert(f);

	unsigned int p, j, m, n, nzmax;
	unsigned int *Ap, *Ai;
	float *Ax;
	m = A->m; n = A->n; Ap = A->p; Ai = A->i; Ax = A->x;
	nzmax = A->nzmax;
	assert(Ax);

	fprintf(f, "%u %u\n", m, n);

	if (cs_is_compressed_col(A))
	{
		for (j = 0; j < n; j++)
		{
			for (p = Ap[j]; p < Ap[j + 1]; p++)
			{
				assert(Ai[p] < m);
				fprintf(f, "%u %u %g\n", Ai[p], j, Ax[p]);
			}
		}
	}
	else
	{
		auto nz = A->nz;
		assert(nz <= (int)nzmax);
		for (p = 0; p < (unsigned int)nz; p++)
		{
			assert(Ai[p] < m);
			assert(Ap[p] < n);
			fprintf(f, "%u %u %g\n", Ai[p], Ap[p], Ax[p]);
		}
	}
}

FUNCTION(int, cs_print, (const cs * const A, int brief = 0), "print a sparse matrix. Similar to cs_dump, but always writes to stdout.")
{
	assert(A);
	unsigned int p, j, m, n, nzmax;
	unsigned int *Ap, *Ai;
	float *Ax;

	m = A->m; n = A->n; Ap = A->p; Ai = A->i; Ax = A->x;
	nzmax = A->nzmax;

	printf("CSparse %s\n",
#ifdef __CUDA_ARCH__
		"on CUDA"
#else
		"on CPU"
#endif
		);
	assert(m > 0 && n > 0);
	if (cs_is_compressed_col(A))
	{
		printf("%d-by-%d, nzmax: %d nnz: %d\n", m, n, nzmax,
			Ap[n]);
		for (j = 0; j < n; j++)
		{
			printf("    col %d : locations %d to %d\n", j, Ap[j], Ap[j + 1] - 1);
			for (p = Ap[j]; p < Ap[j + 1]; p++)
			{
				assert(Ai[p] < m);
				printf("      %d : %g\n", Ai[p], Ax ? Ax[p] : 1);
				if (brief && p > 20) { printf("  ...\n"); return (1); }
			}
		}
	}
	else
	{
		auto nz = A->nz;
		printf("triplet: %d-by-%d, nzmax: %d nnz: %d\n", m, n, nzmax, nz);
		assert(nz <= (int)nzmax);
		for (p = 0; p < (unsigned int)nz; p++)
		{
			printf("    %d %d : %f\n", Ai[p], Ap[p], Ax ? Ax[p] : 1);
			assert(Ap[p] >= 0);
			if (brief && p > 20) { printf("  ...\n"); return (1); }
		}
	}
	return (1);
}


FUNCTION(int, cs_mv, (
	_Inout_ float * __restrict y,

	_In_ const float alpha,
	_In_ const cs * __restrict const  A,
	_In_ const float * __restrict const x,
	_In_ const float beta),
	"y = alpha A x + beta y"
	""
	"the memory for y, x and A cannot overlap"
	"TODO implement a version that can transpose A implicitly"
	"if beta == 0.f, y is never read, only written", PURITY_OUTPUT_POINTERS)
{
	assert(A && x && y);	    /* check inputs */
	assert(cs_is_compressed_col(A));
	assertFinite(beta);
	assertFinite(alpha);

	unsigned int p, j, n; unsigned int *Ap, *Ai;
	float *Ax;
	n = A->n;
	Ap = A->p; Ai = A->i; Ax = A->x;

	// the height of A is the height of y. Premultiply y with beta, then proceed as before, including the alpha factor when needed 
	// TODO (can we do better?)
	// Common special cases
	if (beta == 0)
		memset(y, 0, sizeof(float) * A->m);
	else
		for (unsigned int i = 0; i < A->m; i++) y[i] *= beta;

	if (alpha == 1)
		for (j = 0; j < n; j++) for (p = Ap[j]; p < Ap[j + 1]; p++) y[Ai[p]] += Ax[p] * x[j];
	else if (alpha != 0) // TODO instead of deciding this at runtime, let the developer choose the right function xD
		for (j = 0; j < n; j++) for (p = Ap[j]; p < Ap[j + 1]; p++) y[Ai[p]] += alpha * Ax[p] * x[j];
	// if alpha = 0, we are done

	return (1);
}
// ---


FUNCTION(
	void,
	printJ,
	(cs const * const J),
	"prints a sparse matrix"
	, PURITY_ENVIRONMENT_DEPENDENT) {
	if (dprintEnabled) cs_print(J);
}

// for conjgrad/sparse leastsquares:

// infrastructure like CusparseSolver



// A nonempty vector of finite floats
struct fvector {
	float* x;
	unsigned int n; // > 0

	MEMBERFUNCTION(void, print, (), "print this fvector") {
		printv(x, n);
	}

    MEMBERFUNCTION(float, operator[], (const unsigned int i) const, "member access") {
        assert(i < n);
        return x[i];
    }


    MEMBERFUNCTION(float&, operator[], (const unsigned int i), "member access") {
        assert(i < n);
        return x[i];
    }
};

CPU_FUNCTION(
	void
	, dump
	, (const fvector & v, FILE* f)
	,
	"Similar to fvector::print, cs_dump. dump a vector in the format:\n"
	"n\n"
	"x\n"
	"etc. "
	, PURITY_ENVIRONMENT_DEPENDENT | PURITY_OUTPUT_POINTERS) {
	assert(f);
	fprintf(f, "%u\n", v.n);
	FOREACHC(y, v.x, v.n)
		fprintf(f, "%g\n", y);
}

FUNCTION(bool, assertFinite, (_In_reads_(n) const float* const x, const unsigned int n), "assert that each element in v is finite", PURITY_PURE) {
	assert(n > 0);
	FOREACH(y, x, n)
		assertFinite(y);
	return true;
}

FUNCTION(bool, assertFinite, (const fvector& v), "assert that each element in v is finite", PURITY_PURE) {
	assertFinite(v.x, v.n);
	return true;
}

FUNCTION(fvector, vector_wrapper, (_Inout_ float* const x, const unsigned int n), "create a fvector object pointing to existing memory for convenient accessing", PURITY_OUTPUT_POINTERS) {
	assert(n > 0);
	fvector v;
	v.n = n;
	v.x = x;
	assertFinite(v);
	return v;
}

FUNCTION(fvector, vector_allocate, (const unsigned int n, MEMPOOL), "Create a new fvector. uninitialized: must be written before it is read!") {
	fvector v;
	v.n = n;
	cs_malloc(v.x, sizeof(float) * nextEven(v.n));
	return v;
}

FUNCTION(fvector, vector_allocate_malloc, (const unsigned int n), "Create a new fvector using malloc. uninitialized: must be written before it is read!") {
    fvector v;
    v.n = n;
    v.x = (float*)malloc(sizeof(float) * nextEven(v.n));
    return v;
}

FUNCTION(fvector, vector_copy, (const fvector& other, MEMPOOL), "create a copy of other") {
	fvector v;
	v.n = other.n;
	cs_malloc(v.x, sizeof(float) * nextEven(v.n));
	memcpy(v.x, other.x, sizeof(float) * v.n);
	assertFinite(v);
	return v;
}

struct matrix {
	const cs* const mat; // in compressed column form (transpose does not work with triplets)

	__declspec(property(get = getRows)) unsigned int rows;
	MEMBERFUNCTION(unsigned int, getRows, (), "m", PURITY_PURE) const {
		return mat->m;
	}
	__declspec(property(get = getCols)) unsigned int cols;
	MEMBERFUNCTION(unsigned int, getCols, (), "n", PURITY_PURE) const {
		return mat->n;
	}


	MEMBERFUNCTION(, matrix, (const cs* const mat), "construct a matrix wrapper", PURITY_OUTPUT_POINTERS) : mat(mat) {
		assert(!cs_is_triplet(mat)); assert(cs_is_compressed_col(mat));
		assert(mat->m && mat->n);
		assertFinite(mat->x, cs_x_used_entries(mat));
	}

	MEMBERFUNCTION(void, print, (), "print this matrix") {
		cs_print(mat, 0);
	}

	void operator=(matrix); // undefined
};


FUNCTION(float, dot, (const fvector& x, const fvector& y), "result = <x, y>, aka x.y or x^T y (the dot-product of x and y). Undefined if the addition overflows, finite otherwise.", PURITY_PURE) {
	assert(y.n == x.n);
	float r = 0;
	DO(i, x.n) r += x.x[i] * y.x[i];
	return assertFinite(r);
}

FUNCTION(void, axpy, (_Inout_ fvector& y, const float alpha, const fvector& x), "y = alpha * x + y", PURITY_OUTPUT_POINTERS) {
	assert(y.n == x.n);
	assert(assertFinite(alpha));
	DO(i, x.n) assertFinite(y.x[i] += alpha * x.x[i]);
}

FUNCTION(void, axpy, (_Inout_ fvector& y, const fvector& x), "y = x + y", PURITY_OUTPUT_POINTERS) {
	axpy(y, 1, x);
}

FUNCTION(void, scal, (_Inout_ fvector& x, const float alpha), "x *= alpha", PURITY_OUTPUT_POINTERS) {
	assert(assertFinite(alpha));
	DO(i, x.n) assertFinite(x.x[i] *= alpha);
}

FUNCTION(void, mv, (_Inout_ fvector& y, const float alpha, const matrix& A, const fvector& x, const float beta),
	"y = alpha A x + beta y", PURITY_OUTPUT_POINTERS) {
	assert(A.mat->m && A.mat->n);
	assert(y.n == A.mat->m);
	assert(x.n == A.mat->n);
	cs_mv(y.x, alpha, A.mat, x.x, beta);
}

FUNCTION(void, mv, (_Out_ fvector& y, const matrix& A, const fvector& x), "y = A x", PURITY_OUTPUT_POINTERS) {
	mv(y, 1, A, x, 0);
}

FUNCTION(matrix, transpose, (const matrix& A, MEMPOOL), "A^T" /* conceptually a pure function, but not because of explicit memory-management*/) {
	return matrix(cs_transpose(A.mat, MEMPOOLPARAM));
}

#define memoryPush() const auto old_memoryPool = memoryPool; const auto old_memory_size = memory_size; //savepoint: anything allocated after this can be freed again
#define memoryPop() {memoryPool = old_memoryPool; memory_size = old_memory_size;} // free anything allocated since memoryPush

// core algorithm, adapted from CusparseSolver (developed for Computational Geometry class), originally copied from Wikipedia
/* required operations:
- new fvector of given size
- copy/assign fvector
- mv_T, mv (matrix (transpose) times fvector) -- because I did not come up with a transposing-multiply operation, I just compute AT once instead of using mv_T
- scal (scaling)
- axpy // y = alpha * x + y
*/
//function [x] = conjgrad_normal(A,b,x)
/*The conjugate gradient method can be applied to an arbitrary n-by-m matrix by applying it to normal equations ATA and right-hand side fvector ATb, since ATA is a symmetric positive-semidefinite matrix for any A. The result is conjugate gradient on the normal equations (CGNR).
ATAx = ATb
As an iterative method, it is not necessary to form ATA explicitly in memory but only to perform the matrix-fvector and transpose matrix-fvector multiplications.

x is an n-fvector in this case still

x is used as the initial guess -- it may be 0 but must in any case contain valid numbers
*/
FUNCTION(void, conjgrad_normal, (
	const matrix& A,
	const fvector& b,
	fvector& x,
	MEMPOOL),
	"x = A\b" /* conceptually a pure function, but does memory allocation - however the memory is freed (but not reset) again */
	)
{
	memoryPush(); //savepoint: anything allocated after this can be freed again

	unsigned int m = A.rows, n = A.cols;

	matrix AT = transpose(A, MEMPOOLPARAM); // TODO implement an mv that does transposing in-place

	fvector t = vector_allocate(m, MEMPOOLPARAM);

	fvector r = vector_allocate(n, MEMPOOLPARAM); mv(r, AT, b); mv(t, A, x); mv(r, -1, AT, t, 1);//r=A^T*b; t = A*x; r = -A^T*t + r;//r=A^T*b-A^T*A*x;

	fvector p = vector_copy(r, MEMPOOLPARAM);//p=r;

	float rsold = dot(r, r);//rsold=r'*r;

	fvector Ap;
	if (sqrt(rsold) < 1e-5) goto end; // low residual: solution found

	Ap = vector_allocate(A.cols, MEMPOOLPARAM);

	REPEAT(b.n) {

		mv(t, A, p); mv(Ap, AT, t);//t = A*p;Ap=A^T*t;//Ap=A^T*A*p;

		if (abs(dot(p, Ap)) < 1e-9) { printf("conjgrad_normal emergency exit\n"); goto end; }// avoid almost division by 0
		float alpha = assertFinite(rsold / (dot(p, Ap)));//alpha=rsold/(p'*Ap);

		axpy(x, alpha, p);//x = alpha p + x;//x=x+alpha*p;
		axpy(r, -alpha, Ap);//r = -alpha*Ap + r;//r=r-alpha*Ap;
		float rsnew = dot(r, r);//rsnew=r'*r;
		if (sqrt(rsnew) < 1e-5) goto end; // error tolerance, might also limit amount of iterations or check change in rsnew to rsold...
		float beta = assertFinite(rsnew / rsold);
		scal(p, beta); axpy(p, r);//p*=(rsnew/rsold); p = r + p;//p=r+(rsnew/rsold)*p;
		rsold = rsnew;//rsold=rsnew;

	}

end:
	memoryPop(); // free anything allocated since memory push
}

// solving least-squares problems
FUNCTION(void, cs_cg, (

		const cs * const A, 
		_In_reads_(A->m) const float * const b, 
		_Inout_updates_all_(A->n) float *x, 
		MEMPOOL
	
	),
	"x=A\b"
	""
	"current value of x is used as initial guess"
	"Uses memory pool to allocate transposed copy of A and four vectors with size m or n, but frees all of these temporaries again"
	// PURITY_OUTPUT_POINTERS conceptually
	)
{
	const auto old_memory_size = memory_size;
	assert(A && b && x && memoryPool && memory_size > 0);

	auto xv = vector_wrapper(x, A->n);
	conjgrad_normal(matrix(A), vector_wrapper((float*)b, A->m), xv, MEMPOOLPARAM);

	assert(memory_size == old_memory_size);
}



TEST(test_cg) {
	char memory[1000];
	auto mem = memory;
	int msz = sizeof(memory);
	/*
	LeastSquares[{{1, 0}, {3, 4}} 1., {2., 1.}]
	{2., -1.25}
	*/
	cs* A = cs_spalloc(2,2,3,true, mem, msz);
	cs_entry(A, 0, 0, 1.f);
	// cs_entry(A, 0, 1, 0);
	cs_entry(A, 1, 0, 3.f);
	cs_entry(A, 1, 1, 4.f);

	float b[] = { 2.f, 1.f };

	float x[] = { 0.f, 0.f };

	auto B = cs_triplet(A, mem, msz);
	cs_cg(B, b, x, mem, msz);

	float sol[] = { 2.f, -1.25f };

	assertApproxEqual(x, sol, 2);
	return;
}

/*

CSPARSE library end

*/






























namespace vecmath {
    // Simple mathematical functions

    template<typename T>
    FUNCTION(T, ROUND, (T x),
        "prepare floating-point x for rounding-by-truncation, i.e. such that (int)ROUND(x) is the integer nearest to x"
        "TODO make this obsolete, use built-in rounding instead.", PURITY_PURE){
        return ((x < 0) ? (x - 0.5f) : (x + 0.5f));
    }


    template<typename T>
    FUNCTION(T, MAX, (T x, T y), "TODO use built-in max, make undefined for infinite values", PURITY_PURE) {
        return x > y ? x : y;
    }


    template<typename T>
    FUNCTION(T, MIN, (T x, T y), "C161", PURITY_PURE)  {
        return x < y ? x : y;
    }


    template<typename T>
    FUNCTION(T, CLAMP, (T x, T a, T b), "", PURITY_PURE) {
        return MAX((a), MIN((b), (x)));
    }



	//////////////////////////////////////////////////////////////////////////
	//						Basic Vector Structure
	//////////////////////////////////////////////////////////////////////////

	template <class T> struct Vector2_ {
		union {
			struct { T x, y; }; // standard names for components
			struct { T s, t; }; // standard names for components
			struct { T width, height; };
			T v[2];     // array access
		};
	};

	template <class T> struct Vector3_ {
		union {
			struct { T x, y, z; }; // standard names for components
			struct { T r, g, b; }; // standard names for components
			struct { T s, t, p; }; // standard names for components
			T v[3];
		};
	};

	template <class T> struct Vector4_ {
		union {
			struct { T x, y, z, w; }; // standard names for components
			struct { T r, g, b, a; }; // standard names for components
			struct { T s, t, p, q; }; // standard names for components
			T v[4];
		};
	};

	template <class T> struct Vector6_ {
		//union {
		T v[6];
		//};
	};

	template<class T, unsigned int s> struct VectorX_
	{
		int vsize;
		T v[s];
	};

	//////////////////////////////////////////////////////////////////////////
	// Vector class with math operators: +, -, *, /, +=, -=, /=, [], ==, !=, T*(), etc.
	//////////////////////////////////////////////////////////////////////////
	template <class T> class Vector2 : public Vector2_ < T >
	{
	public:
		typedef T value_type;
		CPU_AND_GPU inline int size() const { return 2; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////
		CPU_AND_GPU Vector2() {} // Default constructor
		CPU_AND_GPU Vector2(const T &t) { this->x = t; this->y = t; } // Scalar constructor
		CPU_AND_GPU Vector2(const T *tp) { this->x = tp[0]; this->y = tp[1]; } // Construct from array			            
		CPU_AND_GPU Vector2(const T v0, const T v1) { this->x = v0; this->y = v1; } // Construct from explicit values
		CPU_AND_GPU Vector2(const Vector2_<T> &v) { this->x = v.x; this->y = v.y; }// copy constructor

		CPU_AND_GPU explicit Vector2(const Vector3_<T> &u) { this->x = u.x; this->y = u.y; }
		CPU_AND_GPU explicit Vector2(const Vector4_<T> &u) { this->x = u.x; this->y = u.y; }

		CPU_AND_GPU inline Vector2<int> toInt() const {
			return Vector2<int>((int)ROUND(this->x), (int)ROUND(this->y));
		}

		CPU_AND_GPU inline Vector2<int> toIntFloor() const {
			return Vector2<int>((int)floor(this->x), (int)floor(this->y));
		}

		CPU_AND_GPU inline Vector2<unsigned char> toUChar() const {
			Vector2<int> vi = toInt(); return Vector2<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255));
		}

		CPU_AND_GPU inline Vector2<float> toFloat() const {
			return Vector2<float>((float)this->x, (float)this->y);
		}

		CPU_AND_GPU const T *getValues() const { return this->v; }
		CPU_AND_GPU Vector2<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; return *this; }

		CPU_AND_GPU T area() const {
			return width * height;
		}
		// indexing operators
		CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
		CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		CPU_AND_GPU operator T *() { return this->v; }
		CPU_AND_GPU operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		CPU_AND_GPU friend Vector2<T> &operator *= (const Vector2<T> &lhs, T d) {
			lhs.x *= d; lhs.y *= d; return lhs;
		}

		// component-wise vector multiply assign
		CPU_AND_GPU friend Vector2<T> &operator *= (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs;
		}

		// scalar divide assign
		CPU_AND_GPU friend Vector2<T> &operator /= (Vector2<T> &lhs, T d) {
			if (d == 0) return lhs; lhs.x /= d; lhs.y /= d; return lhs;
		}

		// component-wise vector divide assign
		CPU_AND_GPU friend Vector2<T> &operator /= (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x /= rhs.x; lhs.y /= rhs.y;	return lhs;
		}

		// component-wise vector add assign
		CPU_AND_GPU friend Vector2<T> &operator += (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x += rhs.x; lhs.y += rhs.y;	return lhs;
		}

		// component-wise vector subtract assign
		CPU_AND_GPU friend Vector2<T> &operator -= (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x -= rhs.x; lhs.y -= rhs.y;	return lhs;
		}

		// unary negate
		CPU_AND_GPU friend Vector2<T> operator - (const Vector2<T> &rhs) {
			Vector2<T> rv;	rv.x = -rhs.x; rv.y = -rhs.y; return rv;
		}

		// vector add
		CPU_AND_GPU friend Vector2<T> operator + (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		CPU_AND_GPU friend Vector2<T> operator - (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector2<T> operator * (const Vector2<T> &lhs, T rhs) {
			Vector2<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector2<T> operator * (T lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend Vector2<T> operator * (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector2<T> operator / (const Vector2<T> &lhs, T rhs) {
			Vector2<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend Vector2<T> operator / (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		CPU_AND_GPU friend bool operator == (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			return (lhs.x == rhs.x) && (lhs.y == rhs.y);
		}

		// inequality
		CPU_AND_GPU friend bool operator != (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			return (lhs.x != rhs.x) || (lhs.y != rhs.y);
		}

		OSTREAM(Vector2<T>) {
			os << o.x << ", " << o.y;
			return os;
		}
	};

	template <class T> class Vector3 : public Vector3_ < T >
	{
	public:
		typedef T value_type;
		CPU_AND_GPU inline int size() const { return 3; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////
		CPU_AND_GPU Vector3() {} // Default constructor
		CPU_AND_GPU Vector3(const T &t) { this->x = t; this->y = t; this->z = t; } // Scalar constructor
		CPU_AND_GPU Vector3(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; } // Construct from array
		CPU_AND_GPU Vector3(const T v0, const T v1, const T v2) { this->x = v0; this->y = v1; this->z = v2; } // Construct from explicit values
		CPU_AND_GPU explicit Vector3(const Vector4_<T> &u) { this->x = u.x; this->y = u.y; this->z = u.z; }
        template<typename T2>
        CPU_AND_GPU explicit Vector3(const Vector3_<T2> &u) { this->x = u.x; this->y = u.y; this->z = u.z; }
		CPU_AND_GPU explicit Vector3(const Vector2_<T> &u, T v0) { this->x = u.x; this->y = u.y; this->z = v0; }

		CPU_AND_GPU inline Vector3<int> toIntRound() const {
			return Vector3<int>((int)ROUND(this->x), (int)ROUND(this->y), (int)ROUND(this->z));
		}

		CPU_AND_GPU inline Vector3<int> toInt() const {
			return Vector3<int>((int)(this->x), (int)(this->y), (int)(this->z));
		}

		CPU_AND_GPU inline Vector3<int> toInt(Vector3<float> &residual) const {
			Vector3<int> intRound = toInt();
			residual = Vector3<float>(this->x - intRound.x, this->y - intRound.y, this->z - intRound.z);
			return intRound;
		}

		CPU_AND_GPU inline Vector3<short> toShortRound() const {
			return Vector3<short>((short)ROUND(this->x), (short)ROUND(this->y), (short)ROUND(this->z));
		}

		CPU_AND_GPU inline Vector3<short> toShortFloor() const {
			return Vector3<short>((short)floor(this->x), (short)floor(this->y), (short)floor(this->z));
		}

		CPU_AND_GPU inline Vector3<int> toIntFloor() const {
			return Vector3<int>((int)floor(this->x), (int)floor(this->y), (int)floor(this->z));
		}

		/// Floors the coordinates to integer values, returns this and the residual float.
		/// Use like
		/// TO_INT_FLOOR3(int_xyz, residual_xyz, xyz)
		/// for xyz === this
		CPU_AND_GPU inline Vector3<int> toIntFloor(Vector3<float> &residual) const {
			Vector3<float> intFloor(floor(this->x), floor(this->y), floor(this->z));
			residual = *this - intFloor;
			return Vector3<int>((int)intFloor.x, (int)intFloor.y, (int)intFloor.z);
		}

		CPU_AND_GPU inline Vector3<unsigned char> toUChar() const {
			Vector3<int> vi = toIntRound(); return Vector3<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255), (unsigned char)CLAMP(vi.z, 0, 255));
		}

		CPU_AND_GPU inline Vector3<float> toFloat() const {
			return Vector3<float>((float)this->x, (float)this->y, (float)this->z);
		}

		CPU_AND_GPU inline Vector3<float> normalised() const {
			float norm = 1.0f / sqrt((float)(this->x * this->x + this->y * this->y + this->z * this->z));
			return Vector3<float>((float)this->x * norm, (float)this->y * norm, (float)this->z * norm);
		}

		CPU_AND_GPU const T *getValues() const { return this->v; }
		CPU_AND_GPU Vector3<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; return *this; }

		// indexing operators
		CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
		CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		CPU_AND_GPU operator T *() { return this->v; }
		CPU_AND_GPU operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		CPU_AND_GPU friend Vector3<T> &operator *= (Vector3<T> &lhs, T d) {
			lhs.x *= d; lhs.y *= d; lhs.z *= d; return lhs;
		}

		// component-wise vector multiply assign
		CPU_AND_GPU friend Vector3<T> &operator *= (Vector3<T> &lhs, const Vector3<T> &rhs) {
			lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs;
		}

		// scalar divide assign
		CPU_AND_GPU friend Vector3<T> &operator /= (Vector3<T> &lhs, T d) {
			lhs.x /= d; lhs.y /= d; lhs.z /= d; return lhs;
		}

		// component-wise vector divide assign
		CPU_AND_GPU friend Vector3<T> &operator /= (Vector3<T> &lhs, const Vector3<T> &rhs) {
			lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs;
		}

		// component-wise vector add assign
		CPU_AND_GPU friend Vector3<T> &operator += (Vector3<T> &lhs, const Vector3<T> &rhs) {
			lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs;
		}

		// component-wise vector subtract assign
		CPU_AND_GPU friend Vector3<T> &operator -= (Vector3<T> &lhs, const Vector3<T> &rhs) {
			lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs;
		}

		// unary negate
		CPU_AND_GPU friend Vector3<T> operator - (const Vector3<T> &rhs) {
			Vector3<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; return rv;
		}

		// vector add
		CPU_AND_GPU friend Vector3<T> operator + (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		CPU_AND_GPU friend Vector3<T> operator - (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector3<T> operator * (const Vector3<T> &lhs, T rhs) {
			Vector3<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector3<T> operator * (T lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend Vector3<T> operator * (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector3<T> operator / (const Vector3<T> &lhs, T rhs) {
			Vector3<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend Vector3<T> operator / (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// inequality
		CPU_AND_GPU friend bool operator != (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z);
		}

		////////////////////////////////////////////////////////////////////////////////
		// dimension specific operations
		////////////////////////////////////////////////////////////////////////////////

		OSTREAM(Vector3<T>) {
			os << o.x << ", " << o.y << ", " << o.z;
			return os;
		}
	};

	////////////////////////////////////////////////////////
	//  Non-member comparison operators
	////////////////////////////////////////////////////////

	// equality
	template <typename T1, typename T2> CPU_AND_GPU inline bool operator == (const Vector3<T1> &lhs, const Vector3<T2> &rhs) {
		return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
	}

	template <class T> class Vector4 : public Vector4_ < T >
	{
	public:
		typedef T value_type;
		CPU_AND_GPU inline int size() const { return 4; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////

		CPU_AND_GPU Vector4() {} // Default constructor
		CPU_AND_GPU Vector4(const T &t) { this->x = t; this->y = t; this->z = t; this->w = t; } //Scalar constructor
		CPU_AND_GPU Vector4(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; this->w = tp[3]; } // Construct from array
		CPU_AND_GPU Vector4(const T v0, const T v1, const T v2, const T v3) { this->x = v0; this->y = v1; this->z = v2; this->w = v3; } // Construct from explicit values
		CPU_AND_GPU explicit Vector4(const Vector3_<T> &u, T v0) { this->x = u.x; this->y = u.y; this->z = u.z; this->w = v0; }
		CPU_AND_GPU explicit Vector4(const Vector2_<T> &u, T v0, T v1) { this->x = u.x; this->y = u.y; this->z = v0; this->w = v1; }

		CPU_AND_GPU inline Vector4<int> toIntRound() const {
			return Vector4<int>((int)ROUND(this->x), (int)ROUND(this->y), (int)ROUND(this->z), (int)ROUND(this->w));
		}

		CPU_AND_GPU inline Vector4<unsigned char> toUChar() const {
			Vector4<int> vi = toIntRound(); return Vector4<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255), (unsigned char)CLAMP(vi.z, 0, 255), (unsigned char)CLAMP(vi.w, 0, 255));
		}

		CPU_AND_GPU inline Vector4<float> toFloat() const {
			return Vector4<float>((float)this->x, (float)this->y, (float)this->z, (float)this->w);
		}

		CPU_AND_GPU inline Vector4<T> homogeneousCoordinatesNormalize() const {
			return (this->w <= 0) ? *this : Vector4<T>(this->x / this->w, this->y / this->w, this->z / this->w, 1);
		}

		CPU_AND_GPU inline Vector3<T> toVector3() const {
			return Vector3<T>(this->x, this->y, this->z);
		}

		CPU_AND_GPU const T *getValues() const { return this->v; }
		CPU_AND_GPU Vector4<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; this->w = rhs[3]; return *this; }

		// indexing operators
		CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
		CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		CPU_AND_GPU operator T *() { return this->v; }
		CPU_AND_GPU operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		CPU_AND_GPU friend Vector4<T> &operator *= (Vector4<T> &lhs, T d) {
			lhs.x *= d; lhs.y *= d; lhs.z *= d; lhs.w *= d; return lhs;
		}

		// component-wise vector multiply assign
		CPU_AND_GPU friend Vector4<T> &operator *= (Vector4<T> &lhs, const Vector4<T> &rhs) {
			lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; lhs.w *= rhs.w; return lhs;
		}

		// scalar divide assign
		CPU_AND_GPU friend Vector4<T> &operator /= (Vector4<T> &lhs, T d) {
			lhs.x /= d; lhs.y /= d; lhs.z /= d; lhs.w /= d; return lhs;
		}

		// component-wise vector divide assign
		CPU_AND_GPU friend Vector4<T> &operator /= (Vector4<T> &lhs, const Vector4<T> &rhs) {
			lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs;
		}

		// component-wise vector add assign
		CPU_AND_GPU friend Vector4<T> &operator += (Vector4<T> &lhs, const Vector4<T> &rhs) {
			lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs;
		}

		// component-wise vector subtract assign
		CPU_AND_GPU friend Vector4<T> &operator -= (Vector4<T> &lhs, const Vector4<T> &rhs) {
			lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs;
		}

		// unary negate
		CPU_AND_GPU friend Vector4<T> operator - (const Vector4<T> &rhs) {
			Vector4<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; rv.w = -rhs.w; return rv;
		}

		// vector add
		CPU_AND_GPU friend Vector4<T> operator + (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		CPU_AND_GPU friend Vector4<T> operator - (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector4<T> operator * (const Vector4<T> &lhs, T rhs) {
			Vector4<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector4<T> operator * (T lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend Vector4<T> operator * (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv *= rhs;
		}

		// scalar divide
		CPU_AND_GPU friend Vector4<T> operator / (const Vector4<T> &lhs, T rhs) {
			Vector4<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise divide
		CPU_AND_GPU friend Vector4<T> operator / (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		CPU_AND_GPU friend bool operator == (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.w == rhs.w);
		}

		// inequality
		CPU_AND_GPU friend bool operator != (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z) || (lhs.w != rhs.w);
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector4<T>& dt) {
			os << dt.x << ", " << dt.y << ", " << dt.z << ", " << dt.w;
			return os;
		}
	};

	template <class T> class Vector6 : public Vector6_ < T >
	{
	public:
		typedef T value_type;
		CPU_AND_GPU inline int size() const { return 6; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////

		CPU_AND_GPU Vector6() {} // Default constructor
		CPU_AND_GPU Vector6(const T &t) { this->v[0] = t; this->v[1] = t; this->v[2] = t; this->v[3] = t; this->v[4] = t; this->v[5] = t; } //Scalar constructor
		CPU_AND_GPU Vector6(const T *tp) { this->v[0] = tp[0]; this->v[1] = tp[1]; this->v[2] = tp[2]; this->v[3] = tp[3]; this->v[4] = tp[4]; this->v[5] = tp[5]; } // Construct from array
		CPU_AND_GPU Vector6(const T v0, const T v1, const T v2, const T v3, const T v4, const T v5) { this->v[0] = v0; this->v[1] = v1; this->v[2] = v2; this->v[3] = v3; this->v[4] = v4; this->v[5] = v5; } // Construct from explicit values
		CPU_AND_GPU explicit Vector6(const Vector4_<T> &u, T v0, T v1) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = u.z; this->v[3] = u.w; this->v[4] = v0; this->v[5] = v1; }
		CPU_AND_GPU explicit Vector6(const Vector3_<T> &u, T v0, T v1, T v2) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = u.z; this->v[3] = v0; this->v[4] = v1; this->v[5] = v2; }
		CPU_AND_GPU explicit Vector6(const Vector2_<T> &u, T v0, T v1, T v2, T v3) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = v0; this->v[3] = v1; this->v[4] = v2, this->v[5] = v3; }

		CPU_AND_GPU inline Vector6<int> toIntRound() const {
			return Vector6<int>((int)ROUND(this[0]), (int)ROUND(this[1]), (int)ROUND(this[2]), (int)ROUND(this[3]), (int)ROUND(this[4]), (int)ROUND(this[5]));
		}

		CPU_AND_GPU inline Vector6<unsigned char> toUChar() const {
			Vector6<int> vi = toIntRound(); return Vector6<unsigned char>((unsigned char)CLAMP(vi[0], 0, 255), (unsigned char)CLAMP(vi[1], 0, 255), (unsigned char)CLAMP(vi[2], 0, 255), (unsigned char)CLAMP(vi[3], 0, 255), (unsigned char)CLAMP(vi[4], 0, 255), (unsigned char)CLAMP(vi[5], 0, 255));
		}

		CPU_AND_GPU inline Vector6<float> toFloat() const {
			return Vector6<float>((float)this[0], (float)this[1], (float)this[2], (float)this[3], (float)this[4], (float)this[5]);
		}

		CPU_AND_GPU const T *getValues() const { return this->v; }
		CPU_AND_GPU Vector6<T> &setValues(const T *rhs) { this[0] = rhs[0]; this[1] = rhs[1]; this[2] = rhs[2]; this[3] = rhs[3]; this[4] = rhs[4]; this[5] = rhs[5]; return *this; }

		// indexing operators
		CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
		CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		CPU_AND_GPU operator T *() { return this->v; }
		CPU_AND_GPU operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		CPU_AND_GPU friend Vector6<T> &operator *= (Vector6<T> &lhs, T d) {
			lhs[0] *= d; lhs[1] *= d; lhs[2] *= d; lhs[3] *= d; lhs[4] *= d; lhs[5] *= d; return lhs;
		}

		// component-wise vector multiply assign
		CPU_AND_GPU friend Vector6<T> &operator *= (Vector6<T> &lhs, const Vector6<T> &rhs) {
			lhs[0] *= rhs[0]; lhs[1] *= rhs[1]; lhs[2] *= rhs[2]; lhs[3] *= rhs[3]; lhs[4] *= rhs[4]; lhs[5] *= rhs[5]; return lhs;
		}

		// scalar divide assign
		CPU_AND_GPU friend Vector6<T> &operator /= (Vector6<T> &lhs, T d) {
			lhs[0] /= d; lhs[1] /= d; lhs[2] /= d; lhs[3] /= d; lhs[4] /= d; lhs[5] /= d; return lhs;
		}

		// component-wise vector divide assign
		CPU_AND_GPU friend Vector6<T> &operator /= (Vector6<T> &lhs, const Vector6<T> &rhs) {
			lhs[0] /= rhs[0]; lhs[1] /= rhs[1]; lhs[2] /= rhs[2]; lhs[3] /= rhs[3]; lhs[4] /= rhs[4]; lhs[5] /= rhs[5]; return lhs;
		}

		// component-wise vector add assign
		CPU_AND_GPU friend Vector6<T> &operator += (Vector6<T> &lhs, const Vector6<T> &rhs) {
			lhs[0] += rhs[0]; lhs[1] += rhs[1]; lhs[2] += rhs[2]; lhs[3] += rhs[3]; lhs[4] += rhs[4]; lhs[5] += rhs[5]; return lhs;
		}

		// component-wise vector subtract assign
		CPU_AND_GPU friend Vector6<T> &operator -= (Vector6<T> &lhs, const Vector6<T> &rhs) {
			lhs[0] -= rhs[0]; lhs[1] -= rhs[1]; lhs[2] -= rhs[2]; lhs[3] -= rhs[3]; lhs[4] -= rhs[4]; lhs[5] -= rhs[5];  return lhs;
		}

		// unary negate
		CPU_AND_GPU friend Vector6<T> operator - (const Vector6<T> &rhs) {
			Vector6<T> rv; rv[0] = -rhs[0]; rv[1] = -rhs[1]; rv[2] = -rhs[2]; rv[3] = -rhs[3]; rv[4] = -rhs[4]; rv[5] = -rhs[5];  return rv;
		}

		// vector add
		CPU_AND_GPU friend Vector6<T> operator + (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		CPU_AND_GPU friend Vector6<T> operator - (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector6<T> operator * (const Vector6<T> &lhs, T rhs) {
			Vector6<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend Vector6<T> operator * (T lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend Vector6<T> operator * (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv *= rhs;
		}

		// scalar divide
		CPU_AND_GPU friend Vector6<T> operator / (const Vector6<T> &lhs, T rhs) {
			Vector6<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise divide
		CPU_AND_GPU friend Vector6<T> operator / (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		CPU_AND_GPU friend bool operator == (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]) && (lhs[3] == rhs[3]) && (lhs[4] == rhs[4]) && (lhs[5] == rhs[5]);
		}

		// inequality
		CPU_AND_GPU friend bool operator != (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			return (lhs[0] != rhs[0]) || (lhs[1] != rhs[1]) || (lhs[2] != rhs[2]) || (lhs[3] != rhs[3]) || (lhs[4] != rhs[4]) || (lhs[5] != rhs[5]);
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector6<T>& dt) {
			os << dt[0] << ", " << dt[1] << ", " << dt[2] << ", " << dt[3] << ", " << dt[4] << ", " << dt[5];
			return os;
		}
	};

	/*
	s - dimensional vector over the field T
	T must provide operators for all the operations used.
	*/
	template <class T, unsigned int s> class VectorX : public VectorX_ < T, s >
	{
	public:
		typedef T value_type;
		CPU_AND_GPU inline int size() const { return this->vsize; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////

		CPU_AND_GPU VectorX() { this->vsize = s; } // Default constructor
		CPU_AND_GPU VectorX(const T &t) { for (int i = 0; i < s; i++) this->v[i] = t; } //Scalar constructor
		CPU_AND_GPU VectorX(const T tp[s]) { for (int i = 0; i < s; i++) this->v[i] = tp[i]; } // Construct from array
		VectorX(std::array<T, s> t) : VectorX(t.data()) { } // Construct from array


		CPU_AND_GPU static inline VectorX<T, s> make_zeros() {
			VectorX<T, s> x;
			x.setZeros();
			return x;
		}

		// indexing operators
		CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
		CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }


		CPU_AND_GPU inline VectorX<int, s> toIntRound() const {
			VectorX<int, s> retv;
			for (int i = 0; i < s; i++) retv[i] = (int)ROUND(this->v[i]);
			return retv;
		}

		CPU_AND_GPU inline VectorX<unsigned char, s> toUChar() const {
			VectorX<int, s> vi = toIntRound();
			VectorX<unsigned char, s> retv;
			for (int i = 0; i < s; i++) retv[i] = (unsigned char)CLAMP(vi[0], 0, 255);
			return retv;
		}

		CPU_AND_GPU inline VectorX<float, s> toFloat() const {
			VectorX<float, s> retv;
			for (int i = 0; i < s; i++) retv[i] = (float) this->v[i];
			return retv;
		}

		CPU_AND_GPU const T *getValues() const { return this->v; }
		CPU_AND_GPU VectorX<T, s> &setValues(const T *rhs) { for (int i = 0; i < s; i++) this->v[i] = rhs[i]; return *this; }
		CPU_AND_GPU void Clear(T v) {
			for (int i = 0; i < s; i++)
				this->v[i] = v;
		}

		CPU_AND_GPU void setZeros() {
			Clear(0);
		}

		// type-cast operators
		CPU_AND_GPU operator T *() { return this->v; }
		CPU_AND_GPU operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		CPU_AND_GPU friend VectorX<T, s> &operator *= (VectorX<T, s> &lhs, T d) {
			for (int i = 0; i < s; i++) lhs[i] *= d; return lhs;
		}

		// component-wise vector multiply assign
		CPU_AND_GPU friend VectorX<T, s> &operator *= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) lhs[i] *= rhs[i]; return lhs;
		}

		// scalar divide assign
		CPU_AND_GPU friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, T d) {
			for (int i = 0; i < s; i++) lhs[i] /= d; return lhs;
		}

		// component-wise vector divide assign
		CPU_AND_GPU friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) lhs[i] /= rhs[i]; return lhs;
		}

		// component-wise vector add assign
		CPU_AND_GPU friend VectorX<T, s> &operator += (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) lhs[i] += rhs[i]; return lhs;
		}

		// component-wise vector subtract assign
		CPU_AND_GPU friend VectorX<T, s> &operator -= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) lhs[i] -= rhs[i]; return lhs;
		}

		// unary negate
		CPU_AND_GPU friend VectorX<T, s> operator - (const VectorX<T, s> &rhs) {
			VectorX<T, s> rv; for (int i = 0; i < s; i++) rv[i] = -rhs[i]; return rv;
		}

		// vector add
		CPU_AND_GPU friend VectorX<T, s> operator + (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv += rhs;
		}

		// vector subtract
		CPU_AND_GPU friend VectorX<T, s> operator - (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend VectorX<T, s> operator * (const VectorX<T, s> &lhs, T rhs) {
			VectorX<T, s> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		CPU_AND_GPU friend VectorX<T, s> operator * (T lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		CPU_AND_GPU friend VectorX<T, s> operator * (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv *= rhs;
		}

		// scalar divide
		CPU_AND_GPU friend VectorX<T, s> operator / (const VectorX<T, s> &lhs, T rhs) {
			VectorX<T, s> rv(lhs); return rv /= rhs;
		}

		// vector component-wise divide
		CPU_AND_GPU friend VectorX<T, s> operator / (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		CPU_AND_GPU friend bool operator == (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) if (lhs[i] != rhs[i]) return false;
			return true;
		}

		// inequality
		CPU_AND_GPU friend bool operator != (const VectorX<T, s> &lhs, const Vector6<T> &rhs) {
			for (int i = 0; i < s; i++) if (lhs[i] != rhs[i]) return true;
			return false;
		}

		friend std::ostream& operator<<(std::ostream& os, const VectorX<T, s>& dt) {
			for (int i = 0; i < s; i++) os << dt[i] << "\n";
			return os;
		}
	};

	////////////////////////////////////////////////////////////////////////////////
	// Generic vector operations
	////////////////////////////////////////////////////////////////////////////////

	template< class T> CPU_AND_GPU inline T sqr(const T &v) { return v*v; }

	// compute the dot product of two vectors
	template<class T> CPU_AND_GPU inline typename T::value_type dot(const T &lhs, const T &rhs) {
		typename T::value_type r = 0;
		for (int i = 0; i < lhs.size(); i++)
			r += lhs[i] * rhs[i];
		return r;
	}

	// return the squared length of the provided vector, i.e. the dot product with itself
	template< class T> CPU_AND_GPU inline typename T::value_type length2(const T &vec) {
		return dot(vec, vec);
	}

	// return the length of the provided vector
	template< class T> CPU_AND_GPU inline typename T::value_type length(const T &vec) {
		return sqrt(length2(vec));
	}

	// return the normalized version of the vector
	template< class T> CPU_AND_GPU inline T normalize(const T &vec) {
		typename T::value_type sum = length(vec);
		return sum == 0 ? T(typename T::value_type(0)) : vec / sum;
	}

	//template< class T> CPU_AND_GPU inline T min(const T &lhs, const T &rhs) {
	//	return lhs <= rhs ? lhs : rhs;
	//}

	//template< class T> CPU_AND_GPU inline T max(const T &lhs, const T &rhs) {
	//	return lhs >= rhs ? lhs : rhs;
	//}

	//component wise min
	template< class T> CPU_AND_GPU inline T minV(const T &lhs, const T &rhs) {
		T rv;
		for (int i = 0; i < lhs.size(); i++)
			rv[i] = min(lhs[i], rhs[i]);
		return rv;
	}

	// component wise max
	template< class T>
	CPU_AND_GPU inline T maxV(const T &lhs, const T &rhs) {
		T rv;
		for (int i = 0; i < lhs.size(); i++)
			rv[i] = max(lhs[i], rhs[i]);
		return rv;
	}


	// cross product
	template< class T>
	CPU_AND_GPU Vector3<T> cross(const Vector3<T> &lhs, const Vector3<T> &rhs) {
		Vector3<T> r;
		r.x = lhs.y * rhs.z - lhs.z * rhs.y;
		r.y = lhs.z * rhs.x - lhs.x * rhs.z;
		r.z = lhs.x * rhs.y - lhs.y * rhs.x;
		return r;
	}























	/************************************************************************/
	/* WARNING: the following 3x3 and 4x4 matrix are using column major, to	*/
	/* be consistent with OpenGL default rather than most C/C++ default.	*/
	/* In all other parts of the code, we still use row major order.		*/
	/************************************************************************/
	template <class T> class Vector2;
	template <class T> class Vector3;
	template <class T> class Vector4;
	template <class T, unsigned int s> class VectorX;

	//////////////////////////////////////////////////////////////////////////
	//						Basic Matrix Structure
	//////////////////////////////////////////////////////////////////////////

	template <class T> struct Matrix4_ {
		union {
			struct { // Warning: see the header in this file for the special matrix order
				T m00, m01, m02, m03;	// |0, 4, 8,  12|    |m00, m10, m20, m30|
				T m10, m11, m12, m13;	// |1, 5, 9,  13|    |m01, m11, m21, m31|
				T m20, m21, m22, m23;	// |2, 6, 10, 14|    |m02, m12, m22, m32|
				T m30, m31, m32, m33;	// |3, 7, 11, 15|    |m03, m13, m23, m33|
			};
			T m[16];
		};
	};

	template <class T> struct Matrix3_ {
		union { // Warning: see the header in this file for the special matrix order
			struct {
				T m00, m01, m02; // |0, 3, 6|     |m00, m10, m20|
				T m10, m11, m12; // |1, 4, 7|     |m01, m11, m21|
				T m20, m21, m22; // |2, 5, 8|     |m02, m12, m22|
			};
			T m[9];
		};
	};

	template<class T, unsigned int s> struct MatrixSQX_ {
		int dim;
		int sq;
		T m[s*s];
	};

	template<class T>
	class Matrix3;
	//////////////////////////////////////////////////////////////////////////
	// Matrix class with math operators
	//////////////////////////////////////////////////////////////////////////
	template<class T>
	class Matrix4 : public Matrix4_ < T >
	{
	public:
		CPU_AND_GPU Matrix4() {}
		CPU_AND_GPU Matrix4(T t) { setValues(t); }
		CPU_AND_GPU Matrix4(const T *m) { setValues(m); }
		CPU_AND_GPU Matrix4(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22, T a23, T a30, T a31, T a32, T a33) {
			this->m00 = a00; this->m01 = a01; this->m02 = a02; this->m03 = a03;
			this->m10 = a10; this->m11 = a11; this->m12 = a12; this->m13 = a13;
			this->m20 = a20; this->m21 = a21; this->m22 = a22; this->m23 = a23;
			this->m30 = a30; this->m31 = a31; this->m32 = a32; this->m33 = a33;
		}

#define Rij(row, col) R.m[row + 3 * col]
		CPU_AND_GPU Matrix3<T> GetR(void) const
		{
			Matrix3<T> R;
			Rij(0, 0) = m[0 + 4 * 0]; Rij(1, 0) = m[1 + 4 * 0]; Rij(2, 0) = m[2 + 4 * 0];
			Rij(0, 1) = m[0 + 4 * 1]; Rij(1, 1) = m[1 + 4 * 1]; Rij(2, 1) = m[2 + 4 * 1];
			Rij(0, 2) = m[0 + 4 * 2]; Rij(1, 2) = m[1 + 4 * 2]; Rij(2, 2) = m[2 + 4 * 2];

			return R;
		}

		CPU_AND_GPU void SetR(const Matrix3<T>& R) {
			m[0 + 4 * 0] = Rij(0, 0); m[1 + 4 * 0] = Rij(1, 0); m[2 + 4 * 0] = Rij(2, 0);
			m[0 + 4 * 1] = Rij(0, 1); m[1 + 4 * 1] = Rij(1, 1); m[2 + 4 * 1] = Rij(2, 1);
			m[0 + 4 * 2] = Rij(0, 2); m[1 + 4 * 2] = Rij(1, 2); m[2 + 4 * 2] = Rij(2, 2);
		}
#undef Rij

		CPU_AND_GPU inline void getValues(T *mp) const { memcpy(mp, this->m, sizeof(T) * 16); }
		CPU_AND_GPU inline const T *getValues() const { return this->m; }
		CPU_AND_GPU inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

		// Element access
		CPU_AND_GPU inline T &operator()(int x, int y) { return at(x, y); }
		CPU_AND_GPU inline const T &operator()(int x, int y) const { return at(x, y); }
		CPU_AND_GPU inline T &operator()(Vector2<int> pnt) { return at(pnt.x, pnt.y); }
		CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const { return at(pnt.x, pnt.y); }
		CPU_AND_GPU inline T &at(int x, int y) { return this->m[y | (x << 2)]; }
		CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[y | (x << 2)]; }

		// set values
		CPU_AND_GPU inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 16); }
		CPU_AND_GPU inline void setValues(T r) { for (int i = 0; i < 16; i++)	this->m[i] = r; }
		CPU_AND_GPU inline void setZeros() { memset(this->m, 0, sizeof(T) * 16); }
		CPU_AND_GPU inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = this->m33 = 1; }
		CPU_AND_GPU inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
		CPU_AND_GPU inline void setScale(const Vector3_<T> &s) { this->m00 = s.v[0]; this->m11 = s.v[1]; this->m22 = s.v[2]; }
		CPU_AND_GPU inline void setTranslate(const Vector3_<T> &t) { for (int y = 0; y < 3; y++) at(3, y) = t.v[y]; }
		CPU_AND_GPU inline void setRow(int r, const Vector4_<T> &t) { for (int x = 0; x < 4; x++) at(x, r) = t.v[x]; }
		CPU_AND_GPU inline void setColumn(int c, const Vector4_<T> &t) { memcpy(this->m + 4 * c, t.v, sizeof(T) * 4); }

		// get values
		CPU_AND_GPU inline Vector3<T> getTranslate() const {
			Vector3<T> T;
			for (int y = 0; y < 3; y++)
				T.v[y] = m[y + 4 * 3];
			return T;
		}
		CPU_AND_GPU inline Vector4<T> getRow(int r) const { Vector4<T> v; for (int x = 0; x < 4; x++) v.v[x] = at(x, r); return v; }
		CPU_AND_GPU inline Vector4<T> getColumn(int c) const { Vector4<T> v; memcpy(v.v, this->m + 4 * c, sizeof(T) * 4); return v; }
		CPU_AND_GPU inline Matrix4 t() { // transpose
			Matrix4 mtrans;
			for (int x = 0; x < 4; x++)	for (int y = 0; y < 4; y++)
				mtrans(x, y) = at(y, x);
			return mtrans;
		}

		CPU_AND_GPU inline friend Matrix4 operator * (const Matrix4 &lhs, const Matrix4 &rhs) {
			Matrix4 r;
			r.setZeros();
			for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) for (int k = 0; k < 4; k++)
				r(x, y) += lhs(k, y) * rhs(x, k);
			return r;
		}

		CPU_AND_GPU inline friend Matrix4 operator + (const Matrix4 &lhs, const Matrix4 &rhs) {
			Matrix4 res(lhs.m);
			return res += rhs;
		}

		CPU_AND_GPU inline Vector4<T> operator *(const Vector4<T> &rhs) const {
			Vector4<T> r;
			r[0] = this->m[0] * rhs[0] + this->m[4] * rhs[1] + this->m[8] * rhs[2] + this->m[12] * rhs[3];
			r[1] = this->m[1] * rhs[0] + this->m[5] * rhs[1] + this->m[9] * rhs[2] + this->m[13] * rhs[3];
			r[2] = this->m[2] * rhs[0] + this->m[6] * rhs[1] + this->m[10] * rhs[2] + this->m[14] * rhs[3];
			r[3] = this->m[3] * rhs[0] + this->m[7] * rhs[1] + this->m[11] * rhs[2] + this->m[15] * rhs[3];
			return r;
		}

		// Used as a projection matrix to multiply with the Vector3
		CPU_AND_GPU inline Vector3<T> operator *(const Vector3<T> &rhs) const {
			Vector3<T> r;
			r[0] = this->m[0] * rhs[0] + this->m[4] * rhs[1] + this->m[8] * rhs[2] + this->m[12];
			r[1] = this->m[1] * rhs[0] + this->m[5] * rhs[1] + this->m[9] * rhs[2] + this->m[13];
			r[2] = this->m[2] * rhs[0] + this->m[6] * rhs[1] + this->m[10] * rhs[2] + this->m[14];
			return r;
		}

		CPU_AND_GPU inline friend Vector4<T> operator *(const Vector4<T> &lhs, const Matrix4 &rhs) {
			Vector4<T> r;
			for (int x = 0; x < 4; x++)
				r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2) + lhs[3] * rhs(x, 3);
			return r;
		}

		CPU_AND_GPU inline Matrix4& operator += (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] += r; return *this; }
		CPU_AND_GPU inline Matrix4& operator -= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] -= r; return *this; }
		CPU_AND_GPU inline Matrix4& operator *= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] *= r; return *this; }
		CPU_AND_GPU inline Matrix4& operator /= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] /= r; return *this; }
		CPU_AND_GPU inline Matrix4 &operator += (const Matrix4 &mat) { for (int i = 0; i < 16; ++i) this->m[i] += mat.m[i]; return *this; }
		CPU_AND_GPU inline Matrix4 &operator -= (const Matrix4 &mat) { for (int i = 0; i < 16; ++i) this->m[i] -= mat.m[i]; return *this; }

		CPU_AND_GPU inline friend bool operator == (const Matrix4 &lhs, const Matrix4 &rhs) {
			bool r = lhs.m[0] == rhs.m[0];
			for (int i = 1; i < 16; i++)
				r &= lhs.m[i] == rhs.m[i];
			return r;
		}

		CPU_AND_GPU inline friend bool operator != (const Matrix4 &lhs, const Matrix4 &rhs) {
			bool r = lhs.m[0] != rhs.m[0];
			for (int i = 1; i < 16; i++)
				r |= lhs.m[i] != rhs.m[i];
			return r;
		}

		CPU_AND_GPU inline Matrix4 getInv() const {
			Matrix4 out;
			this->inv(out);
			return out;
		}
		/// Set out to be the inverse matrix of this.
		CPU_AND_GPU inline bool inv(Matrix4 &out) const {
			T tmp[12], src[16], det;
			T *dst = out.m;
			for (int i = 0; i < 4; i++) {
				src[i] = this->m[i * 4];
				src[i + 4] = this->m[i * 4 + 1];
				src[i + 8] = this->m[i * 4 + 2];
				src[i + 12] = this->m[i * 4 + 3];
			}

			tmp[0] = src[10] * src[15];
			tmp[1] = src[11] * src[14];
			tmp[2] = src[9] * src[15];
			tmp[3] = src[11] * src[13];
			tmp[4] = src[9] * src[14];
			tmp[5] = src[10] * src[13];
			tmp[6] = src[8] * src[15];
			tmp[7] = src[11] * src[12];
			tmp[8] = src[8] * src[14];
			tmp[9] = src[10] * src[12];
			tmp[10] = src[8] * src[13];
			tmp[11] = src[9] * src[12];

			dst[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7]);
			dst[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7]);
			dst[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
			dst[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);

			det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[2] + src[3] * dst[3];
			if (det == 0.0f)
				return false;

			dst[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3]);
			dst[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3]);
			dst[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
			dst[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

			tmp[0] = src[2] * src[7];
			tmp[1] = src[3] * src[6];
			tmp[2] = src[1] * src[7];
			tmp[3] = src[3] * src[5];
			tmp[4] = src[1] * src[6];
			tmp[5] = src[2] * src[5];
			tmp[6] = src[0] * src[7];
			tmp[7] = src[3] * src[4];
			tmp[8] = src[0] * src[6];
			tmp[9] = src[2] * src[4];
			tmp[10] = src[0] * src[5];
			tmp[11] = src[1] * src[4];

			dst[8] = (tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15]) - (tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15]);
			dst[9] = (tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15]) - (tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15]);
			dst[10] = (tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15]) - (tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15]);
			dst[11] = (tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14]) - (tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14]);
			dst[12] = (tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9]) - (tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10]);
			dst[13] = (tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10]) - (tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8]);
			dst[14] = (tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8]) - (tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9]);
			dst[15] = (tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9]) - (tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8]);

			out *= 1 / det;
			return true;
		}

		friend std::ostream& operator<<(std::ostream& os, const Matrix4<T>& dt) {
			for (int y = 0; y < 4; y++)
				os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << ", " << dt(3, y) << "\n";
			return os;
		}

		friend std::istream& operator>>(std::istream& s, Matrix4<T>& dt) {
			for (int y = 0; y < 4; y++)
				s >> dt(0, y) >> dt(1, y) >> dt(2, y) >> dt(3, y);
			return s;
		}
	};

	template<class T>
	class Matrix3 : public Matrix3_ < T >
	{
	public:
		CPU_AND_GPU Matrix3() {}
		CPU_AND_GPU Matrix3(T t) { setValues(t); }
		CPU_AND_GPU Matrix3(const T *m) { setValues(m); }
		CPU_AND_GPU Matrix3(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) {
			this->m00 = a00; this->m01 = a01; this->m02 = a02;
			this->m10 = a10; this->m11 = a11; this->m12 = a12;
			this->m20 = a20; this->m21 = a21; this->m22 = a22;
		}

		CPU_AND_GPU inline void getValues(T *mp) const { memcpy(mp, this->m, sizeof(T) * 9); }
		CPU_AND_GPU inline const T *getValues() const { return this->m; }
		CPU_AND_GPU inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

		// Element access
		CPU_AND_GPU inline T &operator()(int x, int y) { return at(x, y); }
		CPU_AND_GPU inline const T &operator()(int x, int y) const { return at(x, y); }
		CPU_AND_GPU inline T &operator()(Vector2<int> pnt) { return at(pnt.x, pnt.y); }
		CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const { return at(pnt.x, pnt.y); }
		CPU_AND_GPU inline T &at(int x, int y) { return this->m[x * 3 + y]; }
		CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[x * 3 + y]; }

		// set values
		CPU_AND_GPU inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 9); }
		CPU_AND_GPU inline void setValues(const T r) { for (int i = 0; i < 9; i++)	this->m[i] = r; }
		CPU_AND_GPU inline void setZeros() { memset(this->m, 0, sizeof(T) * 9); }
		CPU_AND_GPU inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = 1; }
		CPU_AND_GPU inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
		CPU_AND_GPU inline void setScale(const Vector3_<T> &s) { this->m00 = s[0]; this->m11 = s[1]; this->m22 = s[2]; }
		CPU_AND_GPU inline void setRow(int r, const Vector3_<T> &t) { for (int x = 0; x < 3; x++) at(x, r) = t[x]; }
		CPU_AND_GPU inline void setColumn(int c, const Vector3_<T> &t) { memcpy(this->m + 3 * c, t.v, sizeof(T) * 3); }

		// get values
		CPU_AND_GPU inline Vector3<T> getRow(int r) const { Vector3<T> v; for (int x = 0; x < 3; x++) v[x] = at(x, r); return v; }
		CPU_AND_GPU inline Vector3<T> getColumn(int c) const { Vector3<T> v; memcpy(v.v, this->m + 3 * c, sizeof(T) * 3); return v; }
		CPU_AND_GPU inline Matrix3 t() { // transpose
			Matrix3 mtrans;
			for (int x = 0; x < 3; x++)	for (int y = 0; y < 3; y++)
				mtrans(x, y) = at(y, x);
			return mtrans;
		}

		CPU_AND_GPU inline friend Matrix3 operator * (const Matrix3 &lhs, const Matrix3 &rhs) {
			Matrix3 r;
			r.setZeros();
			for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) for (int k = 0; k < 3; k++)
				r(x, y) += lhs(k, y) * rhs(x, k);
			return r;
		}

		CPU_AND_GPU inline friend Matrix3 operator + (const Matrix3 &lhs, const Matrix3 &rhs) {
			Matrix3 res(lhs.m);
			return res += rhs;
		}

		CPU_AND_GPU inline Vector3<T> operator *(const Vector3<T> &rhs) const {
			Vector3<T> r;
			r[0] = this->m[0] * rhs[0] + this->m[3] * rhs[1] + this->m[6] * rhs[2];
			r[1] = this->m[1] * rhs[0] + this->m[4] * rhs[1] + this->m[7] * rhs[2];
			r[2] = this->m[2] * rhs[0] + this->m[5] * rhs[1] + this->m[8] * rhs[2];
			return r;
		}

		CPU_AND_GPU inline Matrix3& operator *(const T &r) const {
			Matrix3 res(this->m);
			return res *= r;
		}

		CPU_AND_GPU inline friend Vector3<T> operator *(const Vector3<T> &lhs, const Matrix3 &rhs) {
			Vector3<T> r;
			for (int x = 0; x < 3; x++)
				r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2);
			return r;
		}

		CPU_AND_GPU inline Matrix3& operator += (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] += r; return *this; }
		CPU_AND_GPU inline Matrix3& operator -= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] -= r; return *this; }
		CPU_AND_GPU inline Matrix3& operator *= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] *= r; return *this; }
		CPU_AND_GPU inline Matrix3& operator /= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] /= r; return *this; }
		CPU_AND_GPU inline Matrix3& operator += (const Matrix3 &mat) { for (int i = 0; i < 9; ++i) this->m[i] += mat.m[i]; return *this; }
		CPU_AND_GPU inline Matrix3& operator -= (const Matrix3 &mat) { for (int i = 0; i < 9; ++i) this->m[i] -= mat.m[i]; return *this; }

		CPU_AND_GPU inline friend bool operator == (const Matrix3 &lhs, const Matrix3 &rhs) {
			bool r = lhs[0] == rhs[0];
			for (int i = 1; i < 9; i++)
				r &= lhs[i] == rhs[i];
			return r;
		}

		CPU_AND_GPU inline friend bool operator != (const Matrix3 &lhs, const Matrix3 &rhs) {
			bool r = lhs[0] != rhs[0];
			for (int i = 1; i < 9; i++)
				r |= lhs[i] != rhs[i];
			return r;
		}

		/// Matrix determinant
		CPU_AND_GPU inline T det() const {
			return (this->m11*this->m22 - this->m12*this->m21)*this->m00 + (this->m12*this->m20 - this->m10*this->m22)*this->m01 + (this->m10*this->m21 - this->m11*this->m20)*this->m02;
		}

		/// The inverse matrix for float/double type
		CPU_AND_GPU inline bool inv(Matrix3 &out) const {
			T determinant = det();
			if (determinant == 0) {
				out.setZeros();
				return false;
			}

			out.m00 = (this->m11*this->m22 - this->m12*this->m21) / determinant;
			out.m01 = (this->m02*this->m21 - this->m01*this->m22) / determinant;
			out.m02 = (this->m01*this->m12 - this->m02*this->m11) / determinant;
			out.m10 = (this->m12*this->m20 - this->m10*this->m22) / determinant;
			out.m11 = (this->m00*this->m22 - this->m02*this->m20) / determinant;
			out.m12 = (this->m02*this->m10 - this->m00*this->m12) / determinant;
			out.m20 = (this->m10*this->m21 - this->m11*this->m20) / determinant;
			out.m21 = (this->m01*this->m20 - this->m00*this->m21) / determinant;
			out.m22 = (this->m00*this->m11 - this->m01*this->m10) / determinant;
			return true;
		}

		friend std::ostream& operator<<(std::ostream& os, const Matrix3<T>& dt) {
			for (int y = 0; y < 3; y++)
				os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << "\n";
			return os;
		}
	};

	template<class T, unsigned int s>
	class MatrixSQX : public MatrixSQX_ < T, s >
	{
	public:
		CPU_AND_GPU MatrixSQX() { this->dim = s; this->sq = s*s; }
		CPU_AND_GPU MatrixSQX(T t) { this->dim = s; this->sq = s*s; setValues(t); }
		CPU_AND_GPU MatrixSQX(const T *m) { this->dim = s; this->sq = s*s; setValues(m); }
		CPU_AND_GPU MatrixSQX(const T m[s][s]) { this->dim = s; this->sq = s*s; setValues((T*)m); }

		CPU_AND_GPU inline void getValues(T *mp) const { memcpy(mp, this->m, sizeof(T) * 16); }
		CPU_AND_GPU inline const T *getValues() const { return this->m; }

		CPU_AND_GPU static inline MatrixSQX<T, s> make_aaT(const VectorX<float, s>& a) {
			float a_aT[s][s];
			for (int c = 0; c < s; c++)
				for (int r = 0; r < s; r++)
					a_aT[c][r] = a[c] * a[r];
			return a_aT;
		}

		CPU_AND_GPU static inline MatrixSQX<T, s> make_zeros() {
			MatrixSQX<T, s> x;
			x.setZeros();
			return x;
		}

		// Element access
		CPU_AND_GPU inline T &operator()(int x, int y) { return at(x, y); }
		CPU_AND_GPU inline const T &operator()(int x, int y) const { return at(x, y); }
		CPU_AND_GPU inline T &operator()(Vector2<int> pnt) { return at(pnt.x, pnt.y); }
		CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const { return at(pnt.x, pnt.y); }
		CPU_AND_GPU inline T &at(int x, int y) { return this->m[y * s + x]; }
		CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[y * s + x]; }

		// indexing operators
		CPU_AND_GPU T &operator [](int i) { return this->m[i]; }
		CPU_AND_GPU const T &operator [](int i) const { return this->m[i]; }

		// set values
		CPU_AND_GPU inline void setValues(const T *mp) { for (int i = 0; i < s*s; i++) this->m[i] = mp[i]; }
		CPU_AND_GPU inline void setValues(T r) { for (int i = 0; i < s*s; i++)	this->m[i] = r; }
		CPU_AND_GPU inline void setZeros() { for (int i = 0; i < s*s; i++)	this->m[i] = 0; }
		CPU_AND_GPU inline void setIdentity() { setZeros(); for (int i = 0; i < s*s; i++) this->m[i + i*s] = 1; }

		// get values
		CPU_AND_GPU inline VectorX<T, s> getRow(int r) const { VectorX<T, s> v; for (int x = 0; x < s; x++) v[x] = at(x, r); return v; }
		CPU_AND_GPU inline VectorX<T, s> getColumn(int c) const { Vector4<T> v; for (int x = 0; x < s; x++) v[x] = at(c, x); return v; }
		CPU_AND_GPU inline MatrixSQX<T, s> getTranspose()
		{ // transpose
			MatrixSQX<T, s> mtrans;
			for (int x = 0; x < s; x++)	for (int y = 0; y < s; y++)
				mtrans(x, y) = at(y, x);
			return mtrans;
		}

		CPU_AND_GPU inline friend  MatrixSQX<T, s> operator * (const  MatrixSQX<T, s> &lhs, const  MatrixSQX<T, s> &rhs) {
			MatrixSQX<T, s> r;
			r.setZeros();
			for (int x = 0; x < s; x++) for (int y = 0; y < s; y++) for (int k = 0; k < s; k++)
				r(x, y) += lhs(k, y) * rhs(x, k);
			return r;
		}

		CPU_AND_GPU inline friend MatrixSQX<T, s> operator + (const MatrixSQX<T, s> &lhs, const MatrixSQX<T, s> &rhs) {
			MatrixSQX<T, s> res(lhs.m);
			return res += rhs;
		}

		CPU_AND_GPU inline MatrixSQX<T, s>& operator += (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] += r; return *this; }
		CPU_AND_GPU inline MatrixSQX<T, s>& operator -= (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] -= r; return *this; }
		CPU_AND_GPU inline MatrixSQX<T, s>& operator *= (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] *= r; return *this; }
		CPU_AND_GPU inline MatrixSQX<T, s>& operator /= (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] /= r; return *this; }
		CPU_AND_GPU inline MatrixSQX<T, s> &operator += (const MatrixSQX<T, s> &mat) { for (int i = 0; i < s*s; ++i) this->m[i] += mat.m[i]; return *this; }
		CPU_AND_GPU inline MatrixSQX<T, s> &operator -= (const MatrixSQX<T, s> &mat) { for (int i = 0; i < s*s; ++i) this->m[i] -= mat.m[i]; return *this; }

		CPU_AND_GPU inline friend bool operator == (const MatrixSQX<T, s> &lhs, const MatrixSQX<T, s> &rhs) {
			bool r = lhs[0] == rhs[0];
			for (int i = 1; i < s*s; i++)
				r &= lhs[i] == rhs[i];
			return r;
		}

		CPU_AND_GPU inline friend bool operator != (const MatrixSQX<T, s> &lhs, const MatrixSQX<T, s> &rhs) {
			bool r = lhs[0] != rhs[0];
			for (int i = 1; i < s*s; i++)
				r |= lhs[i] != rhs[i];
			return r;
		}

		friend std::ostream& operator<<(std::ostream& os, const MatrixSQX<T, s>& dt) {
			for (int y = 0; y < s; y++)
			{
				for (int x = 0; x < s; x++) os << dt(x, y) << "\t";
				os << "\n";
			}
			return os;
		}
	};





























	// Solve Ax = b for A symmetric positive-definite
	// Usually, A = B^TB and b = B^Ty, 
	// as present in the normal-equations for solving linear least-squares problems
	class Cholesky
	{
	private:
		fvector cholesky;
		int rank, size;

	public:

        FUNCTION(, ~Cholesky, (), "") {
            free(this->cholesky.x);
        }

		//"Solve Ax = b for A symmetric positive-definite of size*size",
		template<unsigned int m>
		static VectorX<float, m> // cannot use FUNCTION because this contains a , that macro expansion cannot handle
			solve
				(
				const MatrixSQX<float, m>& mat,
				const VectorX<float, m>&  b
				)
			//PURITY_PURE
			//)
		{

			auto x = VectorX<float, m>();
			solve((const float*)mat.m, m, (const float*)b.v, x.v);
			return x;

		}

		static 
			FUNCTION(void, solve,(
				_In_reads_(size*size) const float* const mat, 
				const unsigned int size, 
				_In_reads_(size) const float* const b, 
				_Out_writes_(size) float * const result)
				, "Solve Ax = b for A symmetric positive-definite of size*size", PURITY_PURE) {
			Cholesky cholA(mat, size);
			cholA.Backsub(result, b);
		}

		/// \f[A = LL*\f]
		/// Produces Cholesky decomposition of the
		/// symmetric, positive-definite matrix mat of dimension size*size
		/// \f$L\f$ is a lower triangular matrix with real and positive diagonal entries
		///
		/// Note: assertFinite is used to detect singular matrices and other non-supported cases.
		FUNCTION(
            ,Cholesky
            ,(_In_reads_(size*size) const float * const mat, const unsigned int size)
            , "")
		{
            this->size = size;
            this->cholesky = vector_allocate_malloc(size*size);

			for (int i = 0; i < size * size; i++) cholesky[i] = assertFinite(mat[i]);

			for (int c = 0; c < size; c++)
			{
				float inv_diag = 1;
				for (int r = c; r < size; r++)
				{
					float val = cholesky[c + r * size];
					for (int c2 = 0; c2 < c; c2++)
						val -= cholesky[c + c2 * size] * cholesky[c2 + r * size];

					if (r == c)
					{
						cholesky[c + r * size] = assertFinite(val);
						if (val == 0) { rank = r; }
						inv_diag = 1.0f / val;
					}
					else
					{
						cholesky[r + c * size] = assertFinite(val);
						cholesky[c + r * size] = assertFinite(val * inv_diag);
					}
				}
			}

			rank = size;
		}

		/// Solves \f[Ax = b\f]
		/// by
		/// * solving Ly = b for y by forward substitution, and then
		/// * solving L*x = y for x by back substitution.
		FUNCTION(void, Backsub,(
			_Out_writes_(this->size) float * const x,  //!< out \f$x\f$
			_In_reads_(this->size) const float * const b //!< input \f$b\f$
			) const, "")
		{
			// Forward
			fvector y = vector_allocate_malloc(size);
			for (int i = 0; i < size; i++)
			{
				float val = b[i];
				for (int j = 0; j < i; j++) val -= cholesky[j + i * size] * y[j];
				y[i] = val;
			}

			for (int i = 0; i < size; i++) y[i] /= cholesky[i + i * size];

			// Backward
			for (int i = size - 1; i >= 0; i--)
			{
				float val = y[i];
				for (int j = i + 1; j < size; j++) val -= cholesky[i + j * size] * x[j];
				x[i] = val;
			}


            free(y.x);
		}
	};

	TEST(testCholesky) {
		float m[] = {
			1, 0,
			0, 1
		};
		float b[] = { 1, 2 };
		float r[2];
		Cholesky::solve(m, 2, b, r);
		assert(r[0] == b[0] && r[1] == b[1]);
	}

























































	typedef class Matrix3<float> Matrix3f;
	typedef class Matrix4<float> Matrix4f;

	typedef class Vector2<short> Vector2s;
    typedef class Vector2<int> Vector2i;
    typedef class Vector2<unsigned int> Vector2ui;

#ifdef __CUDACC__
	inline dim3 getGridSize(Vector2ui taskSize, dim3 blockSize)
	{
		return getGridSize(dim3(taskSize.x, taskSize.y), blockSize);
	}
#endif

	typedef class Vector2<float> Vector2f;

	typedef class Vector3<short> Vector3s;
	typedef class Vector3<double> Vector3d;
	typedef class Vector3<int> Vector3i;
	typedef class Vector3<unsigned int> Vector3ui;

	FUNCTION(unsigned int, volume, (Vector3ui d), "C110. Undefined where the product would overflow", PURITY_PURE) {
		return d.x*d.y*d.z;
	}

    typedef class Vector3<unsigned char> Vector3u;
	typedef class Vector3<float> Vector3f;

	// Documents that something is a unit-vector (e.g. normal vector), i.e. \in S^2
	typedef Vector3f UnitVector;

	typedef class Vector4<float> Vector4f;
	typedef class Vector4<int> Vector4i;
	typedef class Vector4<short> Vector4s;
    typedef class Vector4<unsigned char> Vector4u;

	typedef class Vector6<float> Vector6f;

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(x) (x).toIntRound()
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(x) (x).toIntRound()
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(inted, coeffs, in) inted = (in).toIntFloor(coeffs)
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (x).toShortFloor()
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (x).toUChar()
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (x).toFloat()
#endif

#ifndef TO_SHORT3
#define TO_SHORT3(p) Vector3s(p.x, p.y, p.z)
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) (a).toVector3()
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif

	inline CPU_AND_GPU Vector4f toFloat(Vector4u c) {
		return c.toFloat();
	}

	inline CPU_AND_GPU Vector4f toFloat(Vector4f c) {
		return c;
	}
	inline CPU_AND_GPU float toFloat(float c) {
		return c;
	}









    /*
    template<unsigned int m>
    FUNCTION(
        void
        , assertApproxEqual
        , (const VectorX<float, m>&  a, const VectorX<float, m>& b, const unsigned int considered_initial_bits = 20)
        , "", PURITY_PURE) {

        DO(i, m) {
            assertApproxEqual(a[i], b[i], considered_initial_bits);
        }

    }
    // causes internal compiler error in nvcc when used in constructExampleEquation
    */
















	/// Alternative/external implementation of axis-angle rotation matrix construction
	/// axis does not need to be normalized.
	/// c.f. ITMPose
	Matrix3f createRotation(const Vector3f & _axis, float angle)
	{
		Vector3f axis = normalize(_axis);
		float si = sinf(angle);
		float co = cosf(angle);

		Matrix3f ret;
		ret.setIdentity();

		ret *= co;
		for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) ret.at(c, r) += (1.0f - co) * axis[c] * axis[r];

		Matrix3f skewmat;
		skewmat.setZeros();
		skewmat.at(1, 0) = -axis.z;
		skewmat.at(0, 1) = axis.z;
		skewmat.at(2, 0) = axis.y;
		skewmat.at(0, 2) = -axis.y;
		skewmat.at(2, 1) = -axis.x; // should be axis.x?! c.f. infinitam
		skewmat.at(1, 2) = axis.x;// should be -axis.x;
		skewmat *= si;
		ret += skewmat;

		return ret;
	}









































	void approxEqual(float a, float b, const float eps = 0.00001) {
		assert(abs(a - b) < eps, "%f != %f mod %f", a, b, eps);
	}


	void approxEqual(Matrix4f a, Matrix4f b, const float eps = 0.00001) {
		for (int i = 0; i < 4 * 4; i++)
			approxEqual(a.m[i], b.m[i], eps);
	}

	void approxEqual(Matrix3f a, Matrix3f b, const float eps = 0.00001) {
		for (int i = 0; i < 3 * 3; i++)
			approxEqual(a.m[i], b.m[i], eps);
	}

	template<int m>
	FUNCTION(void, assertApproxEqual, (VectorX<float, m> a, VectorX<float, m> b, int considered_initial_bits = 20), "", PURITY_PURE) {
		for (int i = 0; i < m; i++) {
			assertApproxEqual(a[i], b[i], considered_initial_bits);
		}
	}


	TEST(testMatrix) {
		// Various ways of accessing matrix elements 
		Matrix4f m;
		m.setZeros();
		// at(x,y), mxy (not myx!, i.e. both syntaxes give the column first, then the row, different from standard maths)
		m.at(1, 0) = 1;
		m.at(0, 1) = 2;

		Matrix4f n;
		n.setZeros();
		n.m10 = 1;
		n.m01 = 2;
		/* m = n =
		0 1 0 0
		2 0 0 0
		0 0 0 0
		0 0 0 0*/

		approxEqual(m, n);

		Vector4f v(1, 8, 1, 2);
		assert(m*v == Vector4f(8, 2, 0, 0));
		assert(n*v == Vector4f(8, 2, 0, 0));
	}


























#define Rij(row, col) R.m[row + 3 * col]

	/** \brief
	Represents a rigid body pose with rotation and translation
	parameters.

	When used as a camera pose, by convention, this represents the world-to-camera transform.
	*/
	class ITMPose
	{
	private:
		void SetRPartOfM(const Matrix3f& R) {
			M.SetR(R);
		}

		/** This is the minimal representation of the pose with
		six parameters. The three rotation parameters are
		the Lie algebra representation of SO3.
		*/
		union UP
		{
			float all[6];
			struct {
				float tx, ty, tz;
				float rx, ry, rz;
			}each;
			struct {
				Vector3f t;
				// r is an "Euler vector", i.e. the vector "axis of rotation (u) * theta" (axis angle representation)
				Vector3f r;
			};
			UP() : t(), r() {} // seems necessary for vs15
		} params;

		/** The pose as a 4x4 transformation matrix (world-to-camera transform, modelview
		matrix).
		*/
		Matrix4f M;

		/** This will update the minimal parameterisation from
		the current modelview matrix.
		*/
		void SetParamsFromModelView();

		/** This will update the "modelview matrix" M from the
		minimal representation.
		*/
		void SetModelViewFromParams();
	public:

		void SetFrom(float tx, float ty, float tz, float rx, float ry, float rz);
		void SetFrom(const Vector3f &translation, const Vector3f &rotation);
		void SetFrom(const Vector6f &tangent);

		/// float tx, float ty, float tz, float rx, float ry, float rz
		void SetFrom(const float pose[6]);
		void SetFrom(const ITMPose *pose);

		/** This will multiply a pose @p pose on the right, i.e.
		this = this * pose.
		*/
		void MultiplyWith(const ITMPose *pose);

		const Matrix4f & GetM(void) const
		{
			return M;
		}

		Matrix3f GetR(void) const;
		Vector3f GetT(void) const;

		void GetParams(Vector3f &translation, Vector3f &rotation);

		void SetM(const Matrix4f & M);

		void SetR(const Matrix3f & R);
		void SetT(const Vector3f & t);
		void SetRT(const Matrix3f & R, const Vector3f & t);

		Matrix4f GetInvM(void) const;
		void SetInvM(const Matrix4f & invM);

		/** This will enforce the orthonormality constraints on
		the rotation matrix. It's recommended to call this
		function after manipulating the matrix M.
		*/
		void Coerce(void);

		ITMPose(const ITMPose & src);
		ITMPose(const Matrix4f & src);
		ITMPose(float tx, float ty, float tz, float rx, float ry, float rz);
		ITMPose(const Vector6f & tangent);
		explicit ITMPose(const float pose[6]);

		ITMPose(void);

		/** This builds a Pose based on its exp representation (c.f. exponential map in lie algebra, matrix exponential...)
		*/
		static ITMPose exp(const Vector6f& tangent);
	};

	ITMPose::ITMPose(void) { this->SetFrom(0, 0, 0, 0, 0, 0); }

	ITMPose::ITMPose(float tx, float ty, float tz, float rx, float ry, float rz)
	{
		this->SetFrom(tx, ty, tz, rx, ry, rz);
	}
	ITMPose::ITMPose(const float pose[6]) { this->SetFrom(pose); }
	ITMPose::ITMPose(const Matrix4f & src) { this->SetM(src); }
	ITMPose::ITMPose(const Vector6f & tangent) { this->SetFrom(tangent); }
	ITMPose::ITMPose(const ITMPose & src) { this->SetFrom(&src); }

	void ITMPose::SetFrom(float tx, float ty, float tz, float rx, float ry, float rz)
	{
		this->params.each.tx = tx;
		this->params.each.ty = ty;
		this->params.each.tz = tz;
		this->params.each.rx = rx;
		this->params.each.ry = ry;
		this->params.each.rz = rz;

		this->SetModelViewFromParams();
	}

	void ITMPose::SetFrom(const Vector3f &translation, const Vector3f &rotation)
	{
		this->params.each.tx = translation.x;
		this->params.each.ty = translation.y;
		this->params.each.tz = translation.z;
		this->params.each.rx = rotation.x;
		this->params.each.ry = rotation.y;
		this->params.each.rz = rotation.z;

		this->SetModelViewFromParams();
	}

	void ITMPose::SetFrom(const Vector6f &tangent)
	{
		this->params.each.tx = tangent[0];
		this->params.each.ty = tangent[1];
		this->params.each.tz = tangent[2];
		this->params.each.rx = tangent[3];
		this->params.each.ry = tangent[4];
		this->params.each.rz = tangent[5];

		this->SetModelViewFromParams();
	}

	void ITMPose::SetFrom(const float pose[6])
	{
		SetFrom(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]);
	}

	void ITMPose::SetFrom(const ITMPose *pose)
	{
		this->params.each.tx = pose->params.each.tx;
		this->params.each.ty = pose->params.each.ty;
		this->params.each.tz = pose->params.each.tz;
		this->params.each.rx = pose->params.each.rx;
		this->params.each.ry = pose->params.each.ry;
		this->params.each.rz = pose->params.each.rz;

		M = pose->M;
	}

	// init M from params
	void ITMPose::SetModelViewFromParams()
	{
		// w is an "Euler vector", i.e. the vector "axis of rotation (u) * theta" (axis angle representation)
		const Vector3f w = params.r;
		const float theta_sq = dot(w, w), theta = sqrt(theta_sq);

		const Vector3f t = params.t;

		float A, B, C;
		/*
		Limit for t approximating theta

		A = lim_{t -> theta} Sin[t]/t
		B = lim_{t -> theta} (1 - Cos[t])/t^2
		C = lim_{t -> theta} (1 - A)/t^2
		*/
		if (theta_sq < 1e-6f) // dont divide by very small or zero theta - use taylor series expansion of involved functions instead
		{
			A = 1 - theta_sq / 6 + theta_sq*theta_sq / 120; // Series[a, {t, 0, 4}]
			B = 1 / 2.f - theta_sq / 24;  //  Series[b, {t, 0, 2}]
			C = 1 / 6.f - theta_sq / 120; // Series[c, {t, 0, 2}]
		}
		else {
			assert(theta != 0.f);
			const float inv_theta = 1.0f / theta;
			A = sinf(theta) * inv_theta;
			B = (1.0f - cosf(theta)) * (inv_theta * inv_theta);
			C = (1.0f - A) * (inv_theta * inv_theta);
		}
		// TODO why isnt T = t?
		const Vector3f crossV = cross(w, t);
		const Vector3f cross2 = cross(w, crossV);
		const Vector3f T = t + B * crossV + C * cross2;

		// w = t u, u \in S^2, t === theta
		// R = exp(w . L) = I + sin(t) (u . L) + (1 - cos(t)) (u . L)^2
		// u . L == [u]_x, the matrix computing the left cross product with u (u x *)
		// L = (L_x, L_y, L_z) the lie algebra basis
		// c.f. https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Exponential_map
		Matrix3f R;
		const float wx2 = w.x * w.x, wy2 = w.y * w.y, wz2 = w.z * w.z;
		Rij(0, 0) = 1.0f - B*(wy2 + wz2);
		Rij(1, 1) = 1.0f - B*(wx2 + wz2);
		Rij(2, 2) = 1.0f - B*(wx2 + wy2);

		float a, b;
		a = A * w.z, b = B * (w.x * w.y);
		Rij(0, 1) = b - a;
		Rij(1, 0) = b + a;

		a = A * w.y, b = B * (w.x * w.z);
		Rij(0, 2) = b + a;
		Rij(2, 0) = b - a;

		a = A * w.x, b = B * (w.y * w.z);
		Rij(1, 2) = b - a;
		Rij(2, 1) = b + a;

		// Copy to M
		SetRPartOfM(R);
		M.setTranslate(T);

		M.m[3 + 4 * 0] = 0.0f; M.m[3 + 4 * 1] = 0.0f; M.m[3 + 4 * 2] = 0.0f; M.m[3 + 4 * 3] = 1.0f;
	}

	// init params from M
	void ITMPose::SetParamsFromModelView()
	{
		// Compute this->params.r = resultRot;
		Vector3f resultRot;
		const Matrix3f R = GetR();

		const float cos_angle = (R.m00 + R.m11 + R.m22 - 1.0f) * 0.5f;
		resultRot.x = (Rij(2, 1) - Rij(1, 2)) * 0.5f;
		resultRot.y = (Rij(0, 2) - Rij(2, 0)) * 0.5f;
		resultRot.z = (Rij(1, 0) - Rij(0, 1)) * 0.5f;

		const float sin_angle_abs = length(resultRot);

		if (cos_angle > M_SQRT1_2)
		{
			if (sin_angle_abs)
			{
				const float p = asinf(sin_angle_abs) / sin_angle_abs;
				resultRot *= p;
			}
		}
		else
		{
			if (cos_angle > -M_SQRT1_2)
			{
				const float p = acosf(cos_angle) / sin_angle_abs;
				resultRot *= p;
			}
			else
			{
				const float angle = (float)M_PI - asinf(sin_angle_abs);
				const float d0 = Rij(0, 0) - cos_angle;
				const float d1 = Rij(1, 1) - cos_angle;
				const float d2 = Rij(2, 2) - cos_angle;

				Vector3f r2;

				if (fabsf(d0) > fabsf(d1) && fabsf(d0) > fabsf(d2)) {
					r2.x = d0;
					r2.y = (Rij(1, 0) + Rij(0, 1)) * 0.5f;
					r2.z = (Rij(0, 2) + Rij(2, 0)) * 0.5f;
				}
				else {
					if (fabsf(d1) > fabsf(d2)) {
						r2.x = (Rij(1, 0) + Rij(0, 1)) * 0.5f;
						r2.y = d1;
						r2.z = (Rij(2, 1) + Rij(1, 2)) * 0.5f;
					}
					else {
						r2.x = (Rij(0, 2) + Rij(2, 0)) * 0.5f;
						r2.y = (Rij(2, 1) + Rij(1, 2)) * 0.5f;
						r2.z = d2;
					}
				}

				if (dot(r2, resultRot) < 0.0f) { r2 *= -1.0f; }

				r2 = normalize(r2);

				resultRot = angle * r2;
			}
		}

		this->params.r = resultRot;

		// Compute this->params.t = rottrans
		const Vector3f T = GetT();
		const float theta = length(resultRot);

		const float shtot = (theta > 0.00001f) ?
			sinf(theta * 0.5f) / theta :
			0.5f; // lim_{t -> theta} sin(t/2)/t, lim_{t -> 0} sin(t/2)/t = 0.5

		const ITMPose halfrotor(
			0.0f, 0.0f, 0.0f,
			resultRot.x * -0.5f, resultRot.y * -0.5f, resultRot.z * -0.5f
			);

		Vector3f rottrans = halfrotor.GetR() * T;

		const float param = dot(T, resultRot) *
			(
				(theta > 0.001f) ?
				(1 - 2 * shtot) / (theta * theta) :
				1 / 24.f // Series[(1 - 2*Sin[t/2]/t)/(t^2), {t, 0, 1}] = 1/24
				);

		rottrans -= resultRot * param;

		rottrans /= 2 * shtot;

		this->params.t = rottrans;
	}

	ITMPose ITMPose::exp(const Vector6f& tangent)
	{
		return ITMPose(tangent);
	}

	void ITMPose::MultiplyWith(const ITMPose *pose)
	{
		M = M * pose->M;
		this->SetParamsFromModelView();
	}

	Matrix3f ITMPose::GetR(void) const
	{
		return M.GetR();
	}

	Vector3f ITMPose::GetT(void) const
	{
		return M.getTranslate();
	}

	void ITMPose::GetParams(Vector3f &translation, Vector3f &rotation)
	{
		translation.x = this->params.each.tx;
		translation.y = this->params.each.ty;
		translation.z = this->params.each.tz;

		rotation.x = this->params.each.rx;
		rotation.y = this->params.each.ry;
		rotation.z = this->params.each.rz;
	}

	void ITMPose::SetM(const Matrix4f & src)
	{
		M = src;
		SetParamsFromModelView();
	}

	void ITMPose::SetR(const Matrix3f & R)
	{
		SetRPartOfM(R);
		SetParamsFromModelView();
	}

	void ITMPose::SetT(const Vector3f & t)
	{
		M.setTranslate(t);

		SetParamsFromModelView();
	}

	void ITMPose::SetRT(const Matrix3f & R, const Vector3f & t)
	{
		SetRPartOfM(R);
		M.setTranslate(t);

		SetParamsFromModelView();
	}

	Matrix4f ITMPose::GetInvM(void) const
	{
		Matrix4f ret;
		M.inv(ret);
		return ret;
	}

	void ITMPose::SetInvM(const Matrix4f & invM)
	{
		invM.inv(M);
		SetParamsFromModelView();
	}

	void ITMPose::Coerce(void)
	{
		SetParamsFromModelView();
		SetModelViewFromParams();
	}

#undef Rij






	TEST(testPose) {
		Matrix3f m = createRotation(Vector3f(0, 0, 0), 0);
		Matrix3f id; id.setIdentity();
		approxEqual(m, id);

		{
			Matrix3f rot = createRotation(Vector3f(0, 0, 1), M_PI);
			Matrix3f rot_ = {
				-1, 0, 0,
				0, -1, 0,
				0, 0, 1
			};
			ITMPose pose(0, 0, 0,
				0, 0, M_PI);
			approxEqual(rot, rot_);
			approxEqual(rot, pose.GetR());
		}
		{
#define ran (rand() / (1.f * RAND_MAX))
			Vector3f axis(ran, ran, ran);
			axis = axis.normalised(); // axis must have unit length for itmPose
			float angle = rand() / (1.f * RAND_MAX);

			ITMPose pose(0, 0, 0,
				axis.x*angle, axis.y*angle, axis.z*angle);

			Matrix3f rot = createRotation(axis, angle);
			approxEqual(rot, pose.GetR());
#undef ran
		}
	}



	const int MAX_REDUCE_BLOCK_SIZE = 4 * 4 * 4; // TODO this actually depends on shared memory demand of Constructor (::m, ExtraData etc.) -- template-specialize on it?

#ifdef __CUDACC__
	/// Exchange information
	__managed__ long long transform_reduce_resultMemory[100 * 100]; // use only in one module

	template <typename T>
	CPU_AND_GPU T& transform_reduce_result() {
		assert(sizeof(T) <= sizeof(transform_reduce_resultMemory));
		return *(T*)transform_reduce_resultMemory;
	}

	template<class Constructor>
	KERNEL transform_reduce_if_device(const unsigned int n) {
        const unsigned int tid = linear_threadIdx();
		assert(tid < MAX_REDUCE_BLOCK_SIZE, "tid %d", tid);
        const unsigned int i = linear_global_threadId();

        const unsigned int _REDUCE_BLOCK_SIZE = volume(blockDim);

		// Whether this thread block needs to compute a prefix sum
		__shared__ bool shouldPrefix;
		shouldPrefix = false;
		__syncthreads();

		__shared__ typename Constructor::ElementType reduced_elements[MAX_REDUCE_BLOCK_SIZE]; // this is pretty heavy on shared memory!

        // Let each thread generate its elements
		typename Constructor::ElementType& ei = reduced_elements[tid];

		if (
            i >= n || 

            !Constructor::generate(i
                , Vector3ui(xyz(blockIdx))
                , Vector3ui(xyz(threadIdx))
                , ei)

            ) {
            ei = Constructor::ElementType(); // use a neutral element when generate fails
		}
        else {
            shouldPrefix = true; // at least one thread has generated a valid element
        }

		__syncthreads();

		if (!shouldPrefix) return;

		// only if at least one thread in the thread block gets here do we do the prefix sum.

        // TODO start out at a later stage in the reduction if all upper
        // elements are neutral (maintain a counter) <- maybe too much overhead

		// tree reduction into reduced_elements[0]
		for (int offset = _REDUCE_BLOCK_SIZE / 2; offset >= 1; offset /= 2) {
			if (tid >= offset) return;

			Constructor::operate(reduced_elements[tid], reduced_elements[tid + offset]);
			// TODO warp reduce on simple additions will bring further performance enhancement.

			__syncthreads();
		}

		assert(0 == tid); // The only remaining thread is 0

		// Sum globally, using atomics
		auto& result = transform_reduce_result<Constructor::ElementType>();
		Constructor::atomicOperate(result, reduced_elements[0]);
	}
#endif

	/**

	Computes

	Fold[
	Constructor::operate doing the same as atomicOperate
	, Constructor::ElementType() <- neutral element = default constructor
	, Constructor::generate(#, blockIdx, threadIdx) /@ Range@n
	]

	where all computed quantities are of type Constructor::ElementType,
	assuming an associative binary operation 'operate'.

	Generate returns its result via reference. If it returns false, 
    the result is not used (it is assumed to be the neutral element).

	Similar to thrust algorithms, inspired by their naming, c.f. issue/suggestion: https://github.com/thrust/thrust/issues/773
	Provides simple parameters to specify/override the CUDA scheduling (more elaborate in actual thrust library).
	No general iterator is passed: generate() always operates on integers, but this could easily be changed.

	Constructor must provide:
	* Constructor::ElementType: The type returned by generate and neutralElement on which operate is a binary operation.
	* Constructor::generate(i) which will be called with i from 0 to n and may return false causing its result to be replaced with
	* CPU_AND_GPU Constructor::neutralElement()
	* Constructor::operate and atomicOperate define the binary operation. atomicOperate must do the same as operate but must modify the result atomically.

	See the examples for more details on the expected signatures.

	Scheduling:

	Constructor::generate is run once in each CUDA thread that is launched.
	tid == linear_threadIdx() < n always holds when it is called.

	The division into threads *can* be manually specified -- doing so will not significantly affect the outcome if the Constructor is agnostic to threadIdx et.al.
	If gridDim and/or blockDim are nonzero, it will be checked for conformance with n (must be bigger than or equal).
	gridDim can be left 0,0,0 in which case it is computed as ceil(n/volume(blockDim)),1,1.

	volume(blockDim) must be a power of 2 (for reduction) and <= MAX_REDUCE_BLOCK_SIZE.
	The limit MAX_REDUCE_BLOCK_SIZE is needed because per-block shared memory is limited. TODO make this parameter externally tunable.

	Both gridDim and blockDim default to being one-dimensional quantities (meaning y and z are 1).
	*/

	template<typename Constructor>
	CPU_FUNCTION(
		typename Constructor::ElementType
		, transform_reduce_if
		,(
			const unsigned int n
			, /*const*/ Vector3ui gridDim = Vector3ui(0, 0, 0)
			, /*const*/ Vector3ui blockDim = Vector3ui(0, 0, 0)
		)
		,""
		, PURITY_PURE
		) 
	{
	
		// Configure kernel scheduling
		if (gridDim.x == 0) {
			assert(gridDim.y == gridDim.z && gridDim.z == 0);

			if (blockDim.x == 0) {
				assert(blockDim.y == blockDim.z && blockDim.z == 0);
				blockDim = Vector3ui(MAX_REDUCE_BLOCK_SIZE, 1, 1); // default to one-dimension
			}

			gridDim = Vector3ui(ceil(n / volume(blockDim)), 1, 1); // default to one-dimension
		}
		assert(isPowerOf2(volume(blockDim))); // needed for reduction on gpu (parallel tree reduction)

		assert(volume(gridDim)*volume(blockDim) >= n, "must have enough threads to generate each element");
		assert(volume(blockDim) > 0);

#ifdef __CUDACC__
		assert(volume(blockDim) <= MAX_REDUCE_BLOCK_SIZE);

		// Set up storage for result
        transform_reduce_result<typename Constructor::ElementType>() = Constructor::ElementType();

		LAUNCH_KERNEL(
			(transform_reduce_if_device<Constructor>),
            dim3(xyz(gridDim)), dim3(xyz(blockDim)),
			n);
		cudaDeviceSynchronize();

		return transform_reduce_result<Constructor::ElementType>();
#else
		// Compute and reduce on CPU
		Constructor::ElementType accumulator; // init to neutral element
		unsigned int i = 0;

		FOR01(unsigned int, bx, gridDim.x)
		FOR01(unsigned int, by, gridDim.y)
		FOR01(unsigned int, bz, gridDim.z)

		FOR01(unsigned int, tx, blockDim.x)
		FOR01(unsigned int, ty, blockDim.y)
		FOR01(unsigned int, tz, blockDim.z) {
			if (i >= n) goto end;
			
			Constructor::ElementType e;
			if (!Constructor::generate(i++, Vector3ui(bx, by, bz), Vector3ui(tx, ty, tz), e))
                continue;
			
			Constructor::operate(accumulator, e);
		}
		end:
		return accumulator;
#endif

	}

	
	// Test
	struct transform_reduce_if_test_count_even {

		struct ElementType {
			unsigned int counter;
			FUNCTION(,ElementType,(unsigned int counter = 0), "") : counter(counter) {}
		};

		static FUNCTION(
			bool
			, generate
			, (
			_In_ const unsigned int i,
			_In_ const Vector3ui& _blockIdx,
			_In_ const Vector3ui& _threadIdx,
            _Out_opt_ ElementType& out)
			
			, ""
			, PURITY_OUTPUT_POINTERS) {

			assert(_blockIdx.x <= 1);
			assert(_blockIdx.y == 0);
			assert(_blockIdx.z == 0);
			
			assert(_threadIdx.x < 8);
			assert(_threadIdx.y == 0);
			assert(_threadIdx.z == 0);

			assert(i < 16);
            if (i % 2 == 1) return false;
			out = ElementType(1);
            return true;
		}

		static FUNCTION(void, operate, (
			_Inout_ ElementType& accumulator, _In_ const ElementType & integrand), "") {
			accumulator.counter += integrand.counter;
		}

		static FUNCTION(void, atomicOperate, (
			_Inout_ ElementType& accumulator, _In_ const ElementType & integrand), "") {
			_atomicAdd(&accumulator.counter, integrand.counter);
		}
	};

	TEST(transform_reduce_if_test_count_even1) {
		auto result = transform_reduce_if<transform_reduce_if_test_count_even>(16
			, Vector3ui(2, 1, 1)
			, Vector3ui(8, 1, 1));
		assert(result.counter == 8);
	}

	
	// leastsquares constructAndSolve GPU independent application of transform_reduce_if
    // Framework for building and solving (linear) least squares fitting problems 
    // (on the GPU)
    // c.f. constructAndSolve.nb

    namespace LeastSquares {



        /// Constructor must define
        /// Constructor::ExtraData(), add, atomicAdd
        /// static const unsigned int Constructor::m
        /// bool Constructor::generate(const unsigned int i, VectorX<float, m>& , float& bi)
        template<typename Constructor>
        struct AtA_Atb_Add {
            static const unsigned int m = Constructor::m;
            typedef typename MatrixSQX<float, m> AtA;
            typedef typename VectorX<float, m> Atb;
            typedef typename Constructor::ExtraData ExtraData;

            struct ElementType {
                typename AtA _AtA;
                typename Atb _Atb;
                typename ExtraData _extraData;
                CPU_AND_GPU ElementType() : _AtA(AtA::make_zeros()), _Atb(Atb::make_zeros()), _extraData(ExtraData()) {}
                CPU_AND_GPU ElementType(AtA _AtA, Atb _Atb, ExtraData _extraData) : _AtA(_AtA), _Atb(_Atb), _extraData(_extraData) {}
            };

            static FUNCTION(
                bool
                , generate
                , (
                _In_ const unsigned int i,
                _In_ const Vector3ui& _blockIdx,
                _In_ const Vector3ui& _threadIdx,
                _Out_opt_ ElementType& out)

                , ""
                , PURITY_OUTPUT_POINTERS) {

                // Some threads contribute zero
                VectorX<float, m> ai; float bi;
                if (!Constructor::generate(i, _blockIdx, _threadIdx, ai, bi, out._extraData)) return false;

                // Construct ai_aiT (an outer product matrix) and ai_bi
                out._AtA = MatrixSQX<float, m>::make_aaT(ai);
                out._Atb = ai * bi;
                return true;
            }

            static FUNCTION(void, operate, (
                _Inout_ ElementType& accumulator, _In_ const ElementType & integrand), "") {

                accumulator._AtA += integrand._AtA;
                accumulator._Atb += integrand._Atb;
                ExtraData::add(accumulator._extraData, integrand._extraData);
            }

            static FUNCTION(void, atomicOperate, (
                _Inout_ ElementType& accumulator, _In_ const ElementType & integrand), "") {
                for (int r = 0; r < m*m; r++)
                    atomicAdd(
                    &accumulator._AtA[r],
                    integrand._AtA[r]);

                for (int r = 0; r < m; r++)
                    atomicAdd(
                    &accumulator._Atb[r],
                    integrand._Atb[r]);

                ExtraData::atomicAdd(accumulator._extraData, integrand._extraData);
            }
        };



        /**
        Build A^T A and A^T b where A is <=n x m and b has <=n elements.

        Row i (0-based) of A and b[i] are generated by bool Constructor::generate(unsigned int i, VectorX<float, m> out_ai, float& out_bi).
        It is thrown away if generate returns false.
        */
        template<class Constructor>
        typename AtA_Atb_Add<Constructor>::ElementType
            construct(const unsigned int n
            , Vector3ui gridDim = Vector3ui(0, 0, 0)
            , Vector3ui blockDim = Vector3ui(0, 0, 0)) {

            assert(Constructor::m < 100);
            return transform_reduce_if<AtA_Atb_Add<Constructor>>(n, gridDim, blockDim);

        }

        /// Given a Constructor with method
        ///     static __device__ Constructor::generate(unsigned int i, VectorX<float, m> out_ai, float& out_bi)
        /// and static unsigned int Constructor::m
        /// build the equation system Ax = b with out_ai, out_bi in the i-th row/entry of A or b
        /// Then solve this in the least-squares sense and return x.
        ///
        /// i goes from 0 to n-1.
        ///
        /// Custom scheduling can be used and any custom Constructor::ExtraData can be summed up over all i.
        ///
        /// \see construct
        template<class Constructor>
        typename AtA_Atb_Add<Constructor>::Atb
            constructAndSolve(
            const unsigned int n, Vector3ui gridDim, Vector3ui blockDim, typename Constructor::ExtraData& out_extra_sum = Constructor::ExtraData()) {

            //auto 
            typename AtA_Atb_Add<Constructor>::ElementType 
                result = construct<Constructor>(n, gridDim, blockDim);

            out_extra_sum = result._extraData;

            cout << result._AtA << endl;
            cout << result._Atb << endl;

            return Cholesky::solve(result._AtA, result._Atb);
        }
    }




    // test constructAndSolve

    struct ConstructExampleEquation {
        // Amount of columns, should be small
        static const unsigned int m = 6;
        struct ExtraData {
            // User specified payload to be summed up alongside:
            unsigned int count;

            // Empty constructor must generate neutral element
            FUNCTION(,ExtraData,(),"") : count(0) {}

            static FUNCTION(void,add,(_Inout_ ExtraData& l, const ExtraData& r),"") {
                l.count += r.count;
            }
            static FUNCTION(void, atomicAdd,(_Inout_ ExtraData& result, const ExtraData& integrand), "") {
                ::_atomicAdd(&result.count, integrand.count);
            }
        };

        static FUNCTION(bool, generate, (
            const unsigned int i, 
            _In_ const Vector3ui& _blockIdx,
            _In_ const Vector3ui& _threadIdx, 
            VectorX<float, m>& out_ai, 
            float& out_bi, //[1], 
            ExtraData& out_extra), "") {

            assert(_threadIdx.y == _threadIdx.z && _threadIdx.y == 0);
            assert(_threadIdx.x < 32);

            for (int j = 0; j < m; j++) {
                out_ai[j] = 0;
                if (i == j || i + 1 == j || i == j + 1 || i == 0 || j == 0 || i % m == j)
                    out_ai[j] = 1;
            }
            out_bi = i + 1;

            bool ok = _blockIdx.x <= 1; // i.e. i <= 63, since blockDim = 32  // or: i % 2 == 0;
            out_extra.count = ok;

            return ok;//
            // i % 2 == 0;
        }
    };

    TEST(constructExampleEquation) {
        // c.f. constructAndSolve.nb
        const unsigned int m = ConstructExampleEquation::m;
        const unsigned int n = 512;


        auto y = LeastSquares::construct<ConstructExampleEquation>(n, Vector3ui(0, 0, 0), Vector3ui(32, 1, 1));

       
        assert(y._extraData.count == 64);
        
        auto x = LeastSquares::constructAndSolve<ConstructExampleEquation>(n, Vector3ui(0, 0, 0), Vector3ui(32, 1, 1));
        
        std::array<float, m> expect = {43.22650547302665, -10.45935368692164,
            -10.438367760716343, -8.180392124176143,
            -10.817604850187244, -11.47948083867651
        };
        
        //assertApproxEqual(x, VectorX<float, m>(expect));
        ::assertApproxEqual((float const * const)x, (float const * const)expect.data(), m);
    }


    struct ConstructExampleEquation2 {
        // Amount of columns, should be small
        static const unsigned int m = 6;
        struct ExtraData {
            // User specified payload to be summed up alongside:
            unsigned int count;

            // Empty constructor must generate neutral element
            FUNCTION(, ExtraData, (), "") : count(0) {}

            static FUNCTION(void, add, (_Inout_ ExtraData& l, const ExtraData& r), "") {
                l.count += r.count;
            }
            static FUNCTION(void, atomicAdd, (_Inout_ ExtraData& result, const ExtraData& integrand), "") {
                ::_atomicAdd(&result.count, integrand.count);
            }
        };

        static FUNCTION(bool, generate, (
            const unsigned int i,
            _In_ const Vector3ui& _blockIdx,
            _In_ const Vector3ui& _threadIdx,
            VectorX<float, m>& out_ai,
            float& out_bi, //[1], 
            ExtraData& out_extra), "") {

            assert(_threadIdx.y == _threadIdx.z && _threadIdx.y == 0);
            assert(_threadIdx.x < 32);

            for (int j = 0; j < m; j++) {
                out_ai[j] = 0;
                if (i == j || i + 1 == j || i == j + 1 || i == 0 || j == 0 || i % m == j)
                    out_ai[j] = 1;
            }
            out_bi = i + 1;

            bool ok = i % 2 == 0; // <- this is the only change
            out_extra.count = ok;

            return ok;//
            // i % 2 == 0;
        }
    };

           

    TEST(constructExampleEquation2) {
        // c.f. constructAndSolve.nb
        const int m = ConstructExampleEquation2::m;
        const int n = 512;

        auto y = LeastSquares::construct<ConstructExampleEquation2>(n, Vector3ui(0, 0, 0), Vector3ui(32, 1, 1));
        assert(y._extraData.count == 256);

        auto x = LeastSquares::constructAndSolve<ConstructExampleEquation2>(n, Vector3ui(0, 0, 0), Vector3ui(32, 1, 1));
        std::array<float, m> expect = {260.0155642023345, -85.33073929961078, -2.0155642023346445, \
            - 86.32295719844366, 0.9766536964980332, -169.6692607003891};

        ::assertApproxEqual((float const * const)x, (float const * const)expect.data(), m);
    }
    

}
using namespace vecmath;











// Camera calibration parameters, used to convert pixel coordinates to camera coordinates/rays
class ITMIntrinsics
{
public:
	FUNCTION(, ITMIntrinsics, (), "")
	{
		// standard calibration parameters for Kinect RGB camera. Not accurate
		fx = 580;
		fy = 580;
		px = 320;
		py = 240;
		sizeX = 640;
		sizeY = 480;
	}

	__declspec(property(get = all_, put = all_)) Vector4f all;

	FUNCTION(Vector4f, all_, () const, "") {
		return Vector4f(fx, fy, px, py);
	}

	FUNCTION(void, all_, (Vector4f v), "") {
		fx = v.x;
		fy = v.y;
		px = v.z;
		py = v.w;
	}

	union {
		struct {
			float fx, fy, px, py;
		};

		struct {
			float focalLength[2], centerPoint[2];
		};
	};
	unsigned int sizeX, sizeY;

    FUNCTION(Vector2ui, imageSize, () const, "") {
        return Vector2ui(sizeX, sizeY);
	}

	FUNCTION(void, imageSize, (Vector2i size), "") {
		sizeX = size.x;
		sizeY = size.y;
	}
};

/** \brief
Represents the extrinsic calibration between RGB and depth
cameras, i.e. the conversion from RGB camera-coordinates to depth-camera-coordinates and back

TODO use Coordinates class
*/
class ITMExtrinsics
{
public:
	/** The transformation matrix representing the
	extrinsic calibration data.
	*/
	Matrix4f calib;
	/** Inverse of the above. */
	Matrix4f calib_inv;

	/** Setup from a given 4x4 matrix, where only the upper
	three rows are used. More specifically, m00...m22
	are expected to contain a rotation and m30...m32
	contain the translation.
	*/
	FUNCTION(void, SetFrom, (const Matrix4f & src), "")
	{
		this->calib = src;
		this->calib_inv.setIdentity();
		for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) this->calib_inv.m[r + 4 * c] = this->calib.m[c + 4 * r];
		for (int r = 0; r < 3; ++r) {
			float & dest = this->calib_inv.m[r + 4 * 3];
			dest = 0.0f;
			for (int c = 0; c < 3; ++c) dest -= this->calib.m[c + 4 * r] * this->calib.m[c + 4 * 3];
		}
	}

	FUNCTION(, ITMExtrinsics, (), "")
	{
		Matrix4f m;
		m.setZeros();
		m.m00 = m.m11 = m.m22 = m.m33 = 1.0;
		SetFrom(m);
	}
};

/** \brief
Represents the joint RGBD calibration parameters.
*/
class ITMRGBDCalib
{
public:
	ITMIntrinsics intrinsics_rgb;
	ITMIntrinsics intrinsics_d;

	/** @brief
	M_d * worldPoint = trafo_rgb_to_depth.calib * M_rgb * worldPoint

	M_rgb * worldPoint = trafo_rgb_to_depth.calib_inv * M_d * worldPoint
	*/
	ITMExtrinsics trafo_rgb_to_depth;
};


























// Pinhole-camera computations, c.f. RayImage, DepthImage in coordinates::
// intrinsics vector is laid out like ITMIntrinsics.params.all, i.e. (fx, fy, cx, cy)

/// Computes a position in camera space given a 2d image coordinate and a depth.
/// \f$ z K^{-1}u\f$
/// \param x,y \f$ u\f$
FUNCTION(Vector4f, depthTo3D, (
	const Vector4f & viewIntrinsics, //!< K
	const int & x, const int & y,
	const float &depth //!< z
	), "") {
	/// Note: division by projection parameters .x, .y, i.e. fx and fy.
	/// The function below takes <inverse projection parameters> which have 1/fx, 1/fy, cx, cy
	Vector4f o;
	o.x = depth * ((float(x) - viewIntrinsics.z) / viewIntrinsics.x);
	o.y = depth * ((float(y) - viewIntrinsics.w) / viewIntrinsics.y);
	o.z = depth;
	o.w = 1.0f;
	return o;
}


FUNCTION(bool, projectNoBounds,(
	Vector4f projParams, Vector4f pt_camera, Vector2f& pt_image), "") {
	if (pt_camera.z <= 0) return false;

	pt_image.x = projParams.x * pt_camera.x / pt_camera.z + projParams.z;
	pt_image.y = projParams.y * pt_camera.y / pt_camera.z + projParams.w;

	return true;
}

/// $$\\pi(K p)$$
/// Projects pt_model, given in camera coordinates to 2d image coordinates (dropping depth).
/// \returns false when point projects outside of image
FUNCTION(bool, project,(
	Vector4f projParams, //!< K 
	const Vector2ui & imgSize,
	Vector4f pt_camera, //!< p
	Vector2f& pt_image), "") {
	if (!projectNoBounds(projParams, pt_camera, pt_image)) return false;

	if (pt_image.x < 0 || pt_image.x > imgSize.x - 1 || pt_image.y < 0 || pt_image.y > imgSize.y - 1) return false;
	// for inner points, when we compute gradients
	// was used like that in computeUpdatedVoxelDepthInfo
	//if ((pt_image.x < 1) || (pt_image.x > imgSize.x - 2) || (pt_image.y < 1) || (pt_image.y > imgSize.y - 2)) return -1;

	return true;
}

/// Reject pixels on the right lower boundary of the image 
// (which have an incomplete forward-neighborhood)
FUNCTION(bool, projectExtraBounds, (
	Vector4f projParams, const Vector2ui & imgSize,
	Vector4f pt_camera, Vector2f& pt_image), "") {
	if (!projectNoBounds(projParams, pt_camera, pt_image)) return false;

	if (pt_image.x < 0 || pt_image.x > imgSize.x - 2 || pt_image.y < 0 || pt_image.y > imgSize.y - 2) return false;

	return true;
}












// ^^ end of basic maths






























// dynamic-sized Memory-block and 'image'/large-matrix management
namespace memory {









    // Unified Memory Management
    // memoryAllocate/Free manage 'universal'/'portable' dynamic memory, 
    // allocatable on the CPU and available to the GPU too

#ifdef __CUDACC__

#define CUDA_CHECK_ERRORS() {auto e = cudaGetLastError(); if (cudaSuccess != e) printf("cudaGetLastError %d %s %s\n", e, cudaGetErrorName(e), cudaGetErrorString(e));}


#define memoryAllocate(ptrtype, ptr, sizeInBytes) {cudaDeviceSynchronize();cudaMallocManaged(&ptr, (sizeInBytes));cudaDeviceSynchronize();assert(ptr);CUDA_CHECK_ERRORS();}
#define memoryFree(ptr) {cudaDeviceSynchronize();cudaFree(ptr);cudaDeviceSynchronize();CUDA_CHECK_ERRORS();/* // did some earlier kernel throw an assert?*/}

#else

#define memoryAllocate(ptrtype, ptr, sizeInBytes) {ptr = (ptrtype)/*decltype not working*/malloc(sizeInBytes); assert(ptr); }
#define memoryFree(ptr) {::free(ptr);}

#endif

    // Convenience wrappers for memoryAllocate
    template<typename T>
    CPU_FUNCTION(T*, tmalloc, (const size_t n), "Allocate space for n T structs") {
        assert(n);
        T* out;
        memoryAllocate(T*, out, sizeof(T) * n);
        return out;
    }

    template<typename T>
    CPU_FUNCTION(T*, tmalloczeroed, (const size_t n), "Allocate spece for n T structs and set the allocate memory to 0") {
        assert(n);
        T* out;
        memoryAllocate(T*, out, sizeof(T) * n);
        memset(out, 0, sizeof(T) * n);
        return out;
    }

    template<typename T>
    CPU_FUNCTION(T*, mallocmemcpy, (_In_reads_(n) T const * const x, const size_t n), "Allocate space for n T structs and initialize it with the n T structs pointed to by x") {
        assert(n);
        auto out = tmalloc<T>(n);
        memcpy(out, x, sizeof(T) * n);
        return out;
    }

    template<typename T>
    CPU_FUNCTION(void, freemalloctmemcpy, (_Inout_ T** dest, _In_reads_(n) const T* const src, const size_t n), "Free *dest if it is nonzero, then reallocate it and initialize from src")  {
        assert(n);
        if (*dest) memoryFree(*dest);

        auto sz = sizeof(T) * n;

        memoryAllocate(T*, *dest, sz);
        memcpy(*dest, src, sz);
    }






    /// Extend this class to declar objects whose memory lies in CUDA managed memory space
    /// which is accessible from CPU and GPU.
    /// Classes extending this must be heap-allocated
    struct Managed {
        CPU_MEMBERFUNCTION(void *, operator new, (size_t len), ""){
            void* ptr;
            memoryAllocate(void*, ptr, len); // if cudaSafeCall fails here check the following: did some earlier kernel throw an assert?
            return ptr;
        }

        CPU_MEMBERFUNCTION(void, operator delete, (void *ptr), "") {
            memoryFree(ptr);
        }
    };


    void testMemblock();

    /// Pointer to nonempty block of mutable unified memory
    template <typename T>
    struct MemoryBlock : public Managed
    {
    private:
        friend void testMemblock();

        T* data; // managed memory, sizeof(T) * dataSize bytes

        size_t dataSize__;
        MEMBERFUNCTION(void, dataSize_, (size_t dataSize), "") { dataSize__ = dataSize; }

        CPU_MEMBERFUNCTION(void,Free,(), ""){
            if (!data) return;
            memoryFree(data);
            data = 0;
            dataSize = 0;
        }
        
        void operator=(MemoryBlock&); // disabled
    public:
        MEMBERFUNCTION(size_t, dataSize_, () const, "") { return dataSize__; }
        __declspec(property(get = dataSize_, put = dataSize_)) size_t dataSize;

        CPU_MEMBERFUNCTION(void, Reallocate,(size_t dataSize),"") {
            Free();

            assert(0 == this->dataSize);
            assert(0 == this->data);
            assert(dataSize);

            this->dataSize = dataSize;

            memoryAllocate(T*, data, dataSizeInBytes());

            Clear(0);
        }

        CPU_MEMBERFUNCTION(,MemoryBlock,(size_t dataSize), "") : data(0), dataSize__(0) {
            Reallocate(dataSize);
        }

        virtual CPU_MEMBERFUNCTION(, ~MemoryBlock, (), "") {
            Free();
        }
        
        MEMBERFUNCTION(bool,operator==,(const MemoryBlock& other) const, "") {
            if (other.dataSizeInBytes() != dataSizeInBytes())
                return false;
            auto x = _memcmp(data, other.data, other.dataSizeInBytes());
            return x == 0;
        }

        CPU_MEMBERFUNCTION(void, SetFrom, (const MemoryBlock& copyFrom), "") {
            Reallocate(copyFrom.dataSize);

            assert(this->dataSizeInBytes() == copyFrom.dataSizeInBytes());

            memcpy(data, copyFrom.data, copyFrom.dataSizeInBytes());

            assert(*this == copyFrom);
        }

        CPU_MEMBERFUNCTION(, MemoryBlock, (const MemoryBlock& copyFrom), "") : data(0), dataSize__(0){
            SetFrom(copyFrom);
        }

        SERIALIZE_VERSION(1);
        /* format:
        version :: int
        dataSize (amount of elements) :: size_t
        sizeof(T) :: size_t
        data :: BYTE[dataSizeInBytes()]
        */
        CPU_MEMBERFUNCTION(void, serialize,(ofstream& file),"") {
            auto p = file.tellp(); // DEBUG

            SERIALIZE_WRITE_VERSION(file);
            bin(file, dataSize);
            bin(file, (size_t)sizeof(T));
            file.write((const char*)data, dataSizeInBytes());

            assert(file.tellp() - p == sizeof(int) + sizeof(size_t) * 2 + dataSizeInBytes(), "%I64d != %I64d",
                file.tellp() - p,
                sizeof(int) + sizeof(size_t) * 2 + dataSizeInBytes());
        }

        CPU_MEMBERFUNCTION(void, SetFrom,(char* data, size_t dataSize), "") {
            Reallocate(dataSize);
            memcpy((char*)this->data, data, dataSizeInBytes());
        }

        // TODO should be able to require it to be the same size (because Scene serialization format expects localVBA to be of a certain size)
        // loses current data
        CPU_MEMBERFUNCTION(void, deserialize,(ifstream& file), "") {
            SERIALIZE_READ_VERSION(file);

            Reallocate(bin<size_t>(file));
            assert(bin<size_t>(file) == sizeof(T));

            file.read((char*)data, dataSizeInBytes());
        }

        MEMBERFUNCTION(size_t,dataSizeInBytes,() const, "") {
            return dataSize * sizeof(T);
        }

        MEMBERFUNCTION(T*, GetData, (), "Get the data pointer")
        {
            return data;
        }

        MEMBERFUNCTION(T const*, GetData, () const, "Get the const data pointer")
        {
            return data;
        }

        // convenience & bounds checking
        MEMBERFUNCTION(T&, operator[], (unsigned int i), "") {
            assert(i < dataSize, "%d >= %d -- MemoryBlock access out of range", i, dataSize);
            return GetData()[i];
        }

        MEMBERFUNCTION(T const&, operator[], (unsigned int i) const, "") {
            assert(i < dataSize, "%d >= %d -- MemoryBlock access out of range", i, dataSize);
            return GetData()[i];
        }

        /** Set all data to the byte given by @p defaultValue. */
        MEMBERFUNCTION(void, Clear,(unsigned char defaultValue = 0),"")
        {
            memset(data, defaultValue, dataSizeInBytes());
        }
    };

    GLOBAL(int*, data, 0, "pointer to unified memory for testing");

#ifdef __CUDACC__
#define SETDATA() LAUNCH_KERNEL(set_data, 1, 1);
#define CHECKDATA() LAUNCH_KERNEL(check_data, 1, 1);
    KERNEL set_data() {
        data[1] = 42;
    }
    KERNEL check_data() {
        assert(data[1] == 42);
    }
#else

#define SETDATA() set_data();
#define CHECKDATA() check_data();
    void set_data() {
        data[1] = 42;
    }
    void check_data() {
        assert(data[1] == 42);
    }
#endif

    TEST(testMemblock) {
        auto mem = new MemoryBlock<int>(10);
        assert(mem->dataSize == 10);

        mem->Clear(0);

        auto const* const cmem = mem;

        mem->GetData()[1] = 42;
        data = mem->GetData();
        CHECKDATA();

        mem->Clear(0);

        assert(mem->GetData()[1] == 0);

        SETDATA();
        CHECKDATA();

        mem->Clear(0);
        data = mem->GetData();

        assert(mem->GetData()[1] == 0);
        SETDATA();
        CHECKDATA();

        assert(mem->GetData()[1] == 42);
    }

#ifdef __CUDACC__
#define GPUCHECK(p, val) LAUNCH_KERNEL(check,1,1,(char* )p,val);
    KERNEL check(char* p, char val) {
        assert(*p == val);
    }
#else
    void GPUCHECK(void* p, char val) {
        assert(*(char*)p == val);
    }
#endif

    TEST(testMemoryBlockSerialize) {
        MemoryBlock<int> b(1);
        b[0] = 0xbadf00d;
        GPUCHECK(b.GetData(), 0x0d);

        /*
        BEGIN_SHOULD_FAIL("testMemoryBlockSerialize");
        GPUCHECK(b.GetData(MEMORYDEVICE_CUDA), 0xba);
        END_SHOULD_FAIL("testMemoryBlockSerialize");
        */
        auto fn = "o.bin";
        {
            b.serialize(binopen_write(fn));
        }

    {
        b.deserialize(binopen_read(fn));
    }
    assert(b[0] == 0xbadf00d);
    assert(b.dataSize == 1);

    b.Clear();

    {
        b.deserialize(binopen_read(fn));
    }
    assert(b[0] == 0xbadf00d);
    GPUCHECK(b.GetData(), 0x0d);
    assert(b.dataSize == 1);


    MemoryBlock<int> c(100);
    assert(c.dataSize == 100);
    GPUCHECK(c.GetData(), 0);

    {
        c.deserialize(binopen_read(fn));
    }
    assert(c[0] == 0xbadf00d);
    assert(c.dataSize == 1);
    GPUCHECK(c.GetData(), 0x0d);

    }
#undef GPUCHECK

    TEST(testMemoryBlockCopyCompare) {
        MemoryBlock<int> ma(100);
        MemoryBlock<int> mb(100);
        MemoryBlock<int> mc(90);

        assert(mb == mb);
        assert(mb == ma);
        ma.Clear(1);
        assert(!(mb == ma));
        assert(!(mb == mc));
        assert(!(mb == ma));

        MemoryBlock<int> md(ma);
        assert(ma == md);
    }











    /// Linearized pixel index
    FUNCTION(unsigned int, pixelLocId, (const unsigned int x, const unsigned int y, const Vector2ui &imgSize), "undefined where overflows occur", PURITY_PURE) {
        return x + y * imgSize.x;
    }

    /** \brief
    Represents images, templated on the pixel type

    Managed
    */
    template <typename T>
    class Image : public MemoryBlock < T >
    {
    public:
        /** Nonzero Size of the image in pixels. */
        Vector2ui noDims;

        /** Initialize an empty image of the given size
        */
        Image(Vector2ui noDims = Vector2ui(1, 1)) : MemoryBlock<T>(noDims.area()), noDims(noDims) {}

        void EnsureDims(Vector2ui noDims) {
            if (this->noDims == noDims) return;
            this->noDims = noDims;
            Reallocate(noDims.area());
        }

        // convenience
        T& operator()(unsigned int x, unsigned int y) {
            return GetData()[pixelLocId(x,y,noDims)];
        }
    };


#define ITMFloatImage Image<float>
#define ITMFloat2Image Image<Vector2f>
#define ITMFloat4Image Image<Vector4f>
#define ITMShortImage Image<short>
#define ITMShort3Image Image<Vector3s>
#define ITMShort4Image Image<Vector4s>
#define ITMUShortImage Image<ushort>
#define ITMUIntImage Image<unsigned int>
#define ITMIntImage Image<int>
#define ITMUCharImage Image<uchar>
#define ITMUChar4Image Image<Vector4u>
#define ITMBoolImage Image<bool>


}
using namespace memory;





// ^^ end of low-level memory management





























// Colors as Vector4f or Vector4u


template<typename T>
struct IllegalColor {
    static /*constexpr*/ FUNCTION(T, make, (), "", PURITY_PURE, TOTALITY_TOTAL);
};
FUNCTION(float, IllegalColor<float>::make, (), "", PURITY_PURE, TOTALITY_TOTAL) {
    return -1;
}
FUNCTION(Vector4f, IllegalColor<Vector4f>::make, (), "", PURITY_PURE, TOTALITY_TOTAL) {
    return Vector4f(0, 0, 0, -1);
}

FUNCTION(bool, isLegalColor, (float c), "", PURITY_PURE, TOTALITY_TOTAL)  {
    return c >= 0;
}
FUNCTION(bool, isLegalColor, (Vector4f c), "", PURITY_PURE, TOTALITY_TOTAL) {
    return c.w >= 0;
}

FUNCTION(bool,isLegalColor,(Vector4u c),"DO NOT USE") {
    // NOTE this should never be called -- withHoles should be false for a Vector4u image
    // implementing this just calms the compiler
    fatalError("isLegalColor is not implemented for Vector4u");
    return false;
}

























// Local/per pixel Image processing library



/// Sample (lookup) image without interpolation at integer location
template<typename T>
FUNCTION(T, sampleNearest,(
    const T * const source,
    const unsigned int x, const unsigned int y,
const Vector2ui & imgSize), "", PURITY_PURE)
{
    return source[pixelLocId(x, y, imgSize)];
}

/// Sample image without interpolation at rounded location
template<typename T> 
FUNCTION(T, sampleNearest,(
const T *source,
const Vector2f & pt_image,
const Vector2ui & imgSize), "", PURITY_PURE) {
    return source[
        pixelLocId(
            (int)(pt_image.x + 0.5f), // TODO use ROUND
            (int)(pt_image.y + 0.5f),
            imgSize)];
}

/// Whether interpolation should return an illegal color when holes make interpolation impossible
#define WITH_HOLES true
/// Sample 4 channel image with bilinear interpolation (T_IN::toFloat must return Vector4f)
/// IF withHoles == WITH_HOLES: returns makeIllegalColor<OUT>() when any of the four surrounding pixels is illegal (has negative w).
template<typename T_OUT, //!< Vector4f or float
    bool withHoles = false, typename T_IN> 
FUNCTION(Vector4f, interpolateBilinear,(
    const T_IN * const source,
    const Vector2f & position, const Vector2ui & imgSize), "", PURITY_PURE)
{
    T_OUT result;
    Vector2ui p; Vector2f delta;

    assert(position.x >= 0 && position.y >= 0); // TODO use >= 0 float type and/or cast with function that fails for negatives ('cast to natural')
    p.x = (unsigned int)floor(position.x); p.y = (unsigned int)floor(position.y);
    delta.x = position.x - p.x; delta.y = position.y - p.y;

#define sample(dx, dy) sampleNearest(source, p.x + dx, p.y + dy, imgSize);
    T_IN a = sample(0, 0);
    T_IN b = sample(1, 0);
    T_IN c = sample(0, 1);
    T_IN d = sample(1, 1);
#undef sample

    if (withHoles && (!isLegalColor(a) || !isLegalColor(b) || !isLegalColor(c) || !isLegalColor(d))) return IllegalColor<T_OUT>::make();

    /**
    ------> dx
    | a b
    | c d
    dy
    \/
    */
    result =
        toFloat(a) * (1.0f - delta.x) * (1.0f - delta.y) +
        toFloat(b) * delta.x * (1.0f - delta.y) +
        toFloat(c) * (1.0f - delta.x) * delta.y +
        toFloat(d) * delta.x * delta.y;

    return result;
}


// === forEachPixelNoImage ===

#ifdef __CUDACC__
template<typename F>
static KERNEL forEachPixelNoImage_device(Vector2ui imgSize) {
    const unsigned int
        x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > imgSize.x - 1 || y > imgSize.y - 1) return;
    const unsigned int locId = pixelLocId(x, y, imgSize);

    F::process(x, y, locId);
}
#endif

#define forEachPixelNoImage_process() static FUNCTION(void, process,(const unsigned int x, const unsigned int y, const unsigned int locId), "")

/** apply
F::process(int x, int y, int locId)
to each (hypothetical) pixel in the image

locId runs through values generated by pixelLocId(x, y, imgSize);
*/
template<typename F>
static void forEachPixelNoImage(Vector2ui imgSize) {
#ifdef __CUDACC__
    const dim3 blockSize(16, 16);
    LAUNCH_KERNEL(
        forEachPixelNoImage_device<F>,
        getGridSize(dim3(xy(imgSize)), blockSize),
        blockSize,
        imgSize);
#else
    DO(y, imgSize.height)
        DO(x, imgSize.width) {
        const unsigned int locId = pixelLocId(x, y, imgSize);
        F::process(x, y, locId);
    }
#endif
}















namespace TestForEachPixel {
    const unsigned int W = 5;
    const unsigned int H = 7;

    GLOBAL(unsigned int,fipcounter, 0, "FOR TESTS");

    struct DoForEachPixel {
        forEachPixelNoImage_process() {
            assert(x < W);
            assert(y < H);
            _atomicAdd(&fipcounter, 1);
        }
    };

    TEST(testForEachPixelNoImage) {
        fipcounter = 0;
        forEachPixelNoImage<DoForEachPixel>(Vector2ui(W, H));
        assert(fipcounter == W * H);
    }

}









// Subsampling (2x2 box averaging downsampling) filters

GLOBAL(void*, _imageData_out, 0, "used in filterSubsample after casting to the right type");
GLOBAL(void*, _imageData_in, 0, "used in filterSubsample after casting to the right type");
GLOBAL_UNINITIALIZED(Vector2ui, newDims, Vector2ui(0, 0), "");
GLOBAL_UNINITIALIZED(Vector2ui, oldDims, Vector2ui(0, 0), "");

// local subsampling filter for pixel (x,y). newDims = oldDims/2
// uses the above globals as implicit parameters, PURITY_ENVIRONMENT_DEPENDANT | _OUTPUT_POINTERS
template<typename T, bool withHoles = false>
struct filterSubsample {
    forEachPixelNoImage_process() {
        // arguments:
        auto imageData_in = (T const * const)_imageData_in;
        auto imageData_out = (T * const)_imageData_out;
        // 

        unsigned int src_pos_x = x * 2, src_pos_y = y * 2;
        T pixel_out = 0.0f, pixel_in;
        float no_good_pixels = 0.0f;

#define sample(dx,dy) \
    pixel_in = imageData_in[(src_pos_x + dx) + (src_pos_y + dy) * oldDims.x]; \
	if (!withHoles || isLegalColor(pixel_in)) { pixel_out += pixel_in; no_good_pixels++; }

        sample(0, 0);
        sample(1, 0);
        sample(0, 1);
        sample(1, 1);
#undef sample

        if (no_good_pixels > 0) pixel_out /= no_good_pixels;
        else if (withHoles) pixel_out = IllegalColor<T>::make();

        imageData_out[pixelLocId(x, y, newDims)] = pixel_out;

    }
};


template<typename T, bool withHoles = false>
CPU_FUNCTION(
    void, 
    FilterSubsample
    , (_Inout_ Image<T> * const image_out, _In_ const Image<T> * const image_in)
    ,"image_out will be set to the right size, half of the input image", PURITY_OUTPUT_POINTERS) {
    
    ::oldDims = image_in->noDims; 
    ::newDims.x = image_in->noDims.x / 2;
    ::newDims.y = image_in->noDims.y / 2; 
    
    image_out->EnsureDims(newDims); 
    
    ::_imageData_in = (void*)image_in->GetData(); 
    ::_imageData_out = (void*)image_out->GetData();
    

    forEachPixelNoImage<filterSubsample<T, withHoles>>(newDims);
}

template<typename T>
CPU_FUNCTION(
    void,
    FilterSubsampleWithHoles, (_Inout_ Image<T> * const image_out, _In_ const Image<T> * const image_in)
    , "image_out will be set to the right size, half of the input image", PURITY_OUTPUT_POINTERS) {
    FilterSubsample<T, WITH_HOLES>(image_out, image_in);
}

TEST(tFilterSubsample) {
    ITMFloatImage* _f = new ITMFloatImage(Vector2ui(2,2)); ITMFloatImage& f = *_f;
    
    f(0,0) = 1.f;
    f(0, 1) = 2.f;
    f(1, 0) = 3.f;
    f(1, 1) = 4.f;
    assert(4.f == f(1, 1));

    ITMFloatImage* _r = new ITMFloatImage(Vector2ui(2, 2)); ITMFloatImage& r = *_r;

    assert(r.noDims == f.noDims);

    FilterSubsample<float>(_r, _f);

    assert(r.noDims == f.noDims / 2);
    assert(r(0,0) == 2.5f);

    delete _f;
    delete _r;
}
















// Concept of a rectilinear, aribtrarily rotated and scaled coordinate system
// we use the following rigid coordinate systems: 
// world, camera (depth and color), voxel, voxel-block coordinates
// all these coordinates cover the whole world and are interconvertible by homogeneous matrix multiplication
// in these spaces, we refer to points and rays

namespace coordinates {


    class Point;
    class Vector;
    class Ray;
    class CoordinateSystem;

    /// Coordinate systems live in managed memory and are identical when their pointers are.
    GLOBAL(CoordinateSystem*, globalcs, 0);

    class CoordinateSystem : public Managed {
    private:
        //CoordinateSystem(const CoordinateSystem&); // TODO should we really allow copying?
        void operator=(const CoordinateSystem&);

        FUNCTION(Point , toGlobalPoint,(Point p)const,"");
        FUNCTION(Point, fromGlobalPoint, (Point p)const, "");
        FUNCTION(Vector, toGlobalVector, (Vector p)const, "");
        FUNCTION(Vector, fromGlobalVector, (Vector p)const, "");
    public:
        const Matrix4f toGlobal;
        const Matrix4f fromGlobal;
        explicit CPU_MEMBERFUNCTION(,CoordinateSystem,(const Matrix4f& toGlobal),"") : toGlobal(toGlobal), fromGlobal(toGlobal.getInv()) {
            assert(toGlobal.GetR().det() != 0);
        }

        /// The world or global space coodinate system.
        /// Measured in meters if cameras and depth computation are calibrated correctly.
        static FUNCTION(CoordinateSystem*, global, (),"") {

            if (!globalcs) {
#if GPU_CODE
                fatalError("Global coordinate system does not yet exist. It cannot be instantiated on the GPU. Aborting.");
#else
                Matrix4f m;
                m.setIdentity();
                globalcs = new CoordinateSystem(m);
#endif
            }

            assert(globalcs);
            return globalcs;
        }

        FUNCTION(Point, convert,(Point p)const, "");
        FUNCTION(Vector, convert,(Vector p)const, "");
        FUNCTION(Ray, convert,(Ray p)const, "");
    };


    // Represents anything that lives in some coordinate system.
    // Entries are considered equal only when they have the same coordinates.
    // They are comparable only if in the same coordinate system.
    class CoordinateEntry {
    public:
        const CoordinateSystem* coordinateSystem;
        friend CoordinateSystem;
        FUNCTION(,CoordinateEntry,(const CoordinateSystem* coordinateSystem), "") : coordinateSystem(coordinateSystem) {}
    };

    // an origin-less direction in some coordinate system
    // not affected by translations
    // might represent a surface normal (if normalized) or the direction from one point to another
    class Vector : public CoordinateEntry {
    private:
        friend Point;
        friend CoordinateSystem;
    public:
        const Vector3f direction;
        // copy constructor ok
        // assignment will not be possible

        explicit FUNCTION(,Vector,(const CoordinateSystem* coordinateSystem, Vector3f direction), "") : CoordinateEntry(coordinateSystem), direction(direction) {
        }
        FUNCTION(bool, operator==,(const Vector& rhs) const, "", PURITY_PURE) {
            assert(coordinateSystem == rhs.coordinateSystem);
            return direction == rhs.direction;
        }
        FUNCTION(Vector, operator*,(const float rhs) const, "", PURITY_PURE) {
            return Vector(coordinateSystem, direction * rhs);
        }
        FUNCTION(float, dot,(const Vector& rhs) const, "", PURITY_PURE) {
            assert(coordinateSystem == rhs.coordinateSystem);
            return vecmath::dot(direction, rhs.direction);
        }
        FUNCTION(Vector, operator-,(const Vector& rhs) const, "", PURITY_PURE){
            assert(coordinateSystem == rhs.coordinateSystem);
            return Vector(coordinateSystem, direction - rhs.direction);
        }
    };

    class Point : public CoordinateEntry {
    private:
        friend CoordinateSystem;
    public:
        const Vector3f location;
        // copy constructor ok

        // Assignment // TODO instead of allowing assignment, rendering Points mutable (!)
        // use a changing reference-to-a-Point instead (a pointer for example)
        FUNCTION(void, operator=,(const Point& rhs), "") {
            coordinateSystem = rhs.coordinateSystem;
            const_cast<Vector3f&>(location) = rhs.location;
        }

        explicit FUNCTION(,Point,(const CoordinateSystem* coordinateSystem, Vector3f location), "") : CoordinateEntry(coordinateSystem), location(location) {
        }
        FUNCTION(bool, operator==, (const Point& rhs) const, "", PURITY_PURE) {
            assert(coordinateSystem == rhs.coordinateSystem);
            return location == rhs.location;
        }
        FUNCTION(Point, operator+, (const Vector& rhs) const, "", PURITY_PURE)  {
            assert(coordinateSystem == rhs.coordinateSystem);
            return Point(coordinateSystem, location + rhs.direction);
        }

        /// Gives a vector that points from rhs to this.
        /// Think 'the location of this as seen from rhs' or 'how to get to this coordinate given one already got to rhs' ('how much energy do we still need to invest in each direction)
        FUNCTION(Vector, operator-, (const Point& rhs) const, "", PURITY_PURE) {
            assert(coordinateSystem == rhs.coordinateSystem);
            return Vector(coordinateSystem, location - rhs.location);
        }
    };

    /// Oriented line segment
    // TODO does this allow scaling or not?
    class Ray {
    public:
        const Point origin;
        const Vector direction;

        FUNCTION(,Ray,(Point& origin, Vector& direction), "") : origin(origin), direction(direction) {
            assert(origin.coordinateSystem == direction.coordinateSystem);
        }
        FUNCTION(Point, endpoint, () const, "", PURITY_PURE) {
            Point p = origin + direction;
            assert(p.coordinateSystem == origin.coordinateSystem);
            return p;
        }
    };


    FUNCTION(Point, CoordinateSystem::toGlobalPoint,(Point p) const, "", PURITY_PURE) {
        return Point(global(), Vector3f(this->toGlobal * Vector4f(p.location, 1)));
    }
    FUNCTION(Point, CoordinateSystem::fromGlobalPoint,(Point p) const, "", PURITY_PURE) {
        assert(p.coordinateSystem == global());
        return Point(this, Vector3f(this->fromGlobal * Vector4f(p.location, 1)));
    }
    FUNCTION(Vector, CoordinateSystem::toGlobalVector,(Vector v) const, "", PURITY_PURE) {
        return Vector(global(), this->toGlobal.GetR() *v.direction);
    }
    FUNCTION(Vector, CoordinateSystem::fromGlobalVector,(Vector v) const, "", PURITY_PURE) {
        assert(v.coordinateSystem == global());
        return Vector(this, this->fromGlobal.GetR() *v.direction);
    }
    FUNCTION(Point, CoordinateSystem::convert,(Point p) const, "", PURITY_PURE) {
        Point o = this->fromGlobalPoint(p.coordinateSystem->toGlobalPoint(p));
        assert(o.coordinateSystem == this);
        return o;
    }
    FUNCTION(Vector, CoordinateSystem::convert, (Vector p) const, "", PURITY_PURE) {
        Vector o = this->fromGlobalVector(p.coordinateSystem->toGlobalVector(p));
        assert(o.coordinateSystem == this);
        return o;
    }
    FUNCTION(Ray, CoordinateSystem::convert, (Ray p) const, "", PURITY_PURE)  {
        return Ray(convert(p.origin), convert(p.direction));
    }


    CPU_FUNCTION(void, initCoordinateSystems, (), "") {
        CoordinateSystem::global(); // access to make sure it exists

    }


}
using namespace coordinates;

























/// Base class storing a camera calibration, eye coordinate system and an image taken with a camera thusly calibrated.
/// Given correct calibration and no scaling, the resulting points are in world-scale, i.e. meter units.
template<typename T>
class CameraImage : public Managed {

private:
    void operator=(const CameraImage& ci);
    CameraImage(const CameraImage&);

public:
    Image<T>*const image; // TODO should the image have to be const?
    const CoordinateSystem* eyeCoordinates; // const ITMPose* const pose // pose->GetM is fromGlobal matrix of coord system; <- inverse is toGlobal // TODO should this encapsulate a copy? // TODO should the eyeCoordinates have to be const?
    const ITMIntrinsics cameraIntrinsics;// const ITMIntrinsics* const cameraIntrinsics;

    CPU_MEMBERFUNCTION(,CameraImage,(
        Image<T>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics),"") :
        image(image), eyeCoordinates(eyeCoordinates), cameraIntrinsics(cameraIntrinsics) {
        //assert(image->noDims.area() > 1); // don't force this, the image we reference is allowed to change later on
    }

    FUNCTION(Vector2ui, imgSize, ()const, "") {
        return image->noDims;
    }

    FUNCTION(Vector4f, projParams, () const, "") {
        return cameraIntrinsics.all;
    }
    /// 0,0,0 in eyeCoordinates
    FUNCTION(Point, location, () const, "") {
        return Point(eyeCoordinates, Vector3f(0, 0, 0));
    }

    /// Returns a ray starting at the camera origin and passing through the virtual camera plane
    /// pixel coordinates must be valid with regards to image size
    FUNCTION(Ray, getRayThroughPixel, (Vector2ui pixel, float depth) const, "") {
        assert(pixel.x >= 0 && pixel.x < image->noDims.width); // imgSize(). causes 1>J:/Masterarbeit/Implementation/InfiniTAM5/main.cu(4691): error : expected a field name -- compiler error?
        assert(pixel.y >= 0 && pixel.y < image->noDims.height);
        Vector4f f = depthTo3D(projParams(), pixel.x, pixel.y, depth);
        assert(f.z == depth);
        return Ray(location(), Vector(eyeCoordinates, f.toVector3()));
    }
#define EXTRA_BOUNDS true
    /// \see project
    /// If extraBounds = EXTRA_BOUNDS is specified, the point is considered to lie outside of the image
    /// if it cannot later be interpolated (the outer bounds are shrinked by one pixel)
    // TODO in a 2x2 image, any point in [0,2]^2 can be given a meaningful, bilinearly interpolated color...
    // did I maybe mean more complex derivatives, for normals?
    /// \returns false when point projects outside of virtual image plane
    FUNCTION(bool, project,(Point p, Vector2f& pt_image, bool extraBounds = false) const, "") {
        Point p_ec = eyeCoordinates->convert(p);
        assert(p_ec.coordinateSystem == eyeCoordinates);
        if (extraBounds)
            return ::projectExtraBounds(projParams(), imgSize(), Vector4f(p_ec.location, 1.f), pt_image);
        else
            return ::project(projParams(), imgSize(), Vector4f(p_ec.location, 1.f), pt_image);
    }
};





/// Constructs getRayThroughPixel endpoints for depths specified in an image.
class DepthImage : public CameraImage<float> {
public:
    CPU_MEMBERFUNCTION(,DepthImage,(
        Image<float>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics), "") :
        CameraImage(image, eyeCoordinates, cameraIntrinsics) {}

    MEMBERFUNCTION(Point, getPointForPixel, (Vector2ui pixel) const, "") {
        float depth = sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize());
        Ray r = getRayThroughPixel(pixel, depth);
        Point p = r.endpoint();
        assert(p.coordinateSystem == eyeCoordinates);
        return p;
    }
};

/// Treats a raster of locations (x,y,z,1) \in Vector4f as points specified in pointCoordinates.
/// The data may have holes, undefined data, which must be Vector4f IllegalColor<Vector4f>::make() (0,0,0,-1 /*only w is checked*/)
///
/// The assumption is that the location in image[x,y] was recorded by
/// intersecting a ray through pixel (x,y) of this camera with something.
/// Thus, the corresponding point lies in the extension of the ray-for-pixel, but possibly in another coordinate system.
///
/// The coordinate system 'pointCoordinates' does not have to be the same as eyeCoordinates, it might be 
/// global() coordinates.
/// getPointForPixel will return a point in pointCoordinates with .location as specified in the image.
///
/// Note: This is a lot different from the depth image, where the assumption is always that the depths are 
/// the z component in eyeCoordinates. Here, the coordinate system of the data in the image (pointCoordinates) can be anything.
///
/// We use this to store intersection points (in world coordinates) obtained by raytracing from a certain camera location.
class PointImage : public CameraImage<Vector4f> {
public:
    CPU_MEMBERFUNCTION(,PointImage,(
        Image<Vector4f>*const image,
        const CoordinateSystem* const pointCoordinates,

        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics), "") :
        CameraImage(image, eyeCoordinates, cameraIntrinsics), pointCoordinates(pointCoordinates) {}

    const CoordinateSystem* const pointCoordinates;

    MEMBERFUNCTION(Point, getPointForPixel, (Vector2ui pixel) const, "") {
        return Point(pointCoordinates, sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize()).toVector3());
    }

    /// Uses bilinear interpolation to deduce points between raster locations.
    /// out_isIllegal is set to true or false depending on whether the given point falls in a 'hole' (undefined/missing data) in the image
    MEMBERFUNCTION(Point, getPointForPixelInterpolated,(Vector2f pixel, bool& out_isIllegal) const, "") {
        out_isIllegal = false;
        // TODO should this always consider holes?
        auto point = interpolateBilinear<Vector4f, WITH_HOLES>(
            image->GetData(),
            pixel,
            imgSize());
        if (!isLegalColor(point)) out_isIllegal = true;
        // TODO handle holes
        return Point(pointCoordinates, point.toVector3());
    }
};

/// Treats a raster of locations and normals as rays, specified in pointCoordinates.
/// Pixel (x,y) is associated to the ray startin at pointImage[x,y] into the direction normalImage[x,y],
/// where both coordinates are taken to be pointCoordinates.
///
/// This data is generated from intersecting rays with a surface.
class RayImage : public PointImage {
public:
    CPU_MEMBERFUNCTION(,RayImage,(
        Image<Vector4f>*const pointImage,
        Image<Vector4f>*const normalImage,
        const CoordinateSystem* const pointCoordinates,

        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics), "") :
        PointImage(pointImage, pointCoordinates, eyeCoordinates, cameraIntrinsics), normalImage(normalImage) {
        assert(normalImage->noDims == pointImage->noDims);
    }
    Image<Vector4f>* const normalImage; // Image may change

    MEMBERFUNCTION(Ray, getRayForPixel, (Vector2ui pixel) const, PURITY_PURE) {
        Point origin = getPointForPixel(pixel);
        auto direction = sampleNearest(normalImage->GetData(), pixel.x, pixel.y, imgSize()).toVector3();
        return Ray(origin, Vector(pointCoordinates, direction));
    }

    /// pixel should have been produced with
    /// project(x,pixel,EXTRA_BOUNDS)
    MEMBERFUNCTION(Ray, getRayForPixelInterpolated, (Vector2f pixel, bool& out_isIllegal) const, "", PURITY_PURE) {
        out_isIllegal = false;
        Point origin = getPointForPixelInterpolated(pixel, out_isIllegal);
        // TODO should this always consider holes?
        auto direction = interpolateBilinear<Vector4f, WITH_HOLES>(
            normalImage->GetData(),
            pixel,
            imgSize());
        if (!isLegalColor(direction)) out_isIllegal = true;
        // TODO handle holes
        return Ray(origin, Vector(pointCoordinates, direction.toVector3()));
    }
};












namespace CoordinateSystemImageTests {

    FUNCTION(void, testCS,(CoordinateSystem* o), "") {
        auto g = CoordinateSystem::global();
        assert(g);

        Point a = Point(o, Vector3f(0.5, 0.5, 0.5));
        assert(a.coordinateSystem == o);
        Point b = g->convert(a);
        assert(b.coordinateSystem == g);
        assert(!(b == Point(g, Vector3f(1, 1, 1))));
        assert((b == Point(g, Vector3f(0.25, 0.25, 0.25))));

        // Thus:
        Point c = o->convert(Point(g, Vector3f(1, 1, 1)));
        assert(c.coordinateSystem == o);
        assert(!(c == Point(o, Vector3f(1, 1, 1))));
        assert((c == Point(o, Vector3f(2, 2, 2))));

        Point d = o->convert(c);
        assert(c == d);

        Point e = o->convert(g->convert(c));
        assert(c == e);
        assert(g->convert(c) == Point(g, Vector3f(1, 1, 1)));

        Point f = o->convert(g->convert(o->convert(c)));
        assert(c == f);

        // +
        Point q = Point(g, Vector3f(1, 1, 2)) + Vector(g, Vector3f(1, 1, 0));
        assert(q.location == Vector3f(2, 2, 2));

        // -
        {
            Vector t = Point(g, Vector3f(0, 0, 0)) - Point(g, Vector3f(1, 1, 2));
            assert(t.direction == Vector3f(-1, -1, -2));
        }

    {
        Vector t = Vector(g, Vector3f(0, 0, 0)) - Vector(g, Vector3f(1, 1, 2));
        assert(t.direction == Vector3f(-1, -1, -2));
    }

    // * scalar
    {
        Vector t = Vector(g, Vector3f(1, 1, 2)) * (-1);
        assert(t.direction == Vector3f(-1, -1, -2));
    }

    // dot (angle if the coordinate system is orthogonal and the vectors unit)
    assert(Vector(o, Vector3f(1, 2, 3)).dot(Vector(o, Vector3f(3, 2, 1))) == 1 * 3 + 2 * 2 + 1 * 3);
    }


#ifdef __CUDACC__
    KERNEL ktestCS(CoordinateSystem* o) {
        testCS(o);
    }

#endif
    GLOBAL(CoordinateSystem*, testedCs, 0, "FOR TESTING");



    TEST(testCS) {
        // o gives points with twice as large coordinates as the global coordinate system
        Matrix4f m;
        m.setIdentity();
        m.setScale(0.5); // scale down by half to get the global coordinates of the point
        auto o = new CoordinateSystem(m);

        testCS(o);
#ifdef __CUDACC__
        LAUNCH_KERNEL(ktestCS, 1, 1, o);
#endif
    }

    FUNCTION(void, testCi,(
        const DepthImage* const di,
        const PointImage* const pi), "") {
        Vector2ui imgSize(640, 480);
        assert(di->location() == Point(testedCs, Vector3f(0, 0, 0)));
        {
            auto r1 = di->getRayThroughPixel(Vector2ui(0, 0), 1);
            assert(r1.origin == Point(testedCs, Vector3f(0, 0, 0)));
            assert(!(r1.direction == Vector(testedCs, Vector3f(0, 0, 1))));

            auto r2 = di->getRayThroughPixel(imgSize / 2, 1);
            assert(r2.origin == Point(testedCs, Vector3f(0, 0, 0)));
            assert(!(r2.direction == r1.direction));
            assert(r2.direction == Vector(testedCs, Vector3f(0, 0, 1)));
        }
    {
        auto r = di->getRayThroughPixel(imgSize / 2, 2);
        assert(r.origin == Point(testedCs, Vector3f(0, 0, 0)));
        assert(r.direction == Vector(testedCs, Vector3f(0, 0, 2)));
    }
    {
        auto r = di->getPointForPixel(Vector2ui(0, 0));
        assert(r == Point(testedCs, Vector3f(0, 0, 0)));
    }
    {
        auto r = di->getPointForPixel(Vector2ui(1, 0));
        assert(!(r == Point(testedCs, Vector3f(0, 0, 0))));
        assert(r.location.z == 1);
        auto ray = di->getRayThroughPixel(Vector2ui(1, 0), 1);
        assert(ray.endpoint() == r);
    }


    assert(pi->location() == Point(testedCs, Vector3f(0, 0, 0)));
    assert(CoordinateSystem::global()->convert(pi->location()) == Point(CoordinateSystem::global(), Vector3f(0, 0, 1)));
    assert(
        testedCs->convert(Point(CoordinateSystem::global(), Vector3f(0, 0, 0)))
        ==
        Point(testedCs, Vector3f(0, 0, -1))
        );

    {
        auto r = pi->getPointForPixel(Vector2ui(0, 0));
        assert(r == Point(testedCs, Vector3f(0, 0, 0)));
    }
    {
        auto r = pi->getPointForPixel(Vector2ui(1, 0));
        assert(r == Point(testedCs, Vector3f(1, 1, 1)));
    }

    Vector2f pt_image;
    assert(pi->project(Point(CoordinateSystem::global(), Vector3f(0, 0, 2)), pt_image));
    assert(pt_image == (1 / 2.f) * imgSize.toFloat());// *(1 / 2.f));

    assert(pi->project(Point(di->eyeCoordinates, Vector3f(0, 0, 1)), pt_image));
    assert(pt_image == (1 / 2.f) * imgSize.toFloat());// *(1 / 2.f));
    assert(!pi->project(Point(CoordinateSystem::global(), Vector3f(0, 0, 0)), pt_image));

    assert(Point(di->eyeCoordinates, Vector3f(0, 0, 1))
        ==
        di->eyeCoordinates->convert(Point(CoordinateSystem::global(), Vector3f(0, 0, 2)))
        );
    }


#ifdef __CUDACC__

    KERNEL ktestCi(
        const DepthImage* const di,
        const PointImage* const pi) {

        testCi(di, pi);
    }
#endif
    TEST(testCameraImage) {
        ITMIntrinsics intrin;
        Vector2ui imgSize(640, 480);
        auto depthImage = new ITMFloatImage(imgSize);
        auto pointImage = new ITMFloat4Image(imgSize);

        depthImage->GetData()[1] = 1;
        pointImage->GetData()[1] = Vector4f(1, 1, 1, 1);
        // must submit manually

        Matrix4f cameraToWorld;
        cameraToWorld.setIdentity();
        cameraToWorld.setTranslate(Vector3f(0, 0, 1));
        testedCs = new CoordinateSystem(cameraToWorld);
        auto di = new DepthImage(depthImage, testedCs, intrin);
        auto pi = new PointImage(pointImage, testedCs, testedCs, intrin);

        testCi(di, pi);

        // must submit manually

#ifdef __CUDACC__
        LAUNCH_KERNEL(ktestCi, 1, 1, di, pi);
#endif
    }

}





















































































// GPU HashMap

// Forward declarations
template<typename Hasher, typename AllocCallback> class HashMap;

#ifdef __CUDACC__
template<typename Hasher, typename AllocCallback>
KERNEL performAllocationKernel(typename HashMap<Hasher, AllocCallback>* hashMap);
#endif

#define hprintf(...) //printf(__VA_ARGS__) // enable for verbose radio debug messages

struct VoidSequenceIdAllocationCallback {
    template<typename T>
    static FUNCTION(void, allocate, (T, int sequenceId), "") {}
};
/**
Implements an unordered_map, i.e.

    key |-> sequence# > 0

mapping, usable on the GPU, where keys for which allocation is requested get assigned unique,
consecutive unsigned integer numbers ('sequence numbers')
starting at 1.

These sequence numbers might be used to index into a pre-allocated list of objects.
This can be used to store sparse key-value data.

Stores at most Hasher::BUCKET_NUM + EXCESS_NUM - 1 entries.

After a series of
requestAllocation(key)
calls,
performAllocations()
must be called to make
getSequenceId(key)
return a unique nonzero value for key.

*Allocation is not guaranteed in only one requestAllocation(key) -> performAllocations() cycle*
At most one entry will be allocated per hash(key) in one such cycle.

c.f. ismar15infinitam.pdf

Note: As getSequenceId(key) reads from global memory it might be advisable to cache results,
especially when it is expected that the same entry is accessed multiple times from the same thread.

In particular, when you index into a custom large datastructure with the result of this, you might want to copy
the accessed data to __shared__ memory for optimum performance.

TODO Can we provide this functionality from here? Maybe force the creation of some object including a cache to access this.
TODO make public
TODO allow freeing entries
TODO allow choosing reliable allocation (using atomics...)
*/
/* Implementation: See HashMap.png */
template<
    typename Hasher, //!< must have static CPU_AND_GPU function unsigned int Hasher::hash(const KeyType&) which generates values from 0 to Hasher::BUCKET_NUM-1 
    typename SequenceIdAllocationCallback = VoidSequenceIdAllocationCallback //!< must have static CPU_AND_GPU void  allocate(KeyType k, int sequenceId) function
>
class HashMap : public Managed {
public:
    typedef typename Hasher::KeyType KeyType;

private:
    static const unsigned int BUCKET_NUM = Hasher::BUCKET_NUM;
    const unsigned int EXCESS_NUM;
    MEMBERFUNCTION(unsigned int, NUMBER_TOTAL_ENTRIES, () const, "", PURITY_PURE) {
        return (BUCKET_NUM + EXCESS_NUM);
    }

    struct HashEntry {
    public:
        typedef typename Hasher::KeyType KeyType;
        MEMBERFUNCTION(bool, isAllocated, ()const, "", PURITY_PURE, TOTALITY_TOTAL) {
            return sequenceId != 0;
        }
        MEMBERFUNCTION(bool, hasNextExcessList, () const, "", PURITY_PURE){
            assert(isAllocated());
            return nextInExcessList != 0;
        }
        MEMBERFUNCTION(unsigned int, getNextInExcessList, ()const, "", PURITY_PURE) {
            assert(hasNextExcessList() && isAllocated());
            return nextInExcessList;
        }

        MEMBERFUNCTION(bool, hasKey, (const KeyType& key) const,
            "undefined for unallocated entries", PURITY_PURE) {
            assert(isAllocated());
            return this->key == key;
        }

        MEMBERFUNCTION(void, linkToExcessListEntry, (const unsigned int excessListId), "") {
            assert(!hasNextExcessList() && isAllocated() && excessListId >= 1);// && excessListId < EXCESS_NUM);
            // also, the excess list entry should exist and this should be the only entry linking to it
            // all entries in the excess list before this one should be allocated
            nextInExcessList = excessListId;
        }

        MEMBERFUNCTION(void, allocate, (const KeyType& key, const unsigned int sequenceId), "") {
            assert(!isAllocated() && 0 == nextInExcessList);
            assert(sequenceId > 0);
            this->key = key;
            this->sequenceId = sequenceId;

            SequenceIdAllocationCallback::allocate(key, sequenceId);

            hprintf("allocated %d\n", sequenceId);
        }

        MEMBERFUNCTION(unsigned int, getSequenceId, () const, "", PURITY_PURE) {
            assert(isAllocated());
            return sequenceId;
        }

    private:
        KeyType key;
        /// any of 1 to lowestFreeExcessListEntry-1
        /// 0 means this entry ends a list of excess entries and/or is not allocated
        unsigned int nextInExcessList;
        /// any of 1 to lowestFreeSequenceNumber-1
        /// 0 means this entry is not allocated
        unsigned int sequenceId;
    };

public:
    /// BUCKET_NUM + EXCESS_NUM many, information for the next round of allocations
    /// Note that equal hashes will clash only once

    /// Whether the corresponding entry should be allocated
    /// 0 or 1 
    /// TODO could save memory (&bandwidth) by using a bitmap
    MemoryBlock<unsigned char> needsAllocation;

    /// With which key the corresponding entry should be allocated
    /// TODO if there where an 'illegal key' entry, the above would not be needed. However, we would need to read 
    // a full key instead of 1 byte in more threads.
    /// State undefined for entries that don't need allocation
    MemoryBlock<KeyType> naKey;

    // TODO how much does separating allocation and requesting allocation really help?

    //private:
    /// BUCKET_NUM + EXCESS_NUM many
    /// Indexed by Hasher::hash() return value
    // or BUCKET_NUM + HashEntry.nextInExcessList (which is any of 1 to lowestFreeExcessListEntry-1)
    // excessListEntry 0 is unused (guard)
    MemoryBlock<HashEntry> hashMap_then_excessList;

private:
    // abstract away how to access the storage
    // hashMap_then_excessList should not be accessed explicitly beyond this point
    MEMBERFUNCTION(HashEntry&, hashMap, (const unsigned int hash), "") {
        assert(hash < BUCKET_NUM);
        return hashMap_then_excessList[hash];
    }
    MEMBERFUNCTION(HashEntry const &, hashMap, (const unsigned int hash) const, "") {
        assert(hash < BUCKET_NUM);
        return hashMap_then_excessList[hash];
    }
    MEMBERFUNCTION(HashEntry&, excessList, (const unsigned int excessListEntry), "") {
        assert(excessListEntry >= 1 && excessListEntry < EXCESS_NUM);
        return hashMap_then_excessList[BUCKET_NUM + excessListEntry];
    }
    MEMBERFUNCTION(HashEntry const &, excessList, (const unsigned int excessListEntry) const, "") {
        assert(excessListEntry >= 1 && excessListEntry < EXCESS_NUM);
        return hashMap_then_excessList[BUCKET_NUM + excessListEntry];
    }


    /// Sequence numbers already used up. Starts at 1 (sequence number 0 is used to signify non-allocated)
    unsigned int lowestFreeSequenceNumber;

    /// Excess list slots already used up. Starts at 1 (one safeguard entry)
    unsigned int lowestFreeExcessListEntry;

    /// Follows the excess list starting at hashMap[Hasher::hash(key)]
    /// until either hashEntry.key == key, returning true
    /// or until hashEntry does not exist or hashEntry.key != key but there is no further entry, returns false in that case.
    MEMBERFUNCTION(bool, findEntry, (
        _In_ const KeyType& key,
        _Out_opt_ HashEntry& hashEntry,
        _Out_opt_ unsigned int& hashMap_then_excessList_entry
        ) const, "", PURITY_PURE) {

        hashMap_then_excessList_entry = Hasher::hash(key);
        hprintf("%d %d\n", hashMap_then_excessList_entry, BUCKET_NUM);
        assert(hashMap_then_excessList_entry < BUCKET_NUM);
        hashEntry = hashMap(hashMap_then_excessList_entry);

        if (!hashEntry.isAllocated()) return false;
        if (hashEntry.hasKey(key)) return true;

        // try excess list
        int safe = 0;
        while (hashEntry.hasNextExcessList()) {
            hashEntry = excessList(hashMap_then_excessList_entry = hashEntry.getNextInExcessList());
            hashMap_then_excessList_entry += BUCKET_NUM; // the hashMap_then_excessList_entry must include the offset by BUCKET_NUM
            if (hashEntry.hasKey(key)) return true;
            if (safe++ > 100) fatalError("excessive amount of steps in excess list, probably cyclic structure created");
        }
        return false;
    }

    MEMBERFUNCTION(void, allocate, (HashEntry& hashEntry, const KeyType & key), "") {
#if GPU_CODE
        hashEntry.allocate(key, atomicAdd(&lowestFreeSequenceNumber, 1));
#else /* assume single-threaded cpu */
        hashEntry.allocate(key, lowestFreeSequenceNumber++);
#endif
    }

#ifdef __CUDACC__
    friend KERNEL performAllocationKernel<Hasher, SequenceIdAllocationCallback>(typename HashMap<Hasher, SequenceIdAllocationCallback>* hashMap);
#endif

    /// Given a key that does not yet exist, find a location in the hashMap_then_excessList
    /// that can be used to insert the key (or is the end of the current excess list for the keys with the same hash as this)
    /// returns (unsigned int)-1 if the key already exists
    MEMBERFUNCTION(unsigned int, findLocationForKey, (_In_ const KeyType& key) const, "", PURITY_PURE) {
        hprintf("findLocationForKey \n");

        HashEntry hashEntry;
        unsigned int hashMap_then_excessList_entry;

        const bool alreadyExists = findEntry(key, hashEntry, hashMap_then_excessList_entry);
        if (alreadyExists) {
            hprintf("already exists\n");
            return -1;
        }
        hprintf("request goes to %d\n", hashMap_then_excessList_entry);

        assert(hashMap_then_excessList_entry != BUCKET_NUM &&
            hashMap_then_excessList_entry < NUMBER_TOTAL_ENTRIES());
        return hashMap_then_excessList_entry;
    }

    /// hashMap_then_excessList_entry is an index into hashMap_then_excessList that is either free or the current
    /// end of the excess list for keys with the same hash as key.
    /// \returns the sequence number > 0 of the newly allocated entry, (unsigned int)-1 if an error occurred
    MEMBERFUNCTION(unsigned int, performSingleAllocation, (
        _In_ const KeyType& key, _In_ const unsigned int hashMap_then_excessList_entry), "") {
        if (hashMap_then_excessList_entry == (unsigned int)-1) return -1;
        if (hashMap_then_excessList_entry >= NUMBER_TOTAL_ENTRIES()) return -1;

        // Allocate in place if not allocated
        HashEntry& hashEntry = hashMap_then_excessList[hashMap_then_excessList_entry];

        unsigned int sequenceId;
        if (!hashEntry.isAllocated()) {
            hprintf("not allocated\n", hashMap_then_excessList_entry);
            allocate(hashEntry, key);
            sequenceId = hashEntry.getSequenceId();
            goto done;
        }

        hprintf("hashEntry %d\n", hashEntry.getSequenceId());

        // If existing, allocate new and link parent to child
#if GPU_CODE
        const unsigned int excessListId = atomicAdd(&lowestFreeExcessListEntry, 1);
#else /* assume single-threaded cpu code */
        const unsigned int excessListId = lowestFreeExcessListEntry++;
#endif

        HashEntry& newHashEntry = excessList(excessListId);
        assert(!newHashEntry.isAllocated());
        hashEntry.linkToExcessListEntry(excessListId);
        assert(hashEntry.getNextInExcessList() == excessListId);

        allocate(newHashEntry, key);
        hprintf("newHashEntry.getSequenceId() = %d\n", newHashEntry.getSequenceId());
        sequenceId = newHashEntry.getSequenceId();

    done:
#ifdef _DEBUG
        // we should now find this entry:
        HashEntry e; unsigned int _;
        bool found = findEntry(key, e, _);
        assert(found && e.getSequenceId() > 0);
        assert(e.getSequenceId() == sequenceId);
        hprintf("%d = findEntry(), e.seqId = %d\n", found, e.getSequenceId());
#endif
        return sequenceId;
    }

    /// Perform allocation per-thread function, extracting key from naKey, then using performSingleAllocation 
    /// at the known _entry location
    /// Should only be called by performAllocation(s)(Kernel)
    /// for each hashMap_then_excessList_entry
    MEMBERFUNCTION(void, performAllocation, (const unsigned int hashMap_then_excessList_entry), "") {
        if (hashMap_then_excessList_entry >= NUMBER_TOTAL_ENTRIES()) return;
        if (!needsAllocation[hashMap_then_excessList_entry]) return;
        assert(hashMap_then_excessList_entry != BUCKET_NUM); // never allocate guard
        hprintf("performAllocation %d\n", hashMap_then_excessList_entry);


        needsAllocation[hashMap_then_excessList_entry] = false;
        KeyType key = naKey[hashMap_then_excessList_entry];

        const auto sid = performSingleAllocation(key, hashMap_then_excessList_entry);
        assert(sid > 0);
    }


public:
    CPU_MEMBERFUNCTION(, HashMap, (const unsigned int EXCESS_NUM //<! must be at least one
        ), "") : EXCESS_NUM(EXCESS_NUM),
        needsAllocation(NUMBER_TOTAL_ENTRIES()),
        naKey(NUMBER_TOTAL_ENTRIES()),
        hashMap_then_excessList(NUMBER_TOTAL_ENTRIES())

    {
        assert(EXCESS_NUM >= 1);
#if __CUDACC__
        cudaDeviceSynchronize();
#endif

        lowestFreeSequenceNumber = lowestFreeExcessListEntry = 1;
    }

    MEMBERFUNCTION(
        unsigned int, getLowestFreeSequenceNumber,() const, "", PURITY_PURE, TOTALITY_TOTAL) {
#if defined(__CUDACC__) && !GPU_CODE
        cudaDeviceSynchronize();
#endif
        return lowestFreeSequenceNumber;
    }
    /*
    unsigned int countAllocatedEntries() {
    return getLowestFreeSequenceNumber() - 1;
    }
    */

    // TODO should this ad-hoc crude serialization be part of this class?

    CPU_MEMBERFUNCTION(void, serialize, (ofstream& file), "", PURITY_PURE) {
        bin(file, NUMBER_TOTAL_ENTRIES());
        bin(file, EXCESS_NUM);
        bin(file, lowestFreeSequenceNumber);
        bin(file, lowestFreeExcessListEntry);

        needsAllocation.serialize(file);
        naKey.serialize(file);
        hashMap_then_excessList.serialize(file);
    }

    /*
    reads from the binary file:
    - lowestFreeSequenceNumber
    - lowestFreeExcessListEntry
    - needsAllocation (ideally this is not in a dirty state currently, i.e. all 0)
    - naKey (ditto)
    - hashMap_then_excessList
    version and size of these structures in the file must match (full binary dump)
    */
    // loses current data
    CPU_MEMBERFUNCTION(void, deserialize, (ifstream& file), "", PURITY_OUTPUT_POINTERS) {
        assert(NUMBER_TOTAL_ENTRIES() == bin<unsigned int>(file));
        assert(EXCESS_NUM == bin<unsigned int>(file));
        bin(file, lowestFreeSequenceNumber);
        bin(file, lowestFreeExcessListEntry);

        needsAllocation.deserialize(file);
        naKey.deserialize(file);
        hashMap_then_excessList.deserialize(file);
    }

    /**
    Requests allocation for a specific key.
    Only one request can be made per hash(key) before performAllocations must be called.
    Further requests will be ignored.
    */
    MEMBERFUNCTION(void, requestAllocation, (const KeyType& key), "") {
        hprintf("requestAllocation \n");

        const unsigned int hashMap_then_excessList_entry = findLocationForKey(key);

        if (hashMap_then_excessList_entry == (unsigned int)-1) {
            hprintf("already exists\n");
            return;
        }

        assert(hashMap_then_excessList_entry != BUCKET_NUM &&
            hashMap_then_excessList_entry < NUMBER_TOTAL_ENTRIES());

        // not strictly necessary, ordering is random anyways
        if (needsAllocation[hashMap_then_excessList_entry]) {
            hprintf("already requested\n");
            return;
        }

        needsAllocation[hashMap_then_excessList_entry] = true;
        naKey[hashMap_then_excessList_entry] = key;
    }

    // during performAllocations
#define THREADS_PER_BLOCK 256 // TODO which value works best?

    /**
    Allocates entries that requested allocation. Allocates at most one entry per hash(key).
    Further requests can allocate colliding entries.
    */
    CPU_MEMBERFUNCTION(void, performAllocations, (), "") {

#ifdef __CUDACC__
        //cudaSafeCall(cudaGetError());
        cudaSafeCall(cudaDeviceSynchronize()); // Managed this is not accessible when still in use?
        LAUNCH_KERNEL(performAllocationKernel, // Note: trivially parallelizable for-each type task
            /// Scheduling strategy: Fixed number of threads per block, working on all entries (to find those that have needsAllocation set)
            (unsigned int)ceil(NUMBER_TOTAL_ENTRIES() / (1. * THREADS_PER_BLOCK)),
            THREADS_PER_BLOCK,
            this);
#ifdef _DEBUG
        cudaSafeCall(cudaDeviceSynchronize());  // detect problems (failed assertions) early where this kernel is called
#endif
        cudaSafeCall(cudaGetLastError());
#else
        DO(hashMap_then_excessList_entry, NUMBER_TOTAL_ENTRIES())
            performAllocation(hashMap_then_excessList_entry);
#endif
    }

    /// Allocate and assign a sequence number for the given key.
    /// Note: Potentially slower than requesting a whole bunch, then allocating all at once, use as fallback.
    /// \returns the sequence number > 0 of the newly allocated entry or (unsigned int)-1 if an error occurred
    MEMBERFUNCTION(unsigned int, performSingleAllocation, (const KeyType& key), "") {
        return performSingleAllocation(key, findLocationForKey(key));
    }

    MEMBERFUNCTION(unsigned int, getSequenceNumber, (const KeyType& key) const,
        "\returns 0 if the key is not allocated, otherwise something greater than 0 unique to this key and less than getLowestFreeSequenceNumber()") {
        HashEntry hashEntry; unsigned int _;
        if (!findEntry(key, hashEntry, _)) return 0;
        return hashEntry.getSequenceId();
    }
};

#ifdef __CUDACC__
template<typename Hasher, typename AllocCallback>
KERNEL performAllocationKernel(typename HashMap<Hasher, AllocCallback>* hashMap) {
    assert(blockDim.x == THREADS_PER_BLOCK && blockDim.y == 1 && blockDim.z == 1);
    assert(
        gridDim.x*blockDim.x >= hashMap->NUMBER_TOTAL_ENTRIES() && // all entries covered
        gridDim.y == 1 &&
        gridDim.z == 1);
    assert(linear_global_threadId() == blockIdx.x*THREADS_PER_BLOCK + threadIdx.x);
    hashMap->performAllocation(blockIdx.x*THREADS_PER_BLOCK + threadIdx.x);
}
#endif





template<typename T>
struct Z3Hasher {
    typedef T KeyType;
    static const unsigned int BUCKET_NUM = 0x1000; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

    static FUNCTION(unsigned int, hash, (const T& blockPos),
        "computes a hash for the given (integral) block position. Tolerates 'overflows' assuming the implementation does it", PURITY_PURE, TOTALITY_TOTAL) {
        return (((unsigned int)blockPos.x * 73856093u) ^ ((unsigned int)blockPos.y * 19349669u) ^ ((unsigned int)blockPos.z * 83492791u))
            & // optimization - has to be % if BUCKET_NUM is not a power of 2 // TODO can the compiler not figure this out?
            (unsigned int)(BUCKET_NUM - 1);
    }
};

namespace HashMapTests {

    GLOBAL(int, sequenceNumberToGet, 0, "testing");
    FUNCTION(void, get, (HashMap<Z3Hasher<Vector3s>>* myHash, Vector3s /* Z3Hasher<Vector3s>::KeyType*/ q), "") {
        sequenceNumberToGet = myHash->getSequenceNumber(q);
    }

    FUNCTION(void, alloc, (HashMap<Z3Hasher<Vector3s>>* myHash, int p),"") {
        myHash->requestAllocation(p);
    }
    
#ifdef __CUDACC__
    KERNEL Kget(HashMap<Z3Hasher<Vector3s>>* myHash, Vector3s q) {
        get(myHash, q);
    }

    KERNEL Kalloc(HashMap<Z3Hasher<Vector3s>>* myHash) {
        int p = blockDim.x * blockIdx.x + threadIdx.x;
        alloc(myHash, p);
    }
#endif

    TEST(testZ3Hasher) {
        // insert a lot of points (n) into a large hash just for fun
        HashMap<Z3Hasher<Vector3s>>* myHash = new HashMap<Z3Hasher<Vector3s>>(0x2000);

        
        const int n = 1000;
#ifdef __CUDACC__
        LAUNCH_KERNEL(Kalloc, n, 1, myHash);
#else
        DO(i, n) alloc(myHash, i);
#endif
        myHash->performAllocations();
        puts("after alloc");
        // should be some permutation of 1:n
        vector<bool> found; found.resize(n + 1);

        for (int i = 0; i < n; i++) {
#ifdef __CUDACC__
            LAUNCH_KERNEL(Kget,
                1, 1,
                myHash, Vector3s(i, i, i));
#else
            get(myHash, Vector3s(i, i, i));
#endif
            printf("Vector3s(%i,%i,%i) -> %d\n", i, i, i, sequenceNumberToGet);

            assert(!found[sequenceNumberToGet]);
            found[sequenceNumberToGet] = 1;
        }

    }

    
    // n hasher test suite
    // trivial hash function n -> n
    struct NHasher{
        typedef int KeyType;
        static const unsigned int BUCKET_NUM = 1; // can play with other values, the tests should support it
        static FUNCTION(unsigned int, hash,(const int& n),"",PURITY_PURE, TOTALITY_TOTAL) {
            return n % BUCKET_NUM;//& (BUCKET_NUM-1);
        }
    };

    FUNCTION(void, get, (HashMap<NHasher>* myHash, NHasher::KeyType q), "") {
        sequenceNumberToGet = myHash->getSequenceNumber(q);
    }

    FUNCTION(void, alloc, (HashMap<NHasher>* myHash, int p), "") {
        myHash->requestAllocation(p);
    }

#ifdef __CUDACC__
    KERNEL Kget(HashMap<NHasher>* myHash, NHasher::KeyType q) {
        get(myHash, q);
    }

    KERNEL Kalloc(HashMap<NHasher>* myHash, int p) {
        alloc(myHash, p);
    }
#endif

    TEST(testNHasher) {
        int n = NHasher::BUCKET_NUM;
        auto myHash = new HashMap<NHasher>(1 + 1); // space for BUCKET_NUM entries only, and 1 collision handling entry


        for (int i = 0; i < n; i++) {
#ifdef __CUDACC__
            LAUNCH_KERNEL(Kalloc,
                1, 1,
                myHash, i);
#else
            alloc(myHash, i);
#endif
        }
        myHash->performAllocations();

        // an additional alloc at another key not previously seen (e.g. BUCKET_NUM) 
        // this will use the excess list

#ifdef __CUDACC__
        LAUNCH_KERNEL(Kalloc, 1, 1, myHash, NHasher::BUCKET_NUM); 
#else
        alloc(myHash, NHasher::BUCKET_NUM);
#endif
        myHash->performAllocations();

        // an additional alloc at another key not previously seen (e.g. BUCKET_NUM + 1) makes it crash cuz no excess list space is left
        //alloc << <1, 1 >> >(myHash, NHasher::BUCKET_NUM + 1, p);

        myHash->performAllocations(); // performAllocations is always fine to call when no extra allocations where made

        puts("after alloc");
        // should be some permutation of 1:BUCKET_NUM
        bool found[NHasher::BUCKET_NUM + 1] = {0};
        for (int i = 0; i < n; i++) {
#ifdef __CUDACC__
            LAUNCH_KERNEL(Kget, 1, 1, myHash, i);
#else
            get(myHash, i);
#endif

            printf("%i -> %d\n", i, sequenceNumberToGet);
            assert(!found[sequenceNumberToGet]);
            //assert(*p != i+1); // numbers are very unlikely to be in order -- nah it happens
            found[sequenceNumberToGet] = 1;
        }

    }
    
    // zero hasher test suite
    // trivial hash function with one bucket.
    // This will allow the allocation of only one block at a time
    // and all blocks will be in the same list.
    // The numbers will be in order.
    struct ZeroHasher{
        typedef int KeyType;
        static const unsigned int BUCKET_NUM = 0x1;
        static FUNCTION(unsigned int, hash,(const int&),"C59 constant 0",PURITY_PURE,TOTALITY_TOTAL) { return 0; }
    };

    FUNCTION(void, get, (HashMap<ZeroHasher>* myHash, ZeroHasher::KeyType q), "") {
        sequenceNumberToGet = myHash->getSequenceNumber(q);
    }

    FUNCTION(void, alloc,(HashMap<ZeroHasher>* myHash, int p), "") {
        myHash->requestAllocation(p);
    }

#ifdef __CUDACC__
    KERNEL Kget(HashMap<ZeroHasher>* myHash, int q) {
        get(myHash, q);
    }

    KERNEL Kalloc(HashMap<ZeroHasher>* myHash, int p) {
        alloc(myHash, p);
    }
#endif

    TEST(testZeroHasher) {
        int n = 10;
        auto myHash = new HashMap<ZeroHasher>(n); // space for BUCKET_NUM(1) + excessnum(n-1) = n entries
        assert(myHash->getLowestFreeSequenceNumber() == 1);
        
        const int extra = 0; // doing one more will crash it at
        // Assertion `excessListEntry >= 1 && excessListEntry < EXCESS_NUM` failed.

        // Keep requesting allocation until all have been granted
        for (int j = 0; j < n + extra; j++) { // request & perform alloc cycle
            for (int i = 0; i < n + extra
                ; i++) {
#ifdef __CUDACC__
                LAUNCH_KERNEL(Kalloc, 1, 1, myHash, i); // only one of these allocations will get through at a time
#else
                alloc(myHash, i);
#endif
            }
            myHash->performAllocations();

            puts("after alloc");
            for (int i = 0; i < n; i++) {
#ifdef __CUDACC__
                LAUNCH_KERNEL(Kget, 1, 1, myHash, i);
#else
                get(myHash, i);
#endif
                printf("%i -> %d\n", i, sequenceNumberToGet);
                // expected result
                assert(i <= j ? sequenceNumberToGet == i + 1 : sequenceNumberToGet == 0);
            }
        }

        assert(myHash->getLowestFreeSequenceNumber() != 1);
    }

}








































// voxel coordinate systems [

/// Size of a voxel, usually given in meters.
/// In world space coordinates. 

#define voxelSize (Scene::getCurrentScene()->getVoxelSize()) // 0.005f
#define oneOverVoxelSize (1.0f / voxelSize)

/** @} */
/** \brief
Encodes the world-space width of the band of the truncated
signed distance transform that is actually stored
in the volume. This is again usually specified in
meters (world coordinates).
Note that thus, the resulting width in voxels is @ref mu
divided by @ref voxelSize (times two -> on both sides of the surface).
Also, a voxel storing the value 1 has world-space-distance mu from the surface.
(the stored -1 to 1 SDF values are understood as fractions of mu)

Must be greater than voxelSize -> defined automatically from voxelSize
*/
#define voxelSize_to_mu(vs) (4*vs)// TODO is this heuristic ok?
#define mu voxelSize_to_mu(voxelSize)//0.02f

/**
Size of the thin shell region for volumetric refinement-from-shading computation.
Must be smaller than mu and should be bigger than voxelSize

In world space coordinates (meters).
*/
#define t_shell (mu/2.f)// TODO is this heuristic ok?


/// In world space coordinates.
#define voxelBlockSize (voxelSize*SDF_BLOCK_SIZE)
#define oneOverVoxelBlockWorldspaceSize (1.0f / (voxelBlockSize))



/// (0,0,0) is the lower corner of the first voxel *block*, (1,1,1) its upper corner,
/// voxelBlockCoordinate (1,1,1) corresponds to (voxelBlockSize, voxelBlockSize, voxelBlockSize) in world coordinates.
#define voxelBlockCoordinates (Scene::getCurrentScene()->voxelBlockCoordinates_)

/// (0,0,0) is the lower corner of the voxel, (1,1,1) its upper corner,
/// a position corresponding to (voxelSize, voxelSize, voxelSize) in world coordinates.
/// aka "voxel-fractional-world-coordinates"
#define voxelCoordinates (Scene::getCurrentScene()->voxelCoordinates_)


// ]















/**
Voxel block coordinates.

This is the coarsest integer grid laid over our 3d space.

Multiply by SDF_BLOCK_SIZE to get voxel coordinates,
and then by ITMSceneParams::voxelSize to get world coordinates.

using short (Vector3*s*) to reduce storage requirements of hash map // TODO could use another type for accessing convenience/alignment speed
*/
typedef Vector3s VoxelBlockPos;
// Default voxel block pos, used for debugging
#define INVALID_VOXEL_BLOCK_POS Vector3s(SHRT_MIN, SHRT_MIN, SHRT_MIN)















//////////////////////////////////////////////////////////////////////////
// Voxel Hashing definition and helper functions
//////////////////////////////////////////////////////////////////////////

// amount of voxels along one side of a voxel block
#define SDF_BLOCK_SIZE 8
/*
// COMPILE TIME CONST
CPU_AND_GPU_CONSTANT unsigned int SDF_BLOCK_SIZE = 8; // > 0
static_assert(SDF_BLOCK_SIZE > 0, "must be positive");
*/

// a common loop

#define XYZ_over(xmax,ymax,zmax) DO(z, zmax) DO(y, ymax) DO(x, xmax)

// zyx order - this matches the order in 'blockVoxels' of the voxel block, so this is more cache efficient
#define XYZ_over_SDF_BLOCK_SIZE XYZ_over(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE)

// use this version when you explicitly care about the linearization order -- less cache efficient
#define XYZ_over_xyz_order(xmax,ymax,zmax) DO(x, xmax) DO(y, ymax) DO(z, zmax) 

#define XYZ_over_SDF_BLOCK_SIZE_xyz_order XYZ_over_xyz_order(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE)


// SDF_BLOCK_SIZE^3, amount of voxels in a voxel block
#define SDF_BLOCK_SIZE3 (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)

// (Maximum) Number of actually stored blocks (i.e. maximum load the hash-table can actually have -- yes, we can never fill all buckets)
// is much smaller than SDF_BUCKET_NUM for efficiency reasons. 
// doesn't make sense for this to be bigger than SDF_GLOBAL_BLOCK_NUM
// localVBA's size
// memory hog if too large, main limitation for scene size
#define SDF_LOCAL_BLOCK_NUM  0x8000 //0x10000	//	0x40000

#define SDF_BUCKET_NUM 0x100000			// Number of Hash Bucket, must be 2^n and bigger than SDF_LOCAL_BLOCK_NUM
#define SDF_HASH_MASK (SDF_BUCKET_NUM-1)// Used for get hashing value of the bucket index, "x & (uint)SDF_HASH_MASK" is the same as "x % SDF_BUCKET_NUM"
#define SDF_EXCESS_LIST_SIZE 0x20000	// Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

// TODO rename to SDF_TOTAL_HASH_POSITIONS
#define SDF_GLOBAL_BLOCK_NUM (SDF_BUCKET_NUM+SDF_EXCESS_LIST_SIZE)	// Number of globally stored blocks == size of ordered + unordered part of hash table












































/** \brief
Stores the information of a single voxel in the volume
*/
class ITMVoxel
{
private:
    // signed distance, fixed comma 16 bit int, converted to snorm [-1, 1] range as described by OpenGL standard (?)/terminology as in DirectX11
    // saving storage
    short sdf;  // aka. 'doriginal, D()'
public:
    /** Value of the truncated signed distance transformation, in [-1, 1] (scaled by truncation distance mu when storing) */
    FUNCTION(void, setSDF_initialValue,(),"",PURITY_OUTPUT_POINTERS) { sdf = 32767; }
    FUNCTION(float, getSDF, () const, "", PURITY_PURE) { return (float)(sdf) / 32767.0f; }
    FUNCTION(void, setSDF,(float x), "", PURITY_OUTPUT_POINTERS) {
        assert(x >= -1 && x <= 1);
        sdf = (short)((x)* 32767.0f);
    }

    /** Number of fused observations that make up @p sdf. */
    unsigned char w_depth;

    /** RGB colour information stored for this voxel, 0-255 per channel (optimizing storage). */
    Vector3u clr; // C(v) 

    /** Number of observations that made up @p clr. */
    unsigned char w_color;

    // for vsfs:

    //! unknowns of our objective
    float luminanceAlbedo; // a(v)
    float refinedDistance; // D'(v), \tilde D(v), 'd', 'd-refined', refinedSDF // NOTE: we don't care about optimizing storage here yet

    // chromaticity and intensity are
    // computed from C(v) on-the-fly -- TODO visualize those two

    // luminance as average of rgb
    // \in [0,1]
    FUNCTION(float, intensity, () const, "", PURITY_PURE) {
        // TODO is this how luminance should be computed?
        Vector3f color = clr.toFloat() / 255.f;
        return (color.r + color.g + color.b) / 3.f;
    }

    /// \f$\Gamma(v)\f$
    // \in [0,255]^3
    FUNCTION(Vector3f, chromaticity,() const, "", PURITY_PURE) {
        return clr.toFloat() / intensity();
    }


    // NOTE not used inially when memory is just allocated and reinterpreted, but used on each allocation
    FUNCTION(,ITMVoxel,(),"")
    {
        setSDF_initialValue();
        w_depth = 0;
        clr = (unsigned char)0;
        w_color = 0;

        // start with constant white albedo
        //luminanceAlbedo = 1.f; // nah, keep uninitialized until initAD, init both to be sure
    }
};

struct ITMVoxelBlock {
    FUNCTION(void, resetVoxels, (), "") {
        for (auto& i : blockVoxels) i = ITMVoxel();
    }

    /// compute voxelLocalId to access blockVoxels
    // TODO Vector3i is too general for the tightly limited range of valid values, c.f. assert statements below
    // TODO unify and document the use of 'localPos'/'globalPos' variable names
    FUNCTION(ITMVoxel const *, getVoxel,(Vector3ui localPos) const, "", PURITY_PURE) {
        assert(localPos.x < SDF_BLOCK_SIZE);
        assert(localPos.y < SDF_BLOCK_SIZE);
        assert(localPos.z < SDF_BLOCK_SIZE);

        return &blockVoxels[
            // Note that x changes fastest here, while in a mathematica 3D array with indices {x,y,z}
            // x changes slowest!
            localPos.x + localPos.y * SDF_BLOCK_SIZE + localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
        ];
    }

    FUNCTION(ITMVoxel*, getVoxel, (Vector3ui localPos), "") {
        return (ITMVoxel*)const_cast<const ITMVoxelBlock*>(this)->getVoxel(localPos);
    }

    FUNCTION(VoxelBlockPos, getPos,() const, "", PURITY_PURE) {
        return pos_;
    }

    __declspec(property(get = getPos)) VoxelBlockPos pos;

    /// Initialize pos and reset data
    FUNCTION(void, reinit, (VoxelBlockPos pos), "") {
        pos_ = pos;

        resetVoxels();
    }

    //private:
    /// pos is Mutable, 
    /// because this voxel block might represent any part of space, and it might be freed and reallocated later to represent a different part
    VoxelBlockPos pos_;

    ITMVoxel blockVoxels[SDF_BLOCK_SIZE3];
};


























// Scene, stores VoxelBlocks
// accessed via 'currentScene' to reduce amount of parameters passed to kernels
// TODO maybe prefer passing (statelessness), remove 'current' notion (is this a pipeline like OpenGL with state?)

/*
* vb: an ITMVoxelBlock, worked on in tandem by the current thread-block
* v: vb->getVoxel(localPos), each thread operates on one voxel
* localPos: can be passed to getVoxel, i.e. no coordinate is bigger than sdf_block_size
TODO then Vector3ui or would be the more appropriate and self-documenting type
* globalPoint:
(vb->pos.toInt() * SDF_BLOCK_SIZE + localPos) * voxelSize
assert(globalPoint.coordinateSystem == CoordinateSystem::global());
world-space position of the voxel
*/
// see doForEachAllocatedVoxel for T
#define doForEachAllocatedVoxel_process() static FUNCTION(void, process, (const ITMVoxelBlock* vb, ITMVoxel* const v, const Vector3ui localPos, const Vector3i globalPos, const Point globalPoint), "")

#define doForEachAllocatedVoxelBlock_process() static FUNCTION(void, process,(ITMVoxelBlock* voxelBlock), "")


template<typename T>
FUNCTION(void, doForAllocatedVoxel, (ITMVoxelBlock* vb, Vector3ui localPos), "getCurrentScene & voxelSize needed by this, therefore defined after Scene");

#ifdef __CUDACC__
template<typename T>
KERNEL doForEachAllocatedVoxel(
    ITMVoxelBlock* localVBA,
    unsigned int nextFreeSequenceId) {
    int index = blockIdx.x;
    if (index <= 0 || index >= nextFreeSequenceId) return;

    ITMVoxelBlock* vb = &localVBA[index];
    Vector3ui localPos(threadIdx_xyz);

    doForAllocatedVoxel<T>(vb, localPos);
}

template<typename T>
KERNEL doForEachAllocatedVoxelBlock(
    ITMVoxelBlock* localVBA, unsigned int nextFreeSequenceId
    ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= 0 /* valid sequence nubmers start at 1 - TODO this knowledge should not be repeated here */
        || index >= nextFreeSequenceId) return;

    ITMVoxelBlock* vb = &localVBA[index];
    T::process(vb);
}
#endif

/// Must be heap-allocated
class Scene : public Managed {
public:

    /// Pos is in voxelCoordinates (global)
    /// \returns NULL when the voxel was not found
    FUNCTION(ITMVoxel*, getVoxel,(Vector3i pos),"");
    FUNCTION(ITMVoxel const *, getVoxel, (Vector3i pos) const, "", PURITY_PURE);

    FUNCTION(bool, voxelExistsQ, (Vector3i pos) const, "", PURITY_PURE) {
        return NULL != getVoxel(pos);
    }

    /* whether all voxels with max-norm distance at most 1 from pos exist */
    FUNCTION(bool, voxel1NeighborhoodExistsQ, (Vector3i pos) const, "",PURITY_PURE){
        for (int x = -1; x <= 1; x++)
            for (int y = -1; y <= 1; y++)
                for (int z = -1; z <= 1; z++)
                    if (!voxelExistsQ(pos + Vector3i(x, y, z))) return false;

        return true;
    }

    /// sequenceNumber > 0 and <= countVoxelBlocks(), i.e. < voxelBlockHash->getLowestFreeSequenceNumber()
    /// \returns a voxel block from the localVBA
    /// Returns NULL if the voxel block is not allocated
    FUNCTION(ITMVoxelBlock*, getVoxelBlockForSequenceNumber, (unsigned int sequenceNumber), "");
    FUNCTION(ITMVoxelBlock const*, getVoxelBlockForSequenceNumber, (unsigned int sequenceNumber) const, "", PURITY_PURE);

    // Two-phase allocation:

    FUNCTION(void, requestVoxelBlockAllocation,(VoxelBlockPos pos), "");
    static CPU_FUNCTION(void, Scene::performCurrentSceneAllocations,(), "");


#define CURRENT_SCENE_SCOPE(s) Scene::CurrentSceneScope currentSceneScope(s);

    /// Immediately allocate a voxel block at the given location.
    /// Perfer requestVoxelBlockAllocation & performCurrentSceneAllocations over this
    /// TODO how much performance is gained by not immediately allocating?
    CPU_FUNCTION(unsigned int, performVoxelBlockAllocation,(VoxelBlockPos pos), "") {
        CURRENT_SCENE_SCOPE(this); // current scene has to be set for AllocateVB::allocate
        const unsigned int sequenceNumber = voxelBlockHash->performSingleAllocation(pos);
        assert(sequenceNumber > 0); // valid sequence numbers are > 0 -- TODO don't repeat this here
        return sequenceNumber;
    }

    CPU_MEMBERFUNCTION(, Scene, (const float newVoxelSize = 0.005f), "");
    virtual CPU_MEMBERFUNCTION(, ~Scene, (), "");

    /// T must have an operator(ITMVoxelBlock*, ITMVoxel*, Vector3i localPos)
    /// where localPos will run from 0,0,0 to (SDF_BLOCK_SIZE-1)^3
    /// runs threadblock per voxel block and thread per voxel
    template<typename T>
    CPU_MEMBERFUNCTION(void, doForEachAllocatedVoxel, (), "") {
#ifdef __CUDACC__
        cudaDeviceSynchronize(); // avoid in-page reading error, might even cause huge startup lag?
        LAUNCH_KERNEL( // fails in Release mode
            ::doForEachAllocatedVoxel<T>,
            voxelBlockHash->getLowestFreeSequenceNumber(),// at most SDF_LOCAL_BLOCK_NUM, // cannot be non power of 2?
            dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE),

            // ITMVoxelBlock* localVBA, uint nextFreeSequenceId
            localVBA.GetData(), voxelBlockHash->getLowestFreeSequenceNumber()
            );
#else
        DO1(sequenceNumber, countVoxelBlocks()) {
            XYZ_over_SDF_BLOCK_SIZE{
                doForAllocatedVoxel<T>(getVoxelBlockForSequenceNumber(sequenceNumber),
                    Vector3ui(x, y, z));
            }
        }
#endif
    }

    /// T must have an operator(ITMVoxelBlock*)
    template<typename T>
    CPU_MEMBERFUNCTION(void, doForEachAllocatedVoxelBlock, (), "") {
#ifdef __CUDACC__
        dim3 blockSize(256);
        dim3 gridSize((int)ceil((float)SDF_LOCAL_BLOCK_NUM / (float)blockSize.x));
        LAUNCH_KERNEL(
            ::doForEachAllocatedVoxelBlock<T>,
            gridSize,
            blockSize,
            localVBA.GetData(),
            voxelBlockHash->getLowestFreeSequenceNumber()
            );
#else
        DO1(sequenceNumber, countVoxelBlocks()) {
            XYZ_over_SDF_BLOCK_SIZE{
                T::process(getVoxelBlockForSequenceNumber(sequenceNumber));
            }
        }
#endif
    }

    static FUNCTION(ITMVoxel*, getCurrentSceneVoxel, (Vector3i pos), "") {
        assert(getCurrentScene());
        return getCurrentScene()->getVoxel(pos);
    }
    static FUNCTION(void, requestCurrentSceneVoxelBlockAllocation,(VoxelBlockPos pos),"") {
        assert(getCurrentScene());
        return getCurrentScene()->requestVoxelBlockAllocation(pos);
    }


    /** !private implementation detail! But has to be placed in public for HashMap to access it - unless we make that a friend */
    struct Z3Hasher {
        typedef VoxelBlockPos KeyType;
        static const unsigned int BUCKET_NUM = SDF_BUCKET_NUM; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

        static CPU_AND_GPU unsigned int hash(const VoxelBlockPos& blockPos) {
            return (((unsigned int)blockPos.x * 73856093u) ^ ((unsigned int)blockPos.y * 19349669u) ^ ((unsigned int)blockPos.z * 83492791u))
                &
                (unsigned int)(BUCKET_NUM - 1);
        }
    };

    /** !private implementation detail! But has to be placed in public for HashMap to access it - unless we make that a friend*/
    struct AllocateVB {
        static FUNCTION(void, allocate, (VoxelBlockPos pos, int sequenceId), "");
    };

    // Scene is mostly fixed. // TODO prefer using a scoping construct that lives together with the call stack!
    // Having it globally accessible heavily reduces having
    // to pass parameters.
    static FUNCTION( Scene*, getCurrentScene,(), "");

    /// Change current scene for the current block/scope
    /*
    TODO the 'current scene' global state simplifies interfaces, especially to GPU code using template parameters with fixed-format static methods,
    at the cost of hidden state/dependencies and side-effects
    */
    class CurrentSceneScope {
    public:
        CPU_MEMBERFUNCTION(,CurrentSceneScope,(Scene* const newCurrentScene), "") :
            oldCurrentScene(Scene::getCurrentScene()) {
            Scene::setCurrentScene(newCurrentScene);
        }
        CPU_MEMBERFUNCTION(,~CurrentSceneScope,(),"") {
            Scene::setCurrentScene(oldCurrentScene);
        }

    private:
        Scene* const oldCurrentScene;
    };

    SERIALIZE_VERSION(3);
    CPU_MEMBERFUNCTION(void, serialize,(ofstream& file), "", PURITY_PURE) {
        SERIALIZE_WRITE_VERSION(file);

        bin(file, voxelSize_);

        voxelBlockHash->serialize(file);

        assert(localVBA.dataSize == SDF_LOCAL_BLOCK_NUM);
        localVBA.serialize(file);
    }

    /*
    reads from the binary file:
    - the voxel size
    - the full voxelBlockHash
    - the full localVBA
    version and size of these structures in the file must match (full binary dump)
    */
    CPU_MEMBERFUNCTION(void, deserialize,(ifstream& file), "", PURITY_OUTPUT_POINTERS) {
        SERIALIZE_READ_VERSION(file);

        setVoxelSize(bin<decltype(voxelSize_)>(file));

        voxelBlockHash->deserialize(file);

        localVBA.deserialize(file);
        assert(localVBA.dataSize == SDF_LOCAL_BLOCK_NUM);

        // Assert that the file ends here
        // TODO assuming a file ends with serialized scene -- not necessarily so
        int x;  file >> x;
        assert(file.bad() || file.eof());
    }



    CPU_MEMBERFUNCTION(void, setVoxelSize,(float newVoxelSize), "") {
        voxelSize_ = newVoxelSize;

        Matrix4f m;
        m.setIdentity(); m.setScale(voxelSize_ * SDF_BLOCK_SIZE/*voxelBlockSize*/);
        voxelBlockCoordinates_ = new CoordinateSystem(m);

        m.setIdentity(); m.setScale(voxelSize_);
        voxelCoordinates_ = new CoordinateSystem(m);
    }

    MEMBERFUNCTION(float, getVoxelSize, () const, "", PURITY_PURE) {
        assert(this, "if this fails, you are calling this on an illegal scene object");
        return voxelSize_;
    }

    MEMBERFUNCTION(unsigned int, countVoxelBlocks, ()const, "", PURITY_PURE)  {
        return voxelBlockHash->getLowestFreeSequenceNumber() - 1;
    }

    /// (0,0,0) is the lower corner of the first voxel block, (1,1,1) its upper corner,
    /// a position corresponding to (voxelBlockSize, voxelBlockSize, voxelBlockSize) in world coordinates.
    CoordinateSystem* voxelBlockCoordinates_ = 0;

    /// (0,0,0) is the lower corner of the voxel, (1,1,1) its upper corner,
    /// a position corresponding to (voxelSize, voxelSize, voxelSize) in world coordinates.
    /// aka "voxel-fractional-world-coordinates"
    CoordinateSystem* voxelCoordinates_ = 0;

    // NULL if not allocated
    MEMBERFUNCTION(ITMVoxelBlock*, getVoxelBlock,(VoxelBlockPos pos), "");
    MEMBERFUNCTION(ITMVoxelBlock const*, getVoxelBlock,(VoxelBlockPos pos) const, "", PURITY_PURE);
private:

    static CPU_FUNCTION(void, setCurrentScene,(Scene* s), "");


    float voxelSize_;

public: // data elements -- these two could be private where it not for testing/debugging

    MemoryBlock<ITMVoxelBlock> localVBA;

    /// Gives indices into localVBA for allocated voxel blocks
    // Cannot use an auto_ptr because this pointer is used on the device.
    HashMap<Z3Hasher, AllocateVB>* voxelBlockHash;

};





template<typename T>
FUNCTION(void, doForAllocatedVoxel, (ITMVoxelBlock* vb, Vector3ui localPos), "getCurrentScene & voxelSize needed by this, therefore defined after Scene") {
    // signature specified in doForEachAllocatedVoxel_process

    // TODO  an optimization would remove the following computations whose result is not used
    // in voxel coordinates (as passed to getVoxel of Scene)
    const Vector3i globalPos = vb->pos.toInt() * SDF_BLOCK_SIZE + localPos.toInt();

    // world-space coordinate position of current voxel
    auto globalPoint = Point(CoordinateSystem::global(), globalPos.toFloat() * voxelSize);

    T::process(
        vb,
        vb->getVoxel(localPos),
        localPos,
        globalPos,
        globalPoint);
}














//__device__ Scene* currentScene;
//__host__
GLOBAL(Scene*, currentScene, 0); // TODO use __const__ memory, since this value is not changeable from gpu!

FUNCTION(Scene*, Scene::getCurrentScene,(), "") {
    return currentScene;
}

CPU_FUNCTION(void, Scene::setCurrentScene, (Scene* s), "") {
#ifdef __CUDACC__
    cudaDeviceSynchronize(); // want to write managed currentScene 
#endif
    currentScene = s;
}


// performAllocations -- private:
FUNCTION(void, Scene::AllocateVB::allocate,(VoxelBlockPos pos, int sequenceId),"") {
    assert(Scene::getCurrentScene());
    assert(sequenceId < SDF_LOCAL_BLOCK_NUM, "%d >= %d -- not enough voxel blocks", sequenceId, SDF_LOCAL_BLOCK_NUM);
    Scene::getCurrentScene()->localVBA[sequenceId].reinit(pos);
}

CPU_FUNCTION(void, Scene::performCurrentSceneAllocations, (), "") {
    assert(Scene::getCurrentScene());
    Scene::getCurrentScene()->voxelBlockHash->performAllocations(); // will call Scene::AllocateVB::allocate for all outstanding allocations
#ifdef __CUDACC__
    cudaDeviceSynchronize(); // want to write managed currentScene 
#endif
}
//

CPU_MEMBERFUNCTION(,Scene::Scene,(const float newVoxelSize),"") : localVBA(SDF_LOCAL_BLOCK_NUM) {
    initCoordinateSystems();
    setVoxelSize(newVoxelSize);
    voxelBlockHash = new HashMap<Z3Hasher, AllocateVB>(SDF_EXCESS_LIST_SIZE);
}

CPU_MEMBERFUNCTION(,Scene::~Scene,(),"") {
    delete voxelBlockHash;
}

FUNCTION(VoxelBlockPos, pointToVoxelBlockPos, (
    const Vector3i & point //!< [in] in voxel coordinates
    ), "partial, undefined for certain points", PURITY_PURE) {
    // "The 3D voxel block location is obtained by dividing the voxel coordinates with the block size along each axis."
    VoxelBlockPos blockPos;
    // if SDF_BLOCK_SIZE == 8, then -3 should go to block -1, so we need to adjust negative values 
    // (C's quotient-remainder division gives -3/8 == 0)
    blockPos.x = ((point.x < 0) ? point.x - SDF_BLOCK_SIZE + 1 : point.x) / SDF_BLOCK_SIZE;
    blockPos.y = ((point.y < 0) ? point.y - SDF_BLOCK_SIZE + 1 : point.y) / SDF_BLOCK_SIZE;
    blockPos.z = ((point.z < 0) ? point.z - SDF_BLOCK_SIZE + 1 : point.z) / SDF_BLOCK_SIZE;
    assert(blockPos != INVALID_VOXEL_BLOCK_POS);
    return blockPos;
}

FUNCTION(ITMVoxel*, Scene::getVoxel, (Vector3i point), "") {
    VoxelBlockPos blockPos = pointToVoxelBlockPos(point);

    ITMVoxelBlock* b = getVoxelBlock(blockPos);
    if (b == NULL) return NULL;

    Vector3ui localPos = Vector3ui(point - blockPos.toInt() * SDF_BLOCK_SIZE); // localized coordinate
    return b->getVoxel(localPos);
}

FUNCTION(ITMVoxel const*, Scene::getVoxel,(Vector3i point) const, "", PURITY_PURE) {
    VoxelBlockPos blockPos = pointToVoxelBlockPos(point);

    ITMVoxelBlock const* b = getVoxelBlock(blockPos);
    if (b == NULL) return NULL;

    Vector3ui localPos = Vector3ui(point - blockPos.toInt() * SDF_BLOCK_SIZE); // localized coordinate
    return b->getVoxel(localPos);
}

FUNCTION(ITMVoxelBlock*, Scene::getVoxelBlockForSequenceNumber,(const unsigned int sequenceNumber), "") {
    assert(sequenceNumber >= 1 && sequenceNumber < SDF_LOCAL_BLOCK_NUM, "illegal sequence number %d (must be >= 1, < %d)", sequenceNumber, SDF_LOCAL_BLOCK_NUM);
    assert(sequenceNumber < voxelBlockHash->getLowestFreeSequenceNumber(),
        "unallocated sequence number %d (lowest free: %d)"
        , sequenceNumber
        , voxelBlockHash->getLowestFreeSequenceNumber()
        );

    return &localVBA[sequenceNumber];
}

FUNCTION(ITMVoxelBlock const*, Scene::getVoxelBlockForSequenceNumber,(const unsigned int sequenceNumber) const, "", PURITY_PURE) {
    assert(sequenceNumber >= 1 && sequenceNumber < SDF_LOCAL_BLOCK_NUM, "illegal sequence number %d (must be >= 1, < %d)", sequenceNumber, SDF_LOCAL_BLOCK_NUM);
    assert(sequenceNumber < voxelBlockHash->getLowestFreeSequenceNumber(),
        "unallocated sequence number %d (lowest free: %d)"
        , sequenceNumber
        , voxelBlockHash->getLowestFreeSequenceNumber()
        );

    return &localVBA[sequenceNumber];
}

/// Returns NULL if the voxel block is not allocated
FUNCTION(ITMVoxelBlock*, Scene::getVoxelBlock,(VoxelBlockPos pos), "") {
    int sequenceNumber = voxelBlockHash->getSequenceNumber(pos); // returns 0 if pos is not allocated
    if (sequenceNumber == 0) return NULL;
    return &localVBA[sequenceNumber];
}

FUNCTION(ITMVoxelBlock const*, Scene::getVoxelBlock,(VoxelBlockPos pos) const, "", PURITY_PURE) {
    int sequenceNumber = voxelBlockHash->getSequenceNumber(pos); // returns 0 if pos is not allocated
    if (sequenceNumber == 0) return NULL;
    return &localVBA[sequenceNumber];
}

FUNCTION(void, Scene::requestVoxelBlockAllocation, (VoxelBlockPos pos), "") {
    voxelBlockHash->requestAllocation(pos);
}



























namespace TestScene {
    TEST(sceneNeighborhoodExistence) {
        Scene* s = new Scene;

        VoxelBlockPos p(0, 0, 0);
        s->performVoxelBlockAllocation(p);
        s->getVoxelBlock(p)->resetVoxels();

        assert(s->voxelExistsQ(Vector3i(0, 0, 0)));
        assert(!s->voxelExistsQ(Vector3i(-2, 0, 0)));
        assert(!s->voxel1NeighborhoodExistsQ(Vector3i(0, 0, 0)));
        assert(!s->voxel1NeighborhoodExistsQ(Vector3i(1, 0, 0)));
        assert(s->voxel1NeighborhoodExistsQ(Vector3i(1, 1, 1)));

        delete s;
    }


    TEST(testSceneSerialize) {
        // TODO this test crashes sometimes at performAllocations (too many? -> kernel killed?)

        Scene* scene = new Scene();
        CURRENT_SCENE_SCOPE(scene);
        // buildSphereScene(0.5); // TODO fails often- in-page readin error or error in performCurrentSceneAllocations

        MemoryBlock<ITMVoxelBlock> mb(scene->localVBA);
        MemoryBlock<unsigned char> cb(scene->voxelBlockHash->needsAllocation);
        // TODO check hash map itself

        assert(cb == scene->voxelBlockHash->needsAllocation);
        assert(mb == scene->localVBA);

        {
            scene->serialize(binopen_write("scene.bin"));
        }

        assert(mb == scene->localVBA);
        {
            scene->deserialize(binopen_read("scene.bin"));
        }

        assert(mb == scene->localVBA);
        {
            scene->serialize(binopen_write("scene2.bin"));
        }
        assert(mb == scene->localVBA);


        {
            scene->deserialize(binopen_read("scene2.bin"));
        }

        assert(mb == scene->localVBA);
        assert(cb == scene->voxelBlockHash->needsAllocation);

        delete scene;
    }


    // [[ procedural scenes
#ifdef __CUDACC__
    // request allocation for the block at offset + blockIdx
    // Run with one thread per block
    // TODO legacy from when only the GPU could request blocks to be allocated
    static KERNEL buildBlockRequests(Vector3i offset) {
        Scene::requestCurrentSceneVoxelBlockAllocation(
            VoxelBlockPos(
            offset.x + blockIdx.x,
            offset.y + blockIdx.y,
            offset.z + blockIdx.z));
    }
#endif

    GLOBAL(float,radiusInWorldCoordinates, 0.f);


    struct BuildSphere {
        doForEachAllocatedVoxel_process() {
            assert(v);
            assert(radiusInWorldCoordinates > 0);

            // world-space coordinate position of current voxel
            Vector3f voxelGlobalPos = globalPoint.location;

            // Compute distance to origin
            const float distanceToOrigin = length(voxelGlobalPos);
            // signed distance to radiusInWorldCoordinates, positive when bigger
            const float dist = distanceToOrigin - radiusInWorldCoordinates;

            // Truncate and convert to -1..1 for band of size mu
            // (Note: voxel blocks with all 1 or all -1 don't usually exist, but they do for this sphere)
            const float eta = dist;
            v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));

            v->clr = Vector3u(255, 255, 0); // yellow sphere for debugging

            v->w_color = 1;
            v->w_depth = 1;
        }
    };


    CPU_FUNCTION(void, buildSphereScene, (const float radiusInWorldCoordinates_), 
        "Makes the current scene represent a sphere of radiusInWorldCoordinates_ size") {
        assert(Scene::getCurrentScene());
        assert(radiusInWorldCoordinates_ > 0);
        radiusInWorldCoordinates = radiusInWorldCoordinates_;
        const float diameterInWorldCoordinates = radiusInWorldCoordinates * 2;
        int offseti = -ceil(radiusInWorldCoordinates / voxelBlockSize) - 1; // -1 for extra space
        assert(offseti < 0);

        Vector3i offset(offseti, offseti, offseti);
        int counti = 2 * -offseti;
        assert(counti > 0);
        assert(offseti + counti == -offseti);

        // repeat allocation a few times to avoid holes
        do {
#ifdef __CUDACC__
            const dim3 count(counti, counti, counti);
            LAUNCH_KERNEL(buildBlockRequests, count, 1, offset);
#else
            XYZ_over(counti, counti, counti)
                Scene::requestCurrentSceneVoxelBlockAllocation(VoxelBlockPos(x, y, z));
#endif

            Scene::performCurrentSceneAllocations();
        } while (Scene::getCurrentScene()->countVoxelBlocks() != counti*counti*counti);

        // Then set up the voxels to represent a sphere
        Scene::getCurrentScene()->doForEachAllocatedVoxel<BuildSphere>();
        return;
    }

    // assumes buildWallRequests has been executed
    // followed by perform allocations
    // builds a solid wall, i.e.
    // an trunctated sdf reaching 0 at 
    // z == (SDF_BLOCK_SIZE / 2)*voxelSize
    // and negative at bigger z.
    struct BuildWall {
        doForEachAllocatedVoxel_process() {
            assert(v);

            float z = (localPos.z) * voxelSize;
            float eta = (SDF_BLOCK_SIZE / 2)*voxelSize - z;
            v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));

            v->clr = Vector3u(255, 255, 0); // YELLOW WALL
        }
    };
    
    CPU_FUNCTION(void, buildWallScene, (), "makes the current scene represent a simple wall scene, i.e. a wall in the xy plane") {
        assert(Scene::getCurrentScene());

#ifdef __CUDACC__
        LAUNCH_KERNEL(buildBlockRequests, dim3(10, 10, 1), 1, Vector3i(0, 0, 0));
#else
        XYZ_over(10, 10, 1)
            Scene::requestCurrentSceneVoxelBlockAllocation(VoxelBlockPos(x, y, z));
#endif


        Scene::performCurrentSceneAllocations();

        Scene::getCurrentScene()->doForEachAllocatedVoxel<BuildWall>();
    }

    // ]]

    // Sets the SDF at each voxel to some value
    struct WriteEach {
        doForEachAllocatedVoxel_process() {
            v->setSDF((
                localPos.x +
                localPos.y * SDF_BLOCK_SIZE +
                localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
                ) / 1024.f);
        }
    };


    GLOBAL(unsigned int, _counter, 0, "used for scene tests");
    GLOBAL(bool, visited[SDF_BLOCK_SIZE][SDF_BLOCK_SIZE][SDF_BLOCK_SIZE], {0});

    // Reads out the values written by WriteEach and checks that they are more or less what was entered
    struct DoForEach {
        doForEachAllocatedVoxel_process() {
            assert(localPos.x >= 0 && localPos.y >= 0 && localPos.z >= 0);
            assert(localPos.x < SDF_BLOCK_SIZE && localPos.y < SDF_BLOCK_SIZE && localPos.z < SDF_BLOCK_SIZE);

            assert(vb);
            assert(vb->pos == VoxelBlockPos(0, 0, 0) ||
                vb->pos == VoxelBlockPos(1, 2, 3));

            visited[localPos.x][localPos.y][localPos.z] = 1;

            printf("%f .. %f\n", v->getSDF(),
                (
                localPos.x +
                localPos.y * SDF_BLOCK_SIZE +
                localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
                ) / 1024.f);
            assert(abs(
                v->getSDF() -
                (
                localPos.x +
                localPos.y * SDF_BLOCK_SIZE +
                localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
                ) / 1024.f) < 0.001 // not perfectly accurate
                );
            _atomicAdd(&_counter, 1);
        }
    };

    // Checks that all existing voxel blocks have coordinates as given in addSceneVB and counts them
    struct DoForEachVoxelBlock {
        static FUNCTION(void, process,(ITMVoxelBlock* vb), "") {
            assert(vb);
            assert(vb->pos == VoxelBlockPos(0, 0, 0) ||
                vb->pos == VoxelBlockPos(1, 2, 3));
            _atomicAdd(&_counter, 1);
        }
    };

    FUNCTION(void,modifyS,(), "makes a small modification to the current scene") {
        Scene::getCurrentSceneVoxel(Vector3i(0, 0, 1))->setSDF(1.0);
    }

    FUNCTION(void, checkModifyS,(), "checks that the result of modifyS was persisted")  {
        assert(Scene::getCurrentSceneVoxel(Vector3i(0, 0, 1))->getSDF() == 1.0);
    }

    FUNCTION(void, addSceneVB,(Scene* scene), "requests allocation of some specific voxel blocks") {
        assert(scene);
        scene->requestVoxelBlockAllocation(VoxelBlockPos(0, 0, 0));
        scene->requestVoxelBlockAllocation(VoxelBlockPos(1, 2, 3));
    }

    FUNCTION(void, allExist,(Scene* scene, Vector3i base), "asserts that SDF_BLOCK_SIZE^3 voxels exist starting at base") {
        XYZ_over_SDF_BLOCK_SIZE{
            ITMVoxel* v = scene->getVoxel(base + Vector3i(x,y,z));
            assert(v != NULL);
        }
    }

    FUNCTION(void, checkS,(Scene* scene), "asserts that the current scene is scene") {
        assert(Scene::getCurrentScene() == scene);
    }

    FUNCTION(void, findSceneVoxel, (Scene* scene), "asserts that all voxels requested by addSceneVB exist") {
        allExist(scene, Vector3i(0, 0, 0));
        allExist(scene, Vector3i(SDF_BLOCK_SIZE, 2 * SDF_BLOCK_SIZE, 3 * SDF_BLOCK_SIZE));

        assert(scene->getVoxel(Vector3i(-1, 0, 0)) == NULL);
    }
    // ^^ TODO old tests launched these as kernels - should not matter

    TEST(testScene) {
        assert(Scene::getCurrentScene() == 0);

        Scene* s = new Scene();
        addSceneVB(s);
        {
            CURRENT_SCENE_SCOPE(s);
            Scene::performCurrentSceneAllocations();
        }
        findSceneVoxel(s);

        // current scene starts out at 0
        checkS(0);

        // change current scene
        {
            checkS(0); // still 0 before scope begins

            CURRENT_SCENE_SCOPE(s);
            checkS(s);
            // Nest
            {
                CURRENT_SCENE_SCOPE(0);
                checkS(0);
            }
            checkS(s);
        }
        checkS(0); // 0 again

        // modify current scene
        {
            CURRENT_SCENE_SCOPE(s);
            modifyS();
            checkModifyS();
        }

        // do for each
        /* TODO this part seems broken
        s->doForEachAllocatedVoxel<WriteEach>(); // hangs in debug build

        _counter = 0;
        for (int x = 0; x < SDF_BLOCK_SIZE; x++)
        for (int y = 0; y < SDF_BLOCK_SIZE; y++)
        for (int z = 0; z < SDF_BLOCK_SIZE; z++)
        assert(!visited[x][y][z]);
        s->doForEachAllocatedVoxel<DoForEach>();
        cudaDeviceSynchronize();
        assert(_counter == 2 * SDF_BLOCK_SIZE3);
        for (int x = 0; x < SDF_BLOCK_SIZE; x++)
        for (int y = 0; y < SDF_BLOCK_SIZE; y++)
        for (int z = 0; z < SDF_BLOCK_SIZE; z++)
        assert(visited[x][y][z]);

        _counter = 0;
        s->doForEachAllocatedVoxelBlock<DoForEachVoxelBlock>();
        cudaDeviceSynchronize();
        assert(_counter == 2);
        */

        delete s;
    }
}
















































GLOBAL(int, x, 0, "x value");

FUNCTION(void, f, (), "sets x to 100") {
    x = 100;
}

#ifdef __CUDACC__
KERNEL Kf() { f(); }

TEST(testfcuda) {
    assert(x == 0);
    LAUNCH_KERNEL(Kf, 1, 1);
    cudaDeviceSynchronize();
    assert(x == 100);
}
#else
TEST(testfcpu) {
    assert(x == 0);
    f();
    assert(x == 100);
}
#endif











































// Wolfram Language Interface (WSTP, Wolfram Symbolic Transfer protocol)
#define WL_WSTP_MAIN
#define WL_ALLOC_CONSOLE
#define WL_WSTP_PRE_MAIN
#define WL_WSTP_EXIT
#include <paulwl.h>



// WSTP sanity check
extern "C" int Get42() {
    return 42;
}

extern "C" int RunTestsM() {
    runTests();
    return 1;
}

void preWsMain() {
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF); // catch malloc errors // TODO this prints a report when the program gracefully terminates. Anything else?
    // TODO I think this program still leaks lots of (cuda and cpu) memory, e.g. images

    _controlfp_s(NULL,
        0, // By default, the run-time libraries mask all floating-point exceptions. (11111...). We set to 0 (unmask) the following:
        _EM_OVERFLOW | _EM_ZERODIVIDE | _EM_INVALID
        ); // enable all floating point exceptions' trapping behaviour except for the following exceptions: denormal-result (can not be changed by _controlfp_s anyways), underflow and inexact which we tolerate
}

void preWsExit() {
    runTests();
    return;
}



