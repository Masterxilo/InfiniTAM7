/*

Heavily modified and extended

InfiniTAM 
https://github.com/victorprad/InfiniTAM

this iteration completes the volumetric refinement pipeline

by Paul Frischknecht, 28.09.2016

Extension on InfiniTAM 5, still supporting everything this supported ('it only adds!')

*/

/* 
How to build

* requires x64 msvc 2013
* CUDA 7.0 build-customization, build this as a CUDA .cu file
    Code generation:compute_50,sm_50
    nvcc commandline 

        --Werror cross-execution-space-call for sanity checking

        -maxregcount 97 (or similar) ensures that no 'Too Many Resources Requested for Launch' is generated:
            -Xptxas -v shows how many registers a kernel uses per thread. At runtime, we may not pack more threads 
            in one block than the registers per block (65k) allow. SDF_BLOCK_SIZE can be reduced to make this issue less pressing.
            limited -maxregcount will make the rest of the memory requirements backed by global memory (spilling)

* run with environment variable NSIGHT_CUDA_DEBUGGER=1 for debugging ( SetEnvironment["NSIGHT_CUDA_DEBUGGER" -> "1"] )

Use highest warning levels possible and Code Analysis (todo: works only with extension .cpp...)

Dependencies:
    Self contained except for standard headers available in a standard installation of the above 

    and 

    paulwl.h & WSTP libraries & wsprep for Mathematica interface, .tm file
        TODO make this optional - a main() function should be able to drive this, and the library should be usable from the outside
        (similarly to how mathematica is exposed to it - opaque scene pointer and view class should be mostly enough)
        (though meshing to files is not ideal)

        Setup:
            * create a .tm file, add it to source files
            * add ..\CPPLibraries\Mathematica to Executable Directories
            *set pre-build command to "wsprep WSTPTemplateFile.tm > WSTPMain.c && trykill $(TargetFileName) " (get trykill.bat)
            * add generated WSTPMain.c to source files

    paul.h
        Utilities

    f.cpp, df.cpp, lengthz, lengthfz, sigmap
        These describe the energy optimized over.
        They are mostly generated from SceneXEnergy and SceneXSelect in mathematica,
        using RIFunctionCForm...

TODO why are there unicode-only characters in this file?
*/


// Standard/system headers

// Windows
#define _CRT_SECURE_NO_WARNINGS // TODO make it build without this
#define NOMINMAX
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>            // for MessageBox, DebugBreak

// MSVC C verification support, document parameter intention
#include <sal.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#pragma comment(lib,"cudart")

// C
#define _USE_MATH_DEFINES   // M_PI, ...
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <memory.h>
#include <string.h>
#include <time.h>

typedef unsigned int uint;

// C++
#include <array>
#include <string> 
#include <vector> 
#include <map>
#include <unordered_map> 
#include <type_traits> // syntax hacks
#include <functional>
#include <algorithm>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iostream>
#include <string>
#include <exception>
#include <iterator>
#include <memory>
using namespace std;

// End of Standard/system headers


// Custom headers

// Wolfram Language Interface (WSTP, Wolfram Symbolic Transfer protocol)
#define WL_WSTP_MAIN
#define WL_ALLOC_CONSOLE
#define WL_WSTP_PRE_MAIN
#define WL_WSTP_EXIT
#include <paulwl.h>






















// assert() - implementation

/// Whether a failed assertion triggers a debugger break.
__managed__ bool breakOnAssertion = true;

/// Whether when an assertions fails std::exception should be thrown.
/// No effect in GPU code. Used for BEGIN_SHOULD_FAIL tests only.
__managed__ bool assertionThrowException = false;

/// Used to track assertion failures on GPU in particular.
/// Must be reset manually (*_SHOULD_FAIL reset it).
__managed__ bool assertionFailed = false;

#pragma warning(disable : 4003) // assert does not need "commentFormat" and its arguments
#undef assert

#ifdef NDEBUG

#define assert(...) {}
// TODO keep tassert (test-assert) around for tests?
#else

#ifdef __CUDA_ARCH__
#define assert(x,commentFormat,...) {if(!(x)) {if (!assertionThrowException) printf("%s(%i) : Assertion failed : %s.\n\tblockIdx %d %d %d, threadIdx %d %d %d\n\t<" commentFormat ">\n", __FILE__, __LINE__, #x, xyz(blockIdx), xyz(threadIdx), __VA_ARGS__); assertionFailed = true; if (breakOnAssertion) *(int*)0 = 0; /* causes a memory checker error when an assertion fails "access violation on store (global memory)" "address = 0x00000000 accessSize = 4"*//* asm("trap;"); just illegal instruction, sometimes unnoticed by debugger?!*/} }
#else
#define assert(x,commentFormat,...) {if(!(x)) {char s[10000]; sprintf_s(s, "%s(%i) : Assertion failed : %s.\n\t<" commentFormat ">\n", __FILE__, __LINE__, #x, __VA_ARGS__); if (!assertionThrowException) {puts(s);MessageBoxA(0,s,"Assertion failed",0);OutputDebugStringA(s);} /*flushStd();*/ assertionFailed = true; if (breakOnAssertion) DebugBreak();  if (assertionThrowException) throw std::exception(s); }}
#endif

#endif

#define fatalError(commentFormat,...) {assert(false, commentFormat, __VA_ARGS__);}


// For assertions within tests that are expected to fail.

/// BEGIN_SHOULD_FAIL starts a block of code that should raise an assertion error.
/// Can only be used together with END_SHOULD_FAIL in the same block
// TODO support SEH via __try, __except (to catch division by 0, null pointer access etc.)
#define BEGIN_SHOULD_FAIL(msg) {OutputDebugStringA("=== BEGIN_SHOULD_FAIL: " msg "\n"); cudaDeviceSynchronize(); assert(!assertionThrowException, "BEGIN_SHOULD_FAIL blocks cannot be nested"); bool ok = false; assertionThrowException = true; breakOnAssertion = false; assert(!assertionFailed); try {

#define END_SHOULD_FAIL(msg) } catch(const std::exception& e) { /*cout << e.what();*/ } cudaDeviceSynchronize(); assertionThrowException = false; breakOnAssertion = true; if (assertionFailed) { ok = true; assertionFailed = false; } assert(ok, "expected an exception but got none"); OutputDebugStringA("=== END_SHOULD_FAIL: " msg "\n");  }




































// Utilities
#define PAUL_NO_ASSERT
#include <paul.h>


// WSTP sanity check
extern "C" int Get42() {
    return 42;
}



FUNCTION(
    void,
    assert_restricted,
    (void const* const a, void const* const b),
    ""){
    assert(a != b, "Pointers where assumed to be different (restricted even for their lifetime), but where the same (initially): %p %p", a, b);
}














#ifndef __CUDACC__
#error This file can only be compiled as a .cu file by nvcc.
#endif

#ifndef _WIN64
#error cudaMallocManaged and __managed__ require 64 bits. Also, this program is made for windows.
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 500
#error Always use the latest cuda arch. Old versions dont support any amount of thread blocks being submitted at once.
#endif
















































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
    "prints a string to stdout"){
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

CUDA_FUNCTION(uint,linear_threadIdx,(),"", PURITY_ENVIRONMENT_DEPENDENT) {
    return toLinearId(blockDim, threadIdx);
}

CUDA_FUNCTION(uint, linear_blockIdx, (), "", PURITY_ENVIRONMENT_DEPENDENT) {
    return toLinearId(gridDim, blockIdx);
}

// PCFunction[uint, volume, {{dim3, d}}, "C110", Pure, d.x*d.y*d.z.w]
FUNCTION(unsigned int, volume, (dim3 d), "C110. Undefined where the product would overflow", PURITY_PURE) {
    return d.x*d.y*d.z;
}


// Universal thread identifier
FUNCTION(uint, linear_global_threadId, (), "", PURITY_ENVIRONMENT_DEPENDENT) {
#if GPU_CODE
    return linear_blockIdx() * volume(blockDim) + linear_threadIdx();
#else
return 0; // TODO support CPU multithreading (multi-blocking - one thread is really one block, there is no cross cooperation ideally to ensure lock-freeness and best performance)
#endif
}


FUNCTION(dim3,getGridSize,(dim3 taskSize, dim3 blockSize),
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






















_Must_inspect_result_
FUNCTION(bool, approximatelyEqual, (float x, float y), "whether x and y differ by at most 1e-5f. Undefined for infinite values.", PURITY_PURE){
    return abs(assertFinite(x) - assertFinite(y)) < 1e-5f;
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
    float target[] = {0.f, 0.f};
    float addedValues[] = {1.f, 2.f};
    unsigned int targetIndices[] = {1, 0};

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
    float source[] = {1.f, 2.f};
    unsigned int sourceIndices[] = {1, 0};

    extract(target, source, 2, sourceIndices, 2);

    assert(target[0] == 2.f);
    assert(target[1] == 1.f);
}
























template<typename T> 
FUNCTION(void,zeroMalloc,(T*& p, const uint count = 1), "Allocate a block of CUDA device memory and memset it to 0.") {
    cudaMalloc(&p, sizeof(T) * count);
    cudaMemset(p, 0, sizeof(T) * count);
}





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
CPU_FUNCTION(T*,tmalloczeroed,(const size_t n),"Allocate spece for n T structs and set the allocate memory to 0") {
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


























































// Simple mathematical functions

template<typename T>
FUNCTION(T,ROUND,(T x),
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














// CUDA Kernel launch and error reporting framework

dim3 _lastLaunch_gridDim, _lastLaunch_blockDim;
#ifndef __CUDACC__

// HACK to make intellisense shut up about illegal C++ 
#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, arguments, ...) ((void)0)

#else

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


#endif


CPU_FUNCTION(void,reportCudaError,(cudaError err, const char * const expr, const char * const file, const int line)
    ,"Pure insofar as it only logs things, not modifying other system/CUDA state."
    , PURITY_PURE);

CPU_FUNCTION(void,cudaCheckLaunch,(const char* const launchCommand, const char * const file, const int line),"") {
    auto err = cudaGetLastError();
    if (err == cudaSuccess) return;
    reportCudaError(err, launchCommand, file, line);
}















































/// Extend this class to declar objects whose memory lies in CUDA managed memory space
/// which is accessible from CPU and GPU.
/// Classes extending this must be heap-allocated
struct Managed {
    CPU_MEMBERFUNCTION(void *, operator new, (size_t len), ""){
        void *ptr;
        cudaMallocManaged(&ptr, len); // if cudaSafeCall fails here check the following: did some earlier kernel throw an assert?
        cudaDeviceSynchronize();
        return ptr;
    }

    CPU_MEMBERFUNCTION(void, operator delete,(void *ptr), "") {
        cudaDeviceSynchronize();  // did some earlier kernel throw an assert?
        cudaFree(ptr);
    }
};





































/// Sums up 64 floats
/// c.f. "Optimizing Parallel Reduction in CUDA" https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
///
/// sdata[0] will contain the sum 
///
/// tid up to tid+32 must be a valid indices into sdata 
/// tid should be 0 to 31
/// ! Must be run by all threads of a single warp (the 32 first threads of a block) simultaneously.
/// sdata must point to __shared__ memory
CUDA_FUNCTION(void,warpReduce,(volatile SHAREDPTR(float* const) sdata, const int tid),"", PURITY_OUTPUT_POINTERS) {
    // Ignore the fact that we compute some unnecessary sums.
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


template<typename T //!< int or float
>
CUDA_FUNCTION(void,warpReduce256,(
const float localValue,
volatile SHAREDPTR(float* const) dim_shared1,
const int locId_local,
DEVICEPTR(T*) outTotal),
"Sums up 256 floats"
"and atomicAdd's the final sum to a float or int in global memory", PURITY_OUTPUT_POINTERS) {
    dim_shared1[locId_local] = localValue;
    __syncthreads();

    if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
    __syncthreads();
    if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
    __syncthreads();

    if (locId_local < 32) warpReduce(dim_shared1, locId_local);

    if (locId_local == 0) atomicAdd(outTotal, (T)dim_shared1[locId_local]);
}






















































// Serialization infrastructure

template<typename T>
CPU_FUNCTION(void,binwrite,(ofstream& f, const T* const x),"") {
    auto p = f.tellp(); // DEBUG
    f.write((char*)x, sizeof(T));
    assert(sizeof(T) == f.tellp() - p);
}

template<typename T>
CPU_FUNCTION(T,binread,(ifstream& f),"") {
    T x;
    f.read((char*)&x, sizeof(T));
    return x;
}

template<typename T>
CPU_FUNCTION(void,binread,(ifstream& f, T* const x),"") {
    f.read((char*)x, sizeof(T));
}

#define SERIALIZE_VERSION(x) static const int serialize_version = x
#define SERIALIZE_WRITE_VERSION(file) bin(file, serialize_version)
#define SERIALIZE_READ_VERSION(file) {const int sv = bin<int>(file); \
assert(sv == serialize_version\
, "Serialized version in file, '%d', does not match expected version '%d'"\
, sv, serialize_version);}

template<typename T>
CPU_FUNCTION(void,bin,(ofstream& f, const T& x),"") {
    binwrite(f, &x);
}
template<typename T>
CPU_FUNCTION(void,bin,(ifstream& f, T& x),"") {
    binread(f, &x);
}
template<typename T>
CPU_FUNCTION(T,bin,(ifstream& f),"") {
    return binread<T>(f);
}

CPU_FUNCTION(ofstream,binopen_write,(string fn),"") {
    return ofstream(fn, ios::binary);
}

CPU_FUNCTION(ifstream,binopen_read,(string fn),"") {
    return ifstream(fn, ios::binary);
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


CPU_FUNCTION(void,reportCudaError,(cudaError err, const char * const expr, const char * const file, const int line),"") {

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


/// \returns true if err is cudaSuccess
/// Fills errmsg in UNIT_TESTING build.
CPU_FUNCTION(bool,cudaSafeCallImpl,(cudaError err, const char * const expr, const char * const file, const int line),"")
{
    if (cudaSuccess == err) return true;

    cudaGetLastError(); // Reset error flag

    reportCudaError(err, expr, file, line);

    //flushStd();

    return false;
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

FUNCTION(bool, cs_is_triplet, (const cs * const A), "whether A is a triplet matrix. Undefined if A does not point to a valid cs instance.",PURITY_PURE) {
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
    "allocate new stuff. can only allocate multiples of 8 bytes to preserve alignment of pointers in cs. Use nextEven to round up when allocating 4 byte stuff (e.g. int)"){
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

FUNCTION(void, cs_free_, (char*& memoryPool, int& memory_size, unsigned int sz), "free the last allocated thing of given size"){
    // assert(sz < INT_MAX) // TODO to be 100% correct, we'd have to check that memory_size + sz doesn't overflow
    assert(divisible(sz, 8));
    assert(aligned(memoryPool, 8));
    memoryPool -= sz;
    memory_size += (int)sz;
    assert(memory_size >= 0);
}

#define cs_free(sz) {cs_free_(MEMPOOLPARAM, (sz));}


FUNCTION(unsigned int, cs_spalloc_size, (const unsigned int m, const unsigned int n, const unsigned int nzmax, bool triplet),
    "amount of bytes a sparse matrix with the given characteristics will occupy", PURITY_PURE){
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
    ,PURITY_OUTPUT_POINTERS)
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
    , PURITY_ENVIRONMENT_DEPENDENT){
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
        assert(!cs_is_triplet(mat));
        assert(mat->m && mat->n);
        assertFinite(mat->x, cs_x_used_entries(mat));
    }

    MEMBERFUNCTION(void, print, (), "print this matrix"){
        cs_print(mat, 0);
    }

    void operator=(matrix); // undefined
};


FUNCTION(float, dot, (const fvector& x, const fvector& y), "result = <x, y>, aka x.y or x^T y (the dot-product of x and y). Undefined if the addition overflows, finite otherwise.", PURITY_PURE){
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

FUNCTION(void, scal, (_Inout_ fvector& x, const float alpha), "x *= alpha", PURITY_OUTPUT_POINTERS){
    assert(assertFinite(alpha));
    DO(i, x.n) assertFinite(x.x[i] *= alpha);
}

FUNCTION(void, mv, (_Inout_ fvector& y, const float alpha, const matrix& A, const fvector& x, const float beta),
    "y = alpha A x + beta y", PURITY_OUTPUT_POINTERS){
    assert(A.mat->m && A.mat->n);
    assert(y.n == A.mat->m);
    assert(x.n == A.mat->n);
    cs_mv(y.x, alpha, A.mat, x.x, beta);
}

FUNCTION(void, mv, (_Out_ fvector& y, const matrix& A, const fvector& x), "y = A x", PURITY_OUTPUT_POINTERS){
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
FUNCTION(void, cs_cg, (const cs * const A, _In_reads_(A->m) const float * const b, _Inout_updates_all_(A->n) float *x, MEMPOOL),
    "x=A\b"
    "current value of x is used as initial guess"
    "Uses memory pool to allocate transposed copy of A and four vectors with size m or n"
    // PURITY_OUTPUT_POINTERS conceptually
    )
{
    assert(A && b && x && memoryPool && memory_size > 0);

    auto xv = vector_wrapper(x, A->n);
    conjgrad_normal(matrix(A), vector_wrapper((float*)b, A->m), xv, MEMPOOLPARAM);
}

/*

CSPARSE library end

*/


























// SparseOptimizationProblem(Decomposed) (SOPD) library
/*
Solves least-squares problems with energies of the form

\sum_{P \in Q} \sum_{p \in P} ||f(select_p(x))||_2^2

Q gives a partitioning of the domain. In the simplest case, there is only one partition.

The solution to this may or may not be close to the solution to

\sum_{p \in \Cup Q} ||f(select_p(x))||_2^2

*/

// --- Memory pool passed to the csparse library ---
// used in buildFxAndJFxAndSolve

// this is ideally some __shared__ memory in CUDA: In CUDA (I think) 
// C-style "stack" memory is first register based but then spills to main memory
// (is shared memory also used for the registers? Just another way to access the register file?)
// this memory does not need to be manually freed

// DEBUG TODO moved memory to global space for debugging -- move to __shared__ again.
// down the stack, no two functions should be calling SOMEMEM at the same time!

//__managed__ char memory[40000/*"Maximum Shared Memory Per Block" -> 49152*/ * 1000]; // TODO could allocate 8 byte sized type, should be aligned then (?)
//__managed__ bool claimedMemory = false; // makes sure that SOMEMEM is only called by one function on the stack

// "A default heap of eight megabytes is allocated if any program uses malloc() without explicitly specifying the heap size." -- want more 

// TODO run this to increase the tiny heap size
CPU_FUNCTION(void,prepareCUDAHeap,(),"") { // using a constructor to do this seems not to work
    int const mb = 400;
    printf("setting cuda malloc heap size to %d mb\n", mb);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, mb * 1000 * 1000); // basically the only memory we will use, so have some! // TODO changes when this is run from InfiniTAM
    CUDA_CHECK_ERRORS();
}
// TODO easily exceeded with lots of partitions on big scenes - small partitions don't need that much memory


#define SOMEMEM() \
    const size_t memory_size = 16 * 1000  * 1000;\
    char* const memory = (char*)malloc(memory_size);/*use global memory afterall*/\
    char* mem = (char*)(((unsigned long long)memory+7) & (~ 0x7ull)); /* align on 8 byte boundary */\
    assert(aligned(mem, 8) && after(mem, memory));\
    int memsz = memory_size - 8;/*be safe*/ \
    assert(memsz>0);\
    bool claimedMemory = true;\
    printf("allocated %d bytes at %#p using malloc\n", memory_size, memory);\
    assert(memory); /*attempting to access a null pointer just gives a kernel launch failure on GPU most of the time - at least when debugger cannot be attached */

#define FREESOMEMEM() {assert(claimedMemory); claimedMemory = false; ::free(memory); mem = 0;}


#define SOMEMEMP mem, memsz

// --- end of memory pool stuff ---

template<typename f> class SOPDProblem;

// f as in SOPDProblem
// one separate SOP (for one P in Q), shares only "x" with the global problem
// has custom y, p and values derived from that
// pointers are all nonzero to __managed__ memory
// instances of SOPPartition shall live in managed memory
//
// Note: Conceptually, F() is another function for each partition P: It is defined as
//   F(x) = (f(sigma_p(x)))_{p in P}
template<typename f>
struct SOPPartition {
    typedef f fType;

    float* minusFx; unsigned int lengthFx; // "-F(x)", length > 0, manually kept up-to-date using xIndices and the x vector in the parent sopd
    float* h; unsigned int lengthY; // > 0 "h, the update to y (subset of x, the parameters currently optimized over)"

    /*
    > 0
    "amount of 'points' at which the function f is evaluated."
    "lengthP * lengthz is the length of xIndices, "
    "and sparseDerivativeZtoYIndices contains lengthP sequences of the form (k [k many z indices] [k many y indices]) "
    */
    unsigned int lengthP;

    // integer matrix of dimensions lengthz x lengthP, indexing into x to find the values to pass to f
    unsigned int* xIndices;

    // Used to construct J, c.f. SOPJF
    unsigned int* sparseDerivativeZtoYIndices; // serialized form of this ragged array

    /*
    "the indices into x that indicate where the y are"
    "needed to write out the final update h to the parameters"
    */
    unsigned int* yIndices; /* lengthY */

    // parent (stores shared x and lengthx)
    SOPDProblem<f> const * /*const*/ sopd;
};

// Extract information about global SOPD from sop partition

// assumes the template-parameter is always called f
#define _lengthz(sop) f::lengthz //((sop)->sopd->lengthz)
#define _lengthfz(sop) f::lengthfz //((sop)->sopd->lengthfz)

#define _lengthx(sop) ((sop)->sopd->lengthx)
#define _x(sop) ((sop)->sopd->x)


template<typename f>
FUNCTION(void, buildFxandJFx, (SOPPartition<f>* const sop, cs* const J, const bool buildFx), "");
template<typename f>
FUNCTION(void, solve, (SOPPartition<f>* const sop, cs const * const J, MEMPOOL), "");
template<typename f>
FUNCTION(float, getPartitionEnergy, (SOPPartition<f>* const sop), "");

/*
The type f provides the following static members:
* const unsigned int lengthz, > 0
* const unsigned int lengthfz, > 0
* void f(_In_reads_(lengthz) const float* const input,  _Out_writes_all_(lengthfz) float* const out)
* void df(_In_range_(0, lengthz-1) int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out)
df(i) must be the derivative of f by the i-th argument

Note: Functions f and df should be CPU and GPU compilable.

Instances of this must live in GPU heap memory, because their pointer to the data vector x is dereferenced.
They can be handled only from the CPU.
*/
template<typename f>
class SOPDProblem : public Managed {
private:
    void operator=(SOPDProblem<f>); // undefined

public:
    typedef f fType;

    CPU_MEMBERFUNCTION(,
    SOPDProblem,(
        _In_ const vector<float>& x,
        _In_ const vector<vector<unsigned int>>& xIndicesPerPartition,
        _In_ const vector<vector<unsigned int>>& yIndicesPerPartition,
        _In_ const vector<vector<unsigned int>>& sparseDerivativeZtoYIndicesPerPartition),"") : 
        partitions(restrictSize(xIndicesPerPartition.size())), 
        lengthx(restrictSize(x.size())), 
        partitionTable(0)
        //,lengthz(f::lengthz), lengthfz(f::lengthfz)
        {


        assert(x.size() > 0);
        // forall(i) assert(xIndicesPerPartition[i].size() >= ps[i].size()); // there are at least as many indices into x as there are points, since f has at least 1 argument
        assert(xIndicesPerPartition.size() > 0);
        assert(yIndicesPerPartition.size() == xIndicesPerPartition.size());
        assert(yIndicesPerPartition.size() == sparseDerivativeZtoYIndicesPerPartition.size());
        static_assert(f::lengthz > 0, "");
        static_assert(f::lengthfz > 0, "");

        assert(partitions >= 0);
        allocatePartitions();
        assert(partitionTable);

        // TODO repeat and or externalize parameter checks (occur at quite a few places now but being safe is free)

        receiveSharedOptimizationData(x.data());
        DO(partition, partitions) {

#define r restrictSize
#if _DEBUG
            receiveAndPrintOptimizationData(f::lengthz, f::lengthfz,
                x.data(), r(x.size()),
                sparseDerivativeZtoYIndicesPerPartition[partition].data(), r(sparseDerivativeZtoYIndicesPerPartition[partition].size()),
                xIndicesPerPartition[partition].data(), r(xIndicesPerPartition[partition].size()),
                yIndicesPerPartition[partition].data(), r(yIndicesPerPartition[partition].size())
                );
#endif
            receiveOptimizationData(partition,
                sparseDerivativeZtoYIndicesPerPartition[partition].data(), r(sparseDerivativeZtoYIndicesPerPartition[partition].size()),
                xIndicesPerPartition[partition].data(), r(xIndicesPerPartition[partition].size()),
                yIndicesPerPartition[partition].data(), r(yIndicesPerPartition[partition].size())
                );
        }
    }

    CPU_MEMBERFUNCTION(float, getEnergy, (), "TODO could be parallelized" /*const, conceptually pure -- but will recompute Fx to be sure*/){
        assert(this);
        assert((__int64)(this) != 0xccccccccccccccccull);
        float e = 0.f;
        DO(i, partitions) e += getPartitionEnergy(&partitionTable[i]);
        return e;
    }

    MEMBERFUNCTION(void, solve, (_In_ const unsigned int iterations = 1), "") {

        dprintf("SOPDProblem::solve\n");
        DO(i, partitions) buildFxAndJFxAndSolveRepeatedly(i, iterations); // TODO GPU parallelize partitions
    }

    CPU_MEMBERFUNCTION(vector<float>, getX, () const, "") {
        auto xv = vector<float>(x, x + lengthx);
        return xv;
    }

    CPU_MEMBERFUNCTION(, ~SOPDProblem, (), "") {
        // free old stuff 
        FOREACH(sop, partitionTable, partitions) {
            memoryFree(sop.sparseDerivativeZtoYIndices);
            memoryFree(sop.xIndices);
            memoryFree(sop.yIndices);
            memoryFree(sop.minusFx);
            memoryFree(sop.h);
        }

        memoryFree(partitionTable);
        memoryFree(x);
    }

    // accessed externally:

    // "stores the current data vector 'x' which is updated to reduce the energy ||F(x)||^2", of length lengthx
    float* /*const*/ x; // to __managed__ memory
    const unsigned int lengthx; // > 0 

    //const unsigned int lengthz; // > 0  const unsigned int lengthfz; // > 0 

private:
    SOPPartition<f>* /*const*/ partitionTable; // to __managed__ memory
    const unsigned int partitions; // > 0



    //  "set the amount of partitions"
    CPU_MEMBERFUNCTION(void,allocatePartitions,(),"") {
        assert(partitions >= 0);
        // allocate
        partitionTable = tmalloczeroed<SOPPartition<f>>(partitions); // pointers not yet initialized
    }

    // "Receives x"
    CPU_MEMBERFUNCTION(void,
        receiveSharedOptimizationData,
        (
        _In_reads_(lengthx) const float* const xI
        ),"") {
        x = mallocmemcpy(xI, lengthx);
    }



    // macro for indexing into partitionTable, sop = partitionTable[partition]
#define extractSop(partition) assert(partition >= 0 && partition < partitions); SOPPartition<f>* const sop = &partitionTable[partition];

    CPU_MEMBERFUNCTION(
        void,
        receiveOptimizationData,
        (
        const unsigned int partition,
        _In_reads_(sparseDerivativeZtoYIndicesLength) const unsigned int* const sparseDerivativeZtoYIndicesI, const unsigned int sparseDerivativeZtoYIndicesLength,
        _In_reads_(xIndicesLength) const unsigned int* const xIndicesI, const unsigned int xIndicesLength,
        _In_reads_(yIndicesLength) const unsigned int* const yIndicesI, const unsigned int yIndicesLength
        ),
        "Receives sparseDerivativeZtoYIndices, xIndices and yIndices"
        "Appropriately sized vectors for receiving these data items are newly allocated in __managed__ memory, hence this is a CPU only function"
        ) {
        extractSop(partition);

        sop->sparseDerivativeZtoYIndices = mallocmemcpy(sparseDerivativeZtoYIndicesI, sparseDerivativeZtoYIndicesLength);
        sop->xIndices = mallocmemcpy(xIndicesI, xIndicesLength);
        sop->yIndices = mallocmemcpy(yIndicesI, yIndicesLength);

        static_assert(f::lengthz > 0, "");
        assert(divisible(xIndicesLength, f::lengthz));
        static_assert(f::lengthfz > 0, "");

        sop->lengthP = xIndicesLength / f::lengthz;
        sop->lengthY = yIndicesLength;
        sop->lengthFx = f::lengthfz * sop->lengthP;

        sop->minusFx = tmalloc<float>(sop->lengthFx);

        sop->h = tmalloc<float>(sop->lengthY);

        sop->sopd = this;
    }

    // trivially parallel over the partitions

    // note that these have to be members for accessing partitions

    // this could be a non-member if I find out how to access lengthz
    MEMBERFUNCTION(
        void,
        buildFxAndJFxAndSolve,
        (SOPPartition<f> * const sop, bool buildFx),
        "using current data, builds JFx (and Fx) and solves the least squares problem"
        "optionally does not compute Fx, assuming it is current with the x data (true after every solve)"
        ""
        "Note that we must do the solving right here, because this function handles the memory needed by J"
        "the solution is then accessible in h for further processing (updating x at yIndices)"
        ""
        "sop is passed here, not partition. Use buildFxAndJFxAndSolveRepeatedly as the external interface"
        )
    {
        // Build F and JF

        const unsigned int maxNNZ = MIN((
            f::lengthfz 
            *
            f::lengthz
            ) * sop->lengthP // very pessimistic estimate/overestimation: assume every derivative figures for every P -- usually not all of them will be needed

            // LIMIT by matrix size (usually much bigger except in very small cases)
            ,
            sop->lengthFx * sop->lengthY
            );

        // ^^ e.g. in vsfs the 3 color channels are all not optimized over, neither doriginal

        // consider using dynamic allocation in SOMEMEM!

        SOMEMEM();
        printf("allocating %d x %d sparse matrix for %d entries, needs %d bytes\n", sop->lengthFx, sop->lengthY, maxNNZ, 
            cs_spalloc_size(sop->lengthFx, sop->lengthY, maxNNZ, 1));
        assert(cs_spalloc_size(sop->lengthFx, sop->lengthY, maxNNZ, 1) < memsz, "%d", memsz);

        cs* J = cs_spalloc(sop->lengthFx, sop->lengthY, maxNNZ, 1, SOMEMEMP); // might run out of memory here

        dprintf("buildFxandJFx\n");
        buildFxandJFx(sop, J, buildFx);

        printf("used %d of %d allocated spaces in J\n", J->nz, J->nzmax);
        assert(J->nz > 0); // there must be at least one (nonzero) entry in the jacobian, otherwise we have taken the derivative only over variables no ocurring (or no variables at all!)

        J = cs_triplet(J, SOMEMEMP); // "optimizes storage of J, after which it may no longer be modified" 
        // TODO recycle memory

        // State
        dprintf("-F(x):\n");
        printv(sop->minusFx, sop->lengthFx);
        dprintf("JF(x):\n");
        printJ(J);

        // Solve
        dprintf("solve:\n");
        ::solve(sop, J, SOMEMEMP); // TODO allocates even more memory

        FREESOMEMEM();
    }

    MEMBERFUNCTION(
        void,
        buildFxAndJFxAndSolveRepeatedly,
        (const unsigned int partition, const unsigned int iterations),
        "using current data, builds JFx (and Fx) and solves the least squares problem"
        "then does a gradient descent step"
        "reapeats this whole process as often as desired"
        )
    {
        extractSop(partition);

        // TODO we might want to do this externally
        printf("\n=== buildFxAndJFxAndSolveRepeatedly %d times in partition %d of %d ===\n", iterations, partition, partitions);
        assert(iterations > 0); // TODO iterations should be size_t

        DO(i, iterations) {
            printf("\n--- iteration %d of %d in partition %d of %d ===\n", i, iterations, partition, partitions);

            bool buildFx = i == 0; // Fx is always up-to date after first iteration

            buildFxAndJFxAndSolve(sop, buildFx);
            const float delta = addContinuouslySmallerMultiplesOfHtoXUntilNorm2FxIsSmallerThanBefore(sop);
            if (delta > -0.001) {
                dprintf("delta was only %f, stopping optimization\n", delta);
                return;
            }
        }
    }

    /*
    FUNCTION(
    void,
    buildFxAndJFxAndSolveRepeatedlyThreadIdPartition,
    (const int iterations),
    "buildFxAndJFxAndSolveRepeatedly on the partition given by linear_global_threadId."
    "does nothing when linear_global_threadId is >= partitions"
    ""
    "TODO this should be the block id, threads in the same block should cooperate in the same partition"
    )
    {
    if (linear_global_threadId() >= partitions) {
    dprintf("\n--- thread id %d has nothing to do  - there are only %d partitions\n", linear_global_threadId(), partitions);
    return;
    }

    printf("\n=== Starting work on partition %d in the thread of the same id ===\n", linear_global_threadId());
    buildFxAndJFxAndSolveRepeatedly(linear_global_threadId(), iterations);
    }
    */
};

FUNCTION(void, writeJFx, (_Inout_ cs* const J, const unsigned int i, const unsigned int j, const float x),
    "set J(i, j) = x"
    ) {
    assert(J);
    assert(cs_is_triplet(J));
    assert(i < J->m && j < J->n);
    assert(J->nz + 1 <= (int)J->nzmax); // matrix should not become overful
    assertFinite(x);

    cs_entry(J, i, j, x);
}

template<typename f>
FUNCTION(void, writeFx, (_Inout_ SOPPartition<f>* const sop, const unsigned int i, const float val), "F(x)_i = val") {
    assert(i < sop->lengthFx);
    assert(sop->minusFx);
    assertFinite(val);

    sop->minusFx[i] = -val;
}

// -----------------------
/*
Given access to :

global:

    int lengthP
    int lengthY
    const int lengthz
    const int lengthfz
    f(fz_out, z)
    df(i, fz_out, z)

    float* x <-- 

per partition:

    int* xIndices (a list of indices into x, lengthfz * n many)
    int* sparseDerivativeZtoYIndices (a list of n lists of integers of the structure {k   (deriveby - k integers from 0 to argcount(f)-1) (store at - k integers from 0 to y_length-1)

This creates the vector
Fx
and the sparse matrix
JFx

By calling

void writeFx(int i, float val)
void writeJFx(int i, int j, float val)

using only elementary C constructs
*/
// TODO move these functions to SOPPartition instead of passing the pointer all the time
template<typename f>
FUNCTION(void, readZ, (
    _In_ SOPPartition<f> const * const sop,

    _Out_writes_all_(lengthz) float* z,
    const size_t rowz
    ), "z = x[[xIndices[[rowz;;rowz+lengthz-1]]]]", PURITY_OUTPUT_POINTERS){
    assert(divisible(rowz, _lengthz(sop)));

    extract(z, sop->sopd->x, sop->sopd->lengthx, sop->xIndices + rowz, _lengthz(sop)); // z = x[[xIndices]] // only place where x & lengthz is accessed
}


template<typename f>
FUNCTION(void, readZandSetFxRow, (
    _Inout_ SOPPartition<f>* const sop,
    _Out_writes_all_(lengthz) float* z,
    const unsigned int rowz,
    const unsigned int rowfz
    ), "compute and store Fx[[rowfz;;rowfz+lengthfz-1]] = f(z) and return the z = x[[xIndices[[rowz;;rowz+lengthz-1]]]] required for that", PURITY_OUTPUT_POINTERS){
    assert(divisible(rowz, _lengthz(sop)));
    assert(divisible(rowfz, _lengthfz(sop)));

    readZ(sop, z, rowz); // z = x[[xIndices]]

    float fz[_lengthfz(sop)];
    f::f(z, fz); // fz = f(z) // the only place f is called

    DO(i, _lengthfz(sop)) writeFx(sop, rowfz + i, fz[i]); // Fx[[rowfz;;rowfz+lengthfz-1]] = fz
}

template<typename f>
FUNCTION(void, setFxRow, (
    _Inout_ SOPPartition<f>* const sop,
    const unsigned int rowz,
    const unsigned int rowfz
    ), "compute and store Fx[[rowfz;;rowfz+lengthfz-1]]", PURITY_OUTPUT_POINTERS){
    float z[_lengthz(sop)];
    readZandSetFxRow(sop, z, rowz, rowfz);
}

template<typename f>
FUNCTION(void, buildFx, (_Inout_ SOPPartition<f>* const sop), "from the current x, computes just F(x)"){
    unsigned int rowz = 0;
    unsigned int rowfz = 0;

    FOR(unsigned int, i, 0, sop->lengthP, (rowz += _lengthz(sop), rowfz += _lengthfz(sop), 1)) MAKE_CONST(rowz) MAKE_CONST(rowfz) {
        DBG_UNREFERENCED_LOCAL_VARIABLE(i);
        setFxRow(sop, rowz, rowfz);
    }
}

template<typename f>
FUNCTION(void, buildFxandJFx, (_Inout_ SOPPartition<f>* const sop, cs* const J, const bool buildFx),
    "from the current x, computes F(x) [if buildFx == true] and JF(x)"
    "Note that J is stored into the matrix pointed to"
    "this J must by in triplet form and have allocated enough space to fill in the computed df"
    ) {
    assert(cs_is_triplet(J));
    auto* currentSparseDerivativeZtoYIndices = sop->sparseDerivativeZtoYIndices;
    unsigned int rowz = 0;
    unsigned int rowfz = 0;

    FOR(unsigned int, i, 0, sop->lengthP, (rowz += _lengthz(sop), rowfz += _lengthfz(sop), 1)) MAKE_CONST(rowz) MAKE_CONST(rowfz) {
        DBG_UNREFERENCED_LOCAL_VARIABLE(i);
        float z[_lengthz(sop)];
        if (buildFx)
            readZandSetFxRow(sop, z, rowz, rowfz);
        else
            readZ(sop, z, rowz);

        // MAKE_CONST(z) {

        // deserialize sparseDerivativeZtoYIndices, c.f. flattenSparseDerivativeZtoYIndices
        // convert back to two lists of integers of the same length (K)
        const unsigned int K = *currentSparseDerivativeZtoYIndices++;
        assert(K <= _lengthz(sop));
        const unsigned int* const zIndices = currentSparseDerivativeZtoYIndices; currentSparseDerivativeZtoYIndices += K;
        const unsigned int* const yIndices = currentSparseDerivativeZtoYIndices; currentSparseDerivativeZtoYIndices += K;

        // construct & insert localJ columnwise
        DO(k, K) {
            const unsigned int zIndex = zIndices[k];
            const unsigned int yIndex = yIndices[k];

            assert(zIndex < _lengthz(sop));
            assert(yIndex < sop->lengthY);

            float localJColumn[_lengthfz(sop)];
            f::df(zIndex, z, localJColumn);// the only place df is called
            // MAKE_CONST(localJColumn) {

            // put in the right place (starting at rowfz, column yIndex)
            DO(j, _lengthfz(sop)) {
                writeJFx(J, rowfz + j, yIndex, localJColumn[j]);
            }
        }
    }
}
// -----------------------

// Core algorithms

template<typename f>
FUNCTION(void,
    solve,
    (_Inout_ SOPPartition<f>* const sop, _In_ cs const * const J, MEMPOOL),
    "assumes x, -Fx and J have been built"
    "computes the adjustment fvector h, which is the least-squares solution to the system"
    "Jh = -Fx"
    , PURITY_OUTPUT_POINTERS) {
    assert(J && sop && _lengthx(sop) && _x(sop) && sop->minusFx && sop->h);
    assert(cs_is_compressed_col(J));

    printf("sparse leastSquares (cg) %d x %d... (this might take a while)\n",
        J->m, J->n);

    // dump J
#define DUMP_LS 1
#if !GPU_CODE && DUMP_LS
    if (J->n > 1000) {
        {
            puts("dumping to J.txt\n");
            FILE* file = fopen("J.txt", "wb");
            assert(file);
            cs_dump(J, file);
            fclose(file);
        }

        {
            puts("dumping to minusFx.txt\n");
            FILE* file = fopen("minusFx.txt", "wb");
            assert(file);
            dump(vector_wrapper(sop->minusFx, sop->lengthFx), file);
            fclose(file);
        }
    }
#endif
    //

    assert(sop->lengthY > 0);

    // h must be initialized -- initial guess -- use 0
    memset(sop->h, 0, sizeof(float) * sop->lengthY); // not lengthFx! -- in page writing error -- use struct fvector to keep fvector always with its length (existing solutions?)

    cs_cg(J, sop->minusFx, sop->h, MEMPOOLPARAM);


#if !GPU_CODE && DUMP_LS
    {
        puts("dumping least squares solution to h.txt\n");
        FILE* file = fopen("h.txt", "wb");
        assert(file);
        dump(vector_wrapper(sop->h, sop->lengthY), file);
        fclose(file);
    }
#endif

    dprintf("h:\n"); printv(sop->h, sop->lengthY);
    assertFinite(sop->h, sop->lengthY);
}

template<typename f>
FUNCTION(
    float,
    norm2Fx,
    (_In_ SOPPartition<f> const * const sop), "Assuming F(x) is already computed, returns ||F(x)||_2^2", PURITY_PURE
    ) {
    assert(sop->minusFx);
    float x = 0;
    FOREACHC(y, sop->minusFx, sop->lengthFx) x += y * y;
    return assertFinite(x);
}


template<typename f>
FUNCTION(float, getPartitionEnergy, (_Inout_ SOPPartition<f>* const sop), "Returns the 2-norm of the Fx vector, computing it from the current x first.", PURITY_OUTPUT_POINTERS) {
    buildFx(sop);
    return norm2Fx(sop);
}

template<typename f>
FUNCTION(
    float,
    addContinuouslySmallerMultiplesOfHtoXUntilNorm2FxIsSmallerThanBefore,
    (_Inout_ SOPPartition<f> * const sop),
    "scales h such that F(x0 + h) < F(x) in the 2-norm and updates x = x0 + h"
    "returns total energy delta achieved which should be negative but might not be when the iteration count is exceeded"
    , PURITY_OUTPUT_POINTERS) {
    assert(sop);
    assert(sop->yIndices);
    assert(sop->minusFx);
    assert(_x(sop));

    // determine old norm
    const float norm2Fatx0 = norm2Fx(sop);
    dprintf("||F[x0]||_2^2 = %f\n", norm2Fatx0);

    // add full h
    float lambda = 1.;
    dprintf("x = "); printv(_x(sop), _lengthx(sop));
    axpyWithReindexing(_x(sop), _lengthx(sop), lambda, sop->h, sop->yIndices, sop->lengthY); // xv = x0 + h
    dprintf("x = x0 + h = "); printv(_x(sop), _lengthx(sop));

    buildFx(sop);
    float norm2Faty0 = norm2Fx(sop);
    dprintf("||F[x0 + h]||_2^2 = %f\n", norm2Faty0);


    // Reduce step-size if chosen step does not lead to reduction by subtracting lambda * h
    size_t n = 0; // safety net, limit iterations
    while (norm2Faty0 > norm2Fatx0 && n++ < 20) {
        lambda /= 2.;
        axpyWithReindexing(_x(sop), _lengthx(sop), -lambda, sop->h, sop->yIndices, sop->lengthY); // xv -= lambda * h // note the -!

        dprintf("x = "); printv(_x(sop), _lengthx(sop));

        buildFx(sop); // rebuild Fx after this change to x
        norm2Faty0 = norm2Fx(sop); // reevaluate norm
        dprintf("reduced stepsize, lambda =  %f, ||F[y0]||_2^2 = %f\n", lambda, norm2Faty0);
    }
    dprintf("optimization finishes, total energy change: %f\n", norm2Faty0 - norm2Fatx0);
    /*assert(norm2Faty0 - norm2Fatx0 <= 0.);*/ // might not be true if early out was used
    return assertFinite(norm2Faty0 - norm2Fatx0);
}

// Interface





// Prototyping functions

FUNCTION(void,
    receiveAndPrintOptimizationData,
    (
    const unsigned int lengthz,
    const unsigned int lengthfz,

    _In_reads_(xLength) const float* const x, const unsigned int xLength,
    _In_reads_(sparseDerivativeZtoYIndicesLength) const unsigned int* const sparseDerivativeZtoYIndices, const unsigned int sparseDerivativeZtoYIndicesLength,
    _In_reads_(xIndicesLength) const unsigned int* const xIndices, const unsigned int xIndicesLength,
    _In_reads_(yIndicesLength) const unsigned int* const yIndices, const unsigned int yIndicesLength
    ),
    "Receives x, sparseDerivativeZtoYIndices, xIndices and yIndices, checks and prints them,"
    "emulating arbitrary lengthz, lengthfz"
    "Note: lengthz, lengthfz are fixed at compile-time for other functions"
    "This is a prototyping function that does not allocate or copy anything"
    "use for testing"
    , PURITY_PURE) {

    const unsigned int lengthP = xIndicesLength / lengthz;
    const unsigned int lengthY = yIndicesLength;
    const unsigned int lengthFx = lengthfz * lengthP;
    const unsigned int maxNNZ = (lengthfz*lengthz) * lengthP; // could go down from lengthz to maximum k in sparseDerivativeZtoYIndices
    // or just the actual sum of all such k

    dprintf("lengthz: %d\n", lengthz);
    dprintf("lengthfz: %d\n", lengthfz);
    dprintf("lengthP: %d\n", lengthP);
    dprintf("lengthY: %d\n", lengthY);
    dprintf("lengthFx: %d\n", lengthFx);
    dprintf("maxNNZ: %d\n", maxNNZ);

    assert(lengthz > 0);
    assert(lengthfz > 0);
    assert(lengthY > 0);

    dprintf("x:\n");
    printv(x, xLength);

    dprintf("sparseDerivativeZtoYIndices:\n");
    const unsigned int* p = sparseDerivativeZtoYIndices;

    REPEAT(lengthP) {
        const unsigned int k = *p++;
        assert(k <= lengthz);
        dprintf("---\n");
        printd(p, k); p += k;
        dprintf("-->\n");
        printd(p, k); p += k;
        dprintf("---\n");
    }
    assert(p == sparseDerivativeZtoYIndices + sparseDerivativeZtoYIndicesLength);

    dprintf("xIndices:\n");
    p = xIndices;
    REPEAT(lengthP) {
        printd(p, lengthz);
        p += lengthz;
    }
    assert(p == xIndices + xIndicesLength);
    assertLess(xIndices, xIndicesLength, xLength);

    dprintf("yIndices:\n");
    printd(yIndices, yIndicesLength);
    assertLess(yIndices, yIndicesLength, xLength);
}



FUNCTION(
    void,
    makeAndPrintSparseMatrix,
    (
    const unsigned int m,
    const unsigned int n,
    _In_reads_(xlen) float const * /*const */x,
    /*const */unsigned int xlen,
    _In_reads_(ijlen) int const * /*const */ ij,
    const unsigned int ijlen
    ),
    "Creates a sparse matrix from a list of values and a list of pairs of (i, j) indices specifying where to put the corresponding values (triplet form)"
    "Note: This is a prototyping function without any further purpose"
    , PURITY_PURE) {
    assert(2 * xlen == ijlen);
    assert(xlen <= m*n); // don't allow repeated entries

    SOMEMEM();
    cs* const A = cs_spalloc(m, n, xlen, 1, SOMEMEMP);

    while (xlen--) {
        int i = *ij++;
        int j = *ij++;
        cs_entry(A, i, j, *x++);
    }

    cs_print(A);

    printf("compress and print again:\n");
    const cs* const B = cs_triplet(A, SOMEMEMP);
    cs_print(B);
    printf("done--\n");


    FREESOMEMEM();
}

TEST(makeAndPrintSparseMatrix1) {
    unsigned int const count = 1;
    float x[] = {1.};
    int ij[] = {0, 0};
    makeAndPrintSparseMatrix(1, 1, x, count, ij, 2 * count);
}








// Misc

// "collection of some tests"
TEST(testMain) {
    float x[] = {1, 2};
    printv(x, 2);
    float y[] = {1};
    printv(y, 1);

    unsigned int to[] = {1};
    axpyWithReindexing(x, 2, 1., y, to, 1); // expect 1.000000 3.000000
    printv(x, 2);
    assert(1.f == x[0]);
    assert(3.f == x[1]);

    float z[] = {0, 0};
    unsigned int from[] = {1, 0};
    extract(z, x, 2, from, 2); // expect 3.000000 1.000000
    printv(z, 2);
    assert(3.f == z[0]);
    assert(1.f == z[1]);

    // expect i: 0-9
    FOR(int, i, 0, 10, 1) {
        dprintf("i: %d\n", i);
        //i = 0; // i is const!
    }

    int i = 0;
    REPEAT(10)
        dprintf("rep i: %d\n", i++);
}


// exercise SOMEMEM and cs_
TEST(mainc) {

    int cij[] = {0, 0};
    int xlen = 1;
    float xc[] = {0.1f};
    float* x = xc;
    int m = 1, n = 1;
    int* ij = cij;

    SOMEMEM();
    cs* A = cs_spalloc(m, n, xlen, 1, SOMEMEMP);

    while (xlen--) {
        int i = *ij++;
        int j = *ij++;
        cs_entry(A, i, j, *x++);
    }

    cs_print(A);
    assert(cs_is_triplet(A));
    assert(!cs_is_compressed_col(A));

    printf("compress and print again:\n");
    A = cs_triplet(A, SOMEMEMP);
    assert(!cs_is_triplet(A));
    assert(cs_is_compressed_col(A));
    cs_print(A);

    FREESOMEMEM();
}


// --- end of SOPCompiled framework ---









// --- SOPCompiled interface for C++ ---





























template<typename A>
CPU_FUNCTION(
void,build_locator,(_In_ const vector<A>& v, _Out_ unordered_map<A, unsigned int>& locator),

"Given nonempty v, build locator such that for all a"
"either locator[a] is undefined or"
"v[locator[a]] == a"
""
"locator should initially be empty"

,PURITY_OUTPUT_POINTERS){
    assert(v.size() > 0);
    assert(locator.size() == 0); // locator's constructor will have been called, so 6001 doesn't apply here

    locator.reserve(v.size());

    unsigned int i = 0;
    for (const auto& a : v) {
        locator[a] = i;
        // now:
        // assert(v[locator[a]] == a);
        i++;
    }
    assert(i == v.size());
    assert(v.size() == locator.size());
}

TEST(build_locator1) {
    typedef int In;
    vector<In> v = {42, 0};
    unordered_map<In, unsigned int> locator;
    build_locator(v, locator);

    // check constraints
    assert(v.size() == locator.size());
    assert(0 == locator[42]);
    assert(1 == locator[0]);
}






























/*
Given the FiniteMapping f, return a vector v of the same size as f and an injective function g such that

f(a) == v[g(a)] for all a for which f is defined

g crashes when supplied an argument for which f was not defined.
v & g should be empty initially.

A and B must be valid parameters for unordered_map.
*/
template<typename A, typename B>
CPU_FUNCTION(
    void, linearize, (_In_ const unordered_map<A, B>& f, _Out_ vector<B>& v, _Out_ unordered_map<A,unsigned int>& g), "", PURITY_OUTPUT_POINTERS) {
    printf("linearize start %d\n", f.size());
    assert(f.size() > 0);
    assert(f.size() <= UINT_MAX);
    assert(v.size() == 0);
    assert(g.size() == 0);

    v.resize(f.size());
    g.reserve(f.size());

    unsigned int i = 0; 
    for (const auto& fa : f) {
        v[i] = fa.second;
        g[fa.first] = i;
        // Now the following holds:
        // assert(v[g_map[fa.first]] == fa.second);

        i++;
    }
    assert(i == f.size());
}

TEST(linearize1) {
    // Construct some f
    typedef int In;
    typedef float Out;

    unordered_map<In, Out> f;
    f[0] = 1.;
    f[1] = 2.;

    // Call linearize
    vector<Out> v; unordered_map<In,unsigned int> g;
    linearize(f, v, g);

    // check constraints
    assert(v.size() == f.size());
    for (auto& fa : f) {
        assert(fa.second == v[g[fa.first]]);
    }
}












/*
Change the FiniteMapping f, such that

f(a) == v[g(a)]

whenever f(a) was already defined.
g must be injective and defined for every a that f is defined for and return a valid index into v for these a.
v must have the same size as f.
g and v returned by linearize satisfy these constraints relative to the input f.

! "Impure"/reference-modifying function for performance. Other parts of the system will be influenced indirectly (side-effect) if f is referenced from elsewhere.
*/
template<typename A, typename B>
CPU_FUNCTION(
void,update_from_linearization, (
_Inout_ unordered_map<A, B>& f, 
_In_ const vector<B>& v, 
_In_ const unordered_map<A, unsigned int>& g
), "", PURITY_OUTPUT_POINTERS) {
    assert(f.size() > 0);

    assert(v.size() == f.size());
    assert(g.size() == f.size());

    for (auto& fa : f) {
        auto i = g.at(fa.first);
        f[fa.first] = v[i];
    }
}

TEST(update_from_linearization1) {
    // Construct some f
    typedef int In;
    typedef float Out;
    unordered_map<In, Out> f;
    f[0] = 1.;
    f[1] = 2.;

    vector<Out> v; unordered_map<In, unsigned int> g;
    linearize(f, v, g);

    // Modify v
    for (auto& x : v) x += 2.;

    // Propagate updates to f

    assert(f.size() == 2);
    assert(f[0] == 1.);
    assert(f[1] == 2.);

    update_from_linearization(f, v, g);

    assert(f.size() == 2);
    assert(f[0] == 3.);
    assert(f[1] == 4.);
}








/*
Evaluate sigma at each p in P to obtain the lengthz many names (list of v \in X) of the variables that should be supplied to f at p.
Convert this list to a set of indices into the variable-name |-> value list x.
Concatenate all of these index-lists.

locateInX should give a valid index into the (implied) variable-value list x or crash when supplied a variable name that does not occur there
It is an injective function.

xIndices should be empty initially
*/
template<typename X, typename PT, unsigned int lengthz>
CPU_FUNCTION(
void ,SOPxIndices,(
    _In_ const unordered_map<X, unsigned int>& locateInX,
    _In_ const vector<PT>& P,
    _In_ const function<void(PT,X*)>& sigma,
    _Out_ vector<unsigned int>& xIndices
    ), "", PURITY_OUTPUT_POINTERS) {

    printf("SOPxIndices\n");
    assert(P.size() > 0);
    assert(locateInX.size());
    assert(sigma);
    assert(xIndices.size() == 0);

    //const unsigned int lengthz_ = restrictSize(sigma(P[0]).size());
    //assert(lengthz_ > 0);

    xIndices.resize(lengthz * P.size());

    X zatp[lengthz]; // sigma(p), read out at each p

    unsigned int i = 0;
    for (const auto& p : P) {
        sigma(p, zatp); //zatp  // const auto zatp = sigma(p);

        for (const auto& z : zatp) {
            assert(definedQ(locateInX, z), "attempted to locate a variable in x that is not there");
            xIndices[i++] = locateInX.at(z); 
            // 'fatal program exit requested' if you call this with a z that is not defined
            // (can happen if the set of data and the set of points are such that sigma tries to access points that don't exist).
        }
    }
    assert(i == xIndices.size());
}

TEST(SOPxIndices1) {
    // X === int, {0, -1} in this order
    // P === short, {0, 1}
    // sigma: at point p, supply variables {-p}
    unordered_map<int, unsigned int> locateInX = {{0, 0}, {-1, 1}};
    vector<short> P = {0, 1};
    auto sigma = [](short p, _Out_writes_(1) int* sigmap) {sigmap[0] = -p; };

    //const unsigned int lengthz = restrictSize(sigma(P[0]).size());
    //assert(1 == lengthz);

    vector<unsigned int> xIndices;
    SOPxIndices<int, short, 1>(locateInX, P, sigma, xIndices);
    //assert(xIndices == {0,1});
    assert(xIndices.size() == 2);
    assert(0 == xIndices[0]);
    assert(1 == xIndices[1]);

    // Retry with different P:
    P = {1, 0};
    xIndices.clear();
    SOPxIndices<int, short, 1>(locateInX, P, sigma, xIndices);
    //assert(xIndices == {1,0});
    assert(xIndices.size() == 2);
    assert(1 == xIndices[0]);
    assert(0 == xIndices[1]);
}

/*
locateInX /@ Y

where Y contains only elements for which locateInX is defined.

locateInX should give a valid index into the variable-value list x or crash when supplied a variable name that does not occur there
locateInX should be injective.

yIndices should be void initially
*/
template<typename X>
CPU_FUNCTION(
void,SOPyIndices,(
    _In_ const unordered_map<X, unsigned int>& locateInX, const _In_ vector<X>& Y,
    _Out_ vector<unsigned int>& yIndices
    ), "", PURITY_OUTPUT_POINTERS) {

    printf("SOPyIndices\n");
    assert(Y.size() > 0);
    assert(locateInX.size());
    assert(yIndices.size() == 0);

    yIndices.resize(Y.size());

    // --
    //transform(Y.begin(), Y.end(),
    //    yIndices.begin(), locateInX);
    // or
    
    unsigned int i = 0;
    for (const auto& y : Y) {
    yIndices[i++] = locateInX.at(y);
    }
    assert(i == yIndices.size());
    
    // --
}

TEST(SOPyIndices1) {
    // X === int, {0, -1} in this order
    // Y = {-1,0}
    unordered_map<int, unsigned int> locateInX = {{0, 0}, {-1, 1}};
    vector<int> Y = {-1, 0};

    vector<unsigned int> yIndices;
    SOPyIndices<int>(locateInX, Y, yIndices);

    //assert(yIndices == {1,0});
    assert(2 == yIndices.size());
    assert(1 == yIndices[0]);
    assert(0 == yIndices[1]);

    // Retry with another Y
    Y = {0, -1};
    yIndices.clear();
    SOPyIndices<int>(locateInX, Y, yIndices);
    assert(2 == yIndices.size());
    assert(0 == yIndices[0]);
    assert(1 == yIndices[1]);
}


/*
for each p in P
* figure out the variables that will be supplied to f (sigma(p))
* create empty lists zIndices and yIndices
* find the indices of these variables, listed as z \in 0:lengthz-1 in Y using locateInY
if locateInY(variable-z) is undefined, continue;
push_back(zIndices, z) push_back(yIndices, locateInY(variable-v))
* let K = size(zIndices) == size(yIndices)
* append the list {K} <> zIndices <> yIndices to sparseDerivativeZtoYIndices

This list of indices will allow us to figure out which derivatives of f we need and where these vectors are to be placed in the
jacobian matrix d F / d y, where F(x) = (f(sigma_p(x))_{p in P}

locateInX should crash for undefined x
locateInY is unordered_map because we need to be able to judge whether something is an y or not

The linearization format in sparseDerivativeZtoYIndices is detailed in SOPCompiledFramework.
{{zindices}, {yindices}} /; both of length k

{k, zindices, yindices}

TODO maybe this could be done more efficiently using the SOPxIndices-result and a function that
translates from x-indices to y-indices.

sparseDerivativeZtoYIndices should be initially empty
*/
template<typename X, typename PT, unsigned int lengthz>
CPU_FUNCTION(
void, SOPsparseDerivativeZtoYIndices, (
    _In_ const unordered_map<X, unsigned int>& locateInY,
    _In_ const vector<PT>& P,
    _In_ const function<void(PT, X*)>& sigma,

    _Out_ vector<unsigned int>& sparseDerivativeZtoYIndices
    ), "", PURITY_OUTPUT_POINTERS) {
    printf("SOPsparseDerivativeZtoYIndices\n");
    assert(P.size() > 0);
    assert(locateInY.size() > 0);
    assert(sigma);
    assert(sparseDerivativeZtoYIndices.size() == 0);

    // for checking sigma-constraint
    //const unsigned int lengthz_ = restrictSize(sigma(P[0]).size());
    //assert(lengthz_ > 0);

    X zatp[lengthz]; // sigma(p), read out at each p

    for (const auto& p : P) {
        sigma(p, zatp); //zatp = sigma(p);
        //assert(zatp.size() == lengthz); // sigma-constraint

        vector<unsigned int> yIndices, zIndices;

        // try to locate each z in y
        unsigned int zIndex = 0;
        for (const auto& z : zatp) {

            unsigned int positionInY;
            if (!definedQ<X, unsigned int>(locateInY, z, positionInY)) goto nextz;

            zIndices.push_back(zIndex);
            yIndices.push_back(positionInY);

        nextz:
            zIndex++;
        }

        assert(zIndex == lengthz);
        const unsigned int K = restrictSize(zIndices.size());
        assert(K <= lengthz);
        assert(K == yIndices.size());
        assertLess(zIndices, lengthz);
        //assertLess(yIndices, Y.size())

        sparseDerivativeZtoYIndices.push_back(K);
        sparseDerivativeZtoYIndices.insert(sparseDerivativeZtoYIndices.end(), zIndices.begin(), zIndices.end());
        sparseDerivativeZtoYIndices.insert(sparseDerivativeZtoYIndices.end(), yIndices.begin(), yIndices.end());
    }

    // Post
    assert(sparseDerivativeZtoYIndices.size() >= P.size()); // at least {0, no z-indices, no y-indices} per point
}

/*
Given the data vector x (implicitly), the variables optimized for (y), the points where f is evaluated (p) and the per-point data selection function (sigma),
prepare the index arrays needed by a single SOPCompiled partition.

* sigma(p) must have the same nonzero length for all p in P. It gives the names of variables in x that should be passed to f for the point p.
* xIndices, sparseDerivativeZtoYIndices and yIndices are defined as for the SOPCompiled framework.
The outputs should initially be empty. They will be initialized for you.

x is given as locator function (locateInX) for efficiency: It can be reused for all partitions.
Use linearize to compute it.

Note that this function is not specific to any function f. However, the result should be used with a SOPCompiledFramework-instance created for a function f
that matches sigma (i.e. same amount of parameters and same meaning & order).
*/
template<typename X, typename PT, unsigned int lengthz>
CPU_FUNCTION(
void, prepareSOPCompiledInputForOnePartition, (
    _In_ const unordered_map<X, unsigned int>& locateInX,
    _In_ const vector<X>& Y,
    _In_ const vector<PT>& P,
    _In_ const function<void(PT, X*)>& sigma,

    _Out_ vector<unsigned int>& xIndices, _Out_ vector<unsigned int>& sparseDerivativeZtoYIndices, _Out_ vector<unsigned int>& yIndices
    ), "", PURITY_OUTPUT_POINTERS) {

    printf("prepareSOPCompiledInputForOnePartition\n");
    assert(locateInX.size());
    assert(Y.size() > 0);
    //assert(y.size() <= x.size()); // at most as many variables optimized over as there are x // cannot be verified because x is given implicitly
    // y should be a set (no repeated elements)
    assert(P.size() > 0);
    assert(sigma);

    assert(xIndices.size() == 0);
    assert(sparseDerivativeZtoYIndices.size() == 0);
    assert(yIndices.size() == 0);

    // Precompute location functions
    unordered_map<X, unsigned int> locateInY;
    build_locator(Y, locateInY);
    assert(locateInY.size() > 0);

    // Compute index sets
    SOPxIndices<X,PT,lengthz>(locateInX, P, sigma,
        xIndices);
    SOPyIndices(locateInX, Y,
        yIndices);
    SOPsparseDerivativeZtoYIndices<X, PT, lengthz>(locateInY, P, sigma,
        sparseDerivativeZtoYIndices);

    // Post
    assert(xIndices.size() >= P.size()); // at least one parameter per point
    assert(sparseDerivativeZtoYIndices.size() >= P.size()); // at least {0, no z-indices, no y-indices} per point
    assert(yIndices.size() == Y.size());
    // assert(allLessThan(yIndices, X.size()))
    // assert(allLessThan(sparseDerivativeZtoYIndices, X.size())) // assuming X is larger than lengthz -- wait it has to be!
}

template<typename X, typename PT, unsigned int lengthz>
CPU_FUNCTION(
    void
    , prepareSOPCompiledInput
    , (
    _In_ const unordered_map<X, float>& x,
    _In_ const vector<vector<PT>>& ps,
    _In_ const vector<vector<X>>& ys,
    _In_ const function<void(PT,X*)>& sigma,

    _Out_ vector<float>& xVector,
    _Out_ vector<vector<unsigned int>>& xIndicesPerPartition,
    _Out_ vector<vector<unsigned int>>& yIndicesPerPartition,
    _Out_ vector<vector<unsigned int>>& sparseDerivativeZtoYIndicesPerPartition,
    _Out_ unordered_map<X, unsigned int>& locateInX
    )
    , ""
    , PURITY_OUTPUT_POINTERS
    ) {

    printf("prepareSOPCompiledInput start\n");
    assert(ps.size() > 0);
    assert(ps.size() == ys.size()); // amount of partitions
    assert(x.size() > 0);
    assert(ys[0].size() > 0);
    assert(ps[0].size() > 0);

    //assert(sigma(ps[0][0]).size() > 0);

    const unsigned int partitions = restrictSize(ps.size());
    assert(partitions > 0);

    // prepare x
    assert(locateInX.size() == 0);
    linearize(x, xVector, locateInX);

    // prepare indices
    assert(0 == xIndicesPerPartition.size());
    assert(0 == yIndicesPerPartition.size());
    assert(0 == sparseDerivativeZtoYIndicesPerPartition.size());
    xIndicesPerPartition.resize(partitions);
    yIndicesPerPartition.resize(partitions);
    sparseDerivativeZtoYIndicesPerPartition.resize(partitions);

    DO(i, partitions) {
        prepareSOPCompiledInputForOnePartition<X, PT, lengthz>(
            /* in */
            locateInX, ys[i], ps[i], /*not &!*/ sigma,
            /* out */
            xIndicesPerPartition[i], sparseDerivativeZtoYIndicesPerPartition[i], yIndicesPerPartition[i]
            );

        assert(xIndicesPerPartition[i].size() > 0);
        assert(sparseDerivativeZtoYIndicesPerPartition[i].size() > 0); // todo there should be at least one row {k, y-ind,z-ind} in this array with k!=0
        assert(yIndicesPerPartition[i].size() > 0);
    }
}

/*
The type fSigma provides the following static members:
* const unsigned int lengthz, > 0
* const unsigned int lengthfz, > 0
* void f(_In_reads_(lengthz) const float* const input,  _Out_writes_all_(lengthfz) float* const out)
* void df(_In_range_(0, lengthz-1) int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out)
df(i) must be the derivative of f by the i-th argument
* void sigma(_In_ P p, _Out_writes(lengthz) X* sigmap) gives names of variables supplied to f at p

Note: Functions f and df should be CPU and GPU compilable.

! Modifies x in-place to contain the solution for efficiency. Other parts of system might be influenced by this *side-effect*!
*/
template<typename X, typename PT, typename fSigma>
class SOPDNamed {
private:
    void operator=(SOPDNamed&);
    SOPDNamed(SOPDNamed&);

public:
    SOPDProblem<fSigma>* sopd;
    _Inout_ unordered_map<X, float>& x;
    unordered_map<X, unsigned int> locateInX;

    CPU_MEMBERFUNCTION(, ~SOPDNamed, (), "") {
        delete sopd;
    }

    CPU_MEMBERFUNCTION(
        , SOPDNamed, (
        _Inout_ unordered_map<X, float>& x,
        _In_ const vector<vector<PT>>& ps,
        _In_ const vector<vector<X>>& ys), "", PURITY_OUTPUT_POINTERS) : x(x) {

        static_assert(fSigma::lengthz > 0, "lengthz must be positive");
        static_assert(fSigma::lengthfz > 0, "lengthfz must be positive");
        /*static_*/assert(&fSigma::sigma, "fSigma must define static function sigma");
        /*static_*/assert(&fSigma::f, "fSigma must define static function f");
        /*static_*/assert(&fSigma::df, "fSigma must define static function df");

        vector<float> xVector;
        vector<vector<unsigned int>> xIndicesPerPartition;
        vector<vector<unsigned int>> yIndicesPerPartition;
        vector<vector<unsigned int>> sparseDerivativeZtoYIndicesPerPartition;
        prepareSOPCompiledInput<X, PT, fSigma::lengthz>(x, ps, ys, &fSigma::sigma,
            xVector, xIndicesPerPartition, yIndicesPerPartition, sparseDerivativeZtoYIndicesPerPartition, locateInX);

        // prepare 
        sopd = new SOPDProblem<fSigma>(xVector, xIndicesPerPartition, yIndicesPerPartition, sparseDerivativeZtoYIndicesPerPartition);
    }

    CPU_MEMBERFUNCTION(void, solve, (unsigned int iterations = 1), "") {
        assert(iterations > 0);
        sopd->solve(iterations);
        auto x1Vector = sopd->getX();
        assert(x1Vector.size() == x.size());

        // copy to output x
        update_from_linearization(x, x1Vector, locateInX);
    }

    CPU_MEMBERFUNCTION(float, energy, (void), "") {
        return sopd->getEnergy();
    }
};































/* example

PTest[

Block[{select, sop, x},

select[i_] := {IdentityRule[x]};
sop = SparseOptimizationProblemDecomposedMake[{1-x},
select, {{0}}, {x -> 2.`}, {{x}}];

SOPDDataAsRules[SOPDSolve[sop, Method -> "SOPCompiled"]]

] (*end of Block*)

,
{x -> 1.`}
]

*/
enum Example1X { e1x };
enum Example1P { e1p0 };

struct Example1fSigma {
    static
        CPU_FUNCTION(void, sigma, (_In_ Example1P p, _Out_writes_(lengthz) Example1X* sigmap), "the per-point data selector function") {
        DBG_UNREFERENCED_PARAMETER(p);
        sigmap[0] = e1x;
    }

    static const unsigned int lengthz = 1, lengthfz = 1;

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy"){
        out[0] = 1.f - input[0];
    }

    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f") {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
        out[0] = -1.f;
    }
};

TEST(SOPDSolve1) {
    unordered_map<Example1X, float> x = {{e1x, 2.f}};
    assert(x[e1x] == 2.f);

    SOPDNamed<Example1X, Example1P, Example1fSigma> sopd(x, {{e1p0}}, {{e1x}});
    sopd.solve();

    assert(1 == x.size());
    assert(approximatelyEqual(x[e1x], 1.f), "%f", x[e1x]);
}


// Energy: 1.f for non solved, 0. for solved
TEST(SOPDEnergy1) {
    unordered_map<Example1X, float> x = {{e1x, 2.f}};

    SOPDNamed<Example1X, Example1P, Example1fSigma> sopd(x, {{e1p0}}, {{e1x}});
    assert(sopd.energy() == 1.f);
    sopd.solve();
    assert(sopd.energy() == 0.f);
}

/* example 2, nonlinear: root of 2.

PTest[
Block[{select, sop, x}, select[i_] := {IdentityRule[x]};
sop = SparseOptimizationProblemDecomposedMake[{2 - x*x},
select, {{0}}, {x -> 2.`}, {{x}}];
SOPDGetX0[SOPDSolve[sop, MaxIterations -> 16]]] (*end of Block*)
, Sqrt@2., {SameTest -> ApproximatelyEqual}]

*/

struct Example2fSigma {
    static
        CPU_FUNCTION(void, sigma, (_In_ Example1P p, _Out_writes_(lengthz) Example1X* sigmap), "the per-point data selector function") {
        Example1fSigma::sigma(p, sigmap);
    }

    static const unsigned int lengthz = 1, lengthfz = 1;

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy"){
        out[0] = 2.f - input[0] * input[0];
    }

    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f") {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
        out[0] = -2.f* input[0];
    }
};

TEST(SOPDSolve2) {
    unordered_map<Example1X, float> x = {{e1x, 2.f}};
    assert(x[e1x] == 2.f);


    SOPDNamed<Example1X, Example1P, Example2fSigma> sopd(x, {{e1p0}}, {{e1x}});
    sopd.solve(16);

    assert(1 == x.size());
    assert(approximatelyEqual(x[e1x], sqrtf(2.f)), "%f", x[e1x]);
}

/* example 3, nonlinear & multi-variable, not optimized over everything: root of 2. and 3.
data:

-1 -> 2.
1 -> x

-2 -> 3.
2 -> y

sigma(p): P = {1,2}

a -> p
b -> -p

f(a,b) = b - a*a

Y = {1,2}
*/
typedef int Example3X;
typedef int Example3P;

struct Example3fSigma {
    static
        CPU_FUNCTION(void, sigma, (_In_ Example3P p, _Out_writes_(lengthz) Example3X* sigmap), "the per-point data selector function") {
        sigmap[0] = p;
        sigmap[1] = -p;
    }

    static const unsigned int lengthz = 2, lengthfz = 1;

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy"){
        out[0] = input[1] - input[0] * input[0];
    }

    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f") {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
        switch (i) {
        case 0: out[0] = -2.f * input[0]; return;
        case 1: out[1] = 1.f;
            fatalError("this derivative should not be needed in the given objective");
            return;
        }
    }
};

TEST(SOPDSolve3) {
    unordered_map<Example3X, float> x = {{-1, 2.f}, {1, 2.f}, {-2, 3.f}, {2, 3.f}};
    SOPDNamed<Example3X, Example3P, Example3fSigma> sopd(x, /*p*/{{1, 2}}, /*Y*/{{1, 2}});
    sopd.solve(16);

    assert(4 == x.size());
    assert(approximatelyEqual(x[1], sqrtf(2.f)), "%f", x[1]);
    assert(approximatelyEqual(x[2], sqrtf(3.f)), "%f", x[2]);
    // rest perfectly unchanged
    assert(x[-1] == 2.f);
    assert(x[-2] == 3.f);
}


/* example 4: same as example 3, but with the Y in two separate partitions
the result should be exactly the same, but the computation is parallelizable now*/

TEST(SOPDSolve4) {
    unordered_map<Example3X, float> x = {{-1, 2.f}, {1, 2.f}, {-2, 3.f}, {2, 3.f}};
    SOPDNamed<Example3X, Example3P, Example3fSigma> sopd(x, /*p*/{{1}, {2}}, /*Y*/{{1}, {2}});
    sopd.solve(/*iterations*/16);


    assert(4 == x.size());
    assert(approximatelyEqual(x[1], sqrtf(2.f)), "%f", x[1]);
    assert(approximatelyEqual(x[2], sqrtf(3.f)), "%f", x[2]);
    // rest perfectly unchanged
    assert(x[-1] == 2.f);
    assert(x[-2] == 3.f);
}

















































KERNEL trueTestKernel() {
    assert(true, "this should not fail on the GPU");
}

KERNEL failTestKernel() {
    fatalError("this should fail on the GPU");
}

void sehDemo() {

    OutputDebugStringA("== causing and catching division by zero structured exception\n");
    __try {
        int x = 5; x /= 0;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) { //< could use GetExceptionCode, GetExceptionInformation here
    }

    OutputDebugStringA("==\n");
}

TEST(trueTest) {
    assert(true, "this should not fail on the CPU");

    LAUNCH_KERNEL(trueTestKernel, 1, 1);

    BEGIN_SHOULD_FAIL("assert false cpu");
    fatalError("this should fail on the CPU");
    END_SHOULD_FAIL("assert false cpu");

    // cannot have the GPU assertion-fail, because this resets the device
    // TODO work around, just *exit* (return -- or exit instruction?) from the kernel on assertion failure instead of illegal instruction,
    // at least if no debugger present
    /*
    BEGIN_SHOULD_FAIL();
    LAUNCH_KERNEL(failTestKernel, 1, 1);
    END_SHOULD_FAIL();
    */

    sehDemo();
}












































// 2 to 4 & X dimensional linear algebra library
// TODO consider using this with range-checked datatypes and overflow-avoiding integers
// Better yet, search a standard library which compiles for CUDA

namespace vecmath {

    //////////////////////////////////////////////////////////////////////////
    //						Basic Vector Structure
    //////////////////////////////////////////////////////////////////////////

    template <class T> struct Vector2_{
        union {
            struct { T x, y; }; // standard names for components
            struct { T s, t; }; // standard names for components
            struct { T width, height; };
            T v[2];     // array access
        };
    };

    template <class T> struct Vector3_{
        union {
            struct{ T x, y, z; }; // standard names for components
            struct{ T r, g, b; }; // standard names for components
            struct{ T s, t, p; }; // standard names for components
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

    template<class T, int s> struct VectorX_
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
        CPU_AND_GPU Vector2(){} // Default constructor
        CPU_AND_GPU Vector2(const T &t) { this->x = t; this->y = t; } // Scalar constructor
        CPU_AND_GPU Vector2(const T *tp) { this->x = tp[0]; this->y = tp[1]; } // Construct from array			            
        CPU_AND_GPU Vector2(const T v0, const T v1) { this->x = v0; this->y = v1; } // Construct from explicit values
        CPU_AND_GPU Vector2(const Vector2_<T> &v) { this->x = v.x; this->y = v.y; }// copy constructor

        CPU_AND_GPU explicit Vector2(const Vector3_<T> &u)  { this->x = u.x; this->y = u.y; }
        CPU_AND_GPU explicit Vector2(const Vector4_<T> &u)  { this->x = u.x; this->y = u.y; }

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
        CPU_AND_GPU friend Vector2<T> operator + (const Vector2<T> &lhs, const Vector2<T> &rhs)  {
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
        CPU_AND_GPU Vector3(){} // Default constructor
        CPU_AND_GPU Vector3(const T &t)	{ this->x = t; this->y = t; this->z = t; } // Scalar constructor
        CPU_AND_GPU Vector3(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; } // Construct from array
        CPU_AND_GPU Vector3(const T v0, const T v1, const T v2) { this->x = v0; this->y = v1; this->z = v2; } // Construct from explicit values
        CPU_AND_GPU explicit Vector3(const Vector4_<T> &u)	{ this->x = u.x; this->y = u.y; this->z = u.z; }
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

        CPU_AND_GPU const T *getValues() const	{ return this->v; }
        CPU_AND_GPU Vector3<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; return *this; }

        // indexing operators
        CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
        CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

        // type-cast operators
        CPU_AND_GPU operator T *()	{ return this->v; }
        CPU_AND_GPU operator const T *() const { return this->v; }

        ////////////////////////////////////////////////////////
        //  Math operators
        ////////////////////////////////////////////////////////

        // scalar multiply assign
        CPU_AND_GPU friend Vector3<T> &operator *= (Vector3<T> &lhs, T d)	{
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
        CPU_AND_GPU friend Vector3<T> &operator /= (Vector3<T> &lhs, const Vector3<T> &rhs)	{
            lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs;
        }

        // component-wise vector add assign
        CPU_AND_GPU friend Vector3<T> &operator += (Vector3<T> &lhs, const Vector3<T> &rhs)	{
            lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs;
        }

        // component-wise vector subtract assign
        CPU_AND_GPU friend Vector3<T> &operator -= (Vector3<T> &lhs, const Vector3<T> &rhs) {
            lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs;
        }

        // unary negate
        CPU_AND_GPU friend Vector3<T> operator - (const Vector3<T> &rhs)	{
            Vector3<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; return rv;
        }

        // vector add
        CPU_AND_GPU friend Vector3<T> operator + (const Vector3<T> &lhs, const Vector3<T> &rhs){
            Vector3<T> rv(lhs); return rv += rhs;
        }

        // vector subtract
        CPU_AND_GPU friend Vector3<T> operator - (const Vector3<T> &lhs, const Vector3<T> &rhs){
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
        CPU_AND_GPU friend Vector3<T> operator * (const Vector3<T> &lhs, const Vector3<T> &rhs)	{
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
    template <typename T1, typename T2> CPU_AND_GPU inline bool operator == (const Vector3<T1> &lhs, const Vector3<T2> &rhs){
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
        CPU_AND_GPU friend Vector4<T> &operator /= (Vector4<T> &lhs, T d){
            lhs.x /= d; lhs.y /= d; lhs.z /= d; lhs.w /= d; return lhs;
        }

        // component-wise vector divide assign
        CPU_AND_GPU friend Vector4<T> &operator /= (Vector4<T> &lhs, const Vector4<T> &rhs) {
            lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs;
        }

        // component-wise vector add assign
        CPU_AND_GPU friend Vector4<T> &operator += (Vector4<T> &lhs, const Vector4<T> &rhs)	{
            lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs;
        }

        // component-wise vector subtract assign
        CPU_AND_GPU friend Vector4<T> &operator -= (Vector4<T> &lhs, const Vector4<T> &rhs)	{
            lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs;
        }

        // unary negate
        CPU_AND_GPU friend Vector4<T> operator - (const Vector4<T> &rhs)	{
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

        friend std::ostream& operator<<(std::ostream& os, const Vector4<T>& dt){
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
        CPU_AND_GPU friend Vector6<T> &operator /= (Vector6<T> &lhs, T d){
            lhs[0] /= d; lhs[1] /= d; lhs[2] /= d; lhs[3] /= d; lhs[4] /= d; lhs[5] /= d; return lhs;
        }

        // component-wise vector divide assign
        CPU_AND_GPU friend Vector6<T> &operator /= (Vector6<T> &lhs, const Vector6<T> &rhs) {
            lhs[0] /= rhs[0]; lhs[1] /= rhs[1]; lhs[2] /= rhs[2]; lhs[3] /= rhs[3]; lhs[4] /= rhs[4]; lhs[5] /= rhs[5]; return lhs;
        }

        // component-wise vector add assign
        CPU_AND_GPU friend Vector6<T> &operator += (Vector6<T> &lhs, const Vector6<T> &rhs)	{
            lhs[0] += rhs[0]; lhs[1] += rhs[1]; lhs[2] += rhs[2]; lhs[3] += rhs[3]; lhs[4] += rhs[4]; lhs[5] += rhs[5]; return lhs;
        }

        // component-wise vector subtract assign
        CPU_AND_GPU friend Vector6<T> &operator -= (Vector6<T> &lhs, const Vector6<T> &rhs)	{
            lhs[0] -= rhs[0]; lhs[1] -= rhs[1]; lhs[2] -= rhs[2]; lhs[3] -= rhs[3]; lhs[4] -= rhs[4]; lhs[5] -= rhs[5];  return lhs;
        }

        // unary negate
        CPU_AND_GPU friend Vector6<T> operator - (const Vector6<T> &rhs)	{
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

        friend std::ostream& operator<<(std::ostream& os, const Vector6<T>& dt){
            os << dt[0] << ", " << dt[1] << ", " << dt[2] << ", " << dt[3] << ", " << dt[4] << ", " << dt[5];
            return os;
        }
    };

    /*
    s - dimensional vector over the field T
    T must provide operators for all the operations used.
    */
    template <class T, int s> class VectorX : public VectorX_ < T, s >
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
        CPU_AND_GPU void Clear(T v){
            for (int i = 0; i < s; i++)
                this->v[i] = v;
        }

        CPU_AND_GPU void setZeros(){
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
        CPU_AND_GPU friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, T d){
            for (int i = 0; i < s; i++) lhs[i] /= d; return lhs;
        }

        // component-wise vector divide assign
        CPU_AND_GPU friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
            for (int i = 0; i < s; i++) lhs[i] /= rhs[i]; return lhs;
        }

        // component-wise vector add assign
        CPU_AND_GPU friend VectorX<T, s> &operator += (VectorX<T, s> &lhs, const VectorX<T, s> &rhs)	{
            for (int i = 0; i < s; i++) lhs[i] += rhs[i]; return lhs;
        }

        // component-wise vector subtract assign
        CPU_AND_GPU friend VectorX<T, s> &operator -= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs)	{
            for (int i = 0; i < s; i++) lhs[i] -= rhs[i]; return lhs;
        }

        // unary negate
        CPU_AND_GPU friend VectorX<T, s> operator - (const VectorX<T, s> &rhs)	{
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

        friend std::ostream& operator<<(std::ostream& os, const VectorX<T, s>& dt){
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
    template< class T> CPU_AND_GPU inline T normalize(const T &vec)	{
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
    CPU_AND_GPU inline T maxV(const T &lhs, const T &rhs)	{
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
    template <class T, int s> class VectorX;

    //////////////////////////////////////////////////////////////////////////
    //						Basic Matrix Structure
    //////////////////////////////////////////////////////////////////////////

    template <class T> struct Matrix4_{
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

    template <class T> struct Matrix3_{
        union { // Warning: see the header in this file for the special matrix order
            struct {
                T m00, m01, m02; // |0, 3, 6|     |m00, m10, m20|
                T m10, m11, m12; // |1, 4, 7|     |m01, m11, m21|
                T m20, m21, m22; // |2, 5, 8|     |m02, m12, m22|
            };
            T m[9];
        };
    };

    template<class T, int s> struct MatrixSQX_{
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
        CPU_AND_GPU Matrix4(const T *m)	{ setValues(m); }
        CPU_AND_GPU Matrix4(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22, T a23, T a30, T a31, T a32, T a33)	{
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

        CPU_AND_GPU inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 16); }
        CPU_AND_GPU inline const T *getValues() const { return this->m; }
        CPU_AND_GPU inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

        // Element access
        CPU_AND_GPU inline T &operator()(int x, int y)	{ return at(x, y); }
        CPU_AND_GPU inline const T &operator()(int x, int y) const	{ return at(x, y); }
        CPU_AND_GPU inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
        CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
        CPU_AND_GPU inline T &at(int x, int y) { return this->m[y | (x << 2)]; }
        CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[y | (x << 2)]; }

        // set values
        CPU_AND_GPU inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 16); }
        CPU_AND_GPU inline void setValues(T r)	{ for (int i = 0; i < 16; i++)	this->m[i] = r; }
        CPU_AND_GPU inline void setZeros() { memset(this->m, 0, sizeof(T) * 16); }
        CPU_AND_GPU inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = this->m33 = 1; }
        CPU_AND_GPU inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
        CPU_AND_GPU inline void setScale(const Vector3_<T> &s) { this->m00 = s.v[0]; this->m11 = s.v[1]; this->m22 = s.v[2]; }
        CPU_AND_GPU inline void setTranslate(const Vector3_<T> &t) { for (int y = 0; y < 3; y++) at(3, y) = t.v[y]; }
        CPU_AND_GPU inline void setRow(int r, const Vector4_<T> &t){ for (int x = 0; x < 4; x++) at(x, r) = t.v[x]; }
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

        CPU_AND_GPU inline friend Matrix4 operator * (const Matrix4 &lhs, const Matrix4 &rhs)	{
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

        CPU_AND_GPU inline friend Vector4<T> operator *(const Vector4<T> &lhs, const Matrix4 &rhs){
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
        CPU_AND_GPU Matrix3(const T *m)	{ setValues(m); }
        CPU_AND_GPU Matrix3(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22)	{
            this->m00 = a00; this->m01 = a01; this->m02 = a02;
            this->m10 = a10; this->m11 = a11; this->m12 = a12;
            this->m20 = a20; this->m21 = a21; this->m22 = a22;
        }

        CPU_AND_GPU inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 9); }
        CPU_AND_GPU inline const T *getValues() const { return this->m; }
        CPU_AND_GPU inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

        // Element access
        CPU_AND_GPU inline T &operator()(int x, int y)	{ return at(x, y); }
        CPU_AND_GPU inline const T &operator()(int x, int y) const	{ return at(x, y); }
        CPU_AND_GPU inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
        CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
        CPU_AND_GPU inline T &at(int x, int y) { return this->m[x * 3 + y]; }
        CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[x * 3 + y]; }

        // set values
        CPU_AND_GPU inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 9); }
        CPU_AND_GPU inline void setValues(const T r)	{ for (int i = 0; i < 9; i++)	this->m[i] = r; }
        CPU_AND_GPU inline void setZeros() { memset(this->m, 0, sizeof(T) * 9); }
        CPU_AND_GPU inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = 1; }
        CPU_AND_GPU inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
        CPU_AND_GPU inline void setScale(const Vector3_<T> &s) { this->m00 = s[0]; this->m11 = s[1]; this->m22 = s[2]; }
        CPU_AND_GPU inline void setRow(int r, const Vector3_<T> &t){ for (int x = 0; x < 3; x++) at(x, r) = t[x]; }
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

        CPU_AND_GPU inline friend Matrix3 operator * (const Matrix3 &lhs, const Matrix3 &rhs)	{
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

        CPU_AND_GPU inline friend Vector3<T> operator *(const Vector3<T> &lhs, const Matrix3 &rhs){
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

        friend std::ostream& operator<<(std::ostream& os, const Matrix3<T>& dt)	{
            for (int y = 0; y < 3; y++)
                os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << "\n";
            return os;
        }
    };

    template<class T, int s>
    class MatrixSQX : public MatrixSQX_ < T, s >
    {
    public:
        CPU_AND_GPU MatrixSQX() { this->dim = s; this->sq = s*s; }
        CPU_AND_GPU MatrixSQX(T t) { this->dim = s; this->sq = s*s; setValues(t); }
        CPU_AND_GPU MatrixSQX(const T *m)	{ this->dim = s; this->sq = s*s; setValues(m); }
        CPU_AND_GPU MatrixSQX(const T m[s][s])	{ this->dim = s; this->sq = s*s; setValues((T*)m); }

        CPU_AND_GPU inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 16); }
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
        CPU_AND_GPU inline T &operator()(int x, int y)	{ return at(x, y); }
        CPU_AND_GPU inline const T &operator()(int x, int y) const	{ return at(x, y); }
        CPU_AND_GPU inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
        CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
        CPU_AND_GPU inline T &at(int x, int y) { return this->m[y * s + x]; }
        CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[y * s + x]; }

        // indexing operators
        CPU_AND_GPU T &operator [](int i) { return this->m[i]; }
        CPU_AND_GPU const T &operator [](int i) const { return this->m[i]; }

        // set values
        CPU_AND_GPU inline void setValues(const T *mp) { for (int i = 0; i < s*s; i++) this->m[i] = mp[i]; }
        CPU_AND_GPU inline void setValues(T r)	{ for (int i = 0; i < s*s; i++)	this->m[i] = r; }
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

        CPU_AND_GPU inline friend  MatrixSQX<T, s> operator * (const  MatrixSQX<T, s> &lhs, const  MatrixSQX<T, s> &rhs)	{
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
    // Usually, A = B^TB and b = B^Ty, as present in the normal-equations for solving linear least-squares problems
    class Cholesky
    {
    private:
        std::vector<float> cholesky;
        int size, rank;

    public:
        // Solve Ax = b for A symmetric positive-definite of size*size
        template<int m>
        static VectorX<float, m> solve(
            const MatrixSQX<float, m>& mat,
            const VectorX<float, m>&  b) {

            auto x = VectorX<float, m>();
            solve((const float*)mat.m, m, (const float*)b.v, x.v);
            return x;

        }

        // Solve Ax = b for A symmetric positive-definite of size*size
        static void solve(const float* mat, int size, const float* b, float* result) {
            Cholesky cholA(mat, size);
            cholA.Backsub(result, b);
        }

        /// \f[A = LL*\f]
        /// Produces Cholesky decomposition of the
        /// symmetric, positive-definite matrix mat of dimension size*size
        /// \f$L\f$ is a lower triangular matrix with real and positive diagonal entries
        ///
        /// Note: assertFinite is used to detect singular matrices and other non-supported cases.
        Cholesky(const float *mat, int size)
        {
            this->size = size;
            this->cholesky.resize(size*size);

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
        void Backsub(
            float *x,  //!< out \f$x\f$
            const float *b //!< input \f$b\f$
            ) const
        {
            // Forward
            std::vector<float> y(size);
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
        }
    };

    TEST(testCholesky) {
        float m[] = {
            1, 0,
            0, 1
        };
        float b[] = {1, 2};
        float r[2];
        Cholesky::solve(m, 2, b, r);
        assert(r[0] == b[0] && r[1] == b[1]);
    }
























































    typedef unsigned char uchar;
    typedef unsigned short ushort;
    typedef unsigned long ulong;

    typedef class Matrix3<float> Matrix3f;
    typedef class Matrix4<float> Matrix4f;

    typedef class Vector2<short> Vector2s;
    typedef class Vector2<int> Vector2i;
    inline dim3 getGridSize(Vector2i taskSize, dim3 blockSize)
    {
        return getGridSize(dim3(taskSize.x, taskSize.y), blockSize);
    }
    typedef class Vector2<float> Vector2f;
    typedef class Vector2<double> Vector2d;

    typedef class Vector3<short> Vector3s;
    typedef class Vector3<double> Vector3d;
    typedef class Vector3<int> Vector3i;
    typedef class Vector3<uint> Vector3ui;
    typedef class Vector3<uchar> Vector3u;
    typedef class Vector3<float> Vector3f;

    // Documents that something is a unit-vector (e.g. normal vector), i.e. \in S^2
    typedef Vector3f UnitVector;

    typedef class Vector4<float> Vector4f;
    typedef class Vector4<int> Vector4i;
    typedef class Vector4<short> Vector4s;
    typedef class Vector4<uchar> Vector4u;

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
        union
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
























    // Framework for building and solving (linear) least squares fitting problems 
	// (on the GPU)
    // c.f. constructAndSolve.nb

    namespace LeastSquares {



        /// Constructor must define
        /// Constructor::ExtraData(), add, atomicAdd
        /// static const uint Constructor::m
        /// bool Constructor::generate(const uint i, VectorX<float, m>& , float& bi)
        template<typename Constructor>
        struct AtA_Atb_Add {
            static const uint m = Constructor::m;
            typedef typename MatrixSQX<float, m> AtA;
            typedef typename VectorX<float, m> Atb;
            typedef typename Constructor::ExtraData ExtraData;

            struct ElementType {
                typename AtA _AtA;
                typename Atb _Atb;
                typename ExtraData _extraData;
                CPU_AND_GPU ElementType() {} // uninitialized on purpose
                CPU_AND_GPU ElementType(AtA _AtA, Atb _Atb, ExtraData _extraData) : _AtA(_AtA), _Atb(_Atb), _extraData(_extraData) {}
            };

            static GPU_ONLY bool generate(const uint i, ElementType& out) {
                // Some threads contribute zero
                VectorX<float, m> ai; float bi;
                if (!Constructor::generate(i, ai, bi, out._extraData)) return false;

                // Construct ai_aiT (an outer product matrix) and ai_bi
                out._AtA = MatrixSQX<float, m>::make_aaT(ai);
                out._Atb = ai * bi;
                return true;
            }

            static CPU_AND_GPU ElementType neutralElement() {
                return ElementType(
                    AtA::make_zeros(),
                    Atb::make_zeros(),
                    ExtraData());
            }

            static GPU_ONLY ElementType operate(ElementType & l, ElementType & r) {
                return ElementType(
                    l._AtA + r._AtA,
                    l._Atb + r._Atb,
                    ExtraData::add(l._extraData, r._extraData)
                    );
            }

            static GPU_ONLY void atomicOperate(DEVICEPTR(ElementType&) result, ElementType & integrand) {
                for (int r = 0; r < m*m; r++)
                    atomicAdd(
                    &result._AtA[r],
                    integrand._AtA[r]);

                for (int r = 0; r < m; r++)
                    atomicAdd(
                    &result._Atb[r],
                    integrand._Atb[r]);

                ExtraData::atomicAdd(result._extraData, integrand._extraData);
            }
        };



        /**
        Build A^T A and A^T b where A is <=n x m and b has <=n elements.

        Row i (0-based) of A and b[i] are generated by bool Constructor::generate(uint i, VectorX<float, m> out_ai, float& out_bi).
        It is thrown away if generate returns false.
        */
        template<class Constructor>
        AtA_Atb_Add<Constructor>::ElementType construct(const uint n, dim3 gridDim = dim3(0, 0, 0), dim3 blockDim = dim3(0, 0, 0)) {
            assert(Constructor::m < 100);
            return transform_reduce_if<AtA_Atb_Add<Constructor>>(n, gridDim, blockDim);
        }

        /// Given a Constructor with method
        ///     static __device__ Constructor::generate(uint i, VectorX<float, m> out_ai, float& out_bi)
        /// and static uint Constructor::m
        /// build the equation system Ax = b with out_ai, out_bi in the i-th row/entry of A or b
        /// Then solve this in the least-squares sense and return x.
        ///
        /// i goes from 0 to n-1.
        ///
        /// Custom scheduling can be used and any custom Constructor::ExtraData can be summed up over all i.
        ///
        /// \see construct
        template<class Constructor>
        AtA_Atb_Add<Constructor>::Atb constructAndSolve(int n, dim3 gridDim, dim3 blockDim, Constructor::ExtraData& out_extra_sum = Constructor::ExtraData()) {
            auto result = construct<Constructor>(n, gridDim, blockDim);
            out_extra_sum = result._extraData;
            cout << result._AtA << endl;
            cout << result._Atb << endl;

            return Cholesky::solve(result._AtA, result._Atb);
        }
    }









    // test constructAndSolve



    struct ConstructExampleEquation {
        // Amount of columns, should be small
        static const uint m = 6;
        struct ExtraData {
            // User specified payload to be summed up alongside:
            uint count;

            // Empty constructor must generate neutral element
            CPU_AND_GPU ExtraData() : count(0) {}

            static GPU_ONLY ExtraData add(const ExtraData& l, const ExtraData& r) {
                ExtraData o;
                o.count = l.count + r.count;
                return o;
            }
            static GPU_ONLY void atomicAdd(DEVICEPTR(ExtraData&) result, const ExtraData& integrand) {
                ::atomicAdd(&result.count, integrand.count);
            }
        };

        static GPU_ONLY bool generate(const uint i, VectorX<float, m>& out_ai, float& out_bi/*[1]*/, ExtraData& out_extra) {
            assert(threadIdx.y == threadIdx.z && threadIdx.y == 0);
            assert(threadIdx.x < 32);

            for (int j = 0; j < m; j++) {
                out_ai[j] = 0;
                if (i == j || i + 1 == j || i == j + 1 || i == 0 || j == 0 || i % m == j)
                    out_ai[j] = 1;
            }
            out_bi = i + 1;

            bool ok = blockIdx.x <= 1 /* i.e. i <= 63, since blockDim = 32 */; // or: i % 2 == 0;
            out_extra.count = ok;

            return ok;//
            // i % 2 == 0;
        }
    };

    TEST(constructExampleEquation) {
        // c.f. constructAndSolve.nb
        const int m = ConstructExampleEquation::m;
        const int n = 512;


        auto y = LeastSquares::construct<ConstructExampleEquation>(n, dim3(0, 0, 0), dim3(32, 1, 1));
        assert(y._extraData.count == 64);

        auto x = LeastSquares::constructAndSolve<ConstructExampleEquation>(n, dim3(0, 0, 0), dim3(32, 1, 1));
        std::array<float, m> expect = {43.22650547302665, -10.45935368692164,
            -10.438367760716343, -8.180392124176143,
            -10.817604850187244, -11.47948083867651
        };

        assertApproxEqual(x, VectorX<float, m>(expect));
    }


    struct ConstructExampleEquation2 {
        // Amount of columns, should be small
        static const uint m = 6;
        struct ExtraData {
            // User specified payload to be summed up alongside:
            uint count;

            // Empty constructor must generate neutral element
            CPU_AND_GPU ExtraData() : count(0) {}

            static GPU_ONLY ExtraData add(const ExtraData& l, const ExtraData& r) {
                ExtraData o;
                o.count = l.count + r.count;
                return o;
            }
            static GPU_ONLY void atomicAdd(DEVICEPTR(ExtraData&) result, const ExtraData& integrand) {
                ::atomicAdd(&result.count, integrand.count);
            }
        };

        static GPU_ONLY bool generate(const uint i, VectorX<float, m>& out_ai, float& out_bi/*[1]*/, ExtraData& out_extra) {
            assert(threadIdx.y == threadIdx.z && threadIdx.y == 0);
            assert(threadIdx.x < 32);

            for (int j = 0; j < m; j++) {
                out_ai[j] = 0;
                if (i == j || i + 1 == j || i == j + 1 || i == 0 || j == 0 || i % m == j)
                    out_ai[j] = 1;
            }
            out_bi = i + 1;

            bool ok = i % 2 == 0; // <- this is the only change
            out_extra.count = ok;

            return ok;
        }
    };

    TEST(constructExampleEquation2) {
        // c.f. constructAndSolve.nb
        const int m = ConstructExampleEquation2::m;
        const int n = 512;

        auto y = LeastSquares::construct<ConstructExampleEquation2>(n, dim3(0, 0, 0), dim3(32, 1, 1));
        assert(y._extraData.count == 256);

        auto x = LeastSquares::constructAndSolve<ConstructExampleEquation2>(n, dim3(0, 0, 0), dim3(32, 1, 1));
        std::array<float, m> expect = {260.0155642023345, -85.33073929961078, -2.0155642023346445, \
            - 86.32295719844366, 0.9766536964980332, -169.6692607003891};

        assertApproxEqual(x, VectorX<float, m>(expect));
    }




}
using namespace vecmath;






















// integration 

/** \brief
Up to @ref maxW observations per voxel are averaged.
Beyond that a sliding average is computed.
*/
#define maxW 100



















/// depth threshold for the  tracker

/// For ITMDepthTracker: ICP distance threshold for lowest resolution (later iterations use lower distances)
/// In world space squared -- TODO maybe define heuristically from voxelsize/mus
#define depthTrackerICPMaxThreshold (0.1f * 0.1f)

/// For ITMDepthTracker: ICP iteration termination threshold
#define depthTrackerTerminationThreshold 1e-3f




// rendering

/** @{ */
/** \brief
Fallback parameters: consider only parts of the
scene from @p viewFrustum_min in front of the camera
to a distance of @p viewFrustum_max (world-space distance). Usually the
actual depth range should be determined
automatically by a ITMLib::Engine::ITMVisualisationEngine.

aka.
viewRange_min depthRange
zmin zmax
*/
#define viewFrustum_min 0.2f
#define viewFrustum_max 6.0f





























// Memory-block and 'image'/large-matrix management
namespace memory {

    enum MemoryCopyDirection { CPU_TO_CPU, CPU_TO_CUDA, CUDA_TO_CPU, CUDA_TO_CUDA };
    enum MemoryDeviceType { MEMORYDEVICE_CPU, MEMORYDEVICE_CUDA };
    enum MemoryBlockState { SYNCHRONIZED, CPU_AHEAD, GPU_AHEAD };

    void testMemblock();

    /// Pointer to block of mutable memory, consistent in GPU and CPU memory.
    /// 
    /*
    Notes:
    * always aquire pointers to the memory anew using GetData
    * when CPU memory is considered ahead, GPU memory must be updated via manual request from the host-side -- GPU cannot pull memory it needs
    * should be more efficient than CUDA managed memory (?)

    */
    template <typename T>
    struct MemoryBlock : public Managed
    {
    private:
        friend void testMemblock();
        mutable MemoryBlockState state;

        T* data_cpu;
        DEVICEPTR(T)* data_cuda;

        size_t dataSize__;
        void dataSize_(size_t dataSize) { dataSize__ = dataSize; }

    public:
        CPU_AND_GPU size_t dataSize_() const { return dataSize__; }
        __declspec(property(get = dataSize_, put = dataSize_)) size_t dataSize;

        void Allocate(size_t dataSize) {
            this->dataSize = dataSize;

            data_cpu = new T[dataSize];
            cudaMalloc((T**)&data_cuda, dataSizeInBytes());

            Clear(0);
        }

        MemoryBlock(size_t dataSize) {
            Allocate(dataSize);
            assert(SYNCHRONIZED == state);
        }

        virtual ~MemoryBlock() {
            delete[] data_cpu;
            cudaFree(data_cuda);

            // HACK DEBUGGING set state to illegal values [
            data_cpu = data_cuda = (T*)0xffffffffffffffffUI64;
            dataSize = 0xffffffffffffffffUI64;
            state = (MemoryBlockState)0xffffffff;
            // ]
        }

        // after a call to this, SYNCHRONIZED == state
        void Synchronize() const /* const since semantically the thing does not change */ {
            if (state == GPU_AHEAD) {
                printf("gpu -> cpu %d\n", dataSizeInBytes());
                cudaMemcpy(data_cpu, data_cuda, dataSizeInBytes(), cudaMemcpyDeviceToHost);
            }
            else if (state == CPU_AHEAD) {
                printf("cpu -> gpu %d\n", dataSizeInBytes());
                cudaMemcpy(data_cuda, data_cpu, dataSizeInBytes(), cudaMemcpyHostToDevice);
            }
            else assert(state == SYNCHRONIZED);
            cudaDeviceSynchronize();
            state = SYNCHRONIZED;
        }

        bool operator==(const MemoryBlock& other) const {

            auto y = memcmp(data_cpu, other.data_cpu, MIN(other.dataSizeInBytes(), dataSizeInBytes())); // DEBUG

            if (other.dataSizeInBytes() != dataSizeInBytes())
                return false;
            Synchronize();
            auto z = memcmp(data_cpu, other.data_cpu, MIN(other.dataSizeInBytes(), dataSizeInBytes())); // DEBUG

            other.Synchronize();

            auto w = memcmp(data_cpu, other.data_cpu, other.dataSizeInBytes());
            cudaDeviceSynchronize(); // memcmp fails otherwise with -1 when cuda is still running
            auto x = memcmp(data_cpu, other.data_cpu, other.dataSizeInBytes());
            assert(x == y);
            return x == 0;
        }

        void SetFrom(const MemoryBlock& copyFrom) {
            copyFrom.Synchronize();
            Allocate(copyFrom.dataSize);
            assert(this->dataSizeInBytes() == copyFrom.dataSizeInBytes());
            memcpy(data_cpu, copyFrom.data_cpu, copyFrom.dataSizeInBytes());
            state = CPU_AHEAD;
            assert(memcmp(data_cpu, copyFrom.data_cpu, copyFrom.dataSizeInBytes()) == 0);

            Synchronize();

            assert(*this == copyFrom);
            assert(SYNCHRONIZED == state);
        }

        MemoryBlock(const MemoryBlock& copyFrom) {
            SetFrom(copyFrom);
            assert(SYNCHRONIZED == state);
        }

        SERIALIZE_VERSION(1);
        /* format:
        version :: int
        dataSize (amount of elements) :: size_t
        sizeof(T) :: size_t
        data :: BYTE[dataSizeInBytes()]
        */
        void serialize(ofstream& file) {
            Synchronize();
            auto p = file.tellp(); // DEBUG

            SERIALIZE_WRITE_VERSION(file);
            bin(file, dataSize);
            bin(file, (size_t)sizeof(T));
            file.write((const char*)data_cpu, dataSizeInBytes());

            assert(file.tellp() - p == sizeof(int) + sizeof(size_t) * 2 + dataSizeInBytes(), "%I64d != %I64d",
                file.tellp() - p,
                sizeof(int) + sizeof(size_t) * 2 + dataSizeInBytes());
        }

        void SetFrom(char* data, size_t dataSize) {
            Allocate(dataSize);
            memcpy((char*)data_cpu, data, dataSizeInBytes());
            state = CPU_AHEAD;
            Synchronize();
        }

        // TODO should be able to require it to be the same size (because Scene serialization format expects localVBA to be of a certain size)
        // loses current data
        void deserialize(ifstream& file) {
            SERIALIZE_READ_VERSION(file);

            Allocate(bin<size_t>(file));
            assert(bin<size_t>(file) == sizeof(T));

            file.read((char*)data_cpu, dataSizeInBytes());
            state = CPU_AHEAD;
            Synchronize();
        }

        CPU_AND_GPU size_t dataSizeInBytes() const {
            return dataSize * sizeof(T);
        }

        /** Get the data pointer on CPU or GPU. */
        CPU_AND_GPU DEVICEPTR(T)* const GetData(MemoryDeviceType memoryType)
        {
            switch (memoryType)
            {
#if !GPU_CODE
            case MEMORYDEVICE_CPU:
                if (state != CPU_AHEAD) Synchronize();
                state = CPU_AHEAD; // let the CPU get arbitrarily far ahead
                return data_cpu;
#endif
            case MEMORYDEVICE_CUDA:
#if !GPU_CODE
                Synchronize(); // can only enforce synchronization from cpu
#endif
                assert(state != CPU_AHEAD, "CPU was ahead when GPU tried to access MemoryBlock data. Call Synchronize manually before entering GPU code.");
                state = GPU_AHEAD;
                return data_cuda;
            }
            fatalError("error on GetData: unknown memory type %d", memoryType);
            return 0;
        }

        CPU_AND_GPU const DEVICEPTR(T)* const GetData(MemoryDeviceType memoryType) const
        {
            switch (memoryType)
            {
#if !GPU_CODE
            case MEMORYDEVICE_CPU:
                Synchronize(); // might as well ensure synchronization now
                return data_cpu;
#endif
            case MEMORYDEVICE_CUDA:
#if !GPU_CODE
                Synchronize();
#endif
                assert(state == SYNCHRONIZED || state == GPU_AHEAD); // GPU-ahead is ok while on gpu
                return data_cuda;
            }
            fatalError("error on const GetData: unknown memory type %d", memoryType);
            return 0;
        }


#ifdef __CUDA_ARCH__
        /** Get the data pointer on CPU or GPU. */
        GPU_ONLY DEVICEPTR(T)* GetData() { return GetData(MEMORYDEVICE_CUDA); }
        GPU_ONLY const DEVICEPTR(T)* GetData() const { return GetData(MEMORYDEVICE_CUDA); }
#else
        inline T* GetData() { return GetData(MEMORYDEVICE_CPU); }
        inline const T* GetData() const { return GetData(MEMORYDEVICE_CPU); }
#endif

        // convenience & bounds checking
        CPU_AND_GPU /*possibly DEVICEPTR */ DEVICEPTR(T&) operator[] (unsigned int i) {
            assert(i < dataSize, "%d >= %d -- MemoryBlock access out of range", i, dataSize);
            return GetData()[i];
        }


        CPU_AND_GPU /*possibly DEVICEPTR */ DEVICEPTR(T const&) operator[] (unsigned int i) const {
            assert(i < dataSize, "%d >= %d -- MemoryBlock access out of range", i, dataSize);
            return GetData()[i];
        }

        /** Set all data to the byte given by @p defaultValue. */
        void Clear(unsigned char defaultValue = 0)
        {
            memset(data_cpu, defaultValue, dataSizeInBytes());
            cudaMemset(data_cuda, defaultValue, dataSizeInBytes());
            state = SYNCHRONIZED;
        }
    };

    __managed__ int* data;
    KERNEL set_data() {
        data[1] = 42;
    }
    KERNEL check_data() {
        assert(data[1] == 42);
    }

    TEST(testMemblock) {
        cudaDeviceSynchronize();
        auto mem = new MemoryBlock<int>(10);
        assert(mem->dataSize == 10);
        assert(mem->state == SYNCHRONIZED);

        mem->GetData(MEMORYDEVICE_CPU);
        assert(mem->state == CPU_AHEAD);

        mem->GetData(MEMORYDEVICE_CUDA);
        assert(mem->state == GPU_AHEAD);

        mem->Clear(0);
        assert(mem->state == SYNCHRONIZED);

        auto const* const cmem = mem;
        cmem->GetData(MEMORYDEVICE_CPU);
        assert(mem->state == SYNCHRONIZED);
        cmem->GetData(MEMORYDEVICE_CUDA);
        assert(mem->state == SYNCHRONIZED);

        mem->GetData()[1] = 42;
        assert(mem->state == CPU_AHEAD);
        data = mem->GetData(MEMORYDEVICE_CUDA);
        assert(mem->state == GPU_AHEAD);
        LAUNCH_KERNEL(check_data, 1, 1);
        cudaDeviceSynchronize();

        mem->Clear(0);

        // NOTE wrongly assumes that everything is still clean because we *reused the pointer* (data) instead of claiming it again
        // consequently, memory will not be equal, but state will still say SYNCHRONIZED!
        LAUNCH_KERNEL(set_data, 1, 1);
        cudaDeviceSynchronize();
        assert(mem->state == SYNCHRONIZED);
        LAUNCH_KERNEL(check_data, 1, 1);
        cudaDeviceSynchronize();
        assert(mem->GetData()[1] == 0);

        // re-requesting fixes the problem and syncs the buffers again
        mem->Clear(0);
        data = mem->GetData(MEMORYDEVICE_CUDA);
        LAUNCH_KERNEL(set_data, 1, 1);
        LAUNCH_KERNEL(check_data, 1, 1);
        cudaDeviceSynchronize();
        assert(mem->GetData()[1] == 42);
    }

#define GPUCHECK(p, val) LAUNCH_KERNEL(check,1,1,(char* )p,val);
    KERNEL check(char* p, char val) {
        assert(*p == val);
    }

    TEST(testMemoryBlockSerialize) {
        MemoryBlock<int> b(1);
        b[0] = 0xbadf00d;
        GPUCHECK(b.GetData(MEMORYDEVICE_CUDA), 0x0d);
        BEGIN_SHOULD_FAIL("testMemoryBlockSerialize");
        GPUCHECK(b.GetData(MEMORYDEVICE_CUDA), 0xba);
        END_SHOULD_FAIL("testMemoryBlockSerialize");

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
    GPUCHECK(b.GetData(MEMORYDEVICE_CUDA), 0x0d);
    assert(b.dataSize == 1);


    MemoryBlock<int> c(100);
    assert(c.dataSize == 100);
    GPUCHECK(c.GetData(MEMORYDEVICE_CUDA), 0);

    {
        c.deserialize(binopen_read(fn));
    }
    assert(c[0] == 0xbadf00d);
    assert(c.dataSize == 1);
    GPUCHECK(c.GetData(MEMORYDEVICE_CUDA), 0x0d);

    }

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












    /** \brief
    Represents images, templated on the pixel type

    Managed
    */
    template <typename T>
    class Image : public MemoryBlock < T >
    {
    public:
        /** Size of the image in pixels. */
        Vector2<int> noDims;

        /** Initialize an empty image of the given size
        */
        Image(Vector2<int> noDims = Vector2<int>(1, 1)) : MemoryBlock<T>(noDims.area()), noDims(noDims) {}

        void EnsureDims(Vector2<int> noDims) {
            if (this->noDims == noDims) return;
            this->noDims = noDims;
            Allocate(noDims.area());
        }

    };


#define ITMFloatImage Image<float>
#define ITMFloat2Image Image<Vector2f>
#define ITMFloat4Image Image<Vector4f>
#define ITMShortImage Image<short>
#define ITMShort3Image Image<Vector3s>
#define ITMShort4Image Image<Vector4s>
#define ITMUShortImage Image<ushort>
#define ITMUIntImage Image<uint>
#define ITMIntImage Image<int>
#define ITMUCharImage Image<uchar>
#define ITMUChar4Image Image<Vector4u>
#define ITMBoolImage Image<bool>


}
using namespace memory;







































































template<typename T>
struct IllegalColor {
    static CPU_AND_GPU T make();
};
inline CPU_AND_GPU float IllegalColor<float>::make() {
    return -1;
}
inline CPU_AND_GPU Vector4f IllegalColor<Vector4f>::make() {
    return Vector4f(0, 0, 0, -1);
}
inline CPU_AND_GPU bool isLegalColor(float c) {
    return c >= 0;
}
inline CPU_AND_GPU bool isLegalColor(Vector4f c) {
    return c.w >= 0;
}
inline CPU_AND_GPU bool isLegalColor(Vector4u c) {
    // NOTE this should never be called -- withHoles should be false for a Vector4u
    // implementing this just calms the compiler
    fatalError("isLegalColor is not implemented for Vector4u");
    return false;
}

























// Local/per pixel Image processing library


/// Linearized pixel index
CPU_AND_GPU inline int pixelLocId(const int x, const int y, const Vector2i &imgSize) {
    return x + y * imgSize.x;
}

/// Sample image without interpolation at integer location
template<typename T> CPU_AND_GPU
inline T sampleNearest(
const T *source,
int x, int y,
const Vector2i & imgSize)
{
    return source[pixelLocId(x, y, imgSize)];
}

/// Sample image without interpolation at rounded location
template<typename T> CPU_AND_GPU
inline T sampleNearest(
const T *source,
const Vector2f & pt_image,
const Vector2i & imgSize) {
    return source[
        pixelLocId(
            (int)(pt_image.x + 0.5f),
            (int)(pt_image.y + 0.5f),
            imgSize)];
}

/// Whether interpolation should return an illegal color when holes make interpolation impossible
#define WITH_HOLES true
/// Sample 4 channel image with bilinear interpolation (T_IN::toFloat must return Vector4f)
/// IF withHoles == WITH_HOLES: returns makeIllegalColor<OUT>() when any of the four surrounding pixels is illegal (has negative w).
template<typename T_OUT, //!< Vector4f or float
    bool withHoles = false, typename T_IN> CPU_AND_GPU inline Vector4f interpolateBilinear(
    const T_IN * const source,
    const Vector2f & position, const Vector2i & imgSize)
{
    T_OUT result;
    Vector2i p; Vector2f delta;

    p.x = (int)floor(position.x); p.y = (int)floor(position.y);
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
template<typename F>
static KERNEL forEachPixelNoImage_device(Vector2i imgSize) {
    const int
        x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > imgSize.x - 1 || y > imgSize.y - 1) return;
    const int locId = pixelLocId(x, y, imgSize);

    F::process(x, y, locId);
}

#define forEachPixelNoImage_process() GPU_ONLY static void process(const int x, const int y, const int locId)
/** apply
F::process(int x, int y, int locId)
to each (hypothetical) pixel in the image

locId runs through values generated by pixelLocId(x, y, imgSize);
*/
template<typename F>
static void forEachPixelNoImage(Vector2i imgSize) {
    const dim3 blockSize(16, 16);
    LAUNCH_KERNEL(
        forEachPixelNoImage_device<F>,
        getGridSize(dim3(xy(imgSize)), blockSize),
        blockSize,
        imgSize);
}















namespace TestForEachPixel {
    const int W = 5;
    const int H = 7;
    __managed__ int fipcounter = 0;
    struct DoForEachPixel {
        forEachPixelNoImage_process() {
            assert(x >= 0 && x < W);
            assert(y >= 0 && y < H);
            atomicAdd(&fipcounter, 1);
        }
    };

    TEST(testForEachPixelNoImage) {
        fipcounter = 0;
        forEachPixelNoImage<DoForEachPixel>(Vector2i(W, H));
        cudaDeviceSynchronize();
        assert(fipcounter == W * H);
    }

}








































template<bool withHoles = false, typename T>
CPU_AND_GPU inline void filterSubsample(
    DEVICEPTR(T) *imageData_out, int x, int y, Vector2i newDims,
    const T *imageData_in, Vector2i oldDims)
{
    int src_pos_x = x * 2, src_pos_y = y * 2;
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

// device functions
#define FILTER(FILTERNAME)\
template<bool withHoles, typename T>\
static KERNEL FILTERNAME ## _device(T *imageData_out, Vector2i newDims, const T *imageData_in, Vector2i oldDims) {\
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;\
    if (x > newDims.x - 1 || y > newDims.y - 1) return;\
    FILTERNAME<withHoles>(imageData_out, x, y, newDims, imageData_in, oldDims);\
}

FILTER(filterSubsample)



// host methods
#define FILTERMETHOD(METHODNAME, WITHHOLES)\
            template<typename T>\
                void METHODNAME (Image<T> *image_out, const Image<T> *image_in) {\
                    Vector2i oldDims = image_in->noDims; \
                    Vector2i newDims; newDims.x = image_in->noDims.x / 2; newDims.y = image_in->noDims.y / 2; \
                    \
                    image_out->EnsureDims(newDims); \
                    \
                    const T *imageData_in = image_in->GetData(MEMORYDEVICE_CUDA); \
                    T *imageData_out = image_out->GetData(MEMORYDEVICE_CUDA); \
                    \
                    dim3 blockSize(16, 16); \
                    dim3 gridSize((int)ceil((float)newDims.x / (float)blockSize.x), (int)ceil((float)newDims.y / (float)blockSize.y)); \
                    \
                    LAUNCH_KERNEL(filterSubsample_device<WITHHOLES>, gridSize, blockSize,\
imageData_out, newDims, imageData_in, oldDims); \
            }

FILTERMETHOD(FilterSubsample, false)
FILTERMETHOD(FilterSubsampleWithHoles, WITH_HOLES)





































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
    __managed__ CoordinateSystem* globalcs = 0;
    class CoordinateSystem : public Managed {
    private:
        //CoordinateSystem(const CoordinateSystem&); // TODO should we really allow copying?
        void operator=(const CoordinateSystem&);

        CPU_AND_GPU Point toGlobalPoint(Point p)const;
        CPU_AND_GPU Point fromGlobalPoint(Point p)const;
        CPU_AND_GPU Vector toGlobalVector(Vector p)const;
        CPU_AND_GPU Vector fromGlobalVector(Vector p)const;
    public:
        const Matrix4f toGlobal;
        const Matrix4f fromGlobal;
        explicit CoordinateSystem(const Matrix4f& toGlobal) : toGlobal(toGlobal), fromGlobal(toGlobal.getInv()) {
            assert(toGlobal.GetR().det() != 0);
        }

        /// The world or global space coodinate system.
        /// Measured in meters if cameras and depth computation are calibrated correctly.
        CPU_AND_GPU static CoordinateSystem* global() {

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

        CPU_AND_GPU Point convert(Point p)const;
        CPU_AND_GPU Vector convert(Vector p)const;
        CPU_AND_GPU Ray convert(Ray p)const;
    };

    // Represents anything that lives in some coordinate system.
    // Entries are considered equal only when they have the same coordinates.
    // They are comparable only if in the same coordinate system.
    class CoordinateEntry {
    public:
        const CoordinateSystem* coordinateSystem;
        friend CoordinateSystem;
        CPU_AND_GPU CoordinateEntry(const CoordinateSystem* coordinateSystem) : coordinateSystem(coordinateSystem) {}
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

        CPU_AND_GPU explicit Vector(const CoordinateSystem* coordinateSystem, Vector3f direction) : CoordinateEntry(coordinateSystem), direction(direction) {
        }
        CPU_AND_GPU bool operator==(const Vector& rhs) const {
            assert(coordinateSystem == rhs.coordinateSystem);
            return direction == rhs.direction;
        }
        CPU_AND_GPU Vector operator*(const float rhs) const {
            return Vector(coordinateSystem, direction * rhs);
        }
        CPU_AND_GPU float dot(const Vector& rhs) const {
            assert(coordinateSystem == rhs.coordinateSystem);
            return vecmath::dot(direction, rhs.direction);
        }
        CPU_AND_GPU Vector operator-(const Vector& rhs) const {
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
        CPU_AND_GPU void operator=(const Point& rhs) {
            coordinateSystem = rhs.coordinateSystem;
            const_cast<Vector3f&>(location) = rhs.location;
        }

        CPU_AND_GPU explicit Point(const CoordinateSystem* coordinateSystem, Vector3f location) : CoordinateEntry(coordinateSystem), location(location) {
        }
        CPU_AND_GPU bool operator==(const Point& rhs) const {
            assert(coordinateSystem == rhs.coordinateSystem);
            return location == rhs.location;
        }
        CPU_AND_GPU Point operator+(const Vector& rhs) const {
            assert(coordinateSystem == rhs.coordinateSystem);
            return Point(coordinateSystem, location + rhs.direction);
        }

        /// Gives a vector that points from rhs to this.
        /// Think 'the location of this as seen from rhs' or 'how to get to this coordinate given one already got to rhs' ('how much energy do we still need to invest in each direction)
        CPU_AND_GPU Vector operator-(const Point& rhs) const {
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

        CPU_AND_GPU Ray(Point& origin, Vector& direction) : origin(origin), direction(direction) {
            assert(origin.coordinateSystem == direction.coordinateSystem);
        }
        CPU_AND_GPU Point endpoint() {
            Point p = origin + direction;
            assert(p.coordinateSystem == origin.coordinateSystem);
            return p;
        }
    };


    inline CPU_AND_GPU Point CoordinateSystem::toGlobalPoint(Point p) const {
        return Point(global(), Vector3f(this->toGlobal * Vector4f(p.location, 1)));
    }
    inline CPU_AND_GPU Point CoordinateSystem::fromGlobalPoint(Point p) const {
        assert(p.coordinateSystem == global());
        return Point(this, Vector3f(this->fromGlobal * Vector4f(p.location, 1)));
    }
    inline CPU_AND_GPU Vector CoordinateSystem::toGlobalVector(Vector v) const {
        return Vector(global(), this->toGlobal.GetR() *v.direction);
    }
    inline CPU_AND_GPU Vector CoordinateSystem::fromGlobalVector(Vector v) const {
        assert(v.coordinateSystem == global());
        return Vector(this, this->fromGlobal.GetR() *v.direction);
    }
    inline CPU_AND_GPU Point CoordinateSystem::convert(Point p) const {
        Point o = this->fromGlobalPoint(p.coordinateSystem->toGlobalPoint(p));
        assert(o.coordinateSystem == this);
        return o;
    }
    inline CPU_AND_GPU Vector CoordinateSystem::convert(Vector p) const {
        Vector o = this->fromGlobalVector(p.coordinateSystem->toGlobalVector(p));
        assert(o.coordinateSystem == this);
        return o;
    }
    inline CPU_AND_GPU Ray CoordinateSystem::convert(Ray p) const {
        return Ray(convert(p.origin), convert(p.direction));
    }


    void initCoordinateSystems() {
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

    CameraImage(
        Image<T>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics) :
        image(image), eyeCoordinates(eyeCoordinates), cameraIntrinsics(cameraIntrinsics) {
        //assert(image->noDims.area() > 1); // don't force this, the image we reference is allowed to change later on
    }

    CPU_AND_GPU Vector2i imgSize()const {
        return image->noDims;
    }

    CPU_AND_GPU Vector4f projParams() const {
        return cameraIntrinsics.all;
    }
    /// 0,0,0 in eyeCoordinates
    CPU_AND_GPU Point location() const {
        return Point(eyeCoordinates, Vector3f(0, 0, 0));
    }

    /// Returns a ray starting at the camera origin and passing through the virtual camera plane
    /// pixel coordinates must be valid with regards to image size
    CPU_AND_GPU Ray getRayThroughPixel(Vector2i pixel, float depth) const {
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
    CPU_AND_GPU bool project(Point p, Vector2f& pt_image, bool extraBounds = false) const {
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
    DepthImage(
        Image<float>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics) {}

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
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
    PointImage(
        Image<Vector4f>*const image,
        const CoordinateSystem* const pointCoordinates,

        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics), pointCoordinates(pointCoordinates) {}

    const CoordinateSystem* const pointCoordinates;

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
        return Point(pointCoordinates, sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize()).toVector3());
    }

    /// Uses bilinear interpolation to deduce points between raster locations.
    /// out_isIllegal is set to true or false depending on whether the given point falls in a 'hole' (undefined/missing data) in the image
    CPU_AND_GPU Point getPointForPixelInterpolated(Vector2f pixel, bool& out_isIllegal) const {
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
    RayImage(
        Image<Vector4f>*const pointImage,
        Image<Vector4f>*const normalImage,
        const CoordinateSystem* const pointCoordinates,

        const CoordinateSystem* const eyeCoordinates,
        const ITMIntrinsics cameraIntrinsics) :
        PointImage(pointImage, pointCoordinates, eyeCoordinates, cameraIntrinsics), normalImage(normalImage) {
        assert(normalImage->noDims == pointImage->noDims);
    }
    Image<Vector4f>* const normalImage; // Image may change

    CPU_AND_GPU Ray getRayForPixel(Vector2i pixel) const {
        Point origin = getPointForPixel(pixel);
        auto direction = sampleNearest(normalImage->GetData(), pixel.x, pixel.y, imgSize()).toVector3();
        return Ray(origin, Vector(pointCoordinates, direction));
    }

    /// pixel should have been produced with
    /// project(x,pixel,EXTRA_BOUNDS)
    CPU_AND_GPU Ray getRayForPixelInterpolated(Vector2f pixel, bool& out_isIllegal) const {
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
























CPU_AND_GPU void testCS(CoordinateSystem* o) {
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

KERNEL ktestCS(CoordinateSystem* o) {
    testCS(o);
}

__managed__ CoordinateSystem* testedCs;

CPU_AND_GPU void testCi(
    const DepthImage* const di,
    const PointImage* const pi) {
    Vector2i imgSize(640, 480);
    assert(di->location() == Point(testedCs, Vector3f(0, 0, 0)));
    {
        auto r1 = di->getRayThroughPixel(Vector2i(0, 0), 1);
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
        auto r = di->getPointForPixel(Vector2i(0, 0));
        assert(r == Point(testedCs, Vector3f(0, 0, 0)));
    }
    {
        auto r = di->getPointForPixel(Vector2i(1, 0));
        assert(!(r == Point(testedCs, Vector3f(0, 0, 0))));
        assert(r.location.z == 1);
        auto ray = di->getRayThroughPixel(Vector2i(1, 0), 1);
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
        auto r = pi->getPointForPixel(Vector2i(0, 0));
        assert(r == Point(testedCs, Vector3f(0, 0, 0)));
    }
    {
        auto r = pi->getPointForPixel(Vector2i(1, 0));
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

KERNEL ktestCi(
    const DepthImage* const di,
    const PointImage* const pi) {

    testCi(di, pi);
}
TEST(testCameraImage) {
    ITMIntrinsics intrin;
    Vector2i imgSize(640, 480);
    auto depthImage = new ITMFloatImage(imgSize);
    auto pointImage = new ITMFloat4Image(imgSize);

    depthImage->GetData()[1] = 1;
    pointImage->GetData()[1] = Vector4f(1, 1, 1, 1);
    // must submit manually
    depthImage->Synchronize();
    pointImage->Synchronize();

    Matrix4f cameraToWorld;
    cameraToWorld.setIdentity();
    cameraToWorld.setTranslate(Vector3f(0, 0, 1));
    testedCs = new CoordinateSystem(cameraToWorld);
    auto di = new DepthImage(depthImage, testedCs, intrin);
    auto pi = new PointImage(pointImage, testedCs, testedCs, intrin);

    testCi(di, pi);

    // must submit manually
    depthImage->Synchronize();
    pointImage->Synchronize();

    LAUNCH_KERNEL(ktestCi, 1, 1, di, pi);

}


TEST(testCS) {
    // o gives points with twice as large coordinates as the global coordinate system
    Matrix4f m;
    m.setIdentity();
    m.setScale(0.5); // scale down by half to get the global coordinates of the point
    auto o = new CoordinateSystem(m);

    testCS(o);
    LAUNCH_KERNEL(ktestCS, 1, 1, o);
}













































/** \brief
Represents a single "view", i.e. RGB and depth images along
with all intrinsic, relative and extrinsic calibration information

This defines a point-cloud with 'valid half-space-pseudonormals' for each point:
We know that the observed points have a normal that lies in the same half-space as the direction towards the camera,
otherwise we could not have observed them.
*/
class ITMView : public Managed {
    /// RGB colour image.
    ITMUChar4Image * const rgbData;

    /// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    ITMFloatImage * const depthData;

    Vector2i imgSize_d() const {
        assert(depthImage->imgSize().area() > 1);
        return depthImage->imgSize();
    }
public:

    /// Intrinsic calibration information for the view.
    ITMRGBDCalib const * const calib;

    CameraImage<Vector4u> * const colorImage;

    /// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    DepthImage * const depthImage;

    // \param M_d world-to-view matrix
    void ChangePose(Matrix4f M_d) {
        assert(&M_d);
        assert(abs(M_d.GetR().det() - 1) < 0.00001);
        // TODO delete old ones!
        auto depthCs = new CoordinateSystem(M_d.getInv());
        depthImage->eyeCoordinates = depthCs;

        Matrix4f M_rgb = calib->trafo_rgb_to_depth.calib_inv * M_d;
        auto colorCs = new CoordinateSystem(M_rgb.getInv());
        colorImage->eyeCoordinates = colorCs;
    }

    ITMView(const ITMRGBDCalib &calibration) : ITMView(&calibration) {};

    ITMView(const ITMRGBDCalib *calibration) :
        calib(new ITMRGBDCalib(*calibration)),
        rgbData(new ITMUChar4Image(calibration->intrinsics_rgb.imageSize())),

        depthData(new ITMFloatImage(calibration->intrinsics_d.imageSize())),

        depthImage(new DepthImage(depthData, CoordinateSystem::global(), calib->intrinsics_d)),
        colorImage(new CameraImage<Vector4u>(rgbData, CoordinateSystem::global(), calib->intrinsics_rgb)) {
        assert(colorImage->eyeCoordinates == CoordinateSystem::global());
        assert(depthImage->eyeCoordinates == CoordinateSystem::global());

        Matrix4f M; M.setIdentity();
        ChangePose(M);
        assert(!(colorImage->eyeCoordinates == CoordinateSystem::global()));
        assert(!(depthImage->eyeCoordinates == CoordinateSystem::global()));
        assert(!(colorImage->eyeCoordinates == depthImage->eyeCoordinates));
    }

    void ITMView::ChangeImages(ITMUChar4Image *rgbImage, ITMFloatImage *depthImage);
};

/// current depth & color image
__managed__ ITMView * currentView = 0;

#define INVALID_DEPTH (-1.f)


void ITMView::ChangeImages(ITMUChar4Image *rgbImage, ITMFloatImage *depthImage)
{
    rgbData->SetFrom(*rgbImage);
    depthData->SetFrom(*depthImage);
}
























































/// Allocate sizeof(T) * count bytes and initialize it from device memory.
/// Used for debugging GPU memory from host 
template<typename T>
auto_ptr<T*> mallocAsDeviceCopy(DEVICEPTR(T*) p, uint count) {
    T* x = new T[count];
    cudaMemcpy(x, p, count * sizeof(T), cudaMemcpyDeviceToHost);
    return x;
}





















// GPU HashMap

// Forward declarations
template<typename Hasher, typename AllocCallback> class HashMap;
template<typename Hasher, typename AllocCallback>
KERNEL performAllocationKernel(typename HashMap<Hasher, AllocCallback>* hashMap);

#define hprintf(...) //printf(__VA_ARGS__) // enable for verbose radio debug messages

struct VoidSequenceIdAllocationCallback {
    template<typename T>
    static CPU_AND_GPU void allocate(T, int sequenceId) {}
};
/**
Implements a

key -> sequence#

mapping on the GPU, where keys for which allocation is requested get assigned unique,
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
    typename Hasher, //!< must have static CPU_AND_GPU function uint Hasher::hash(const KeyType&) which generates values from 0 to Hasher::BUCKET_NUM-1 
    typename SequenceIdAllocationCallback = VoidSequenceIdAllocationCallback //!< must have static CPU_AND_GPU void  allocate(KeyType k, int sequenceId) function
>
class HashMap : public Managed {
public:
    typedef Hasher::KeyType KeyType;

private:
    static const uint BUCKET_NUM = Hasher::BUCKET_NUM;
    const uint EXCESS_NUM;
    CPU_AND_GPU uint NUMBER_TOTAL_ENTRIES() const {
        return (BUCKET_NUM + EXCESS_NUM);
    }

    struct HashEntry {
    public:
        CPU_AND_GPU bool isAllocated()const {
            return sequenceId != 0;
        }
        CPU_AND_GPU bool hasNextExcessList() const{
            assert(isAllocated());
            return nextInExcessList != 0;
        }
        CPU_AND_GPU uint getNextInExcessList()const {
            assert(hasNextExcessList() && isAllocated());
            return nextInExcessList;
        }

        CPU_AND_GPU bool hasKey(const KeyType& key)const {
            assert(isAllocated());
            return this->key == key;
        }

        CPU_AND_GPU void linkToExcessListEntry(const uint excessListId) {
            assert(!hasNextExcessList() && isAllocated() && excessListId >= 1);// && excessListId < EXCESS_NUM);
            // also, the excess list entry should exist and this should be the only entry linking to it
            // all entries in the excess list before this one should be allocated
            nextInExcessList = excessListId;
        }

        CPU_AND_GPU void allocate(const KeyType& key, const uint sequenceId) {
            assert(!isAllocated() && 0 == nextInExcessList);
            assert(sequenceId > 0);
            this->key = key;
            this->sequenceId = sequenceId;

            SequenceIdAllocationCallback::allocate(key, sequenceId);

            hprintf("allocated %d\n", sequenceId);
        }

        CPU_AND_GPU uint getSequenceId() const {
            assert(isAllocated());
            return sequenceId;
        }
    private:
        KeyType key;
        /// any of 1 to lowestFreeExcessListEntry-1
        /// 0 means this entry ends a list of excess entries and/or is not allocated
        uint nextInExcessList;
        /// any of 1 to lowestFreeSequenceNumber-1
        /// 0 means this entry is not allocated
        uint sequenceId;
    };

public:
    /// BUCKET_NUM + EXCESS_NUM many, information for the next round of allocations
    /// Note that equal hashes will clash only once

    /// Whether the corresponding entry should be allocated
    /// 0 or 1 
    /// TODO could save memory (&bandwidth) by using a bitmap
    MemoryBlock<uchar> needsAllocation;

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
    MemoryBlock<HashEntry> hashMap_then_excessList;

private:
    CPU_AND_GPU HashEntry& hashMap(const uint hash) {
        assert(hash < BUCKET_NUM);
        return hashMap_then_excessList[hash];
    }
    CPU_AND_GPU HashEntry const & hashMap(const uint hash) const {
        assert(hash < BUCKET_NUM);
        return hashMap_then_excessList[hash];
    }
    CPU_AND_GPU HashEntry& excessList(const uint excessListEntry) {
        assert(excessListEntry >= 1 && excessListEntry < EXCESS_NUM);
        return hashMap_then_excessList[BUCKET_NUM + excessListEntry];
    }
    CPU_AND_GPU HashEntry const & excessList(const uint excessListEntry) const {
        assert(excessListEntry >= 1 && excessListEntry < EXCESS_NUM);
        return hashMap_then_excessList[BUCKET_NUM + excessListEntry];
    }


    /// Sequence numbers already used up. Starts at 1 (sequence number 0 is used to signify non-allocated)
    uint lowestFreeSequenceNumber;

    /// Excess list slots already used up. Starts at 1 (one safeguard entry)
    uint lowestFreeExcessListEntry;

    /// Follows the excess list starting at hashMap[Hasher::hash(key)]
    /// until either hashEntry.key == key, returning true
    /// or until hashEntry does not exist or hashEntry.key != key but there is no further entry, returns false in that case.
    CPU_AND_GPU bool findEntry(const KeyType& key,//!< [in]
        HashEntry& hashEntry, //!< [out]
        uint& hashMap_then_excessList_entry //!< [out]
        ) const {
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
            if (safe++ > 100) fatalError("excessive amount of steps in excess list");
        }
        return false;
    }

    CPU_AND_GPU void allocate(HashEntry& hashEntry, const KeyType & key) {
#if GPU_CODE
        hashEntry.allocate(key, atomicAdd(&lowestFreeSequenceNumber, 1));
#else /* assume single-threaded cpu */
        hashEntry.allocate(key, lowestFreeSequenceNumber++);
#endif
    }

    friend KERNEL performAllocationKernel<Hasher, SequenceIdAllocationCallback>(typename HashMap<Hasher, SequenceIdAllocationCallback>* hashMap);


    /// Given a key that does not yet exist, find a location in the hashMap_then_excessList
    /// that can be used to insert the key (or is the end of the current excess list for the keys with the same hash as this)
    /// returns (uint)-1 if the key already exists
    CPU_AND_GPU uint findLocationForKey(const KeyType& key) {
        hprintf("findLocationForKey \n");

        HashEntry hashEntry;
        uint hashMap_then_excessList_entry;

        bool alreadyExists = findEntry(key, hashEntry, hashMap_then_excessList_entry);
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
    /// \returns the sequence number of the newly allocated entry
    CPU_AND_GPU uint performSingleAllocation(const KeyType& key, const uint hashMap_then_excessList_entry) {
        if (hashMap_then_excessList_entry == (uint)-1) return -1;
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
        const uint excessListId = atomicAdd(&lowestFreeExcessListEntry, 1);
#else /* assume single-threaded cpu code */
        const uint excessListId = lowestFreeExcessListEntry++;
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
        HashEntry e; uint _;
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
    GPU_ONLY void performAllocation(const uint hashMap_then_excessList_entry) {
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
    HashMap(const uint EXCESS_NUM //<! must be at least one
        ) : EXCESS_NUM(EXCESS_NUM),
        needsAllocation(NUMBER_TOTAL_ENTRIES()),
        naKey(NUMBER_TOTAL_ENTRIES()),
        hashMap_then_excessList(NUMBER_TOTAL_ENTRIES())

    {
        assert(EXCESS_NUM >= 1);
        cudaDeviceSynchronize();

        lowestFreeSequenceNumber = lowestFreeExcessListEntry = 1;
    }

    CPU_AND_GPU // could be CPU only if it where not for debugging (and dumping) - do we need it at all?
        uint getLowestFreeSequenceNumber() const {
#if !GPU_CODE
        cudaDeviceSynchronize();
#endif
        return lowestFreeSequenceNumber;
    }
    /*
    uint countAllocatedEntries() {
    return getLowestFreeSequenceNumber() - 1;
    }
    */

    // TODO should this ad-hoc crude serialization be part of this class?

    void serialize(ofstream& file) {
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
    void deserialize(ifstream& file) {
        assert(NUMBER_TOTAL_ENTRIES() == bin<uint>(file));
        assert(EXCESS_NUM == bin<uint>(file));
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
    GPU_ONLY void requestAllocation(const KeyType& key) {
        hprintf("requestAllocation \n");

        uint hashMap_then_excessList_entry = findLocationForKey(key);

        if (hashMap_then_excessList_entry == (uint)-1) {
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
    CPU_MEMBERFUNCTION(void,performAllocations,(),"") {
        //cudaSafeCall(cudaGetError());
        cudaSafeCall(cudaDeviceSynchronize()); // Managed this is not accessible when still in use?
        LAUNCH_KERNEL(performAllocationKernel, // Note: trivially parallelizable for-each type task
            /// Scheduling strategy: Fixed number of threads per block, working on all entries (to find those that have needsAllocation set)
            (uint)ceil(NUMBER_TOTAL_ENTRIES() / (1. * THREADS_PER_BLOCK)),
            THREADS_PER_BLOCK,
            this);
#ifdef _DEBUG
        cudaSafeCall(cudaDeviceSynchronize());  // detect problems (failed assertions) early where this kernel is called
#endif
        cudaSafeCall(cudaGetLastError());
    }

    /// Allocate and assign a sequence number for the given key.
    /// Note: Potentially slower than requesting a whole bunch, then allocating all at once, use as fallback.
    /// \returns the sequence number of the newly allocated entry
    MEMBERFUNCTION(uint,performSingleAllocation,(const KeyType& key),"") {
        return performSingleAllocation(key, findLocationForKey(key));
    }

    MEMBERFUNCTION(uint, getSequenceNumber,(const KeyType& key) const,
        "\returns 0 if the key is not allocated, otherwise something greater than 0 unique to this key and less than getLowestFreeSequenceNumber()") {
        HashEntry hashEntry; uint _;
        if (!findEntry(key, hashEntry, _)) return 0;
        return hashEntry.getSequenceId();
    }
};

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







namespace HashMapTests {


    template<typename T>
    struct Z3Hasher {
        typedef T KeyType;
        static const uint BUCKET_NUM = 0x1000; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

        static CPU_AND_GPU uint hash(const T& blockPos) {
            return (((uint)blockPos.x * 73856093u) ^ ((uint)blockPos.y * 19349669u) ^ ((uint)blockPos.z * 83492791u))
                & // optimization - has to be % if BUCKET_NUM is not a power of 2 // TODO can the compiler not figure this out?
                (uint)(BUCKET_NUM - 1);
        }
    };


    KERNEL get(HashMap<Z3Hasher<Vector3s>>* myHash, Vector3s q, int* o) {
        *o = myHash->getSequenceNumber(q);
    }

    KERNEL alloc(HashMap<Z3Hasher<Vector3s>>* myHash) {
        int p = blockDim.x * blockIdx.x + threadIdx.x;
        myHash->requestAllocation(p);
    }

    KERNEL assertfalse() {
        assert(false);
    }


    TEST(testZ3Hasher) {
        // insert a lot of points (n) into a large hash just for fun
        HashMap<Z3Hasher<Vector3s>>* myHash = new HashMap<Z3Hasher<Vector3s>>(0x2000);

        const int n = 1000;
        LAUNCH_KERNEL(alloc, n, 1, myHash);

        myHash->performAllocations();
        puts("after alloc");
        // should be some permutation of 1:n
        vector<bool> found; found.resize(n + 1);
        int* p; cudaMallocManaged(&p, sizeof(int));
        for (int i = 0; i < n; i++) {
            LAUNCH_KERNEL(get,
                1, 1,
                myHash, Vector3s(i, i, i), p);
            cudaSafeCall(cudaDeviceSynchronize()); // to read managed p
            printf("Vector3s(%i,%i,%i) -> %d\n", i, i, i, *p);

            assert(!found[*p]);
            found[*p] = 1;
        }
    }

    // n hasher test suite
    // trivial hash function n -> n
    struct NHasher{
        typedef int KeyType;
        static const uint BUCKET_NUM = 1; // can play with other values, the tests should support it
        static CPU_AND_GPU uint hash(const int& n) {
            return n % BUCKET_NUM;//& (BUCKET_NUM-1);
        }
    };

    KERNEL get(HashMap<NHasher>* myHash, int p, int* o) {
        *o = myHash->getSequenceNumber(p);
    }

    KERNEL alloc(HashMap<NHasher>* myHash, int p, int* o) {
        myHash->requestAllocation(p);
    }

    TEST(testNHasher) {
        int n = NHasher::BUCKET_NUM;
        auto myHash = new HashMap<NHasher>(1 + 1); // space for BUCKET_NUM entries only, and 1 collision handling entry

        int* p; cudaMallocManaged(&p, sizeof(int));

        for (int i = 0; i < n; i++) {

            LAUNCH_KERNEL(alloc,
                1, 1,
                myHash, i, p);
        }
        myHash->performAllocations();

        // an additional alloc at another key not previously seen (e.g. BUCKET_NUM) 
        // this will use the excess list
        LAUNCH_KERNEL(alloc, 1, 1, myHash, NHasher::BUCKET_NUM, p);
        myHash->performAllocations();

        // an additional alloc at another key not previously seen (e.g. BUCKET_NUM + 1) makes it crash cuz no excess list space is left
        //alloc << <1, 1 >> >(myHash, NHasher::BUCKET_NUM + 1, p);
        myHash->performAllocations(); // performAllocations is always fine to call when no extra allocations where made

        puts("after alloc");
        // should be some permutation of 1:BUCKET_NUM
        bool found[NHasher::BUCKET_NUM + 1] = {0};
        for (int i = 0; i < n; i++) {
            LAUNCH_KERNEL(get, 1, 1, myHash, i, p);
            cudaDeviceSynchronize();
            printf("%i -> %d\n", i, *p);
            assert(!found[*p]);
            //assert(*p != i+1); // numbers are very unlikely to be in order -- nah it happens
            found[*p] = 1;
        }

    }

    // zero hasher test suite
    // trivial hash function with one bucket.
    // This will allow the allocation of only one block at a time
    // and all blocks will be in the same list.
    // The numbers will be in order.
    struct ZeroHasher{
        typedef int KeyType;
        static const uint BUCKET_NUM = 0x1;
        static CPU_AND_GPU uint hash(const int&) { return 0; }
    };

    KERNEL get(HashMap<ZeroHasher>* myHash, int p, int* o) {
        *o = myHash->getSequenceNumber(p);
    }

    KERNEL alloc(HashMap<ZeroHasher>* myHash, int p, int* o) {
        myHash->requestAllocation(p);
    }

    TEST(testZeroHasher) {
        int n = 10;
        auto myHash = new HashMap<ZeroHasher>(n); // space for BUCKET_NUM(1) + excessnum(n-1) = n entries
        assert(myHash->getLowestFreeSequenceNumber() == 1);
        int* p; cudaMallocManaged(&p, sizeof(int));

        const int extra = 0; // doing one more will crash it at
        // Assertion `excessListEntry >= 1 && excessListEntry < EXCESS_NUM` failed.

        // Keep requesting allocation until all have been granted
        for (int j = 0; j < n + extra; j++) { // request & perform alloc cycle
            for (int i = 0; i < n + extra
                ; i++) {
                LAUNCH_KERNEL(alloc, 1, 1, myHash, i, p); // only one of these allocations will get through at a time
            }
            myHash->performAllocations();

            puts("after alloc");
            for (int i = 0; i < n; i++) {
                LAUNCH_KERNEL(get, 1, 1, myHash, i, p);
                cudaDeviceSynchronize();
                printf("%i -> %d\n", i, *p);
                // expected result
                assert(i <= j ? *p == i + 1 : *p == 0);
            }
        }

        assert(myHash->getLowestFreeSequenceNumber() != 1);
    }
}


























































































namespace TestScene {
    TEST(sceneNeighborhoodExistence) {
        Scene* s = new Scene;

        VoxelBlockPos p(0, 0, 0);
        s->performVoxelBlockAllocation(p);
        s->getVoxelBlock(p)->resetVoxels();
        s->localVBA.Synchronize();
        assert(s->voxelExistsQ(Vector3i(0, 0, 0)));
        assert(!s->voxelExistsQ(Vector3i(-2, 0, 0)));
        assert(!s->voxel1NeighborhoodExistsQ(Vector3i(0, 0, 0)));
        assert(!s->voxel1NeighborhoodExistsQ(Vector3i(1, 0, 0)));
        assert(s->voxel1NeighborhoodExistsQ(Vector3i(1,1,1)));

        delete s;
    }



}























































/// === ITMBlockhash methods (readVoxel) ===

// isFound is assumed true initially and set to false when a requested voxel is not found
// a new voxel is returned in that case
GPU_ONLY inline ITMVoxel readVoxel(
    const Vector3i & point,
    bool &isFound, Scene* scene = Scene::getCurrentScene())
{
    ITMVoxel* v = scene->getVoxel(point);
    if (!v) {
        isFound = false;
        return ITMVoxel();
    }
    return *v;
}


/// === Generic methods (readSDF) ===

// isFound is set to true or false
GPU_ONLY inline float readFromSDF_float_uninterpolated(
    Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    bool &isFound)
{
    isFound = true;
    ITMVoxel res = readVoxel(TO_INT_ROUND3(point), isFound);
    return res.getSDF();
}

#define COMPUTE_COEFF_POS_FROM_POINT() \
    /* Coeff are the sub-block coordinates, used for interpolation*/\
    Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

// Given point in voxel-fractional-world-coordinates, this calls
// f(Vector3i globalPos, Vector3i lerpCoeff) on each globalPos (voxel-world-coordinates) bounding the given point
// in no specific order
template<typename T>
GPU_ONLY
void forEachBoundingVoxel(
Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
T& f
) {

    COMPUTE_COEFF_POS_FROM_POINT();
    float lerpCoeff;

#define access(dx,dy,dz) \
    lerpCoeff = \
    (dx ? coeff.x : 1.0f - coeff.x) *\
    (dy ? coeff.y : 1.0f - coeff.y) *\
    (dz ? coeff.z : 1.0f - coeff.z);\
    f(pos + Vector3i(dx, dy, dz), lerpCoeff);


    access(0, 0, 0);
    access(0, 0, 1);
    access(0, 1, 0);
    access(0, 1, 1);
    access(1, 0, 0);
    access(1, 0, 1);
    access(1, 1, 0);
    access(1, 1, 1);

#undef access
}

struct InterpolateSDF {
    float result;
    bool& isFound;
    GPU_ONLY InterpolateSDF(bool& isFound) : result(0), isFound(isFound) {}

    GPU_ONLY void operator()(Vector3i globalPos, float lerpCoeff) {
        result += lerpCoeff * readVoxel(globalPos, isFound).getSDF();
    }
};

GPU_ONLY inline float readFromSDF_float_interpolated(
    Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    bool &isFound)
{
    InterpolateSDF interpolator(isFound);
    forEachBoundingVoxel(point, interpolator);
    return interpolator.result;
}

struct InterpolateColor {
    Vector3f result;
    bool& isFound;
    GPU_ONLY InterpolateColor(bool& isFound) : result(0, 0, 0), isFound(isFound) {}

    GPU_ONLY void operator()(Vector3i globalPos, float lerpCoeff) {
        result += lerpCoeff * readVoxel(globalPos, isFound).clr.toFloat();
    }
};

// TODO should this also have a isFound parameter?
/// Assumes voxels store color in some type convertible to Vector3f (e.g. Vector3u)
GPU_ONLY inline Vector3f readFromSDF_color4u_interpolated(
    const Vector3f & point //!< in voxel-fractional world coordinates, comes e.g. from raycastResult
    )
{
    bool isFound = true;
    InterpolateColor interpolator(isFound);
    forEachBoundingVoxel(point, interpolator);
    return interpolator.result;
}


#define lookup(dx,dy,dz) readVoxel(pos + Vector3i(dx,dy,dz), isFound).getSDF()

// TODO test and visualize
// e.g. round to voxel position when rendering and draw this
GPU_ONLY inline UnitVector computeSingleNormalFromSDFByForwardDifference(
    const Vector3i &pos, //!< [in] global voxel position
    bool& isFound //!< [out] whether all values needed existed;
    ) {
    float sdf0 = lookup(0, 0, 0);
    if (!isFound) return Vector3f();

    // TODO handle !isFound
    Vector3f n(
        lookup(1, 0, 0) - sdf0,
        lookup(0, 1, 0) - sdf0,
        lookup(0, 0, 1) - sdf0
        );
    return n.normalised(); // TODO in a distance field, normalization should not be necessary? But this is not a true distance field.
}

/// Compute SDF normal by interpolated symmetric differences
/// Used in processPixelGrey
// Note: this gets the localVBA list, not just a *single* voxel block.
GPU_ONLY inline Vector3f computeSingleNormalFromSDF(
    const Vector3f &point)
{

    Vector3f ret;
    COMPUTE_COEFF_POS_FROM_POINT();
    Vector3f ncoeff = Vector3f(1, 1, 1) - coeff;

    bool isFound; // swallow

    /*
    x direction gradient at point is evaluated by computing interpolated sdf value in next (1 -- 2, v2) and previous (-1 -- 0, v1) cell:

    -1  0   1   2
    *---*---*---*
    |v1 |   | v2|
    *---*---*---*

    0 z is called front, 1 z is called back

    gradient is then
    v2 - v1
    */

    /* using xyzw components of vector4f to store 4 sdf values as follows:

    *---0--1-> x
    |
    0   x--y
    |   |  |
    1   z--w
    \/
    y
    */

    // all 8 values are going to be reused several times
    Vector4f front, back;
    front.x = lookup(0, 0, 0);
    front.y = lookup(1, 0, 0);
    front.z = lookup(0, 1, 0);
    front.w = lookup(1, 1, 0);
    back.x = lookup(0, 0, 1);
    back.y = lookup(1, 0, 1);
    back.z = lookup(0, 1, 1);
    back.w = lookup(1, 1, 1);

    Vector4f tmp;
    float p1, p2, v1;
    // gradient x
    // v1
    // 0-layer
    p1 = front.x * ncoeff.y * ncoeff.z +
        front.z *  coeff.y * ncoeff.z +
        back.x  * ncoeff.y *  coeff.z +
        back.z  *  coeff.y *  coeff.z;
    // (-1)-layer
    tmp.x = lookup(-1, 0, 0);
    tmp.y = lookup(-1, 1, 0);
    tmp.z = lookup(-1, 0, 1);
    tmp.w = lookup(-1, 1, 1);
    p2 = tmp.x * ncoeff.y * ncoeff.z +
        tmp.y *  coeff.y * ncoeff.z +
        tmp.z * ncoeff.y *  coeff.z +
        tmp.w *  coeff.y *  coeff.z;

    v1 = p1 * coeff.x + p2 * ncoeff.x;

    // v2
    // 1-layer
    p1 = front.y * ncoeff.y * ncoeff.z +
        front.w *  coeff.y * ncoeff.z +
        back.y  * ncoeff.y *  coeff.z +
        back.w  *  coeff.y *  coeff.z;
    // 2-layer
    tmp.x = lookup(2, 0, 0);
    tmp.y = lookup(2, 1, 0);
    tmp.z = lookup(2, 0, 1);
    tmp.w = lookup(2, 1, 1);
    p2 = tmp.x * ncoeff.y * ncoeff.z +
        tmp.y *  coeff.y * ncoeff.z +
        tmp.z * ncoeff.y *  coeff.z +
        tmp.w *  coeff.y *  coeff.z;

    ret.x = (
        p1 * ncoeff.x + p2 * coeff.x // v2
        -
        v1);

    // gradient y
    p1 = front.x * ncoeff.x * ncoeff.z +
        front.y *  coeff.x * ncoeff.z +
        back.x  * ncoeff.x *  coeff.z +
        back.y  *  coeff.x *  coeff.z;
    tmp.x = lookup(0, -1, 0);
    tmp.y = lookup(1, -1, 0);
    tmp.z = lookup(0, -1, 1);
    tmp.w = lookup(1, -1, 1);
    p2 = tmp.x * ncoeff.x * ncoeff.z +
        tmp.y *  coeff.x * ncoeff.z +
        tmp.z * ncoeff.x *  coeff.z +
        tmp.w *  coeff.x *  coeff.z;
    v1 = p1 * coeff.y + p2 * ncoeff.y;

    p1 = front.z * ncoeff.x * ncoeff.z +
        front.w *  coeff.x * ncoeff.z +
        back.z  * ncoeff.x *  coeff.z +
        back.w  *  coeff.x *  coeff.z;
    tmp.x = lookup(0, 2, 0);
    tmp.y = lookup(1, 2, 0);
    tmp.z = lookup(0, 2, 1);
    tmp.w = lookup(1, 2, 1);
    p2 = tmp.x * ncoeff.x * ncoeff.z +
        tmp.y *  coeff.x * ncoeff.z +
        tmp.z * ncoeff.x *  coeff.z +
        tmp.w *  coeff.x *  coeff.z;

    ret.y = (p1 * ncoeff.y + p2 * coeff.y - v1);

    // gradient z
    p1 = front.x * ncoeff.x * ncoeff.y +
        front.y *  coeff.x * ncoeff.y +
        front.z * ncoeff.x *  coeff.y +
        front.w *  coeff.x *  coeff.y;
    tmp.x = lookup(0, 0, -1);
    tmp.y = lookup(1, 0, -1);
    tmp.z = lookup(0, 1, -1);
    tmp.w = lookup(1, 1, -1);
    p2 = tmp.x * ncoeff.x * ncoeff.y +
        tmp.y *  coeff.x * ncoeff.y +
        tmp.z * ncoeff.x *  coeff.y +
        tmp.w *  coeff.x *  coeff.y;
    v1 = p1 * coeff.z + p2 * ncoeff.z;

    p1 = back.x * ncoeff.x * ncoeff.y +
        back.y *  coeff.x * ncoeff.y +
        back.z * ncoeff.x *  coeff.y +
        back.w *  coeff.x *  coeff.y;
    tmp.x = lookup(0, 0, 2);
    tmp.y = lookup(1, 0, 2);
    tmp.z = lookup(0, 1, 2);
    tmp.w = lookup(1, 1, 2);
    p2 = tmp.x * ncoeff.x * ncoeff.y +
        tmp.y *  coeff.x * ncoeff.y +
        tmp.z * ncoeff.x *  coeff.y +
        tmp.w *  coeff.x *  coeff.y;

    ret.z = (p1 * ncoeff.z + p2 * coeff.z - v1);
#undef lookup
    return ret;
}

#undef COMPUTE_COEFF_POS_FROM_POINT
#undef lookup

































namespace rendering {



    /**
    the 3D intersection locations generated by the latest raycast
    in voxelCoordinates
    */
    __managed__ PointImage* raycastResult;

    // for ICP
    //!< [out] receives output points in world coordinates
    //!< [out] receives world space normals computed from points (image space)
    // __managed__ DEVICEPTR(RayImage) * lastFrameICPMap = 0; // -- defined earlier

    // for RenderImage. Transparent where nothing is hit, otherwise computed by any of the DRAWFUNCTIONs
    __managed__ CameraImage<Vector4u>* outRendering = 0;
    __managed__ Vector3f towardsCamera;

    // written by rendering, world-space, 0 for invalid depths
    __managed__ ITMFloatImage* outDepth;

    // === raycasting, rendering ===
    /// \param x,y [in] camera space pixel determining ray direction
    //!< [out] raycastResult[locId]: the intersection point. 
    // w is 1 for a valid point, 0 for no intersection; in voxel-fractional-world-coordinates
    struct castRay {
        forEachPixelNoImage_process()
        {
            // Find 3d position of depth pixel xy, in eye coordinates
            auto pt_camera_f = raycastResult->getRayThroughPixel(Vector2i(x, y), viewFrustum_min);
            assert(pt_camera_f.origin.coordinateSystem == raycastResult->eyeCoordinates);
            auto l = pt_camera_f.endpoint().location;
            assert(l.z == viewFrustum_min);

            // Length given in voxel-fractional-coordinates (such that one voxel has size 1)
            auto pt_camera_f_vc = voxelCoordinates->convert(pt_camera_f);
            float totalLength = length(pt_camera_f_vc.direction.direction);
            assert(voxelSize < 1);
            assert(totalLength > length(pt_camera_f.direction.direction));
            assert(abs(
                totalLength - length(pt_camera_f.direction.direction) / voxelSize) < 0.001f);

            // in voxel-fractional-world-coordinates (such that one voxel has size 1)
            assert(pt_camera_f.endpoint().coordinateSystem == raycastResult->eyeCoordinates);
            assert(!(pt_camera_f_vc.endpoint().coordinateSystem == raycastResult->eyeCoordinates));
            const auto pt_block_s = pt_camera_f_vc.endpoint();

            // End point
            auto pt_camera_e = raycastResult->getRayThroughPixel(Vector2i(x, y), viewFrustum_max);
            auto pt_camera_e_vc = voxelCoordinates->convert(pt_camera_e);
            const float totalLengthMax = length(pt_camera_e_vc.direction.direction);
            const auto pt_block_e = pt_camera_e_vc.endpoint();

            assert(totalLength < totalLengthMax);
            assert(pt_block_s.coordinateSystem == voxelCoordinates);
            assert(pt_block_e.coordinateSystem == voxelCoordinates);

            // Raymarching
            const auto rayDirection = Vector(voxelCoordinates, normalize(pt_block_e.location - pt_block_s.location));
            auto pt_result = pt_block_s; // Current position in voxel-fractional-world-coordinates
            const float stepScale = mu * oneOverVoxelSize; // sdf values are distances in world-coordinates, normalized by division through mu. This is the factor to convert to voxelCoordinates.

            // TODO use caching, we will access the same voxel block multiple times
            float sdfValue = 1.0f;
            bool hash_found;

            // in voxel-fractional-world-coordinates (1.0f means step one voxel)
            float stepLength;

            while (totalLength < totalLengthMax) {
                // D(X)
                sdfValue = readFromSDF_float_uninterpolated(pt_result.location, hash_found);

                if (!hash_found) {
                    //  First we try to find an allocated voxel block, and the length of the steps we take is determined by the block size
                    stepLength = SDF_BLOCK_SIZE;
                }
                else {
                    // If we found an allocated block, 
                    // [Once we are inside the truncation band], the values from the SDF give us conservative step lengths.

                    // using trilinear interpolation only if we have read values in the range −0.5 ≤ D(X) ≤ 0.1
                    if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
                        sdfValue = readFromSDF_float_interpolated(pt_result.location, hash_found);
                    }
                    // once we read a negative value from the SDF, we found the intersection with the surface.
                    if (sdfValue <= 0.0f) break;

                    stepLength = MAX(
                        sdfValue * stepScale,
                        1.0f // if we are outside the truncation band µ, our step size is determined by the truncation band 
                        // (note that the distance is normalized to lie in [-1,1] within the truncation band)
                        );
                }

                pt_result = pt_result + rayDirection * stepLength;
                totalLength += stepLength;
            }

            bool pt_found;
            //  If the T - SDF value is negative after such a trilinear interpolation, the surface
            //  has indeed been found and we terminate the ray, performing one last
            //  trilinear interpolation step for a smoother appearance.
            if (sdfValue <= 0.0f)
            {
                // Refine position
                stepLength = sdfValue * stepScale;
                pt_result = pt_result + rayDirection * stepLength;

                // Read again
                sdfValue = readFromSDF_float_interpolated(pt_result.location, hash_found);
                // Refine position
                stepLength = sdfValue * stepScale;
                pt_result = pt_result + rayDirection * stepLength;

                pt_found = true;
            }
            else pt_found = false;

            raycastResult->image->GetData()[locId] = Vector4f(pt_result.location, (pt_found) ? 1.0f : 0.0f);
            assert(raycastResult->pointCoordinates == voxelCoordinates);
            assert(pt_result.coordinateSystem == voxelCoordinates);
        }
    };

    /// Compute normal in the distance field via the gradient.
    /// c.f. computeSingleNormalFromSDF
    GPU_ONLY inline void computeNormalAndAngle(
        bool & foundPoint, //!< [in,out]
        const Vector3f & point, //!< [in]
        Vector3f& outNormal,//!< [out] 
        float& angle //!< [out] outNormal . towardsCamera
        )
    {
        if (!foundPoint) return;

        outNormal = normalize(computeSingleNormalFromSDF(point));

        angle = dot(outNormal, towardsCamera);
        // dont consider points not facing the camera (raycast will hit these, do backface culling now)
        if (!(angle > 0.0)) foundPoint = false;
    }


    // PIXEL SHADERS
    // " Finally a coloured or shaded rendering of the surface is trivially computed, as desired for the visualisation."

#define DRAWFUNCTION \
    GPU_ONLY static void draw(\
    /*out*/ DEVICEPTR(Vector4u) & dest,\
        const Vector3f & point, /* point is in voxel-fractional world coordinates, comes from raycastResult*/\
        const Vector3f & normal_obj, \
        const float & angle)

    struct renderGrey {
        DRAWFUNCTION{
            const float outRes = (0.8f * angle + 0.2f) * 255.0f;
            dest = Vector4u((uchar)outRes);
            dest.a = 255;
        }
    };

    struct renderColourFromNormal {
        DRAWFUNCTION{
            dest.r = (uchar)((0.3f + (normal_obj.r + 1.0f)*0.35f)*255.0f);
            dest.g = (uchar)((0.3f + (normal_obj.g + 1.0f)*0.35f)*255.0f);
            dest.b = (uchar)((0.3f + (normal_obj.b + 1.0f)*0.35f)*255.0f);

            dest.a = 255;
        }
    };

    struct renderColour {
        DRAWFUNCTION{
            const Vector3f clr = readFromSDF_color4u_interpolated(point);
            dest = Vector4u(TO_UCHAR3(clr), 255);
        }
    };

    template<typename T_DRAWFUNCTION>
    struct PROCESSFUNCTION {
        forEachPixelNoImage_process() {
            DEVICEPTR(Vector4u) &outRender = outRendering->image->GetData()[locId];
            Point voxelCoordinatePoint = raycastResult->getPointForPixel(Vector2i(x, y));
            assert(voxelCoordinatePoint.coordinateSystem == voxelCoordinates);

            const Vector3f point = voxelCoordinatePoint.location;

            float& outZ = outDepth->GetData()[locId];

            auto a = outRendering->eyeCoordinates->convert(voxelCoordinatePoint);
            outZ = a.location.z; /* in world / eye coordinates (distance) */

            bool foundPoint = raycastResult->image->GetData()[locId].w > 0;

            Vector3f outNormal;
            float angle;
            computeNormalAndAngle(foundPoint, point, outNormal, angle);
            if (foundPoint) {/*assert(outZ >= viewFrustum_min && outZ <= viewFrustum_max); -- approx*/
                T_DRAWFUNCTION::draw(outRender, point, outNormal, angle);
            }
            else {
                outRender = Vector4u((uchar)0);
                outZ = 0;
            }
        }
    };


    /// Initializes raycastResult
    static void Common(
        const ITMPose pose,
        const ITMIntrinsics intrinsics
        ) {
        Vector2i imgSize = intrinsics.imageSize();
        assert(imgSize.area() > 1);
        auto raycastImage = new ITMFloat4Image(imgSize);
        auto invPose_M = pose.GetInvM();
        auto cameraCs = new CoordinateSystem(invPose_M);
        raycastResult = new PointImage(
            raycastImage,
            voxelCoordinates,
            cameraCs,
            intrinsics
            );

        // (negative camera z axis)
        towardsCamera = -Vector3f(invPose_M.getColumn(2));

        forEachPixelNoImage<castRay>(imgSize);
    }

    struct Camera {
    public:
        ITMPose pose;
        ITMIntrinsics intrinsics;
    };

    /** Render an image using raycasting. */
    ITMView* RenderImage(const ITMPose pose, const ITMIntrinsics intrinsics,
        const string shader // any of the DRAWFUNCTION
        )
    {
        const Vector2i imgSize = intrinsics.imageSize();
        assert(imgSize.area() > 1);

        ITMRGBDCalib* outCalib = new ITMRGBDCalib();
        outCalib->intrinsics_d = outCalib->intrinsics_rgb = intrinsics;
        auto outView = new ITMView(outCalib);
        outView->ChangePose(pose.GetM());
        rendering::outDepth = outView->depthImage->image;
        assert(rendering::outDepth);
        assert(rendering::outDepth->noDims == imgSize);

        auto outImage = new ITMUChar4Image(imgSize);
        auto outCs = new CoordinateSystem(pose.GetInvM());
        outRendering = outView->colorImage;

        assert(outRendering->imgSize() == intrinsics.imageSize());
        Common(pose, intrinsics);
        cudaDeviceSynchronize(); // want to read imgSize

#define isShader(s) if (shader == #s) {forEachPixelNoImage<PROCESSFUNCTION<s>>(outRendering->imgSize());cudaDeviceSynchronize(); return outView;}
        isShader(renderColour);
        isShader(renderColourFromNormal);
        isShader(renderGrey);
#undef isShader

        fatalError("unkown shader %s", shader.c_str());

        return nullptr;
    }

    /// Computing the surface normal in image space given raycasted image (raycastResult).
    ///
    /// In image space, since the normals are computed on a regular grid,
    /// there are only 4 uninterpolated read operations followed by a cross-product.
    /// (here we might do more when useSmoothing is true, and we step 2 pixels wide to find // //further-away neighbors)
    ///
    /// \returns normal_out[idx].w = sigmaZ_out[idx] = -1 on error where idx = x + y * imgDims.x
    template <bool useSmoothing>
    GPU_ONLY inline void computeNormalImageSpace(
        bool& foundPoint, //!< [in,out] Set to false when the normal cannot be computed
        const int &x, const int&y,
        Vector3f & outNormal
        )
    {
        if (!foundPoint) return;
        const Vector2i imgSize = raycastResult->imgSize();

        // Lookup world coordinates of points surrounding (x,y)
        // and compute forward difference vectors
        Vector4f xp1_y, xm1_y, x_yp1, x_ym1;
        Vector4f diff_x(0.0f, 0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f, 0.0f);

        // If useSmoothing, use positions 2 away
        int extraDelta = useSmoothing ? 1 : 0;

#define d(x) (x + extraDelta)

        if (y <= d(1) || y >= imgSize.y - d(2) || x <= d(1) || x >= imgSize.x - d(2)) { foundPoint = false; return; }

#define lookupNeighbors() \
    xp1_y = sampleNearest(raycastResult->image->GetData(), x + d(1), y, imgSize);\
    x_yp1 = sampleNearest(raycastResult->image->GetData(), x, y + d(1), imgSize);\
    xm1_y = sampleNearest(raycastResult->image->GetData(), x - d(1), y, imgSize);\
    x_ym1 = sampleNearest(raycastResult->image->GetData(), x, y - d(1), imgSize);\
    diff_x = xp1_y - xm1_y;\
    diff_y = x_yp1 - x_ym1;

        lookupNeighbors();

#define isAnyPointIllegal() (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0)

        float length_diff = MAX(length2(diff_x.toVector3()), length2(diff_y.toVector3()));
        bool lengthDiffTooLarge = (length_diff * voxelSize * voxelSize > (0.15f * 0.15f));

        if (isAnyPointIllegal() || (lengthDiffTooLarge && useSmoothing)) {
            if (!useSmoothing) { foundPoint = false; return; }

            // In case we used smoothing, try again without extra delta
            extraDelta = 0;
            lookupNeighbors();

            if (isAnyPointIllegal()){ foundPoint = false; return; }
        }

#undef d
#undef isAnyPointIllegal
#undef lookupNeighbors

        // TODO why the extra minus? -- it probably does not matter because we compute the distance to a plane which would be the same with the inverse normal
        outNormal = normalize(-cross(diff_x.toVector3(), diff_y.toVector3()));

        float angle = dot(outNormal, towardsCamera);
        // dont consider points not facing the camera (raycast will hit these, do backface culling now)
        if (!(angle > 0.0)) foundPoint = false;
    }

#define useSmoothing true

    static __managed__ RayImage* outIcpMap = 0;
    /// Produces a shaded image (outRendering) and a point cloud for e.g. tracking.
    /// Uses image space normals.
    /// \param useSmoothing whether to compute normals by forward differences two pixels away (true) or just one pixel away (false)
    struct processPixelICP {
        forEachPixelNoImage_process() {
            const Vector4f point = raycastResult->image->GetData()[locId];
            assert(raycastResult->pointCoordinates == voxelCoordinates);

            bool foundPoint = point.w > 0.0f;

            Vector3f outNormal;
            // TODO could we use the world space normals here? not without change
            computeNormalImageSpace<useSmoothing>(
                foundPoint, x, y, outNormal);

#define pointsMap outIcpMap->image->GetData()
#define normalsMap outIcpMap->normalImage->GetData()

            if (!foundPoint)
            {
                pointsMap[locId] = normalsMap[locId] = IllegalColor<Vector4f>::make();
                return;
            }

            // Convert point to world coordinates
            pointsMap[locId] = Vector4f(point.toVector3() * voxelSize, 1);
            // Normals are the same whether in world or voxel coordinates
            normalsMap[locId] = Vector4f(outNormal, 0);
#undef pointsMap
#undef normalsMap
        }
    };

    // 1. raycast scene from current viewpoint 
    // to create point cloud for tracking
    RayImage * CreateICPMapsForCurrentView() {
        assert(currentView);

        auto imgSize_d = currentView->depthImage->imgSize();
        assert(imgSize_d.area() > 1);
        auto pointsMap = new ITMFloat4Image(imgSize_d);
        auto normalsMap = new ITMFloat4Image(imgSize_d);

        assert(!outIcpMap);
        outIcpMap = new RayImage(
            pointsMap,
            normalsMap,
            CoordinateSystem::global(),

            currentView->depthImage->eyeCoordinates,
            currentView->depthImage->cameraIntrinsics
            );

        assert(Scene::getCurrentScene());

        // TODO reduce conversion friction
        ITMPose pose; pose.SetM(currentView->depthImage->eyeCoordinates->fromGlobal);
        ITMIntrinsics intrin = currentView->calib->intrinsics_d;
        assert(intrin.imageSize() == imgSize_d);
        assert(intrin.all == currentView->depthImage->cameraIntrinsics.all);
        assert(intrin.imageSize() == currentView->depthImage->cameraIntrinsics.imageSize());
        Common(
            pose, //trackingState->pose_d,
            intrin
            );
        cudaDeviceSynchronize();

        approxEqual(raycastResult->eyeCoordinates->fromGlobal, currentView->depthImage->eyeCoordinates->fromGlobal);
        assert(raycastResult->pointCoordinates == voxelCoordinates);

        // Create ICP maps
        forEachPixelNoImage<processPixelICP>(imgSize_d);
        cudaDeviceSynchronize();

        // defensive
        assert(outIcpMap->eyeCoordinates == currentView->depthImage->eyeCoordinates);
        assert(outIcpMap->pointCoordinates == CoordinateSystem::global());
        assert(outIcpMap->imgSize() == imgSize_d);
        assert(outIcpMap->normalImage->noDims == imgSize_d);
        auto icpMap = outIcpMap;
        outIcpMap = 0;
        return icpMap;
    }

}
using namespace rendering;





































namespace fusion {


#define weightedCombine(oldX, oldW, newX, newW) \
    newX = (float)oldW * oldX + (float)newW * newX; \
    newW = oldW + newW;\
    newX /= (float)newW;\
    newW = MIN(newW, maxW);

    CPU_AND_GPU inline void updateVoxelColorInformation(
        DEVICEPTR(ITMVoxel) & voxel,
        const Vector3f oldC, const int oldW, Vector3f newC, int newW)
    {
        weightedCombine(oldC, oldW, newC, newW);

        // write back
        /// C(X) <-  
        voxel.clr = TO_UCHAR3(newC);
        voxel.w_color = (uchar)newW;
    }

    CPU_AND_GPU inline void updateVoxelDepthInformation(
        DEVICEPTR(ITMVoxel) & voxel,
        const float oldF, const int oldW, float newF, int newW)
    {
        weightedCombine(oldF, oldW, newF, newW);

        // write back
        /// D(X) <-  (4)
        voxel.setSDF(newF);
        voxel.w_depth = (uchar)newW;
    }
#undef weightedCombine

    /// Fusion Stage - Camera Data Integration
    /// \returns \f$\eta\f$, -1 on failure
    // Note that the stored T-SDF values are normalized to lie
    // in [-1,1] within the truncation band.
    GPU_ONLY inline float computeUpdatedVoxelDepthInfo(
        DEVICEPTR(ITMVoxel) &voxel, //!< X
        const Point & pt_model //!< in world space
        )
    {

        // project point into depth image
        /// X_d, depth camera coordinate system
        const Vector4f pt_camera = Vector4f(
            currentView->depthImage->eyeCoordinates->convert(pt_model).location,
            1);
        /// \pi(K_dX_d), projection into the depth image
        Vector2f pt_image;
        if (!currentView->depthImage->project(pt_model, pt_image))
            return -1;

        // get measured depth from image, no interpolation
        /// I_d(\pi(K_dX_d))
        auto p = currentView->depthImage->getPointForPixel(pt_image.toInt());
        const float depth_measure = p.location.z;
        if (depth_measure <= 0.0) return -1;

        /// I_d(\pi(K_dX_d)) - X_d^(z)          (3)
        float const eta = depth_measure - pt_camera.z;
        // check whether voxel needs updating
        if (eta < -mu) return eta;

        // compute updated SDF value and reliability (number of observations)
        /// D(X), w(X)
        float const oldF = voxel.getSDF();
        int const oldW = voxel.w_depth;

        // newF, normalized for -1 to 1
        float const newF = MIN(1.0f, eta / mu);
        int const newW = 1;

        updateVoxelDepthInformation(
            voxel,
            oldF, oldW, newF, newW);

        return eta;
    }

    /// \returns early on failure
    GPU_ONLY inline void computeUpdatedVoxelColorInfo(
        DEVICEPTR(ITMVoxel) &voxel,
        const Point & pt_model)
    {
        Vector2f pt_image;
        if (!currentView->colorImage->project(pt_model, pt_image))
            return;

        int oldW = (float)voxel.w_color;
        const Vector3f oldC = TO_FLOAT3(voxel.clr);

        /// Like formula (4) for depth
        const Vector3f newC = TO_VECTOR3(interpolateBilinear<Vector4f>(currentView->colorImage->image->GetData(), pt_image, currentView->colorImage->imgSize()));
        int newW = 1;

        updateVoxelColorInformation(
            voxel,
            oldC, oldW, newC, newW);
    }


    GPU_ONLY static void computeUpdatedVoxelInfo(
        DEVICEPTR(ITMVoxel) & voxel, //!< [in, out] updated voxel
        const Point & pt_model) {
        const float eta = computeUpdatedVoxelDepthInfo(voxel, pt_model);

        // Only the voxels within +- 25% mu of the surface get color
        if ((eta > mu) || (fabs(eta / mu) > 0.25f)) return;
        computeUpdatedVoxelColorInfo(voxel, pt_model);
    }

    /// Determine the blocks around a given depth sample that are currently visible
    /// and need to be allocated.
    /// Builds hashVisibility and entriesAllocType.
    /// \param x,y [in] loop over depth image.
    struct buildHashAllocAndVisibleTypePP {
        forEachPixelNoImage_process() {
            // Find 3d position of depth pixel xy, in eye coordinates
            auto pt_camera = currentView->depthImage->getPointForPixel(Vector2i(x, y));

            const float depth = pt_camera.location.z;
            if (depth <= 0 || (depth - mu) < 0 || (depth - mu) < viewFrustum_min || (depth + mu) > viewFrustum_max) return;

            // the found point +- mu
            const Vector pt_camera_v = (pt_camera - currentView->depthImage->location());
            const float norm = length(pt_camera_v.direction);
            const Vector pt_camera_v_minus_mu = pt_camera_v*(1.0f - mu / norm);
            const Vector pt_camera_v_plus_mu = pt_camera_v*(1.0f + mu / norm);

            // Convert to voxel block coordinates  
            // the initial point pt_camera_v_minus_mu
            Point point = voxelBlockCoordinates->convert(currentView->depthImage->location() + pt_camera_v_minus_mu);
            // the direction towards pt_camera_v_plus_mu in voxelBlockCoordinates
            const Vector vector = voxelBlockCoordinates->convert(pt_camera_v_plus_mu - pt_camera_v_minus_mu);

            // We will step along point -> point_e and add all voxel blocks we encounter to the visible list
            // "Create a segment on the line of sight in the range of the T-SDF truncation band"
            const int noSteps = (int)ceil(2.0f* length(vector.direction)); // make steps smaller than 1, maybe even < 1/2 to really land in all blocks at least once
            const Vector direction = vector * (1.f / (float)(noSteps - 1));

            //add neighbouring blocks
            for (int i = 0; i < noSteps; i++)
            {
                // "take the block coordinates of voxels on this line segment"
                const VoxelBlockPos blockPos = TO_SHORT_FLOOR3(point.location);
                Scene::requestCurrentSceneVoxelBlockAllocation(blockPos);

                point = point + direction;
            }
        }
    };

    struct IntegrateVoxel {
        doForEachAllocatedVoxel_process() {
            computeUpdatedVoxelInfo(*v, globalPoint);
        }
    };


    /// Fusion stage of the system, depth integration process
    void Fuse()
    {
        cudaDeviceSynchronize();
        assert(Scene::getCurrentScene());
        assert(currentView);

        // allocation request
        forEachPixelNoImage<buildHashAllocAndVisibleTypePP>(currentView->depthImage->imgSize());
        cudaDeviceSynchronize();

        // allocation
        Scene::performCurrentSceneAllocations();

        // camera data integration
        cudaDeviceSynchronize();
        Scene::getCurrentScene()->doForEachAllocatedVoxel<IntegrateVoxel>();
    }



}
using namespace fusion;
















namespace tracking {



    /** Performing ICP based depth tracking.
    Implements the original KinectFusion tracking algorithm.

    c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

    6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
    */
    void ImprovePose();
    /** \file c.f. newcombe_etal_ismar2011.pdf
    * T_{g,k} denotes the transformation from frame k's view space to global space
    * T_{k,g} is the inverse
    */
    /// \file Depth Tracker, c.f. newcombe_etal_ismar2011.pdf Sensor Pose Estimation
    // The current implementation ignores the possible optimizations/special iterations with 
    // rotation estimation only ("At the coarser levels we optimise only for the rotation matrix R.")

    struct AccuCell : public Managed {
        int noValidPoints;
        float f;
        // ATb
        float ATb[6];
        // AT_A (note that this is actually a symmetric matrix, so we could save some effort and memory)
        float AT_A[6][6];
        void reset() {
            memset(this, 0, sizeof(AccuCell));
        }
    };

    /// The tracker iteration type used to define the tracking iteration regime
    enum TrackerIterationType
    {
        /// Update only the current rotation estimate. This is preferable for the coarse solution stages.
        TRACKER_ITERATION_ROTATION = 1,
        TRACKER_ITERATION_BOTH = 3,
        TRACKER_ITERATION_NONE = 4
    };
    struct TrackingLevel : public Managed {
        /// FilterSubsampleWithHoles result of one level higher
        /// Half of the intrinsics of one level higher
        /// Coordinate system is defined by the matrix M_d (this is the world-to-eye transform, i.e. 'fromGlobal')
        /// which we are optimizing for.
        DepthImage* depthImage;

        // Tweaking
        const float distanceThreshold;
        const int numberOfIterations;
        const TrackerIterationType iterationType;

        TrackingLevel(int numberOfIterations, TrackerIterationType iterationType, float distanceThreshold) :
            numberOfIterations(numberOfIterations), iterationType(iterationType), distanceThreshold(distanceThreshold),
            depthImage(0) {
        }
    };
    // ViewHierarchy, 0 is highest resolution
    static std::vector<TrackingLevel*> trackingLevels;
    struct ITMDepthTracker_
    {
        ITMDepthTracker_() {
            // Tweaking
            // Tracking strategy:
            const int noHierarchyLevels = 5;
            const float distThreshStep = depthTrackerICPMaxThreshold / noHierarchyLevels;
            // starting with highest resolution (lowest level, last to be executed)
#define iterations
            trackingLevels.push_back(new TrackingLevel(2  iterations, TRACKER_ITERATION_BOTH, depthTrackerICPMaxThreshold - distThreshStep * 4));
            trackingLevels.push_back(new TrackingLevel(4  iterations, TRACKER_ITERATION_BOTH, depthTrackerICPMaxThreshold - distThreshStep * 3));
            trackingLevels.push_back(new TrackingLevel(6  iterations, TRACKER_ITERATION_ROTATION, depthTrackerICPMaxThreshold - distThreshStep * 2));
            trackingLevels.push_back(new TrackingLevel(8  iterations, TRACKER_ITERATION_ROTATION, depthTrackerICPMaxThreshold - distThreshStep));
            trackingLevels.push_back(new TrackingLevel(10 iterations, TRACKER_ITERATION_ROTATION, depthTrackerICPMaxThreshold));
            assert(trackingLevels.size() == noHierarchyLevels);
#undef iterations
        }
    } _;

    static __managed__ /*const*/ TrackingLevel* currentTrackingLevel;

    static TrackerIterationType iterationType() {
        return currentTrackingLevel->iterationType;
    }
    static bool shortIteration() {
        return (iterationType() == TRACKER_ITERATION_ROTATION);
    }
    /// In world-coordinates squared
    //!< \f$\epsilon_d\f$
    GPU_ONLY static float distThresh() {
        return currentTrackingLevel->distanceThreshold;
    }

    static __managed__ /*const*/ AccuCell accu;
    /// In world coordinates, points map, normals map, for frame k-1, \f$V_{k-1}\f$
    static __managed__ DEVICEPTR(RayImage) * lastFrameICPMap = 0;


    /**
    Computes
    \f{eqnarray*}{
    b &:=& n_{k-1}^\top(p_{k-1} - p_k)  \\
    A^T &:=& G(u)^T . n_{k-1}\\
    \f}

    where \f$G(u) = [ [p_k]_\times \;|\; Id ]\f$ a 3 x 6 matrix and \f$A^T\f$ is a 6 x 1 column vector.

    \f$p_{k-1}\f$ is the point observed in the last frame in
    the direction in which \f$p_k\f$ is observed (projective data association).

    \f$n_{k-1}\f$ is the normal that was observed at that location

    \f$b\f$ is the point-plane alignment energy for the point under consideration

    \param x,y \f$\mathbf u\f$
    \return false on failure
    \see newcombe_etal_ismar2011.pdf Sensor Pose Estimation
    */
    GPU_ONLY static inline bool computePerPointGH_Depth_Ab(
        float AT[6], //!< [out]
        float &b,//!< [out]
        const int x, const int y
        )
    {
        // p_k := T_{g,k}V_k(u) = V_k^g(u)
        Point V_ku = currentTrackingLevel->depthImage->getPointForPixel(Vector2i(x, y));
        if (V_ku.location.z <= 1e-8f) return false;
        assert(V_ku.coordinateSystem == currentTrackingLevel->depthImage->eyeCoordinates);
        Point p_k = CoordinateSystem::global()->convert(V_ku);

        // hat_u = \pi(K T_{k-1,g} T_{g,k}V_k(u) )
        Vector2f hat_u;
        if (!lastFrameICPMap->project(
            p_k,
            hat_u,
            EXTRA_BOUNDS))
            return false;

        bool isIllegal = false;
        Ray ray = lastFrameICPMap->getRayForPixelInterpolated(hat_u, isIllegal);
        if (isIllegal) return false;

        // p_km1 := V_{k-1}(\hat u)
        const Point p_km1 = ray.origin;

        // n_km1 := N_{k-1}(\hat u)
        const Vector n_km1 = ray.direction;

        // d := p_km1 - p_k
        const Vector d = p_km1 - p_k;

        // [
        // Projective data assocation rejection test, "\Omega_k(u) != 0"
        // TODO check whether normal matches normal from image, done in the original paper, but does not seem to be required
        if (length2(d.direction) > distThresh()) return false;
        // ]

        // (2) Point-plane ICP computations

        // b = n_km1 . (p_km1 - p_k)
        b = n_km1.dot(d);

        // Compute A^T = G(u)^T . n_{k-1}
        // Where G(u) = [ [p_k]_x Id ] a 3 x 6 matrix
        // [v]_x denotes the skew symmetric matrix such that for all w, [v]_x w = v \cross w
        int counter = 0;
        {
            const Vector3f pk = p_k.location;
            const Vector3f nkm1 = n_km1.direction;
            // rotationPart
            AT[counter++] = +pk.z * nkm1.y - pk.y * nkm1.z;
            AT[counter++] = -pk.z * nkm1.x + pk.x * nkm1.z;
            AT[counter++] = +pk.y * nkm1.x - pk.x * nkm1.y;
            // translationPart
            AT[counter++] = nkm1.x;
            AT[counter++] = nkm1.y;
            AT[counter++] = nkm1.z;
        }

        return true;
    }


#define REDUCE_BLOCK_SIZE 256 // must be power of 2. Used for reduction of a sum.
    static KERNEL depthTrackerOneLevel_g_rt_device_main()
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

        int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

        __shared__ bool should_prefix; // set to true if any point is valid

        should_prefix = false;
        __syncthreads();

        float A[6];
        float b;
        bool isValidPoint = false;

        auto viewImageSize = currentTrackingLevel->depthImage->imgSize();
        if (x < viewImageSize.width && y < viewImageSize.height
            )
        {
            isValidPoint = computePerPointGH_Depth_Ab(
                A, b, x, y);
            if (isValidPoint) should_prefix = true;
        }

        if (!isValidPoint) {
            for (int i = 0; i < 6; i++) A[i] = 0.0f;
            b = 0.0f;
        }

        __syncthreads();

        if (!should_prefix) return;

        __shared__ float dim_shared1[REDUCE_BLOCK_SIZE];

        { //reduction for noValidPoints
            warpReduce256<int>(
                isValidPoint,
                dim_shared1,
                locId_local,
                &(accu.noValidPoints));
        }
#define reduce(what, into) warpReduce256<float>((what),dim_shared1,locId_local,&(into));
    { //reduction for energy function value
        reduce(b*b, accu.f);
    }

    //reduction for nabla
    for (unsigned char paraId = 0; paraId < 6; paraId++)
    {
        reduce(b*A[paraId], accu.ATb[paraId]);
    }

    float AT_A[6][6];
    int counter = 0;
    for (int r = 0; r < 6; r++)
    {
        for (int c = 0; c < 6; c++) {
            AT_A[r][c] = A[r] * A[c];

            //reduction for hessian
            reduce(AT_A[r][c], accu.AT_A[r][c]);
        }
    }
    }

    // host methods

    AccuCell ComputeGandH(Matrix4f T_g_k_estimate) {
        cudaDeviceSynchronize(); // prepare writing to __managed__

        assert(lastFrameICPMap->pointCoordinates == CoordinateSystem::global());
        assert(!(lastFrameICPMap->eyeCoordinates == CoordinateSystem::global()));
        assert(lastFrameICPMap->eyeCoordinates == currentView->depthImage->eyeCoordinates);

        //::depth = currentTrackingLevel->depth->GetData(MEMORYDEVICE_CUDA);
        //::viewIntrinsics = currentTrackingLevel->intrinsics;
        auto viewImageSize = currentTrackingLevel->depthImage->imgSize();

        //::T_g_k = T_g_k_estimate;
        std::auto_ptr<CoordinateSystem> depthCoordinateSystemEstimate(new CoordinateSystem(T_g_k_estimate)); // TODO should this be deleted when going out of scope?
        // do we really need to recreate it every time? Should it be the same instance ('new depth eye coordinate system') for all resolutions?
        currentTrackingLevel->depthImage->eyeCoordinates = depthCoordinateSystemEstimate.get();

        dim3 blockSize(16, 16); // must equal REDUCE_BLOCK_SIZE
        assert(16 * 16 == REDUCE_BLOCK_SIZE);

        dim3 gridSize(
            (int)ceil((float)viewImageSize.x / (float)blockSize.x),
            (int)ceil((float)viewImageSize.y / (float)blockSize.y));

        assert(!(currentTrackingLevel->depthImage->eyeCoordinates == CoordinateSystem::global()));

        accu.reset();
        LAUNCH_KERNEL(depthTrackerOneLevel_g_rt_device_main, gridSize, blockSize);

        cudaDeviceSynchronize(); // for later access of accu
        return accu;
    }

    /// evaluate error function at the supplied T_g_k_estimate, 
    /// compute sum_ATb and sum_AT_A, the system we need to solve to compute the
    /// next update step (note: this system is not yet solved and we don't know the new energy yet!)
    /// \returns noValidPoints
    int ComputeGandH(
        float &f,
        float sum_ATb[6],
        float sum_AT_A[6][6],
        Matrix4f T_g_k_estimate) {
        AccuCell accu = ComputeGandH(T_g_k_estimate);

        memcpy(sum_ATb, accu.ATb, sizeof(float) * 6);
        assert(sum_ATb[4] == accu.ATb[4]);
        memcpy(sum_AT_A, accu.AT_A, sizeof(float) * 6 * 6);
        assert(sum_AT_A[3][4] == accu.AT_A[3][4]);

        // Output energy -- if we have very few points, output some high energy
        f = (accu.noValidPoints > 100) ? sqrt(accu.f) / accu.noValidPoints : 1e5f;

        return accu.noValidPoints;
    }

    /// Solves hessian.step = nabla
    /// \param delta output array of 6 floats 
    /// \param hessian 6x6
    /// \param delta 3 or 6
    /// \param nabla 3 or 6
    /// \param shortIteration whether there are only 3 parameters
    void ComputeDelta(float step[6], float nabla[6], float hessian[6][6])
    {
        for (int i = 0; i < 6; i++) step[i] = 0;

        if (shortIteration())
        {
            // Keep only upper 3x3 part of hessian
            float smallHessian[3][3];
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) smallHessian[r][c] = hessian[r][c];

            Cholesky::solve((float*)smallHessian, 3, nabla, step);

            // check
            /*float result[3];
            matmul((float*)smallHessian, step, result, 3, 3);
            for (int r = 0; r < 3; r++)
            assert(abs(result[r] - nabla[r]) / abs(result[r]) < 0.0001);
            */
        }
        else
        {
            Cholesky::solve((float*)hessian, 6, nabla, step);
        }
    }

    bool HasConverged(float *step)
    {
        // Compute ||step||_2^2
        float stepLength = 0.0f;
        for (int i = 0; i < 6; i++) stepLength += step[i] * step[i];

        // heuristic? Why /6?
        if (sqrt(stepLength) / 6 < depthTrackerTerminationThreshold) return true; //converged

        return false;
    }

    Matrix4f ComputeTinc(const float delta[6])
    {
        // step is T_inc, expressed as a parameter vector 
        // (beta, gamma, alpha, tx,ty, tz)
        // beta, gamma, alpha parametrize the rotation axis and angle
        float step[6];

        // Depending on the iteration type, fill in 0 for values that where not computed.
        switch (currentTrackingLevel->iterationType)
        {
        case TRACKER_ITERATION_ROTATION:
            step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
            step[3] = 0.0f; step[4] = 0.0f; step[5] = 0.0f;
            break;
        default:
        case TRACKER_ITERATION_BOTH:
            step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
            step[3] = (float)(delta[3]); step[4] = (float)(delta[4]); step[5] = (float)(delta[5]);
            break;
        }

        // Incremental pose update assuming small angles.
        // c.f. (18) in newcombe_etal_ismar2011.pdf
        // step = (beta, gamma, alpha, tx, ty, tz)
        // Tinc = 
        /*
        1       alpha   -gamma tx
        -alpha      1     beta ty
        gamma   -beta        1 tz
        i.e.

        1         step[2]   -step[1] step[3]
        -step[2]        1    step[0] step[4]
        step[1]  -step[0]          1 step[5]
        */
        Matrix4f Tinc;

        Tinc.m00 = 1.0f;		Tinc.m10 = step[2];		Tinc.m20 = -step[1];	Tinc.m30 = step[3];
        Tinc.m01 = -step[2];	Tinc.m11 = 1.0f;		Tinc.m21 = step[0];		Tinc.m31 = step[4];
        Tinc.m02 = step[1];		Tinc.m12 = -step[0];	Tinc.m22 = 1.0f;		Tinc.m32 = step[5];
        Tinc.m03 = 0.0f;		Tinc.m13 = 0.0f;		Tinc.m23 = 0.0f;		Tinc.m33 = 1.0f;
        return Tinc;
    }

    /** Performing ICP based depth tracking.
    Implements the original KinectFusion tracking algorithm.

    c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

    6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
    */
    /// \file c.f. newcombe_etal_ismar2011.pdf, Sensor Pose Estimation section
    void ImprovePose() {
        assert(currentView);
        //assert(!lastFrameICPMap);
        lastFrameICPMap = rendering::CreateICPMapsForCurrentView();

        /// Initialize one tracking event base data. Init hierarchy level 0 (finest).
        cudaDeviceSynchronize(); // prepare writing to __managed__

        /// Init image hierarchy levels
        assert(currentView->depthImage->imgSize().area() > 1);
        trackingLevels[0]->depthImage = //currentView->depthImage;
            new DepthImage(
            currentView->depthImage->image,
            new CoordinateSystem(*currentView->depthImage->eyeCoordinates),
            currentView->depthImage->cameraIntrinsics
            );

        for (int i = 1; i < trackingLevels.size(); i++)
        {
            TrackingLevel* currentLevel = trackingLevels[i];
            TrackingLevel* previousLevel = trackingLevels[i - 1];

            auto subsampledDepthImage = new ITMFloatImage();
            FilterSubsampleWithHoles(subsampledDepthImage, previousLevel->depthImage->image);
            cudaDeviceSynchronize();

            ITMIntrinsics subsampledIntrinsics;
            subsampledIntrinsics.imageSize(subsampledDepthImage->noDims);
            subsampledIntrinsics.all = previousLevel->depthImage->projParams() * 0.5;

            currentLevel->depthImage = new DepthImage(
                subsampledDepthImage,
                CoordinateSystem::global(), // will be set correctly later
                subsampledIntrinsics
                );

            assert(currentLevel->depthImage->imgSize() == previousLevel->depthImage->imgSize() / 2);
            assert(currentLevel->depthImage->imgSize().area() < currentView->depthImage->imgSize().area());
        }

        ITMPose T_k_g_estimate;
        T_k_g_estimate.SetM(currentView->depthImage->eyeCoordinates->fromGlobal);
        {
            Matrix4f M_d = T_k_g_estimate.GetM();
            assert(M_d == currentView->depthImage->eyeCoordinates->fromGlobal);
        }
        // Coarse to fine
        for (int levelId = trackingLevels.size() - 1; levelId >= 0; levelId--)
        {
            currentTrackingLevel = trackingLevels[levelId];
            if (iterationType() == TRACKER_ITERATION_NONE) continue;

            // T_{k,g} transforms global (g) coordinates to eye or view coordinates of the k-th frame
            // T_g_k_estimate caches T_k_g_estimate->GetInvM()
            Matrix4f T_g_k_estimate = T_k_g_estimate.GetInvM();

#define set_T_k_g_estimate(x)\
T_k_g_estimate.SetFrom(&x);
            T_g_k_estimate = T_k_g_estimate.GetInvM();

#define set_T_k_g_estimate_from_T_g_k_estimate(x) \
T_k_g_estimate.SetInvM(x);\
T_k_g_estimate.Coerce(); /* and make sure we've got an SE3*/\
T_g_k_estimate = T_k_g_estimate.GetInvM();

            // We will 'accept' updates into trackingState->pose_d and T_g_k_estimate
            // before we know whether they actually decrease the energy.
            // When they did not in fact, we will revert to this value that was known to have less energy 
            // than all previous estimates.
            ITMPose least_energy_T_k_g_estimate(T_k_g_estimate);

            // Track least energy we measured so far to see whether we improved
            float f_old = 1e20f;

            // current levenberg-marquart style damping parameter, often called mu.
            float lambda = 1.0;

            // Iterate as required
            for (int iterNo = 0; iterNo < currentTrackingLevel->numberOfIterations; iterNo++)
            {
                // [ this takes most time. 
                // Computes f(x) as well as A^TA and A^Tb for next computation of delta_x as
                // (A^TA + lambda * diag(A^TA)) delta_x = A^T b
                // if f decreases, the delta is applied definitely, otherwise x is reset.
                // So we do:
                /*
                x = x_best;
                lambda = 1;
                f_best = infinity

                repeat:
                compute f_new, A^TA_new, A^T b_new

                if (f_new > f_best) {x = x_best; lambda *= 10;}
                else {
                x_best = x;
                A^TA = A^TA_new
                A^Tb = A^Tb_new
                }

                solve (A^TA + lambda * diag(A^TA)) delta_x = A^T b
                x += delta_x;

                */


                // evaluate error function at currently accepted
                // T_g_k_estimate
                // and compute information for next update
                float f_new;
                int noValidPoints;
                float new_sum_ATb[6];
                float new_sum_AT_A[6][6];
                noValidPoints = ComputeGandH(f_new, new_sum_ATb, new_sum_AT_A, T_g_k_estimate);
                // ]]

                float least_energy_sum_AT_A[6][6],
                    damped_least_energy_sum_AT_A[6][6];
                float least_energy_sum_ATb[6];

                // check if energy actually *increased* with the last update
                // Note: This happens rarely, namely when the blind 
                // gauss-newton step actually leads to an *increase in energy
                // because the damping was too small
                if ((noValidPoints <= 0) || (f_new > f_old)) {
                    // If so, revert pose and discard/ignore new_sum_AT_A, new_sum_ATb
                    // TODO would it be worthwhile to not compute these when they are not going to be used?
                    set_T_k_g_estimate(least_energy_T_k_g_estimate);
                    // Increase damping, then solve normal equations again with old matrix (see below)
                    lambda *= 10.0f;
                }
                else {
                    f_old = f_new;
                    least_energy_T_k_g_estimate.SetFrom(&T_k_g_estimate);

                    // Prepare to solve a new system

                    // Preconditioning: Normalize by noValidPoints
                    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) least_energy_sum_AT_A[i][j] = new_sum_AT_A[i][j] / noValidPoints;
                    for (int i = 0; i < 6; ++i) least_energy_sum_ATb[i] = new_sum_ATb[i] / noValidPoints;

                    // Accept and decrease damping
                    lambda /= 10.0f;
                }
                // Solve normal equations

                // Apply levenberg-marquart style damping (multiply diagonal of ATA by 1.0f + lambda)
                for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) damped_least_energy_sum_AT_A[i][j] = least_energy_sum_AT_A[i][j];
                for (int i = 0; i < 6; ++i) damped_least_energy_sum_AT_A[i][i] *= 1.0f + lambda;

                // compute the update step parameter vector x
                float x[6];
                ComputeDelta(x,
                    least_energy_sum_ATb,
                    damped_least_energy_sum_AT_A);

                // Apply the corresponding Tinc
                set_T_k_g_estimate_from_T_g_k_estimate(
                    /* T_g_k_estimate = */
                    ComputeTinc(x) * T_g_k_estimate
                    );

                // if step is small, assume it's going to decrease the error and finish
                if (HasConverged(x)) break;
            }
        }

        //delete lastFrameICPMap;
        //lastFrameICPMap = 0;

        // Apply new guess
        Matrix4f M_d = T_k_g_estimate.GetM();

        // DEBUG sanity check
        Matrix4f id; id.setIdentity();
        assert(M_d != id);

        cudaDeviceSynchronize(); // necessary here?
        assert(currentView->depthImage->eyeCoordinates);
        assert(M_d != currentView->depthImage->eyeCoordinates->fromGlobal);
        assert(&M_d);
        currentView->ChangePose(M_d); // TODO crashes in Release mode -- maybe relying on some uninitialized variable, or accessing while not cudaDeviceSynchronized
    }


}
using namespace tracking;































CPU_FUNCTION(
    void
    ,DepthToUchar4
    ,(ITMUChar4Image *dst, const ITMFloatImage *src)
    ,"Utility. Simple visualization of float image, considered a depth image, by color. c.f. DepthToUchar4.png for the color gradient used"
    , PURITY_OUTPUT_POINTERS)
{
    assert(src);
    assert(dst);
    assert(dst->noDims == src->noDims);
    Vector4u *dest = dst->GetData(MEMORYDEVICE_CPU);
    const float *source = src->GetData(MEMORYDEVICE_CPU);
    int dataSize = static_cast<int>(dst->dataSize);
    assert(dataSize > 1);
    memset(dst->GetData(MEMORYDEVICE_CPU), 0, dataSize * 4);

    Vector4u *destUC4;
    float lims[2], scale;

    destUC4 = (Vector4u*)dest;
    lims[0] = 100000.0f; lims[1] = -100000.0f;

    for (int idx = 0; idx < dataSize; idx++)
    {
        float sourceVal = source[idx]; // only depths greater than 0 are considered
        if (sourceVal > 0.0f) { lims[0] = MIN(lims[0], sourceVal); lims[1] = MAX(lims[1], sourceVal); }
    }

    scale = ((lims[1] - lims[0]) != 0) ? 1.0f / (lims[1] - lims[0]) : 1.0f / lims[1];

    if (lims[0] == lims[1])
        return;// assert(false);

    for (int idx = 0; idx < dataSize; idx++)
    {
        float sourceVal = source[idx];

        if (sourceVal > 0.0f)
        {
            sourceVal = (sourceVal - lims[0]) * scale;


            auto interpolate = [&](float val, float y0, float x0, float y1, float x1) {
                return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
            };

            auto base = [&](float val) {
                if (val <= -0.75f) return 0.0f;
                else if (val <= -0.25f) return interpolate(val, 0.0f, -0.75f, 1.0f, -0.25f);
                else if (val <= 0.25f) return 1.0f;
                else if (val <= 0.75f) return interpolate(val, 1.0f, 0.25f, 0.0f, 0.75f);
                else return 0.0f;
            };

            destUC4[idx].r = (uchar)(base(sourceVal - 0.5f) * 255.0f);
            destUC4[idx].g = (uchar)(base(sourceVal) * 255.0f);
            destUC4[idx].b = (uchar)(base(sourceVal + 0.5f) * 255.0f);
            destUC4[idx].a = 255;
        }
    }
}









































/// c.f. chapter "Lighting Estimation with Signed Distance Fields"
namespace Lighting {

    struct LightingModel {
        static const int b2 = 9;

        /// \f[a \sum_{m = 1}^{b^2} l_m H_m(n)\f]
        /// \f$v\f$ is some voxel (inside the truncation band)
        // TODO wouldn't the non-refined voxels interfere with the updated, refined voxels, 
        // if we just cut them off hard from the computation?
        CPU_MEMBERFUNCTION(float,getReflectedIrradiance,(float albedo, //!< \f$a(v)\f$
            Vector3f normal //!< \f$n(v)\f$
            ) const, "", PURITY_PURE){
            assert(albedo >= 0);
            float o = 0;
            for (int m = 0; m < b2; m++) {
                o += l[m] * sphericalHarmonicHi(m, normal);
            }
            return albedo * o;
        }

        // original paper uses svd to compute the solution to the linear system, but how this is done should not matter
        LightingModel(std::array<float, b2>& l) : l(l){
            assert(l[0] > 0); // constant term should be positive - otherwise the lighting will be negative in some places (?)
        }
        LightingModel(const LightingModel& m) : l(m.l){}

        static FUNCTION(float,sphericalHarmonicHi,(const int i, const Vector3f& n),"", PURITY_PURE) {
            assert(i >= 0 && i < b2);
            switch (i) {
            case 0: return 1.f;
            case 1: return n.y;
            case 2: return n.z;
            case 3: return n.x;
            case 4: return n.x * n.y;
            case 5: return n.y * n.z;
            case 6: return -n.x * n.x - n.y * n.y + 2.f * n.z * n.z;
            case 7: return n.z * n.x;
            case 8: return n.x - n.y * n.y;

            default: fatalError("sphericalHarmonicHi not defined for i = %d", i);
            }
            return 0.f;
        }


        const std::array<float, b2> l;

        OSTREAM(LightingModel) {
            for (auto & x : o.l) os << x << ", ";
            return os;
        }
    };

    /// for constructAndSolve
    struct ConstructLightingModelEquationRow {
        // Amount of columns, should be small
        static const unsigned int m = LightingModel::b2;

        /*not really needed */
        struct ExtraData {
            // User specified payload to be summed up alongside:
            uint count;

            // Empty constructor must generate neutral element
            CPU_AND_GPU ExtraData() : count(0) {}

            static GPU_ONLY ExtraData add(const ExtraData& l, const ExtraData& r) {
                ExtraData o;
                o.count = l.count + r.count;
                return o;
            }
            static GPU_ONLY void atomicAdd(DEVICEPTR(ExtraData&) result, const ExtraData& integrand) {
                ::atomicAdd(&result.count, integrand.count);
            }
        };

        // This operates on half-blocks:
        /// should be executed with (blockIdx.x/2) == valid localVBA index (0 ignored) 
        /// annd blockIdx.y,z from 0 to 1 (parts of one block)
        /// and threadIdx <==> voxel localPos / 2..
        /// Thus, SDF_BLOCK_SIZE must be even
        static GPU_ONLY bool generate(const uint i, VectorX<float, m>& out_ai, float& out_bi/*[1]*/, ExtraData& extra_count /*not really needed */) {
            static_assert(0 == SDF_BLOCK_SIZE % 2, "SDF_BLOCK_SIZE must be even");

            const uint blockSequenceId = blockIdx.x / 2;
            if (blockSequenceId == 0) return false; // unused
            assert(blockSequenceId < SDF_LOCAL_BLOCK_NUM);

            assert(blockSequenceId < Scene::getCurrentScene()->voxelBlockHash->getLowestFreeSequenceNumber());

            assert(threadIdx.x < SDF_BLOCK_SIZE / 2 &&
                threadIdx.y < SDF_BLOCK_SIZE / 2 &&
                threadIdx.z < SDF_BLOCK_SIZE / 2);

            assert(blockIdx.y <= 1);
            assert(blockIdx.z <= 1);
            // voxel position
            const Vector3i localPos = Vector3i(threadIdx_xyz) + Vector3i(blockIdx.x % 2, blockIdx.y % 2, blockIdx.z % 2) * SDF_BLOCK_SIZE / 2;

            assert(localPos.x >= 0 &&
                localPos.y >= 0 &&
                localPos.z >= 0);
            assert(localPos.x < SDF_BLOCK_SIZE  &&
                localPos.y < SDF_BLOCK_SIZE &&
                localPos.z < SDF_BLOCK_SIZE);

            ITMVoxelBlock const * const voxelBlock = Scene::getCurrentScene()->getVoxelBlockForSequenceNumber(blockSequenceId);

            const ITMVoxel* const voxel = voxelBlock->getVoxel(localPos);
            const Vector3i globalPos = (voxelBlock->pos.toInt() * SDF_BLOCK_SIZE + localPos);
            /*
            const Vector3i globalPos = vb->pos.toInt() * SDF_BLOCK_SIZE;

            const THREADPTR(Point) & voxel_pt_world =  Point(
            CoordinateSystem::global(),
            (globalPos.toFloat() + localPos.toFloat()) * voxelSize
            ));

            .toFloat();
            Vector3f worldPos = CoordinateSystems::global()->convert(globalPos);
            */
            const float worldSpaceDistanceToSurface = abs(voxel->getSDF() * mu);
            assert(worldSpaceDistanceToSurface <= mu);

            // Is this voxel within the truncation band? Otherwise discard this term (consider it unreliable for lighting estimation)
            if (worldSpaceDistanceToSurface > t_shell) return false;

            // also fail if we cannot compute the normal
            bool found = true;
            const UnitVector normal = computeSingleNormalFromSDFByForwardDifference(globalPos, found);
            if (!found) return false;

            assert(abs(length(normal) - 1) < 0.01, "invalid normal n = (%f %f %f), |n| = %f", xyz(normal), length(normal));

            // i-th (voxel-th) row of A shall contain H_{0..b^2-1}(n(v))
            DO(j, LightingModel::b2) {
                out_ai[j] = LightingModel::sphericalHarmonicHi(j, normal);
            }

            // corresponding entry of b is I(v) / a(v)
            out_bi = voxel->intensity() / voxel->luminanceAlbedo;

            assert(out_bi >= 0); // && out_bi <= 1);

            // Set extra data
            // TODO not really needed
            extra_count.count = 1;
            return true;
        }

    };

    // todo should we really discard the existing lighting model the next time? maybe we could use it as an initialization
    // when solving
    LightingModel estimateLightingModel() {
        assert(Scene::getCurrentScene());
        // Maximum number of entries

        const int validBlockNum = Scene::getCurrentScene()->voxelBlockHash->getLowestFreeSequenceNumber();

        const auto gridDim = dim3(validBlockNum * 2, 2, 2);
        const auto blockDim = dim3(SDF_BLOCK_SIZE / 2, SDF_BLOCK_SIZE / 2, SDF_BLOCK_SIZE / 2); // cannot use full SDF_BLOCK_SIZE: too much shared data (in reduction) -- TODO could just use the naive reduction algorithm. Could halve the memory use even now with being smart.

        const int n = validBlockNum * SDF_BLOCK_SIZE3; // maximum number of entries: total amount of currently allocated voxels (unlikely)
        assert(n == volume(gridDim) * volume(blockDim));

        ConstructLightingModelEquationRow::ExtraData extra_count;
        const auto l_harmonicCoefficients = LeastSquares::constructAndSolve<ConstructLightingModelEquationRow>(n, gridDim, blockDim, extra_count);
        assert(extra_count.count > 0 && extra_count.count <= n); // sanity check
        assert(l_harmonicCoefficients.size() == LightingModel::b2);

        // VectorX to std::array
        std::array<float, LightingModel::b2> l_harmonicCoefficients_a;
        DO(i, LightingModel::b2)
            l_harmonicCoefficients_a[i] = assertFinite(l_harmonicCoefficients[i]);

        LightingModel lightingModel(l_harmonicCoefficients_a);
        return lightingModel;
    }


    // Computing artificial lighting

    template<typename F>
    struct ComputeLighting {
        doForEachAllocatedVoxel_process() {
            // skip voxels without computable normal
            bool found = true;
            const UnitVector normal = computeSingleNormalFromSDFByForwardDifference(globalPos, found);
            if (!found) return;

            v->clr = F::operate(normal);

            v->w_color = 1;
        }
    };

    // Direction towards directional light source
    static __managed__ Vector3f lightNormal;
    struct DirectionalArtificialLighting {
        static GPU_ONLY Vector3u operate(UnitVector normal) {
            const float cos = MAX(0.f, dot(normal, lightNormal));
            return Vector3u(cos * 255, cos * 255, cos * 255);
        }
    };

    // compute voxel color according to given functor f from normal
    // c(v) := F::operate(n(v))
    // 'lighting shader baking' (lightmapping)
    template<typename F>
    void computeArtificialLighting() {
        cudaDeviceSynchronize();
        assert(Scene::getCurrentScene());

        Scene::getCurrentScene()->doForEachAllocatedVoxel<ComputeLighting<F>>();

        cudaDeviceSynchronize();
    }
}






























































namespace meshing {

    static const CPU_AND_GPU_CONSTANT int edgeTable[256] = {0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a,
        0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66,
        0x86a, 0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca,
        0x3c6, 0x2cf, 0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256,
        0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
        0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596,
        0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};

    // edge number 0 to 11, or -1 for unused
    static const CPU_AND_GPU_CONSTANT int triangleTable[256][16] = {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1}, {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1}, {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1}, {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1}, {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1}, {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1}, {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1}, {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1}, {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1}, {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1}, {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1}, {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1}, {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1}, {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1}, {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1}, {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1}, {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1}, {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1}, {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1}, {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1}, {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1}, {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1}, {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1}, {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1}, {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1}, {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1}, {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1}, {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1}, {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1}, {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1}, {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1}, {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1}, {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1}, {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1}, {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1}, {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1}, {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1}, {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1}, {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1}, {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1}, {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1}, {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1}, {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1}, {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1}, {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1}, {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1}, {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1}, {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1}, {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1}, {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1}, {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1}, {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1}, {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1}, {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1}, {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1}, {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1}, {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1}, {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1}, {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1}, {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1}, {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1}, {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1}, {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1}, {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1}, {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1}, {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1}, {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1}, {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1}, {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1}, {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1}, {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1}, {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1}, {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1}, {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1}, {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1}, {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1}, {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1}, {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1}, {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1}, {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1}, {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1}, {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1}, {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1}, {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1}, {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1}, {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1}, {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1}, {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1}, {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1}, {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1}, {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1}, {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1}, {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1}, {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1}, {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1}, {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};


    struct Vertex {
        Vector3f p, // voxel position in voxel-world-coordinates (integral)
            c // color from 0 to 255.f, converted later (when writing the obj) to 0-1
            ;
    };
    struct Triangle { Vertex v[3]; };

    __managed__ float shaderParam; // for all current shaders, this says which level-set to extract

    // shaders (Shader) for computing sdf and color value
    struct mesh_doriginal_c {
        static GPU_ONLY void getSDFAndC(_In_ const ITMVoxel* v, _Out_ float& sdf, _Out_ Vector3f& c) {
            c = v->clr.toFloat();
            sdf = v->getSDF() - shaderParam;
        }
    };

    struct mesh_d_a {
        static GPU_ONLY void getSDFAndC(_In_ const ITMVoxel* v, _Out_ float& sdf, _Out_ Vector3f& c) {
            c = Vector3f(v->luminanceAlbedo * 255.f /* needs to have high range now */);
            sdf = v->refinedDistance - shaderParam;
        }
    };

    template<typename Shader> // has to be threaded through till here
    GPU_ONLY void get(
        const Vector3i voxelLocation, // global voxel coordinate
        bool& isFound,
        float& sdf,
        Vertex& vert) {

        assert(isFound);
        auto v = Scene::getCurrentSceneVoxel(voxelLocation);
        if (!v) {
            isFound = false;
            return;
        }

        vert.p = voxelLocation.toFloat();

        // 'shader' specifying which data from the current scene is read to drive coloring and sdf (level-set) of output
        // this can be fully parametrized to allow visualizing any (derived) voxel data in the mesh
        /*vert.c = v->clr.toFloat();
        sdf = v->getSDF();*/
        Shader::getSDFAndC(v, /* -> */ sdf, vert.c);
    }

    template<typename Shader>
    GPU_ONLY inline bool findPointNeighbors(
        /*out*/Vertex v[8], /*out*/ float *sdf,
        const Vector3i globalPos)
    {
        bool isFound;

#define access(dx, dy, dz, id) \
        isFound = true; get<Shader>(globalPos + Vector3i(dx, dy, dz), isFound, sdf[id], v[id]);\
        if (!isFound || sdf[id] == 1.0f) return false;

        access(0, 0, 0, 0);
        access(1, 0, 0, 1);
        access(1, 1, 0, 2);
        access(0, 1, 0, 3);
        access(0, 0, 1, 4);
        access(1, 0, 1, 5);
        access(1, 1, 1, 6);
        access(0, 1, 1, 7);
#undef access
        return true;
    }

    GPU_ONLY inline Vector3f sdfInterp(const Vector3f &p1, const Vector3f &p2, float valp1, float valp2)
    {
        if (fabs(0.0f - valp1) < 0.00001f) return p1;
        if (fabs(0.0f - valp2) < 0.00001f) return p2;
        if (fabs(valp1 - valp2) < 0.00001f) return p1;

        return p1 + ((0.0f - valp1) / (valp2 - valp1)) * (p2 - p1);
    }

    GPU_ONLY inline void sdfInterp(Vertex &out, const Vertex &p1, const Vertex &p2, float valp1, float valp2)
    {
        out.p = sdfInterp(p1.p, p2.p, valp1, valp2);
        out.c = sdfInterp(p1.c, p2.c, valp1, valp2);
    }

    template<typename Shader>
    GPU_ONLY inline int buildVertList(Vertex finalVertexList[12], Vector3i globalPos)
    {
        Vertex vertices[8];  float sdfVals[8];

        if (!findPointNeighbors<Shader>(vertices, sdfVals, globalPos)) return -1;

        int cubeIndex = 0;
        if (sdfVals[0] < 0) cubeIndex |= 1; if (sdfVals[1] < 0) cubeIndex |= 2;
        if (sdfVals[2] < 0) cubeIndex |= 4; if (sdfVals[3] < 0) cubeIndex |= 8;
        if (sdfVals[4] < 0) cubeIndex |= 16; if (sdfVals[5] < 0) cubeIndex |= 32;
        if (sdfVals[6] < 0) cubeIndex |= 64; if (sdfVals[7] < 0) cubeIndex |= 128;

        if (edgeTable[cubeIndex] == 0) return -1;


#define access(id, mask, a, b) \
        if (edgeTable[cubeIndex] & mask) sdfInterp(finalVertexList[id], vertices[a], vertices[b], sdfVals[a], sdfVals[b]);

        access(0, 1, 0, 1);
        access(1, 2, 1, 2);
        access(2, 4, 2, 3);
        access(3, 8, 3, 0);
        access(4, 16, 4, 5);
        access(5, 32, 5, 6);
        access(6, 64, 6, 7);
        access(7, 128, 7, 4);
        access(8, 256, 0, 4);
        access(9, 512, 1, 5);
        access(10, 1024, 2, 6);
        access(11, 2048, 3, 7);

#undef access

        return cubeIndex;
    }

    const uint noMaxTriangles = 10 * 1000 * 1000;//SDF_LOCAL_BLOCK_NUM * 32; // heuristic ?! // if anything, consider allocated blocks
    // and max triangles per marching cube and assume moderate occupation of blocks (like, half)

    __managed__ Triangle *triangles;
    __managed__ unsigned int noTriangles;

    template<typename Shader>
    struct MeshVoxel {
        doForEachAllocatedVoxel_process() {
            Vertex finalVertexList[12];
            int cubeIndex = buildVertList<Shader>(finalVertexList, globalPos);

            if (cubeIndex < 0) return;

            for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
            {
                int triangleId = atomicAdd(&noTriangles, 1);

                if (triangleId < noMaxTriangles - 1)
                {
                    for (int k = 0; k < 3; k++) {
                        triangles[triangleId].v[k] = finalVertexList[triangleTable[cubeIndex][i + k]];
                        //assert(triangles[triangleId].c[k].x >= 0.f && triangles[triangleId].c[k].x <= 255.001);
                    }
                }
            }
        }
    };

    void MeshScene(string baseFileName, Scene */*const * const <- not supported in currentScene framework */ scene = Scene::getCurrentScene(), string shader = "mesh_doriginal_c", float shaderParam = 0.f)
    {
        CURRENT_SCENE_SCOPE(scene);

        auto_ptr<MemoryBlock<Triangle>> triangles(new MemoryBlock<Triangle>(noMaxTriangles));

        meshing::shaderParam = shaderParam;
        meshing::noTriangles = 0;
        meshing::triangles = triangles->GetData(MEMORYDEVICE_CUDA);

#define isShader(s) if (shader == #s) { Scene::getCurrentScene()->doForEachAllocatedVoxel<MeshVoxel<s>>(); }
        
        isShader(mesh_doriginal_c)
        else isShader(mesh_d_a)
        else fatalError("unkown shader %s", shader.c_str());
#undef isShader
        //Scene::getCurrentScene()->doForEachAllocatedVoxel<MeshVoxel>();


        cudaDeviceSynchronize();
        assert(noTriangles, "there should have been some triangles generated (cannot mesh empty scenes) but there where none");
        assert(noMaxTriangles);
        assert(noTriangles < noMaxTriangles);


        // Write
        cout << "writing file " << baseFileName << endl;
        Triangle *triangleArray = triangles->GetData();

        FILE *f = fopen((baseFileName + ".obj").c_str(), "wb");
        assert(f, "could not open file");

        int j = 1;
        for (uint i = 0; i < noTriangles; i++)
        {
            // Walk through vertex list in reverse for correct orientation (is the table flipped?)
            for (int k = 2; k >= 0; k--) {
                Vector3f p = triangleArray[i].v[k].p * voxelSize; // coordinates where voxel coordinates
                Vector3f c = triangleArray[i].v[k].c / 255.0f; // colors in obj are 0 to 1

                fprintf(f, "v %f %f %f %f %f %f\n", xyz(p), xyz(c));

                assert(c.x >= 0.f && c.x <= 1.001, "%f", c.x); // color sanity check
            }

            fprintf(f, "f %d %d %d\n", j, j + 1, j + 2);
            j += 3;
        }

        fclose(f);
    }

} // meshing namespace
using namespace meshing;





























// primitive uniform random number
float random01f() {
    return rand()*1.f / RAND_MAX;
}

Vector3f random01v3() {
    return Vector3f(random01f(), random01f(), random01f());
}













// Dumping lists of points
template<typename T>
void writePoints(const Vector3f& color01rgb, vector<Vector3<T>> const & points, FILE* f)
{
    assert(points.size() > 0);
    assert(color01rgb.r >= 0.f && color01rgb.r <= 1.f);

    for (auto& p : points) 
        fprintf(f, "v %f %f %f %f %f %f\n", xyz(p.toFloat()), xyz(color01rgb));
}


template<typename T>
void writePointsRandomColor(vector<Vector3<T>> const & points, FILE* f)
{
    writePoints(random01v3(), points, f);
}

template<typename T>
void writePointsRandomColor(const string baseFileName, vector<vector<Vector3<T>>> const & pointsLists) {
    assert(pointsLists.size() > 0);
    FILE *f = fopen((baseFileName + ".obj").c_str(), "wb");
    assert(f);

    for (auto& x : pointsLists)
        writePointsRandomColor(x, f);

    fclose(f);
}

template<typename T>
void writePointsRandomColor(const string baseFileName, vector<Vector3<T>> const & pointsLists) {
    vector<vector<Vector3<T>>> pl = {pointsLists};
    writePointsRandomColor(baseFileName, pl);
}

TEST(testWritePoints) {
    vector<Vector3i> v = {Vector3i(0, 0, 0)};
    writePointsRandomColor("test.points", v);
}















// Dumping voxel positions

// adds SDF_BLOCK_SIZE3 points
void getVoxelPositions(_Inout_ vector<Vector3i>& p, ITMVoxelBlock const * const vb) {
    XYZ_over_SDF_BLOCK_SIZE {
        Vector3i localPos(x, y, z);
        p.push_back(vb->getPos().toInt()  * SDF_BLOCK_SIZE + localPos);
    }
}    

vector<Vector3i> getVoxelPositions(Scene const* const scene) {
    vector<Vector3i> p;
    p.reserve(scene->countVoxelBlocks() * SDF_BLOCK_SIZE3); // optimization hint
    DO1(i, scene->countVoxelBlocks()) {
        getVoxelPositions(p, scene->getVoxelBlockForSequenceNumber(i));
    }
    assert(p.size() > 0);
    return p;
}

vector<vector<Vector3i>> getVoxelPositionsBlockwise(Scene const* const scene) {

    printf("getVoxelPositionsBlockwise start\n");
    vector<vector<Vector3i>> ps;
    ps.reserve(scene->countVoxelBlocks()); // optimization hint
    DO1(i, scene->countVoxelBlocks()) {
        vector<Vector3i> p;
        p.reserve(SDF_BLOCK_SIZE3);// optimization hint
        getVoxelPositions(p, scene->getVoxelBlockForSequenceNumber(i));
        assert(p.size() > 0);
        ps.push_back(p);
    }
    assert(ps.size() > 0);
    printf("getVoxelPositionsBlockwise end\n");
    return ps;
}

void dumpVoxelPositions(Scene const* const scene, const string baseFileName) {
    writePointsRandomColor(baseFileName, getVoxelPositions(scene));
}


void dumpVoxelPositionsBlockwise(Scene const* const scene, const string baseFileName) {
    writePointsRandomColor(baseFileName, getVoxelPositionsBlockwise(scene));
}



















// returns empty list if any offset block point does not have full 1-neighborhood
// TODO optimization would test whether neighborhood exists only for outer points? (but couldn't there be interior holes? probably not
// by how the blocks are arranged
CPU_FUNCTION(
    vector<Vector3i>, getOptimizationBlock,
    (Scene const*  /* const*/ scene, ITMVoxelBlock const * const vb, Vector3i const offset), "") {
    vector<Vector3i> p;
    p.reserve(SDF_BLOCK_SIZE3);// optimization hint

    // --
    // as long as the blocks are as big as the data-structure blocks, it is sufficient to check the inner and outer corners
    // if the blocks are smaller than the datastructure blocks (SDF_BLOCK_SIZE), it would be enough even to check the outer corners 

    // check
    Vector3i globalPos = vb->getPos().toInt()  * SDF_BLOCK_SIZE + offset;
#define corner(x,y,z)       (globalPos + Vector3i(x,y,z) * SDF_BLOCK_SIZE)
#define outer_corner(x,y,z) (corner(x,y,z) + Vector3i(-1+2*x,-1+2*y,-1+2*z))

    if (
        !scene->voxelExistsQ(corner(0, 0, 0)) ||
        !scene->voxelExistsQ(corner(0, 1, 0)) ||
        !scene->voxelExistsQ(corner(1, 0, 0)) ||
        !scene->voxelExistsQ(corner(1, 1, 0)) ||
        !scene->voxelExistsQ(corner(0, 0, 1)) ||
        !scene->voxelExistsQ(corner(0, 1, 1)) ||
        !scene->voxelExistsQ(corner(1, 0, 1)) ||
        !scene->voxelExistsQ(corner(1, 1, 1)) ||
        
        !scene->voxelExistsQ(outer_corner(0, 0, 0)) ||
        !scene->voxelExistsQ(outer_corner(0, 1, 0)) ||
        !scene->voxelExistsQ(outer_corner(1, 0, 0)) ||
        !scene->voxelExistsQ(outer_corner(1, 1, 0)) ||
        !scene->voxelExistsQ(outer_corner(0, 0, 1)) ||
        !scene->voxelExistsQ(outer_corner(0, 1, 1)) ||
        !scene->voxelExistsQ(outer_corner(1, 0, 1)) ||
        !scene->voxelExistsQ(outer_corner(1, 1, 1))
        ) 
        return vector<Vector3i>();

#undef corner
#undef outer_corner

    // enter


    XYZ_over_SDF_BLOCK_SIZE {
            const Vector3i localPos(x, y, z);
            Vector3i globalPos = vb->getPos().toInt()  * SDF_BLOCK_SIZE + localPos;
            globalPos += offset;
                
            p.push_back(globalPos);
    }

    // or
    /*
    for (int z = -1; z < SDF_BLOCK_SIZE+1; z++)
        for (int y = -1; y < SDF_BLOCK_SIZE+1; y++)
            for (int x = -1; x < SDF_BLOCK_SIZE+1; x++) {

            const Vector3i localPos(x, y, z);
            Vector3i globalPos = vb->getPos().toInt()  * SDF_BLOCK_SIZE + localPos;
            globalPos += offset;

            if (!scene->voxelExistsQ(globalPos)) return vector<Vector3i>();

            // push only inner positions
            if (x >= 0 && x < SDF_BLOCK_SIZE &&
                y >= 0 && y < SDF_BLOCK_SIZE &&
                z >= 0 && z < SDF_BLOCK_SIZE)
                    p.push_back(globalPos);
    }
    */
    // or
    /*
    XYZ_over_SDF_BLOCK_SIZE {
        const Vector3i localPos(x, y, z);
        Vector3i globalPos = vb->getPos().toInt()  * SDF_BLOCK_SIZE + localPos;
        globalPos += offset;

        if (!scene->voxel1NeighborhoodExistsQ(globalPos)) return vector<Vector3i>();

        p.push_back(globalPos);
    }*/
    // --

    return p;
}

// Finding the blocks (ps) to be optimized individually
CPU_FUNCTION(
    vector<vector<Vector3i>>, getOptimizationBlocks, (Scene const* const scene, Vector3i const offset), "") {

    printf("getOptimizationBlocks %d %d %d takes a while\n", xyz(offset));

    assert(abs(offset.x) < SDF_BLOCK_SIZE);
    assert(abs(offset.y) < SDF_BLOCK_SIZE);
    assert(abs(offset.z) < SDF_BLOCK_SIZE);

    vector<vector<Vector3i>> ps;
    ps.reserve(scene->countVoxelBlocks()); // optimization hint -- note that there might be less blocks in the end

    DO1(i, scene->countVoxelBlocks()) {

        auto p = getOptimizationBlock(scene, scene->getVoxelBlockForSequenceNumber(i), offset);

        if (p.size() > 0) {
            assert(p.size() == SDF_BLOCK_SIZE3);
            ps.push_back(p);
        }
        //else
         //   printf("cannot optimize %d %d %d\n", xyz(scene->getVoxelBlockForSequenceNumber(i)->getPos()));
    }

    printf("can optimize %d of %d blocks\n", ps.size(), scene->countVoxelBlocks());
    assert(ps.size() > 0); // there should be SOMETHING to optimize over
    return ps;
}



TEST(getOptimizationBlocks1) {
    Scene* scene = new Scene();

    // exactly one block should be optimizable here
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 0, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 0, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 1, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 1, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 0, 1));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 0, 1));
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 1, 1));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 1, 1));

    auto blocks = getOptimizationBlocks(scene, Vector3i(1,1,1));
    assert(blocks.size() == 1);
    assert(blocks[0][0] == Vector3i(1, 1, 1));

    delete scene;
}
































namespace resample {
    __managed__ Scene* coarse, *fine;

    struct InterpolateCoarseToFine {
        float resultSDF;
        Vector3f resultColor;
        bool& isFound;

        GPU_ONLY InterpolateCoarseToFine(bool& isFound) : 
            resultSDF(0), resultColor(0, 0, 0), // must start with 0 since we add things up to compute a weighted average
            isFound(isFound) {}

        GPU_ONLY void operator()(const Vector3i globalPos, const float lerpCoeff) {
            assert(lerpCoeff >= 0 && lerpCoeff <= 1, "%f", lerpCoeff);

            const auto v = readVoxel(globalPos, isFound, coarse); // IMPORTANT must read from coarse scene here
            assert(v.getSDF() >= -1 && v.getSDF() <= 1, "%f", v.getSDF());
            resultSDF += lerpCoeff * v.getSDF();
            resultColor += lerpCoeff * v.clr.toFloat();
        }
    };

    // call with current scene == fine and fine and coarse pointers initialized correctly
    struct CoarseToFine {
        doForEachAllocatedVoxel_process() {
            assert(Scene::getCurrentScene() == fine);
            assert(coarse != Scene::getCurrentScene());
            assert(coarse);
            assert(fine);

            // how much bigger is the coarse voxel Size?
            const float factor = coarse->getVoxelSize() / fine->getVoxelSize(); // optimization would do this once

            // read and interpolate coarse
            assert(globalPoint.coordinateSystem == CoordinateSystem::global());
            const auto coarseVoxelCoord = coarse->voxelCoordinates_->convert(globalPoint);

            const Vector3f coarsePoint = coarseVoxelCoord.location;

            bool isFound = true;
            InterpolateCoarseToFine interpolator(isFound);
            forEachBoundingVoxel(coarsePoint, interpolator);

            if (!isFound) interpolator.resultSDF = 1;
            assert(interpolator.resultSDF >= -1.0001 && interpolator.resultSDF <= 1.001, "%f", interpolator.resultSDF);

            // Implementation: Since voxels store a normalized SDF value, it has to be rescaled to the fine voxelSize beforehand.

            // rescale SDF
            // to world-distance
            const float coarseMu = voxelSize_to_mu(coarse->getVoxelSize());
            float sdf = interpolator.resultSDF * coarseMu; // multiply by normalization factor
            // to fine
            sdf /= mu; // voxelSize_to_mu(Scene::getCurrentScene()->getVoxelSize); // renormalize to fine's mu

            assert(abs(sdf) <= factor*1.0001); // this will result in values that are at most factor too big

            sdf = CLAMP(sdf, -1.f, 1.f); // truncate again

            // set fine
            v->setSDF(sdf);
            v->w_depth = 1; // we don't care about the amount of samples anymore after integration anyways
            v->clr = (interpolator.resultColor).toUChar();//isFound ? Vector3u(0,255,0) : Vector3u(255,0,0);// 
            v->w_color = 1;
        }
    };

    template<typename T>
    void initFineFromCoarseGen(Scene* fine_, Scene* coarse_) {
        assert(fine_ != coarse_);
        coarse = coarse_;
        fine = fine_;

        assert(fine, "%p", fine); assert(coarse, "%p", coarse);
        assert(coarse->getVoxelSize() > fine->getVoxelSize());
        CURRENT_SCENE_SCOPE(fine);

        fine->doForEachAllocatedVoxel<T>();
    }

    /*
    UPSAMPLE

    Initialize allocated voxels of *fine* by interpolating
    the values (doriginal, c) defined for the corresponding coarse voxels.
    */
    void initFineFromCoarse(Scene* fine_, Scene* coarse_) {
        initFineFromCoarseGen<CoarseToFine>(fine_, coarse_);
    }


    // -- upsampling A and D (TODO: this is basically the same as the above, make more abstract to support interpolating any attributes)
    struct InterpolateCoarseToFineAD {
        float resultRefinedSDF;
        float resultA;

        bool& isFound;

        GPU_ONLY InterpolateCoarseToFineAD(bool& isFound) : 
            resultRefinedSDF(0), resultA(0), // must start with 0 since we add things up to compute a weighted average
            isFound(isFound) {}

        GPU_ONLY void operator()(const Vector3i globalPos, const float lerpCoeff) {
            assert(lerpCoeff >= 0 && lerpCoeff <= 1, "%f", lerpCoeff);

            const auto v = readVoxel(globalPos, isFound, coarse); // IMPORTANT must read from coarse scene here

            // TODO this is only true if we initialize ad first, even though we will reinitialize them just now?
            // nah, they just have to be initialized for the mesh we are copying from
            //assert(v.refinedDistance >= -1.f && v.refinedDistance <= 1.f, "%f", v.refinedDistance); // TODO make sure copying the result of the optimization back this is enforced
            //assert(v.luminanceAlbedo >= 0.f && v.luminanceAlbedo <= 1.f, "%f", v.luminanceAlbedo); // TODO albedo should also be in 0-1, but our optimization has no such constraint // TODO but when copying the result back we should enforce it
            // but we will clamp the lighting model to this

            resultRefinedSDF += lerpCoeff * v.refinedDistance;
            resultA          += lerpCoeff * v.luminanceAlbedo;
        }
    };

    // call with current scene == fine and fine and coarse pointers initialized correctly
    struct CoarseToFineAD {
        doForEachAllocatedVoxel_process() {
            assert(Scene::getCurrentScene() == fine);
            assert(coarse != Scene::getCurrentScene());

            // how much bigger is the coarse voxel Size?
            const float factor = coarse->getVoxelSize() / fine->getVoxelSize(); // optimization would do this once

            // read and interpolate coarse
            //assert(globalPoint.coordinateSystem == CoordinateSystem::global()); // TODO one of the assertions causes this to hang (=== CUDA exception that is not reported) -- hanging is acceptable undefined behaviour...
            const auto coarseVoxelCoord = coarse->voxelCoordinates_->convert(globalPoint);

            const Vector3f coarsePoint = coarseVoxelCoord.location;

            bool isFound = true;
            InterpolateCoarseToFineAD interpolator(isFound);
            forEachBoundingVoxel(coarsePoint, interpolator);

            if (!isFound) interpolator.resultRefinedSDF = v->refinedDistance;
            //assert(interpolator.resultRefinedSDF >= -1.0001 && interpolator.resultRefinedSDF <= 1.001, "%f", interpolator.resultRefinedSDF);

            // rescale SDF
            // to world-distance
            const float coarseMu = voxelSize_to_mu(coarse->getVoxelSize());
            float rsdf = interpolator.resultRefinedSDF * coarseMu; // multiply by normalization factor
            // to fine
            rsdf /= mu; // voxelSize_to_mu(Scene::getCurrentScene()->getVoxelSize); // renormalize to fine's mu

            //assert(abs(rsdf) <= factor*1.0001); // this will result in values that are at most factor too big

            rsdf = CLAMP(rsdf, -1.f, 1.f); // truncate again

            // set fine
            v->refinedDistance = rsdf;
            v->luminanceAlbedo = CLAMP(interpolator.resultA, 0.f, 1.f); // interpolator.resultA; // interpolator might not work perfectly everywhere, so better clamp this too // TODO we didn't have to do this on color -- because it is of limited range anyways/by design
        }
    };

    /*
    UPSAMPLE

    Initialize allocated voxels of *fine* by interpolating
    the values (d, a) defined for the corresponding coarse voxels.

    Since some a and d might not be initializable, call initAD first anyways to ensure later meshability!
    */
    void initFineADFromCoarseAD(Scene* fine_, Scene* coarse_) {
        printf("initFineADFromCoarseAD begins\n");
        initFineFromCoarseGen<CoarseToFineAD>(fine_, coarse_);
        printf("initFineADFromCoarseAD ends\n");
    }
}















struct InitAD {
    doForEachAllocatedVoxel_process() {
        v->refinedDistance = v->getSDF();
        v->luminanceAlbedo = 1.f;
    }
};

void initAD(Scene* scene) {
    assert(scene);
    CURRENT_SCENE_SCOPE(scene);

    scene->doForEachAllocatedVoxel<InitAD>();
    cudaDeviceSynchronize();
}




















// Converting scene data to named x-vector (data linearization), suitable for passing to SOP-framework
// TODO very efficient code would skip the conversion process
// an intermediate speedup would be to do it in parallel
struct XVarName;
template <>
struct hash<XVarName>;

// Set of all possible variable names
class XVarName {
public:
    MEMBERFUNCTION(,XVarName,(),"") {} // default constructor, so that we can fill an array of these later

    MEMBERFUNCTION(bool,operator==,(_In_ const XVarName& o) const, "", PURITY_PURE) {
        return key == o.key;
    }

    static FUNCTION(XVarName, d, (Vector3i globalPos), "", PURITY_PURE) {
        return Vector4i(xyz(globalPos), 0);
    }

    static FUNCTION(XVarName, a, (Vector3i globalPos), "", PURITY_PURE) {
        return Vector4i(xyz(globalPos), 1);
    }

    static FUNCTION(XVarName, doriginal, (Vector3i globalPos), "", PURITY_PURE) {
        return Vector4i(xyz(globalPos), 2);
    }

    static FUNCTION(XVarName, c, (Vector3i globalPos, unsigned int channel), "", PURITY_PURE) {
        assert(channel < 3);
        return Vector4i(xyz(globalPos), 3 + channel);
    }

    static FUNCTION(XVarName, eg, (), "", PURITY_PURE)  { return Vector4i(0, 0, 0, 10); }
    static FUNCTION(XVarName, er, (), "", PURITY_PURE)  { return Vector4i(0, 0, 0, 11); }
    static FUNCTION(XVarName, es, (), "", PURITY_PURE)  { return Vector4i(0, 0, 0, 12); }
    static FUNCTION(XVarName, ea, (), "", PURITY_PURE)  { return Vector4i(0, 0, 0, 13); }
    static FUNCTION(XVarName, l, (unsigned int i), "", PURITY_PURE)  { return Vector4i(0, 0, 0, 14 + i); }

private:

    MEMBERFUNCTION(,XVarName,(Vector4i v),"") : key(v)  {}

    friend hash<XVarName>;
    Vector4i key;
};

template <>
struct hash<XVarName> // TODO make HashMap take this kind of standard hasher, maybe implement unordered_map interface
{
    size_t operator()(const XVarName& k) const
    {
        // adapted from z3 hasher
        // TODO check performance of this (unordered_map collision factor...)
        return 
            (((uint)k.key.x * 73856093u) ^ ((uint)k.key.y * 19349669u) ^ ((uint)k.key.z * 83492791u)) ^ (hash<int>()(k.key.w) << 1);
    }
};

TEST(XVarName1) {
    auto x1 = XVarName::d(Vector3i(1, 1, 1));
    auto x2 = XVarName::d(Vector3i(1, 1, 2));

    assert(x1 == x1);
    assert(!(x1 == x2));

    hash<XVarName> h;
    assert(h(x1) != h(x2));

    unordered_map<XVarName, int> m;
    m[x1] = 0;
    m[x2] = 1;
    assert(0 == m[x1]);
    assert(1 == m[x2]);

    auto x3 = XVarName::doriginal(Vector3i(1, 1, 1));
    assert(!(x1 == x3));
    assert(!(x2 == x3));
}


void enterVoxelData(_Inout_ unordered_map<XVarName, float>& x, VoxelBlockPos const vbp, Vector3i localPos, ITMVoxel const * const v) {
    Vector3i pos = vbp.toInt() * SDF_BLOCK_SIZE + localPos; // globalPos

    // convert to naming used by the Mathematica research
    // TODO unify this


    x[XVarName::d(pos)] = v->refinedDistance;

    // TODO unless the conditions enforce this (which they cannot)
    // we cannot either -- must make SDF encoder aware of out-of-range data (clamp)

    //assert(x[XVarName::d(pos)] >= -1.001f, "%f", x[XVarName::d(pos)]);
    //assert(x[XVarName::d(pos)] <= 1.001f, "%f", x[XVarName::d(pos)]);



    x[XVarName::doriginal(pos)] = v->getSDF();
    //assert(x[XVarName::doriginal(pos)] >= -1.001f, "%f", x[XVarName::doriginal(pos)]);
    //assert(x[XVarName::doriginal(pos)] <= 1.001f, "%f", x[XVarName::doriginal(pos)]);

    x[XVarName::a(pos)] = v->luminanceAlbedo;
    //assert(x[XVarName::a(pos)] >= 0.f, "%f", x[XVarName::a(pos)]);
    //assert(x[XVarName::a(pos)] <= 1.f, "%f", x[XVarName::a(pos)]);

    DO(i, 3) {
        x[XVarName::c(pos, i)] = v->clr[i] / 255.f;
        //assert(x[XVarName::c(pos, i)] >= 0.f, "%f", x[XVarName::c(pos, i)]);
        //assert(x[XVarName::c(pos, i)] <= 1.f, "%f", x[XVarName::c(pos, i)]);
    }
}



void enterVoxelBlockData(_Inout_ unordered_map<XVarName, float>& xm, ITMVoxelBlock const * const vb) {
    XYZ_over_SDF_BLOCK_SIZE {
        Vector3i localPos(x, y, z);
        enterVoxelData(xm, vb->getPos(), localPos, vb->getVoxel(localPos));
    }
}

// when calling this, d and a must have been initialized
unordered_map<XVarName, float> getSceneData(const Scene* const scene) {
    printf("getSceneData start\n");
    scene->localVBA.Synchronize();

    unordered_map<XVarName, float> x;
    x.reserve(scene->countVoxelBlocks() * SDF_BLOCK_SIZE3);

    DO1(i, scene->countVoxelBlocks()) {
        enterVoxelBlockData(x, scene->getVoxelBlockForSequenceNumber(i));
    }

    printf("getSceneData end\n");

    return x;
}

TEST(getSceneData1) {
    Scene* scene = new Scene();
    VoxelBlockPos p(1, 2, 3);
    scene->performVoxelBlockAllocation(p);
    auto vb = scene->getVoxelBlock(p);

    vb->resetVoxels();
    initAD(scene);

    auto data = getSceneData(scene);

    assert(SDF_BLOCK_SIZE3 * (3 + 3 /* d a doriginal 3*c*/) == data.size());
    assert(definedQ(data, XVarName::d(p.toInt() * SDF_BLOCK_SIZE)));
    assert(definedQ(data, XVarName::doriginal(p.toInt() * SDF_BLOCK_SIZE)));
    assert(1.f == data[XVarName::a(p.toInt() * SDF_BLOCK_SIZE)]);
    assert(!definedQ(data, XVarName::d(Vector3i(0,0,0))));

    delete scene;
}


// updating, de-linearization
void updateVoxelData(_In_ unordered_map<XVarName, float> const & x, VoxelBlockPos const vbp, const Vector3i localPos, ITMVoxel * const v) {
    Vector3i pos = vbp.toInt() * SDF_BLOCK_SIZE + localPos; // globalPos

    // convert to naming used by the Mathematica research
    // TODO unify this


    // TODO unless the conditions enforce this (which they cannot)
    // we cannot either -- must make SDF encoder aware of out-of-range data (clamp)

    //assert(x[XVarName::d(pos)] >= -1.001f, "%f", x[XVarName::d(pos)]);
    //assert(x[XVarName::d(pos)] <= 1.001f, "%f", x[XVarName::d(pos)]);


    v->refinedDistance = CLAMP( x.at(XVarName::d(pos)), -1.f, 1.f); // TODO is this ok?
    //assert(x.at(XVarName::d(pos)) >= -1.001f, "%f", v->refinedDistance);
    //assert(x.at(XVarName::d(pos)) <= 1.001f, "%f", v->refinedDistance);

    v->setSDF(CLAMP(x.at(XVarName::doriginal(pos)),-1.f,1.f));
    //assert(x.at(XVarName::doriginal(pos)) >= -1.001f, "%f", x.at//(XVarName::doriginal(pos)));
    //assert(x.at(XVarName::doriginal(pos)) <= 1.001f, "%f", x.at(XVarName::doriginal(pos)));

    v->luminanceAlbedo = CLAMP(x.at(XVarName::a(pos)),0.f,1.f);
    //assert(x.at(XVarName::a(pos)) >= 0.f, "%f", x.at(XVarName::a(pos)));
   // assert(x.at(XVarName::a(pos)) <= 1.f, "%f", x.at(XVarName::a(pos)));

    DO(i, 3) {
        v->clr[i] = CLAMP(x.at(XVarName::c(pos, i)),0.f,1.f) * 255.f; // TODO we don't need to clamp this if we don't optimize over it (it stays unchanged)
        //assert(x.at(XVarName::c(pos, i)) >= 0.f, "%f", x.at(XVarName::c(pos, i)));
        //assert(x.at(XVarName::c(pos, i)) <= 1.f, "%f", x.at(XVarName::c(pos, i)));
    }
}

void updateVoxelBlockData(_In_ unordered_map<XVarName, float> const & xm, ITMVoxelBlock * const vb) {
    for (int x = 0; x < SDF_BLOCK_SIZE; x++)
        for (int y = 0; y < SDF_BLOCK_SIZE; y++)
            for (int z = 0; z < SDF_BLOCK_SIZE; z++) {
                Vector3i localPos(x, y, z);
                updateVoxelData(xm, vb->getPos(), localPos, vb->getVoxel(localPos));
            }
}

void updateFromSceneData(Scene* scene, unordered_map<XVarName, float> const & x) {

    printf("updateFromSceneData start\n");

    DO1(i, scene->countVoxelBlocks()) {
        updateVoxelBlockData(x, scene->getVoxelBlockForSequenceNumber(i));
    }

    scene->localVBA.Synchronize();

    printf("updateFromSceneData end\n");
}

TEST(updateSceneData1) {
    Scene* scene = new Scene();
    VoxelBlockPos p(1, 2, 3);
    scene->performVoxelBlockAllocation(p);
    auto vb = scene->getVoxelBlock(p);

    vb->resetVoxels();
    initAD(scene);
    
    scene->localVBA.Synchronize(); // TODO it is not clear why this is necessary to do manually here - getting voxel blocks locally should do this automatically

    assert(1.f == vb->getVoxel(Vector3i(0, 0, 0))->luminanceAlbedo, "%f", vb->getVoxel(Vector3i(0, 0, 0))->luminanceAlbedo);

    auto data = getSceneData(scene);

    auto vp = p.toInt() * SDF_BLOCK_SIZE;

    data[XVarName::a(vp)] = 0.5f;

    updateFromSceneData(scene, data);

    auto vbu = scene->getVoxelBlock(p);
    assert(vb == vbu); // still at the same place
    assert(.5f == vb->getVoxel(Vector3i(0, 0, 0))->luminanceAlbedo);

    delete scene;
}


























#include "$CFormDefines.cpp"  /* generated for problem, rarely changes */  // Required for including *working* definitions of f and df -- this defines what times(x,y) etc. mean


// select energy to optimize over
#define fSigma fSigmaSimple//  fSigmaReal /*fSigmaSimple*/

/*
Local energy function defining the refinement objective.

Mostly generated from mathematica code defining f in a generic way, c.f.
*/

struct fSigmaSimple {

    static
        MEMBERFUNCTION(void, sigma, (Vector3i p, _Out_writes_(lengthz) XVarName* sigmap), "the per-point data selector function", PURITY_OUTPUT_POINTERS) {
        DBG_UNREFERENCED_PARAMETER(p);

#define sigmap(i) ([]{static_assert(0 <= i && i < fSigmaSimple::lengthz, #i);}, sigmap[i]) /* definitions of f/df use x(i) to refer to input[], c.f. RIFunctionCForm* */

#include "simplef/sigmap.cpp"
            ;
#undef sigmap
    }

    static const unsigned int lengthz =
#include "simplef/lengthz.cpp"
        , lengthfz =
#include "simplef/lengthfz.cpp"
        ;

#define x(i) ([]{static_assert(i < fSigmaSimple::lengthz,"");}, input[i]) /* definitions of f/df use x(i) to refer to input[], c.f. RIFunctionCForm* */
#define out(i) ([]{static_assert(0 <= i && i < fSigmaSimple::lengthfz, #i);}, out[i]) 

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy", PURITY_OUTPUT_POINTERS){
#include "simplef/f.cpp"
    }
    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f", PURITY_OUTPUT_POINTERS) {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
#include "simplef/df.cpp"
    }
#undef x
};


struct fSigmaReal {

    static
        MEMBERFUNCTION(void, sigma, (Vector3i p, _Out_writes_(lengthz) XVarName* sigmap), "the per-point data selector function", PURITY_OUTPUT_POINTERS) {
        DBG_UNREFERENCED_PARAMETER(p);

        // prepare syntax used in generated sigmap, c.f. realf_prepare
#define x p.x
#define y p.y
#define z p.z
#define d XVarName::d
#define doriginal XVarName::doriginal
#define a XVarName::a
#define c XVarName::c
#define l XVarName::l

#define eg XVarName::eg
#define er XVarName::er
#define es XVarName::es
#define ea XVarName::ea

#define sigmap(i) ([]{static_assert(0 <= i && i < fSigmaReal::lengthz, #i);}, sigmap[i]) /* definitions of f/df use x(i) to refer to input[], c.f. RIFunctionCForm* */


#include "realf/sigmap.cpp"
        ;

#undef sigmap
#undef x
#undef y
#undef z
#undef d
#undef doriginal
#undef a
#undef c
#undef l
#undef eg 
#undef er 
#undef es 
#undef ea 

    }

    static const unsigned int lengthz =
#include "realf/lengthz.cpp"
        , lengthfz =
#include "realf/lengthfz.cpp"
        ;


#define x(i) ([]{static_assert(0 <= i && i < fSigmaReal::lengthz, #i);}, input[i]) /* definitions of f/df use x(i) to refer to input[], c.f. RIFunctionCForm* */
#define out(i) ([]{static_assert(0 <= i && i < fSigmaReal::lengthfz, #i);}, out[i]) 

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy", PURITY_OUTPUT_POINTERS){
#include "realf/f.cpp"
    }
    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f", PURITY_OUTPUT_POINTERS) {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
#include "realf/df.cpp"
    }
#undef x
};

#undef x

// Computes x, points and ys as needed for the SOP describing the optimization
// Output lists should initially be empty.
CPU_FUNCTION(void,refinement_x_points_ys,(
    _In_ Scene const * const scene,
    // energy term weights // TODO make more generic
    _In_ const float eg, _In_ const float er, _In_ const float es, _In_ const float ea,
    // lighting model parameters
    _In_ const const float* const l, _In_ const unsigned int lSize,
    
    _Out_ unordered_map<XVarName, float>& x,
    _Out_ vector<vector<Vector3i>>& points,
    _Out_ vector<vector<XVarName>>& ys
    
    ), PURITY_OUTPUT_POINTERS) {
    printf("refinement_x_points_ys start\n");
    assert(0 == x.size());
    assert(0 == points.size());
    assert(0 == ys.size());

    // 1. translate infinitam data to x vector
    assert(lSize < 100); // sanity check
    assert(eg >= 0); // sanity check
    assert(er >= 0); // sanity check
    assert(es >= 0); // sanity check
    assert(ea >= 0); // sanity check

    printf("refineScene start\n");
    x = getSceneData(scene);
    assert(x.size() > 0);

    x[XVarName::eg()] = eg;
    x[XVarName::er()] = er;
    x[XVarName::es()] = es;
    x[XVarName::ea()] = ea;

    DO(i, lSize) x[XVarName::l(i)] = l[i];

    printf("scene data stats: size %d, load_factor %f\n", x.size(), x.load_factor());

    // 2. determine points at which the energy can be computed, decomposed into independent blocks
    //points = getVoxelPositionsBlockwise(scene);// TODO should use getOptimizationBlocks for real energy, which accesses the whole 1-neighborhood

    points = getOptimizationBlocks(scene, Vector3i(1,1,1));// TODO should let the user supply the offset

    assert(points.size() > 0, "no points where found over which we can optimize!");

    // 3. determine variables over which we optimize (y)

    // TODO externalize to a function
    // optimizing over a and d at each point where the energy is computed


    printf("building y start\n");
    for (const auto& block : points) {
        vector<XVarName> y;
        for (const auto& p : block) {
            y.push_back(XVarName::a(p));
            y.push_back(XVarName::d(p));
        }
        ys.push_back(y);
    }
    printf("building y end\n");

    assert(ys.size() > 0);
    assert(ys.size() == points.size());
    printf("refinement_x_points_ys end\n");
}

// d and a must have been initialized (sanity checks for their range, even if not used in optimization)
CPU_FUNCTION(
void ,refineScene,(Scene* scene, 
    // energy term weights // TODO make more generic
    float eg, float er, float es, float ea,
    // lighting model parameters
    const float* const l, unsigned int lSize
    ), "") {
    printf("refineScene start\n");
    unordered_map<XVarName, float> x;
    vector<vector<Vector3i>> points;
    vector<vector<XVarName>> ys;
    refinement_x_points_ys(scene, eg, er, es, ea, l, lSize,
        x,points, ys);

    // 4. solve, enhance x using fSigma
    SOPDNamed<XVarName, Vector3i, fSigma> sopd(x, points, ys);
    sopd.solve();

    // 5. Convert back the result
    updateFromSceneData(scene, x);
    printf("refineScene end\n");
}

CPU_FUNCTION(
float,
sceneEnergy,(Scene* scene,
    // energy term weights // TODO make more generic
    float eg, float er, float es, float ea,
    // lighting model parameters
    const float* const l, unsigned int lSize
    ),"") {
    printf("sceneEnergy start\n");
    unordered_map<XVarName, float> x;
    vector<vector<Vector3i>> points;
    vector<vector<XVarName>> ys;
    refinement_x_points_ys(scene, eg, er, es, ea, l, lSize,
    x, points, ys);

    SOPDNamed<XVarName, Vector3i, fSigma> sopd(x, points, ys);

    float energy = sopd.energy();
    printf("sceneEnergy = %f\n", energy);
    return energy;
}

TEST(refineScene1) {
    Scene* scene = new Scene();

    // at least one block should be optimizable here
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 0, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 0, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 1, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 1, 0));
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 0, 1));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 0, 1));
    scene->performVoxelBlockAllocation(VoxelBlockPos(0, 1, 1));
    scene->performVoxelBlockAllocation(VoxelBlockPos(1, 1, 1));

    initAD(scene);

    const unsigned int exampleLSize = 3;
    const float exampleL[] = {1., 1., 1.};
    const float eg = 1.f, er = 1.f, es = 1.f, ea = 1.f;

    const float e0 = sceneEnergy(scene, eg, er, es, ea, exampleL, exampleLSize);
    assert(assertFinite(e0) >= 0.f);

    refineScene(scene, eg, er, es, ea, exampleL, exampleLSize);

    const float e1 = sceneEnergy(scene, eg, er, es, ea, exampleL, exampleLSize);
    assert(e1 <= /* <, but no reduction for some possible energies */ e0, "%f %f", e1, e0);
    assert(0.f <= e1 );
    delete scene;
}




























// InfiniTAMScene[id] - management
vector<Scene*> scenes;
int addScene(Scene* s) {
    assert(s);
    scenes.push_back(s);
    return scenes.size() - 1;
}
Scene* getScene(int id) {
    assert(id >= 0 && id < scenes.size());
    auto s = scenes[id];
    assert(s, "%p, %d", s, id);
    return s;
}

KERNEL gpuFalse() {
    assert(false);
}


namespace WSTP {
#define getTensor(TYPE, WLTYPE, expectDepth) int* dims; TYPE* a; int depth; char** heads; WSGet ## WLTYPE ## Array(stdlink, &a, &dims, &heads, &depth); assert(depth == expectDepth); 

#define releaseTensor(TYPE) WSRelease ## TYPE ## Array(stdlink, a, dims, heads, depth);


    void putImageRGBA8(ITMUChar4Image* i) {
        int dims[] = {i->noDims.height, i->noDims.width, 4};
        const char* heads[] = {"List", "List", "List"};
        WSPutInteger8Array(stdlink,
            (unsigned char*)i->GetData(),
            dims, heads, 3);
    }

    void putImageFloat(ITMFloatImage* i) {
        int dims[] = {i->noDims.height, i->noDims.width};
        const char* heads[] = {"List", "List"};
        WSPutReal32Array(stdlink,
            (float*)i->GetData(),
            dims, heads, 2);
    }

    void putImageFloat4(ITMFloat4Image* i) {
        int dims[] = {i->noDims.height, i->noDims.width, 4};
        const char* heads[] = {"List", "List", "List"};
        WSPutReal32Array(stdlink,
            (float*)i->GetData(),
            dims, heads, 3);
    }

    // putFloatList rather
    void putFloatArray(const float* a, const int n) {
        int dims[] = {n};
        const char* heads[] = {"List"};
        WSPutReal32Array(stdlink, a, dims, heads, 1); // use List function
    }

    void putUnorm(unsigned char c) {
        WSPutReal(stdlink, 1. * c / UCHAR_MAX);
    }

    unsigned char getUnormUC() {
        double d;  WSGetDouble(stdlink, &d);
        return d * UCHAR_MAX;
    }

    void putColor(Vector3u c) {
        WSPutFunction(stdlink, "List", 3);
        putUnorm(c.r);
        putUnorm(c.g);
        putUnorm(c.b);
    }

    Vector3u getColor() {
        long args; WSCheckFunction(stdlink, "List", &args);
        assert(args == 3);

        unsigned char r = getUnormUC(), g = getUnormUC(), b = getUnormUC();
        return Vector3u(r, g, b);
    }

    // {normalizedSDF_?SNormQ, sdfSampleCount_Integer?NonNegative, color : {r_?UNormQ, g_?UNormQ, b_?UNormQ}, colorSampleCount_Integer?NonNegative}
    void putVoxel(const ITMVoxel& v) {
        WSPutFunction(stdlink, "List", 4);

        WSPutReal(stdlink, v.getSDF());
        WSPutInteger(stdlink, v.w_depth);
        putColor(v.clr);
        WSPutInteger(stdlink, v.w_color);
    }

#include <sal.h>
    void getVoxel(_Out_ ITMVoxel& v) {
        long args;
        WSCheckFunction(stdlink, "List", &args);
        assert(args == 4);

        float f; WSGetFloat(stdlink, &f); v.setSDF(f);
        WSGetInteger8(stdlink, &v.w_depth);
        v.clr = getColor();
        WSGetInteger8(stdlink, &v.w_color);
    }


    // {{x_Integer,y_Integer,z_Integer}, {__Voxel}}
    void putVoxelBlock(ITMVoxelBlock& v) {
        WSPutFunction(stdlink, "List", 2);

        WSPutInteger16List(stdlink, (short*)&v.pos_, 3); // TODO are these packed correctly?

        WSPutFunction(stdlink, "List", SDF_BLOCK_SIZE3);
        /*for (int i = 0; i < SDF_BLOCK_SIZE3; i++)
        putVoxel(v.blockVoxels[i]);*/
        // change ordering such that {x,y,z} indices in mathematica correspond to xyz here:
        // make z vary fastest
        XYZ_over_SDF_BLOCK_SIZE_xyz_order
            putVoxel(*v.getVoxel(Vector3i(x, y, z)));

    }

    // receives {{x_Integer,y_Integer,z_Integer}, {__Voxel}}
    void getVoxelBlock(_Out_ ITMVoxelBlock& v) {
        long args;
        WSCheckFunction(stdlink, "List", &args);
        assert(args == 2);

        int count; short* ppos; WSGetInteger16List(stdlink, &ppos, &count);
        assert(count == 3);
        v.reinit(VoxelBlockPos(ppos));
        assert(v.getPos() != INVALID_VOXEL_BLOCK_POS);
        WSReleaseInteger16List(stdlink, ppos, count);

        WSCheckFunction(stdlink, "List", &args);
        assert(args == SDF_BLOCK_SIZE3);

        /*
        for (int i = 0; i < SDF_BLOCK_SIZE3; i++)
        getVoxel(v.blockVoxels[i]);
        */
        // change ordering such that {x,y,z} indices in mathematica correspond to xyz here:
        // make z vary fastest
        XYZ_over_SDF_BLOCK_SIZE_xyz_order
            getVoxel(*v.getVoxel(Vector3i(x, y, z)));
    }

    Matrix4f getMatrix4f()
    {
        getTensor(double, Real64, 2);
        assert(dims[0] == 4); assert(dims[1] == 4);

        // read row-major matrix
        Matrix4f m;
        for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) m(x, y) = a[y * 4 + x];

        releaseTensor(Real64);
        return m;
    }

    Vector4f getVector4f()
    {
        getTensor(float, Real32, 1);
        assert(dims[0] == 4);
        Vector4f m(a);
        releaseTensor(Real32);
        return m;
    }

    template<int X>
    VectorX<float, X> getVectorXf()
    {
        getTensor(float, Real32, 1);
        assert(dims[0] == X);
        VectorX<float, X> m(a);
        releaseTensor(Real32);
        return m;
    }



    void putMatrix4f(Matrix4f m) {
        int dims[] = {4, 4};
        const char* heads[] = {"List", "List"};

        // write row-major matrix
        float a[4 * 4];
        for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) a[y * 4 + x] = m(x, y);

        WSPutReal32Array(stdlink, a, dims, heads, 2);
    }
}
using namespace WSTP;





extern "C" {

    void assertFalse() {
        assert(false);
        WL_RETURN_VOID();
    }
    void assertGPUFalse() {
        LAUNCH_KERNEL(gpuFalse, 1, 1);
        WL_RETURN_VOID();
    }

    int createScene(double voxelSize_) {
        return addScene(new Scene(voxelSize_));
    }

    double getSceneVoxelSize(int id) {
        return getScene(id)->getVoxelSize();
    }

    int countVoxelBlocks(int id) {
        return getScene(id)->countVoxelBlocks();
    }

    // Manually insert a *new* voxel block
    // TODO what should happen when it already exists?
    // format: c.f. getVoxelBlock
    void putVoxelBlock(int id) {
        auto s = getScene(id);

        int vbCountBefore = s->countVoxelBlocks();

        ITMVoxelBlock receivedVb;
        getVoxelBlock(receivedVb);
        assert(receivedVb.getPos() != INVALID_VOXEL_BLOCK_POS);

        auto& sceneVb = *s->getVoxelBlockForSequenceNumber(s->performVoxelBlockAllocation(receivedVb.getPos()));
        assert(sceneVb.getPos() == receivedVb.getPos());
        sceneVb = receivedVb; // copy

        // synchronize (otherwise, gpu code hangs) - TODO why exactly?
        s->voxelBlockHash->naKey.Synchronize();
        s->voxelBlockHash->needsAllocation.Synchronize();
        s->voxelBlockHash->hashMap_then_excessList.Synchronize();
        s->localVBA.Synchronize();

        assert(vbCountBefore + 1 == s->countVoxelBlocks());
        cudaDeviceSynchronize();
        WL_RETURN_VOID();
    }

    // {__VoxelBlock} at most max many. 0 or negative numbers mean all
    void getVoxelBlock(int id, int i) {
        auto s = getScene(id);
        const int n = s->voxelBlockHash->getLowestFreeSequenceNumber();
        assert(i >= 1 /* valid sequence nubmers start at 1 - TODO this knowledge should not be repeated here */ && i < n, "there is no voxelBlock with index %d, valid indices are 1 to %d", i, n - 1);
        putVoxelBlock(s->localVBA[i]);
    }

    void serializeScene(int id, char* fn) {
        getScene(id)->serialize(binopen_write(fn));
        WL_RETURN_VOID();
    }

    void deserializeScene(int id, char* fn) {
        auto s = getScene(id);
        s->deserialize(binopen_read(fn));
        WL_RETURN_VOID();
    }

    void meshScene(int id, char* fn) {
        cudaDeviceSynchronize();
        MeshScene(fn, getScene(id));
        WL_RETURN_VOID();
    }

    void meshSceneWithShader(int id, char* fn, char* shader, double shaderParam) {
        cudaDeviceSynchronize();
        MeshScene(fn, getScene(id), shader, (float)shaderParam);
        WL_RETURN_VOID();
    }

    // init fine doriginal and c
    // TODO fix
    void initFineFromCoarse(int idFine, int idCoarse) {
        assert(idFine != idCoarse);
        cudaDeviceSynchronize();

        resample::initFineFromCoarse(getScene(idFine), getScene(idCoarse));

        WL_RETURN_VOID();
    }

    // init fine d(refined) and a by supersampling coarse
    void initFineADFromCoarseAD(int idFine, int idCoarse) {
        assert(idFine != idCoarse);

        cudaDeviceSynchronize();

        resample::initFineADFromCoarseAD(getScene(idFine), getScene(idCoarse));

        WL_RETURN_VOID();
    }

    // init d from doriginal and a = 1
    // TODO all these int parameters should be unsigned int
    void initAD(int id) {
        assert(id >= 0);
        initAD(getScene(id));
        WL_RETURN_VOID();
    }

    void computeArtificialLighting(int id, double* dir, long n) {
        assert(n == 3);

        using namespace Lighting;
        lightNormal = Vector3f(comp012(dir)).normalised();
        assert(abs(length(lightNormal) - 1) < 0.1);
        CURRENT_SCENE_SCOPE(getScene(id));
        computeArtificialLighting<DirectionalArtificialLighting>();
        WL_RETURN_VOID();
    }

    void estimateLighting(int id) {

        using namespace Lighting;
        CURRENT_SCENE_SCOPE(getScene(id));
        LightingModel l = estimateLightingModel();

        putFloatArray(l.l.data(), l.l.size());
        WL_RETURN_VOID();
    }

    void buildSphereScene(int id, double rad) {
        CURRENT_SCENE_SCOPE(getScene(id));
        TestScene::buildSphereScene(rad);
        WL_RETURN_VOID();
    }


    ITMView* render(ITMPose pose, string shader, const ITMIntrinsics intrin) {
        assert(intrin.imageSize().area(), "calibration must be loaded");
        assert(Scene::getCurrentScene());
        Vector2i sz = intrin.imageSize();
        auto outputImage = new ITMUChar4Image(sz);
        auto outputDepthImage = new ITMFloatImage(sz);

        return RenderImage(pose, intrin, shader);
    }

    ITMIntrinsics getIntrinsics() {

        ITMIntrinsics intrin;
        auto i = getVectorXf<6>();
        intrin.fx = i[0];
        intrin.fy = i[1];
        intrin.px = i[2];
        intrin.py = i[3];
        intrin.sizeX = i[4];
        intrin.sizeY = i[5];
        return intrin;
    }

    void renderScene(int id, char* shader) {
        CURRENT_SCENE_SCOPE(getScene(id));

        ITMPose p(getMatrix4f());

        auto intrin = getIntrinsics();

        auto v = render(p, shader, intrin);

        // Output: {rgbImage, depthImageData}
        WSPutFunction(stdlink, "List", 2);
        // rgb(a)
        {
            putImageRGBA8(v->colorImage->image);
        }

        // depth
        {
            putImageFloat(v->depthImage->image);
        }
    }

    int sceneExistsQ(int id) {
        return id >= 0 && id < scenes.size();
    }

    void refineScene(int id,
        // energy term weights // TODO make more generic
        double eg, double er, double es, double ea,
        // lighting model parameters
        double* l, long lSize
        ) {
        assert(lSize >= 1);
        auto lf = new float[lSize];
        DO(i, lSize) lf[i] = l[i];

        refineScene(getScene(id), eg, er, es, ea, lf, lSize);

        delete[] lf;


        WL_RETURN_VOID(); // don't forget
    }

    double sceneEnergy(int id,
        // energy term weights // TODO make more generic
        double eg, double er, double es, double ea,
        // lighting model parameters
        double* l, long lSize
        ) {
        assert(lSize >= 1);
        auto lf = new float[lSize];
        DO(i, lSize) lf[i] = l[i];

        float y = sceneEnergy(getScene(id), eg, er, es, ea, lf, lSize);

        delete[] lf;

        return y;
    }

    void processFrame(int doTracking, int id) {
        // rgbaByteImage
        Vector2i rgbSz;
        ITMUChar4Image* rgbaImage;
        {
            // rgbaByteImage_ /;TensorQ[rgbaByteImage, IntegerQ] &&Last@Dimensions@rgbaByteImage == 4
            getTensor(unsigned char, Integer8, 3);
            assert(dims[0] > 1); assert(dims[1] > 1); assert(dims[2] == 4);

            rgbSz = Vector2i(dims[1], dims[0]);
            assert(rgbSz.width > rgbSz.height);

            rgbaImage = new ITMUChar4Image(rgbSz);
            rgbaImage->SetFrom((char*)a, rgbSz.area());

            releaseTensor(Integer8);
        }

        //depthImage_
        Vector2i depthSz;
        ITMFloatImage* depthImage;
        {
            // depthData_?NumericMatrixQ
            getTensor(float, Real32, 2);
            assert(dims[0] > 1); assert(dims[1] > 1);
            depthSz = Vector2i(dims[1], dims[0]);
            assert(depthSz.width > depthSz.height);

            depthImage = new ITMFloatImage(depthSz);
            depthImage->SetFrom((char*)a, depthSz.area());

            releaseTensor(Real32);
        }
        assert(depthSz.area() <= rgbSz.area());

        // poseWorldToView_?poseMatrixQ
        auto poseWorldToView = getMatrix4f();

        ITMRGBDCalib calib;
        // intrinsicsRgb : NamelessIntrinsicsPattern[]
        calib.intrinsics_rgb = getIntrinsics();
        assert(calib.intrinsics_rgb.imageSize() == rgbSz);
        // intrinsicsD : NamelessIntrinsicsPattern[]
        calib.intrinsics_d = getIntrinsics();
        assert(calib.intrinsics_d.imageSize() == depthSz);

        // rgbToDepth_?poseMatrixQ
        calib.trafo_rgb_to_depth.SetFrom(getMatrix4f());

        CURRENT_SCENE_SCOPE(getScene(id));

        // Finally
        assert(rgbaImage->noDims.area() > 1);
        assert(depthImage->noDims.area() > 1);

        cudaDeviceSynchronize();
        if (currentView) delete currentView;
        currentView = new ITMView(calib);

        currentView->ChangeImages(rgbaImage, depthImage);
        currentView->ChangePose(poseWorldToView);

        if (doTracking) {
            Matrix4f old_M_d = currentView->depthImage->eyeCoordinates->fromGlobal;
            assert(old_M_d == currentView->depthImage->eyeCoordinates->fromGlobal);
            ImprovePose();
            assert(old_M_d != currentView->depthImage->eyeCoordinates->fromGlobal);

            cudaDeviceSynchronize();
            // [
            WSPutFunction(stdlink, "List", 3);
            putImageFloat4(rendering::raycastResult->image);
            putImageFloat4(tracking::lastFrameICPMap->image);
            putImageFloat4(tracking::lastFrameICPMap->normalImage);
            // ]
        }
        cudaDeviceSynchronize();

        Fuse();

        //if (doTracking)
        // ;// putMatrix4f(currentView->depthImage->eyeCoordinates->fromGlobal);
        //else 
        WL_RETURN_VOID(); // only changes state, no immediate result
    }

    // dumpSceneVoxelPositions[id_Integer ? NonNegative, fn_String]
    void dumpSceneVoxelPositions(int id, const char* const fn) {
        dumpVoxelPositions(getScene(id), fn);
        WL_RETURN_VOID();
    }

    void dumpSceneVoxelPositionsBlockwise(int id, const char* const fn) {
        dumpVoxelPositionsBlockwise(getScene(id), fn);
        WL_RETURN_VOID();
    }
    
    // dumpSceneOptimizationBlocks[id_Integer ? NonNegative, fn_String, {offsetx_Integer, offsety_Integer, offsett_Integer}]
    void dumpSceneOptimizationBlocks(int id, const char* const fn, int ox, int oy, int oz) {
        assert(abs(ox) < SDF_BLOCK_SIZE);
        assert(abs(oy) < SDF_BLOCK_SIZE);
        assert(abs(oz) < SDF_BLOCK_SIZE);

        vector<vector<Vector3i>> optimizationBlocks = getOptimizationBlocks(getScene(id), Vector3i(ox, oy, oz));
        writePointsRandomColor(fn, optimizationBlocks);

        WL_RETURN_VOID();
    }
}

void preWsMain() {
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF); // catch malloc errors // TODO this prints a report when the program gracefully terminates. Anything else?
    // TODO I think this program still leaks lots of (cuda and cpu) memory, e.g. images

    _controlfp_s(NULL,
        0, // By default, the run-time libraries mask all floating-point exceptions. (11111...). We set to 0 (unmask) the following:
        _EM_OVERFLOW | _EM_ZERODIVIDE | _EM_INVALID
        ); // enable all floating point exceptions' trapping behaviour except for the following exceptions: denormal-result (can not be changed by _controlfp_s anyways), underflow and inexact which we tolerate


    prepareCUDAHeap();
}

void preWsExit() {
    runTests();
}