

#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h> // skipped

// We need to #include "bli_type_defs.h" in order to pull in the
// definition of f77_int. But in order to #include that header, we
// also need to pull in the headers that precede it in blis.h.
// begin bli_system.h


#ifndef BLIS_SYSTEM_H
#define BLIS_SYSTEM_H

// NOTE: If not yet defined, we define _POSIX_C_SOURCE to make sure that
// various parts of POSIX are defined and made available.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h> // skipped
#include <stdlib.h> // skipped
#include <math.h> // skipped
#include <string.h> // skipped
#include <stdarg.h> // skipped
#include <float.h> // skipped
#include <errno.h> // skipped
#include <ctype.h> // skipped

// Determine the compiler (hopefully) and define conveniently named macros
// accordingly.
#if   defined(__ICC) || defined(__INTEL_COMPILER)
  #define BLIS_ICC
#elif defined(__clang__)
  #define BLIS_CLANG
#elif defined(__GNUC__)
  #define BLIS_GCC
#endif

// Determine if we are on a 64-bit or 32-bit architecture.
#if defined(_M_X64) || defined(__x86_64) || defined(__aarch64__) || \
    defined(_ARCH_PPC64)
  #define BLIS_ARCH_64
#else
  #define BLIS_ARCH_32
#endif

// Determine the target operating system.
#if defined(_WIN32) || defined(__CYGWIN__)
  #define BLIS_OS_WINDOWS 1
#elif defined(__gnu_hurd__)
  #define BLIS_OS_GNU 1
#elif defined(__APPLE__) || defined(__MACH__)
  #define BLIS_OS_OSX 1
#elif defined(__ANDROID__)
  #define BLIS_OS_ANDROID 1
#elif defined(__linux__)
  #define BLIS_OS_LINUX 1
#elif defined(__bgq__)
  #define BLIS_OS_BGQ 1
#elif defined(__bg__)
  #define BLIS_OS_BGP 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__bsdi__) || defined(__DragonFly__) || \
      defined(__FreeBSD_kernel__) || defined(__HAIKU__)
  #define BLIS_OS_BSD 1
#elif defined(EMSCRIPTEN)
  #define BLIS_OS_EMSCRIPTEN
#else
  #error "Cannot determine operating system"
#endif

// A few changes that may be necessary in Windows environments.
#if BLIS_OS_WINDOWS

  // Include Windows header file.
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
#include <windows.h> // skipped

  #if !defined(__clang__) && !defined(__GNUC__)
    // Undefine attribute specifiers in Windows.
    #define __attribute__(x)

    // Undefine restrict.
    #define restrict
  #endif

#endif

// time.h provides clock_gettime().
#if BLIS_OS_WINDOWS
#include <time.h> // skipped
#elif BLIS_OS_OSX
#include <mach/mach_time.h> // skipped
#else
  //#include <sys/time.h>

#include <time.h> // skipped
#endif

// POSIX threads are unconditionally required, regardless of whether
// multithreading is enabled via pthreads or OpenMP (or disabled).
// If pthreads is not available (Windows), then fake it.
//#include "bli_pthread_wrap.h"


#endif
// end bli_system.h
// begin bli_config.h


#ifndef BLIS_CONFIG_H
#define BLIS_CONFIG_H

// Enabled configuration "family" (config_name)
#define BLIS_FAMILY_ZEN3


// Enabled sub-configurations (config_list)
#define BLIS_CONFIG_ZEN3


// Enabled kernel sets (kernel_list)
#define BLIS_KERNELS_ZEN3
#define BLIS_KERNELS_ZEN2
#define BLIS_KERNELS_ZEN
#define BLIS_KERNELS_HASWELL


//This macro is enabled only for ZEN family configurations.
//This enables us to use different cache-blocking sizes for TRSM instead of common level-3 cache-block sizes.
#if 1
#define AOCL_BLIS_ZEN
#endif

#if 0
#define BLIS_ENABLE_OPENMP
#endif

#if 0
#define BLIS_ENABLE_PTHREADS
#endif

#if 1
#define BLIS_ENABLE_JRIR_SLAB
#endif

#if 0
#define BLIS_ENABLE_JRIR_RR
#endif

#if 1
#define BLIS_ENABLE_PBA_POOLS
#else
#define BLIS_DISABLE_PBA_POOLS
#endif

#if 1
#define BLIS_ENABLE_SBA_POOLS
#else
#define BLIS_DISABLE_SBA_POOLS
#endif

#if 0
#define BLIS_ENABLE_MEM_TRACING
#else
#define BLIS_DISABLE_MEM_TRACING
#endif

#if 0 == 64
#define BLIS_INT_TYPE_SIZE 64
#elif 0 == 32
#define BLIS_INT_TYPE_SIZE 32
#else
// determine automatically
#endif

#if 32 == 64
#define BLIS_BLAS_INT_TYPE_SIZE 64
#elif 32 == 32
#define BLIS_BLAS_INT_TYPE_SIZE 32
#else
// determine automatically
#endif

#ifndef BLIS_ENABLE_BLAS
#ifndef BLIS_DISABLE_BLAS
#if 1
#define BLIS_ENABLE_BLAS
#else
#define BLIS_DISABLE_BLAS
#endif
#endif
#endif

#ifndef BLIS_ENABLE_CBLAS
#ifndef BLIS_DISABLE_CBLAS
#if 1
#define BLIS_ENABLE_CBLAS
#else
#define BLIS_DISABLE_CBLAS
#endif
#endif
#endif

#ifndef BLIS_ENABLE_MIXED_DT
#ifndef BLIS_DISABLE_MIXED_DT
#if 1
#define BLIS_ENABLE_MIXED_DT
#else
#define BLIS_DISABLE_MIXED_DT
#endif
#endif
#endif

#ifndef BLIS_ENABLE_MIXED_DT_EXTRA_MEM
#ifndef BLIS_DISABLE_MIXED_DT_EXTRA_MEM
#if 1
#define BLIS_ENABLE_MIXED_DT_EXTRA_MEM
#else
#define BLIS_DISABLE_MIXED_DT_EXTRA_MEM
#endif
#endif
#endif

#if 1
#define BLIS_ENABLE_SUP_HANDLING
#else
#define BLIS_DISABLE_SUP_HANDLING
#endif

#if 0
#define BLIS_ENABLE_MEMKIND
#else
#define BLIS_DISABLE_MEMKIND
#endif

#if 1
#define BLIS_ENABLE_PRAGMA_OMP_SIMD
#else
#define BLIS_DISABLE_PRAGMA_OMP_SIMD
#endif

#if 0
#define BLIS_ENABLE_SANDBOX
#else
#define BLIS_DISABLE_SANDBOX
#endif

#if 1
#define BLIS_ENABLE_SHARED
#else
#define BLIS_DISABLE_SHARED
#endif


#endif
// end bli_config.h
// begin bli_config_macro_defs.h


#ifndef BLIS_CONFIG_MACRO_DEFS_H
#define BLIS_CONFIG_MACRO_DEFS_H


// -- INTEGER PROPERTIES -------------------------------------------------------

// The bit size of the integer type used to track values such as dimensions,
// strides, diagonal offsets. A value of 32 results in BLIS using 32-bit signed
// integers while 64 results in 64-bit integers. Any other value results in use
// of the C99 type "long int". Note that this ONLY affects integers used
// internally within BLIS as well as those exposed in the native BLAS-like BLIS
// interface.
#ifndef BLIS_INT_TYPE_SIZE
  #ifdef BLIS_ARCH_64
    #define BLIS_INT_TYPE_SIZE 64
  #else
    #define BLIS_INT_TYPE_SIZE 32
  #endif
#endif


// -- FLOATING-POINT PROPERTIES ------------------------------------------------

// Enable use of built-in C99 "float complex" and "double complex" types and
// associated overloaded operations and functions? Disabling results in
// scomplex and dcomplex being defined in terms of simple structs.
// NOTE: AVOID USING THIS FEATURE. IT IS PROBABLY BROKEN.
#ifdef BLIS_ENABLE_C99_COMPLEX
  // No additional definitions needed.
#else
  // Default behavior is disabled.
#endif


// -- MULTITHREADING -----------------------------------------------------------

// Enable multithreading via POSIX threads.
#ifdef BLIS_ENABLE_PTHREADS
  // No additional definitions needed.
#else
  // Default behavior is disabled.
#endif

// Enable multithreading via OpenMP.
#ifdef BLIS_ENABLE_OPENMP
  // No additional definitions needed.
#else
  // Default behavior is disabled.
#endif

// Perform a sanity check to make sure the user doesn't try to enable
// both OpenMP and pthreads.
#if defined ( BLIS_ENABLE_OPENMP ) && \
    defined ( BLIS_ENABLE_PTHREADS )
  #error "BLIS_ENABLE_OPENMP and BLIS_ENABLE_PTHREADS may not be simultaneously defined."
#endif

// Here, we define BLIS_ENABLE_MULTITHREADING if either OpenMP
// or pthreads are enabled. This macro is useful in situations when
// we want to detect use of either OpenMP or pthreads (as opposed
// to neither being used).
#if defined ( BLIS_ENABLE_OPENMP ) || \
    defined ( BLIS_ENABLE_PTHREADS )
  #define BLIS_ENABLE_MULTITHREADING
#endif


// -- MIXED DATATYPE SUPPORT ---------------------------------------------------

// Enable mixed datatype support?
#ifdef BLIS_DISABLE_MIXED_DT
  #undef BLIS_ENABLE_GEMM_MD
#else
  // Default behavior is enabled.
  #define BLIS_ENABLE_GEMM_MD
#endif

// Enable memory-intensive optimizations for mixed datatype support?
#ifdef BLIS_DISABLE_MIXED_DT_EXTRA_MEM
  #undef BLIS_ENABLE_GEMM_MD_EXTRA_MEM
#else
  // Default behavior is enabled.
  #define BLIS_ENABLE_GEMM_MD_EXTRA_MEM
#endif


// -- MISCELLANEOUS OPTIONS ----------------------------------------------------

// Do NOT require the cross-blocksize constraints. That is, do not enforce
// MC % NR = 0 and NC % MR = 0 in bli_kernel_macro_defs.h. These are ONLY
// needed when implementing trsm_r by allowing the right-hand matrix B to
// be triangular.
#ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
  #define BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
#endif

// Stay initialized after auto-initialization, unless and until the user
// explicitly calls bli_finalize().
#ifdef BLIS_DISABLE_STAY_AUTO_INITIALIZED
  #undef BLIS_ENABLE_STAY_AUTO_INITIALIZED
#else
  // Default behavior is enabled.
  #undef  BLIS_ENABLE_STAY_AUTO_INITIALIZED // In case user explicitly enabled.
  #define BLIS_ENABLE_STAY_AUTO_INITIALIZED
#endif


// -- BLAS COMPATIBILITY LAYER -------------------------------------------------

// Enable the BLAS compatibility layer?
#ifdef BLIS_DISABLE_BLAS
  #undef BLIS_ENABLE_BLAS
#else
  // Default behavior is enabled.
  #undef  BLIS_ENABLE_BLAS // In case user explicitly enabled.
  #define BLIS_ENABLE_BLAS
#endif

// The bit size of the integer type used to track values such as dimensions and
// leading dimensions (ie: column strides) within the BLAS compatibility layer.
// A value of 32 results in the compatibility layer using 32-bit signed integers
// while 64 results in 64-bit integers. Any other value results in use of the
// C99 type "long int". Note that this ONLY affects integers used within the
// BLAS compatibility layer.
#ifndef BLIS_BLAS_INT_TYPE_SIZE
  #define BLIS_BLAS_INT_TYPE_SIZE 32
#endif

// By default, the level-3 BLAS routines are implemented by directly calling
// the BLIS object API. Alternatively, they may first call the typed BLIS
// API, which will then call the object API.
//#define BLIS_BLAS3_CALLS_TAPI
#ifdef BLIS_BLAS3_CALLS_TAPI
  #undef  BLIS_BLAS3_CALLS_OAPI
#else
  // Default behavior is to call object API directly.
  #undef  BLIS_BLAS3_CALLS_OAPI // In case user explicitly enabled.
  #define BLIS_BLAS3_CALLS_OAPI
#endif


// -- CBLAS COMPATIBILITY LAYER ------------------------------------------------

// Enable the CBLAS compatibility layer?
// NOTE: Enabling CBLAS will automatically enable the BLAS compatibility layer
// regardless of whether or not it was explicitly enabled above. Furthermore,
// the CBLAS compatibility layer will use the integer type size definition
// specified above when defining the size of its own integers (regardless of
// whether the BLAS layer was enabled directly or indirectly).
#ifdef BLIS_ENABLE_CBLAS
  // No additional definitions needed.
#else
  // Default behavior is disabled.
#endif


// -- SHARED LIBRARY SYMBOL EXPORT ---------------------------------------------

// When building shared libraries, we can control which symbols are exported for
// linking by external applications. BLIS annotates all function prototypes that
// are meant to be "public" with BLIS_EXPORT_BLIS (with BLIS_EXPORT_BLAS playing
// a similar role for BLAS compatibility routines). Which symbols are exported
// is controlled by the default symbol visibility, as specifed by the gcc option
// -fvisibility=[default|hidden]. The default for this option is 'default', or,
// "public", which, if allowed to stand, causes all symbols in BLIS to be
// linkable from the outside. But when compiling with -fvisibility=hidden, all
// symbols start out hidden (that is, restricted only for internal use by BLIS),
// with that setting overridden only for function prototypes or variable
// declarations that are annotated with BLIS_EXPORT_BLIS.

#ifndef BLIS_EXPORT
  #if !defined(BLIS_ENABLE_SHARED)
    #define BLIS_EXPORT
  #else
    #if defined(_WIN32) || defined(__CYGWIN__)
      #ifdef BLIS_IS_BUILDING_LIBRARY
        #define BLIS_EXPORT __declspec(dllexport)
      #else
        #define BLIS_EXPORT __declspec(dllimport)
      #endif
    #elif defined(__GNUC__) && __GNUC__ >= 4
      #define BLIS_EXPORT __attribute__ ((visibility ("default")))
    #else
      #define BLIS_EXPORT
    #endif
  #endif
#endif

#define BLIS_EXPORT_BLIS BLIS_EXPORT
#define BLIS_EXPORT_BLAS BLIS_EXPORT


// -- STATIC INLINE FUNCTIONS --------------------------------------------------

// C and C++ have different semantics for defining "inline" functions. In C,
// the keyword phrase "static inline" accomplishes this, though the "inline"
// is optional. In C++, the "inline" keyword is required and obviates "static"
// altogether. Why does this matter? While BLIS is compiled in C99, blis.h may
// be #included by a source file that is compiled with C++.
#ifdef __cplusplus
  #define BLIS_INLINE inline
#else
  //#define BLIS_INLINE static inline
  #define BLIS_INLINE static
#endif


#endif

// end bli_config_macro_defs.h
// begin bli_type_defs.h


#ifndef BLIS_TYPE_DEFS_H
#define BLIS_TYPE_DEFS_H


//
// -- BLIS basic types ---------------------------------------------------------
//

#ifdef __cplusplus
  // For C++, include stdint.h.
#include <stdint.h> // skipped
#elif __STDC_VERSION__ >= 199901L
  // For C99 (or later), include stdint.h.
#include <stdint.h> // skipped
#include <stdbool.h> // skipped
#else
  // When stdint.h is not available, manually typedef the types we will use.
  #ifdef _WIN32
    typedef          __int32  int32_t;
    typedef unsigned __int32 uint32_t;
    typedef          __int64  int64_t;
    typedef unsigned __int64 uint64_t;
  #else
    #error "Attempting to compile on pre-C99 system without stdint.h."
  #endif
#endif

// -- General-purpose integers --

// If BLAS integers are 64 bits, mandate that BLIS integers also be 64 bits.
// NOTE: This cpp guard will only meaningfully change BLIS's behavior on
// systems where the BLIS integer size would have been automatically selected
// to be 32 bits, since explicit selection of 32 bits is prohibited at
// configure-time (and explicit or automatic selection of 64 bits is fine
// and would have had the same result).
#if BLIS_BLAS_INT_SIZE == 64
  #undef  BLIS_INT_TYPE_SIZE
  #define BLIS_INT_TYPE_SIZE 64
#endif

// Define integer types depending on what size integer was requested.
#if   BLIS_INT_TYPE_SIZE == 32
typedef           int32_t  gint_t;
typedef          uint32_t guint_t;
#elif BLIS_INT_TYPE_SIZE == 64
typedef           int64_t  gint_t;
typedef          uint64_t guint_t;
#else
typedef   signed long int  gint_t;
typedef unsigned long int guint_t;
#endif

// -- Boolean type --

// NOTE: bool_t is no longer used and has been replaced with C99's bool type.
//typedef bool bool_t;

// BLIS uses TRUE and FALSE macro constants as possible boolean values, but we
// define these macros in terms of true and false, respectively, which are
// defined by C99 in stdbool.h.
#ifndef TRUE
  #define TRUE  true
#endif

#ifndef FALSE
  #define FALSE false
#endif

// -- Special-purpose integers --

// This cpp guard provides a temporary hack to allow libflame
// interoperability with BLIS.
#ifndef _DEFINED_DIM_T
#define _DEFINED_DIM_T
typedef   gint_t dim_t;      // dimension type
#endif
typedef   gint_t inc_t;      // increment/stride type
typedef   gint_t doff_t;     // diagonal offset type
typedef  guint_t siz_t;      // byte size type
typedef uint32_t objbits_t;  // object information bit field

// -- Real types --

// Define the number of floating-point types supported, and the size of the
// largest type.
#define BLIS_NUM_FP_TYPES   4
#define BLIS_MAX_TYPE_SIZE  sizeof(dcomplex)

// There are some places where we need to use sizeof() inside of a C
// preprocessor #if conditional, and so here we define the various sizes
// for those purposes.
#define BLIS_SIZEOF_S      4  // sizeof(float)
#define BLIS_SIZEOF_D      8  // sizeof(double)
#define BLIS_SIZEOF_C      8  // sizeof(scomplex)
#define BLIS_SIZEOF_Z      16 // sizeof(dcomplex)

// -- Complex types --

#ifdef BLIS_ENABLE_C99_COMPLEX

    #if __STDC_VERSION__ >= 199901L
#include <complex.h> // skipped

        // Typedef official complex types to BLIS complex type names.
        typedef  float complex scomplex;
        typedef double complex dcomplex;
    #else
        #error "Configuration requested C99 complex types, but C99 does not appear to be supported."
    #endif

#else // ifndef BLIS_ENABLE_C99_COMPLEX

    // This cpp guard provides a temporary hack to allow libflame
    // interoperability with BLIS.
    #ifndef _DEFINED_SCOMPLEX
    #define _DEFINED_SCOMPLEX
    typedef struct
    {
        float  real;
        float  imag;
    } scomplex;
    #endif

    // This cpp guard provides a temporary hack to allow libflame
    // interoperability with BLIS.
    #ifndef _DEFINED_DCOMPLEX
    #define _DEFINED_DCOMPLEX
    typedef struct
    {
        double real;
        double imag;
    } dcomplex;
    #endif

#endif // BLIS_ENABLE_C99_COMPLEX

// -- Atom type --

// Note: atom types are used to hold "bufferless" scalar object values. Note
// that it needs to be as large as the largest possible scalar value we might
// want to hold. Thus, for now, it is a dcomplex.
typedef dcomplex atom_t;

// -- Fortran-77 types --

// Note: These types are typically only used by BLAS compatibility layer, but
// we must define them even when the compatibility layer isn't being built
// because they also occur in bli_slamch() and bli_dlamch().

// Define f77_int depending on what size of integer was requested.
#if   BLIS_BLAS_INT_TYPE_SIZE == 32
typedef int32_t   f77_int;
#elif BLIS_BLAS_INT_TYPE_SIZE == 64
typedef int64_t   f77_int;
#else
typedef long int  f77_int;
#endif

typedef char      f77_char;
typedef float     f77_float;
typedef double    f77_double;
typedef scomplex  f77_scomplex;
typedef dcomplex  f77_dcomplex;

// -- Void function pointer types --

// Note: This type should be used in any situation where the address of a
// *function* will be conveyed or stored prior to it being typecast back
// to the correct function type. It does not need to be used when conveying
// or storing the address of *data* (such as an array of float or double).

//typedef void (*void_fp)( void );
typedef void* void_fp;


//
// -- BLIS info bit field offsets ----------------------------------------------
//



// info
#define BLIS_DATATYPE_SHIFT                0
#define   BLIS_DOMAIN_SHIFT                0
#define   BLIS_PRECISION_SHIFT             1
#define BLIS_CONJTRANS_SHIFT               3
#define   BLIS_TRANS_SHIFT                 3
#define   BLIS_CONJ_SHIFT                  4
#define BLIS_UPLO_SHIFT                    5
#define   BLIS_UPPER_SHIFT                 5
#define   BLIS_DIAG_SHIFT                  6
#define   BLIS_LOWER_SHIFT                 7
#define BLIS_UNIT_DIAG_SHIFT               8
#define BLIS_INVERT_DIAG_SHIFT             9
#define BLIS_TARGET_DT_SHIFT               10
#define   BLIS_TARGET_DOMAIN_SHIFT         10
#define   BLIS_TARGET_PREC_SHIFT           11
#define BLIS_EXEC_DT_SHIFT                 13
#define   BLIS_EXEC_DOMAIN_SHIFT           13
#define   BLIS_EXEC_PREC_SHIFT             14
#define BLIS_PACK_SCHEMA_SHIFT             16
#define   BLIS_PACK_RC_SHIFT               16
#define   BLIS_PACK_PANEL_SHIFT            17
#define   BLIS_PACK_FORMAT_SHIFT           18
#define   BLIS_PACK_SHIFT                  22
#define BLIS_PACK_REV_IF_UPPER_SHIFT       23
#define BLIS_PACK_REV_IF_LOWER_SHIFT       24
#define BLIS_PACK_BUFFER_SHIFT             25
#define BLIS_STRUC_SHIFT                   27
#define BLIS_COMP_DT_SHIFT                 29
#define   BLIS_COMP_DOMAIN_SHIFT           29
#define   BLIS_COMP_PREC_SHIFT             30

// info2
#define BLIS_SCALAR_DT_SHIFT                0
#define   BLIS_SCALAR_DOMAIN_SHIFT          0
#define   BLIS_SCALAR_PREC_SHIFT            1

//
// -- BLIS info bit field masks ------------------------------------------------
//

// info
#define BLIS_DATATYPE_BITS                 ( 0x7  << BLIS_DATATYPE_SHIFT )
#define   BLIS_DOMAIN_BIT                  ( 0x1  << BLIS_DOMAIN_SHIFT )
#define   BLIS_PRECISION_BIT               ( 0x1  << BLIS_PRECISION_SHIFT )
#define BLIS_CONJTRANS_BITS                ( 0x3  << BLIS_CONJTRANS_SHIFT )
#define   BLIS_TRANS_BIT                   ( 0x1  << BLIS_TRANS_SHIFT )
#define   BLIS_CONJ_BIT                    ( 0x1  << BLIS_CONJ_SHIFT )
#define BLIS_UPLO_BITS                     ( 0x7  << BLIS_UPLO_SHIFT )
#define   BLIS_UPPER_BIT                   ( 0x1  << BLIS_UPPER_SHIFT )
#define   BLIS_DIAG_BIT                    ( 0x1  << BLIS_DIAG_SHIFT )
#define   BLIS_LOWER_BIT                   ( 0x1  << BLIS_LOWER_SHIFT )
#define BLIS_UNIT_DIAG_BIT                 ( 0x1  << BLIS_UNIT_DIAG_SHIFT )
#define BLIS_INVERT_DIAG_BIT               ( 0x1  << BLIS_INVERT_DIAG_SHIFT )
#define BLIS_TARGET_DT_BITS                ( 0x7  << BLIS_TARGET_DT_SHIFT )
#define   BLIS_TARGET_DOMAIN_BIT           ( 0x1  << BLIS_TARGET_DOMAIN_SHIFT )
#define   BLIS_TARGET_PREC_BIT             ( 0x1  << BLIS_TARGET_PREC_SHIFT )
#define BLIS_EXEC_DT_BITS                  ( 0x7  << BLIS_EXEC_DT_SHIFT )
#define   BLIS_EXEC_DOMAIN_BIT             ( 0x1  << BLIS_EXEC_DOMAIN_SHIFT )
#define   BLIS_EXEC_PREC_BIT               ( 0x1  << BLIS_EXEC_PREC_SHIFT )
#define BLIS_PACK_SCHEMA_BITS              ( 0x7F << BLIS_PACK_SCHEMA_SHIFT )
#define   BLIS_PACK_RC_BIT                 ( 0x1  << BLIS_PACK_RC_SHIFT )
#define   BLIS_PACK_PANEL_BIT              ( 0x1  << BLIS_PACK_PANEL_SHIFT )
#define   BLIS_PACK_FORMAT_BITS            ( 0xF  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_PACK_BIT                    ( 0x1  << BLIS_PACK_SHIFT )
#define BLIS_PACK_REV_IF_UPPER_BIT         ( 0x1  << BLIS_PACK_REV_IF_UPPER_SHIFT )
#define BLIS_PACK_REV_IF_LOWER_BIT         ( 0x1  << BLIS_PACK_REV_IF_LOWER_SHIFT )
#define BLIS_PACK_BUFFER_BITS              ( 0x3  << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_STRUC_BITS                    ( 0x3  << BLIS_STRUC_SHIFT )
#define BLIS_COMP_DT_BITS                  ( 0x7  << BLIS_COMP_DT_SHIFT )
#define   BLIS_COMP_DOMAIN_BIT             ( 0x1  << BLIS_COMP_DOMAIN_SHIFT )
#define   BLIS_COMP_PREC_BIT               ( 0x1  << BLIS_COMP_PREC_SHIFT )

// info2
#define BLIS_SCALAR_DT_BITS                ( 0x7  << BLIS_SCALAR_DT_SHIFT )
#define   BLIS_SCALAR_DOMAIN_BIT           ( 0x1  << BLIS_SCALAR_DOMAIN_SHIFT )
#define   BLIS_SCALAR_PREC_BIT             ( 0x1  << BLIS_SCALAR_PREC_SHIFT )


//
// -- BLIS enumerated type value definitions -----------------------------------
//

#define BLIS_BITVAL_REAL                      0x0
#define BLIS_BITVAL_COMPLEX                   BLIS_DOMAIN_BIT
#define BLIS_BITVAL_SINGLE_PREC               0x0
#define BLIS_BITVAL_DOUBLE_PREC               BLIS_PRECISION_BIT
#define   BLIS_BITVAL_FLOAT_TYPE              0x0
#define   BLIS_BITVAL_SCOMPLEX_TYPE           BLIS_DOMAIN_BIT
#define   BLIS_BITVAL_DOUBLE_TYPE             BLIS_PRECISION_BIT
#define   BLIS_BITVAL_DCOMPLEX_TYPE         ( BLIS_DOMAIN_BIT | BLIS_PRECISION_BIT )
#define   BLIS_BITVAL_INT_TYPE                0x04
#define   BLIS_BITVAL_CONST_TYPE              0x05
#define BLIS_BITVAL_NO_TRANS                  0x0
#define BLIS_BITVAL_TRANS                     BLIS_TRANS_BIT
#define BLIS_BITVAL_NO_CONJ                   0x0
#define BLIS_BITVAL_CONJ                      BLIS_CONJ_BIT
#define BLIS_BITVAL_CONJ_TRANS              ( BLIS_CONJ_BIT | BLIS_TRANS_BIT )
#define BLIS_BITVAL_ZEROS                     0x0
#define BLIS_BITVAL_UPPER                   ( BLIS_UPPER_BIT | BLIS_DIAG_BIT )
#define BLIS_BITVAL_LOWER                   ( BLIS_LOWER_BIT | BLIS_DIAG_BIT )
#define BLIS_BITVAL_DENSE                     BLIS_UPLO_BITS
#define BLIS_BITVAL_NONUNIT_DIAG              0x0
#define BLIS_BITVAL_UNIT_DIAG                 BLIS_UNIT_DIAG_BIT
#define BLIS_BITVAL_INVERT_DIAG               BLIS_INVERT_DIAG_BIT
#define BLIS_BITVAL_NOT_PACKED                0x0
#define   BLIS_BITVAL_4MI                   ( 0x1  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_3MI                   ( 0x2  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_4MS                   ( 0x3  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_3MS                   ( 0x4  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_RO                    ( 0x5  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_IO                    ( 0x6  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_RPI                   ( 0x7  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_1E                    ( 0x8  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_1R                    ( 0x9  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_PACKED_UNSPEC         ( BLIS_PACK_BIT                                                            )
#define   BLIS_BITVAL_PACKED_ROWS           ( BLIS_PACK_BIT                                                            )
#define   BLIS_BITVAL_PACKED_COLUMNS        ( BLIS_PACK_BIT                                         | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS     ( BLIS_PACK_BIT                   | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS     ( BLIS_PACK_BIT                   | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_4MI ( BLIS_PACK_BIT | BLIS_BITVAL_4MI | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_4MI ( BLIS_PACK_BIT | BLIS_BITVAL_4MI | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_3MI ( BLIS_PACK_BIT | BLIS_BITVAL_3MI | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_3MI ( BLIS_PACK_BIT | BLIS_BITVAL_3MI | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_4MS ( BLIS_PACK_BIT | BLIS_BITVAL_4MS | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_4MS ( BLIS_PACK_BIT | BLIS_BITVAL_4MS | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_3MS ( BLIS_PACK_BIT | BLIS_BITVAL_3MS | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_3MS ( BLIS_PACK_BIT | BLIS_BITVAL_3MS | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_RO  ( BLIS_PACK_BIT | BLIS_BITVAL_RO  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_RO  ( BLIS_PACK_BIT | BLIS_BITVAL_RO  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_IO  ( BLIS_PACK_BIT | BLIS_BITVAL_IO  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_IO  ( BLIS_PACK_BIT | BLIS_BITVAL_IO  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_RPI ( BLIS_PACK_BIT | BLIS_BITVAL_RPI | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_RPI ( BLIS_PACK_BIT | BLIS_BITVAL_RPI | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_1E  ( BLIS_PACK_BIT | BLIS_BITVAL_1E  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_1E  ( BLIS_PACK_BIT | BLIS_BITVAL_1E  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_1R  ( BLIS_PACK_BIT | BLIS_BITVAL_1R  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_1R  ( BLIS_PACK_BIT | BLIS_BITVAL_1R  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define BLIS_BITVAL_PACK_FWD_IF_UPPER         0x0
#define BLIS_BITVAL_PACK_REV_IF_UPPER         BLIS_PACK_REV_IF_UPPER_BIT
#define BLIS_BITVAL_PACK_FWD_IF_LOWER         0x0
#define BLIS_BITVAL_PACK_REV_IF_LOWER         BLIS_PACK_REV_IF_LOWER_BIT
#define BLIS_BITVAL_BUFFER_FOR_A_BLOCK        0x0
#define BLIS_BITVAL_BUFFER_FOR_B_PANEL      ( 0x1 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_BUFFER_FOR_C_PANEL      ( 0x2 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_BUFFER_FOR_GEN_USE      ( 0x3 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_GENERAL                   0x0
#define BLIS_BITVAL_HERMITIAN               ( 0x1 << BLIS_STRUC_SHIFT )
#define BLIS_BITVAL_SYMMETRIC               ( 0x2 << BLIS_STRUC_SHIFT )
#define BLIS_BITVAL_TRIANGULAR              ( 0x3 << BLIS_STRUC_SHIFT )


//
// -- BLIS enumerated type definitions -----------------------------------------
//

// -- Operational parameter types --

typedef enum
{
    BLIS_NO_TRANSPOSE      = 0x0,
    BLIS_TRANSPOSE         = BLIS_BITVAL_TRANS,
    BLIS_CONJ_NO_TRANSPOSE = BLIS_BITVAL_CONJ,
    BLIS_CONJ_TRANSPOSE    = BLIS_BITVAL_CONJ_TRANS
} trans_t;

typedef enum
{
    BLIS_NO_CONJUGATE      = 0x0,
    BLIS_CONJUGATE         = BLIS_BITVAL_CONJ
} conj_t;

typedef enum
{
    BLIS_ZEROS             = BLIS_BITVAL_ZEROS,
    BLIS_LOWER             = BLIS_BITVAL_LOWER,
    BLIS_UPPER             = BLIS_BITVAL_UPPER,
    BLIS_DENSE             = BLIS_BITVAL_DENSE
} uplo_t;

typedef enum
{
    BLIS_LEFT              = 0x0,
    BLIS_RIGHT
} side_t;

typedef enum
{
    BLIS_NONUNIT_DIAG      = 0x0,
    BLIS_UNIT_DIAG         = BLIS_BITVAL_UNIT_DIAG
} diag_t;

typedef enum
{
    BLIS_NO_INVERT_DIAG    = 0x0,
    BLIS_INVERT_DIAG       = BLIS_BITVAL_INVERT_DIAG
} invdiag_t;

typedef enum
{
    BLIS_GENERAL           = BLIS_BITVAL_GENERAL,
    BLIS_HERMITIAN         = BLIS_BITVAL_HERMITIAN,
    BLIS_SYMMETRIC         = BLIS_BITVAL_SYMMETRIC,
    BLIS_TRIANGULAR        = BLIS_BITVAL_TRIANGULAR
} struc_t;


// -- Data type --

typedef enum
{
    BLIS_FLOAT             = BLIS_BITVAL_FLOAT_TYPE,
    BLIS_DOUBLE            = BLIS_BITVAL_DOUBLE_TYPE,
    BLIS_SCOMPLEX          = BLIS_BITVAL_SCOMPLEX_TYPE,
    BLIS_DCOMPLEX          = BLIS_BITVAL_DCOMPLEX_TYPE,
    BLIS_INT               = BLIS_BITVAL_INT_TYPE,
    BLIS_CONSTANT          = BLIS_BITVAL_CONST_TYPE,
    BLIS_DT_LO             = BLIS_FLOAT,
    BLIS_DT_HI             = BLIS_DCOMPLEX
} num_t;

typedef enum
{
    BLIS_REAL              = BLIS_BITVAL_REAL,
    BLIS_COMPLEX           = BLIS_BITVAL_COMPLEX
} dom_t;

typedef enum
{
    BLIS_SINGLE_PREC       = BLIS_BITVAL_SINGLE_PREC,
    BLIS_DOUBLE_PREC       = BLIS_BITVAL_DOUBLE_PREC
} prec_t;


// -- Pack schema type --

typedef enum
{
    BLIS_NOT_PACKED            = BLIS_BITVAL_NOT_PACKED,
    BLIS_PACKED_UNSPEC         = BLIS_BITVAL_PACKED_UNSPEC,
    BLIS_PACKED_VECTOR         = BLIS_BITVAL_PACKED_UNSPEC,
    BLIS_PACKED_ROWS           = BLIS_BITVAL_PACKED_ROWS,
    BLIS_PACKED_COLUMNS        = BLIS_BITVAL_PACKED_COLUMNS,
    BLIS_PACKED_ROW_PANELS     = BLIS_BITVAL_PACKED_ROW_PANELS,
    BLIS_PACKED_COL_PANELS     = BLIS_BITVAL_PACKED_COL_PANELS,
    BLIS_PACKED_ROW_PANELS_4MI = BLIS_BITVAL_PACKED_ROW_PANELS_4MI,
    BLIS_PACKED_COL_PANELS_4MI = BLIS_BITVAL_PACKED_COL_PANELS_4MI,
    BLIS_PACKED_ROW_PANELS_3MI = BLIS_BITVAL_PACKED_ROW_PANELS_3MI,
    BLIS_PACKED_COL_PANELS_3MI = BLIS_BITVAL_PACKED_COL_PANELS_3MI,
    BLIS_PACKED_ROW_PANELS_4MS = BLIS_BITVAL_PACKED_ROW_PANELS_4MS,
    BLIS_PACKED_COL_PANELS_4MS = BLIS_BITVAL_PACKED_COL_PANELS_4MS,
    BLIS_PACKED_ROW_PANELS_3MS = BLIS_BITVAL_PACKED_ROW_PANELS_3MS,
    BLIS_PACKED_COL_PANELS_3MS = BLIS_BITVAL_PACKED_COL_PANELS_3MS,
    BLIS_PACKED_ROW_PANELS_RO  = BLIS_BITVAL_PACKED_ROW_PANELS_RO,
    BLIS_PACKED_COL_PANELS_RO  = BLIS_BITVAL_PACKED_COL_PANELS_RO,
    BLIS_PACKED_ROW_PANELS_IO  = BLIS_BITVAL_PACKED_ROW_PANELS_IO,
    BLIS_PACKED_COL_PANELS_IO  = BLIS_BITVAL_PACKED_COL_PANELS_IO,
    BLIS_PACKED_ROW_PANELS_RPI = BLIS_BITVAL_PACKED_ROW_PANELS_RPI,
    BLIS_PACKED_COL_PANELS_RPI = BLIS_BITVAL_PACKED_COL_PANELS_RPI,
    BLIS_PACKED_ROW_PANELS_1E  = BLIS_BITVAL_PACKED_ROW_PANELS_1E,
    BLIS_PACKED_COL_PANELS_1E  = BLIS_BITVAL_PACKED_COL_PANELS_1E,
    BLIS_PACKED_ROW_PANELS_1R  = BLIS_BITVAL_PACKED_ROW_PANELS_1R,
    BLIS_PACKED_COL_PANELS_1R  = BLIS_BITVAL_PACKED_COL_PANELS_1R
} pack_t;

// We combine row and column packing into one "type", and we start
// with BLIS_PACKED_ROW_PANELS, _COLUMN_PANELS. We also count the
// schema pair for "4ms" (4m separated), because its bit value has
// been reserved, even though we don't use it.
#define BLIS_NUM_PACK_SCHEMA_TYPES 10


// -- Pack order type --

typedef enum
{
    BLIS_PACK_FWD_IF_UPPER = BLIS_BITVAL_PACK_FWD_IF_UPPER,
    BLIS_PACK_REV_IF_UPPER = BLIS_BITVAL_PACK_REV_IF_UPPER,

    BLIS_PACK_FWD_IF_LOWER = BLIS_BITVAL_PACK_FWD_IF_LOWER,
    BLIS_PACK_REV_IF_LOWER = BLIS_BITVAL_PACK_REV_IF_LOWER
} packord_t;


// -- Pack buffer type --

typedef enum
{
    BLIS_BUFFER_FOR_A_BLOCK = BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
    BLIS_BUFFER_FOR_B_PANEL = BLIS_BITVAL_BUFFER_FOR_B_PANEL,
    BLIS_BUFFER_FOR_C_PANEL = BLIS_BITVAL_BUFFER_FOR_C_PANEL,
    BLIS_BUFFER_FOR_GEN_USE = BLIS_BITVAL_BUFFER_FOR_GEN_USE
} packbuf_t;


// -- Partitioning direction --

typedef enum
{
    BLIS_FWD,
    BLIS_BWD
} dir_t;


// -- Subpartition type --

typedef enum
{
    BLIS_SUBPART0,
    BLIS_SUBPART1,
    BLIS_SUBPART2,
    BLIS_SUBPART1AND0,
    BLIS_SUBPART1AND2,
    BLIS_SUBPART1A,
    BLIS_SUBPART1B,
    BLIS_SUBPART00,
    BLIS_SUBPART10,
    BLIS_SUBPART20,
    BLIS_SUBPART01,
    BLIS_SUBPART11,
    BLIS_SUBPART21,
    BLIS_SUBPART02,
    BLIS_SUBPART12,
    BLIS_SUBPART22
} subpart_t;


// -- Matrix dimension type --

typedef enum
{
    BLIS_M = 0,
    BLIS_N = 1
} mdim_t;


// -- Machine parameter types --

typedef enum
{
    BLIS_MACH_EPS = 0,
    BLIS_MACH_SFMIN,
    BLIS_MACH_BASE,
    BLIS_MACH_PREC,
    BLIS_MACH_NDIGMANT,
    BLIS_MACH_RND,
    BLIS_MACH_EMIN,
    BLIS_MACH_RMIN,
    BLIS_MACH_EMAX,
    BLIS_MACH_RMAX,
    BLIS_MACH_EPS2
} machval_t;

#define BLIS_NUM_MACH_PARAMS   11
#define BLIS_MACH_PARAM_FIRST  BLIS_MACH_EPS
#define BLIS_MACH_PARAM_LAST   BLIS_MACH_EPS2


// -- Induced method types --

typedef enum
{
    BLIS_3MH       = 0,
    BLIS_3M1,
    BLIS_4MH,
    BLIS_4M1B,
    BLIS_4M1A,
    BLIS_1M,
    BLIS_NAT,
    BLIS_IND_FIRST = 0,
    BLIS_IND_LAST  = BLIS_NAT
} ind_t;

#define BLIS_NUM_IND_METHODS (BLIS_NAT+1)

// These are used in bli_*_oapi.c to construct the ind_t values from
// the induced method substrings that go into function names.
#define bli_3mh  BLIS_3MH
#define bli_3m1  BLIS_3M1
#define bli_4mh  BLIS_4MH
#define bli_4mb  BLIS_4M1B
#define bli_4m1  BLIS_4M1A
#define bli_1m   BLIS_1M
#define bli_nat  BLIS_NAT


// -- Kernel ID types --

typedef enum
{
    BLIS_ADDV_KER  = 0,
    BLIS_AMAXV_KER,
    BLIS_AMINV_KER,
    BLIS_AXPBYV_KER,
    BLIS_AXPYV_KER,
    BLIS_COPYV_KER,
    BLIS_DOTV_KER,
    BLIS_DOTXV_KER,
    BLIS_INVERTV_KER,
    BLIS_SCALV_KER,
    BLIS_SCAL2V_KER,
    BLIS_SETV_KER,
    BLIS_SUBV_KER,
    BLIS_SWAPV_KER,
    BLIS_XPBYV_KER
} l1vkr_t;

#define BLIS_NUM_LEVEL1V_KERS 15


typedef enum
{
    BLIS_AXPY2V_KER = 0,
    BLIS_DOTAXPYV_KER,
    BLIS_AXPYF_KER,
    BLIS_DOTXF_KER,
    BLIS_DOTXAXPYF_KER
} l1fkr_t;

#define BLIS_NUM_LEVEL1F_KERS 5


typedef enum
{
    BLIS_PACKM_0XK_KER  = 0,
    BLIS_PACKM_1XK_KER  = 1,
    BLIS_PACKM_2XK_KER  = 2,
    BLIS_PACKM_3XK_KER  = 3,
    BLIS_PACKM_4XK_KER  = 4,
    BLIS_PACKM_5XK_KER  = 5,
    BLIS_PACKM_6XK_KER  = 6,
    BLIS_PACKM_7XK_KER  = 7,
    BLIS_PACKM_8XK_KER  = 8,
    BLIS_PACKM_9XK_KER  = 9,
    BLIS_PACKM_10XK_KER = 10,
    BLIS_PACKM_11XK_KER = 11,
    BLIS_PACKM_12XK_KER = 12,
    BLIS_PACKM_13XK_KER = 13,
    BLIS_PACKM_14XK_KER = 14,
    BLIS_PACKM_15XK_KER = 15,
    BLIS_PACKM_16XK_KER = 16,
    BLIS_PACKM_17XK_KER = 17,
    BLIS_PACKM_18XK_KER = 18,
    BLIS_PACKM_19XK_KER = 19,
    BLIS_PACKM_20XK_KER = 20,
    BLIS_PACKM_21XK_KER = 21,
    BLIS_PACKM_22XK_KER = 22,
    BLIS_PACKM_23XK_KER = 23,
    BLIS_PACKM_24XK_KER = 24,
    BLIS_PACKM_25XK_KER = 25,
    BLIS_PACKM_26XK_KER = 26,
    BLIS_PACKM_27XK_KER = 27,
    BLIS_PACKM_28XK_KER = 28,
    BLIS_PACKM_29XK_KER = 29,
    BLIS_PACKM_30XK_KER = 30,
    BLIS_PACKM_31XK_KER = 31,

    BLIS_UNPACKM_0XK_KER  = 0,
    BLIS_UNPACKM_1XK_KER  = 1,
    BLIS_UNPACKM_2XK_KER  = 2,
    BLIS_UNPACKM_3XK_KER  = 3,
    BLIS_UNPACKM_4XK_KER  = 4,
    BLIS_UNPACKM_5XK_KER  = 5,
    BLIS_UNPACKM_6XK_KER  = 6,
    BLIS_UNPACKM_7XK_KER  = 7,
    BLIS_UNPACKM_8XK_KER  = 8,
    BLIS_UNPACKM_9XK_KER  = 9,
    BLIS_UNPACKM_10XK_KER = 10,
    BLIS_UNPACKM_11XK_KER = 11,
    BLIS_UNPACKM_12XK_KER = 12,
    BLIS_UNPACKM_13XK_KER = 13,
    BLIS_UNPACKM_14XK_KER = 14,
    BLIS_UNPACKM_15XK_KER = 15,
    BLIS_UNPACKM_16XK_KER = 16,
    BLIS_UNPACKM_17XK_KER = 17,
    BLIS_UNPACKM_18XK_KER = 18,
    BLIS_UNPACKM_19XK_KER = 19,
    BLIS_UNPACKM_20XK_KER = 20,
    BLIS_UNPACKM_21XK_KER = 21,
    BLIS_UNPACKM_22XK_KER = 22,
    BLIS_UNPACKM_23XK_KER = 23,
    BLIS_UNPACKM_24XK_KER = 24,
    BLIS_UNPACKM_25XK_KER = 25,
    BLIS_UNPACKM_26XK_KER = 26,
    BLIS_UNPACKM_27XK_KER = 27,
    BLIS_UNPACKM_28XK_KER = 28,
    BLIS_UNPACKM_29XK_KER = 29,
    BLIS_UNPACKM_30XK_KER = 30,
    BLIS_UNPACKM_31XK_KER = 31

} l1mkr_t;

#define BLIS_NUM_PACKM_KERS   32
#define BLIS_NUM_UNPACKM_KERS 32


typedef enum
{
    BLIS_GEMM_UKR = 0,
    BLIS_GEMMTRSM_L_UKR,
    BLIS_GEMMTRSM_U_UKR,
    BLIS_TRSM_L_UKR,
    BLIS_TRSM_U_UKR
} l3ukr_t;

#define BLIS_NUM_LEVEL3_UKRS 5


typedef enum
{
    BLIS_REFERENCE_UKERNEL = 0,
    BLIS_VIRTUAL_UKERNEL,
    BLIS_OPTIMIZED_UKERNEL,
    BLIS_NOTAPPLIC_UKERNEL
} kimpl_t;

#define BLIS_NUM_UKR_IMPL_TYPES 4


#if 0
typedef enum
{
	// RV = row-stored, contiguous vector-loading
	// RG = row-stored, non-contiguous gather-loading
	// CV = column-stored, contiguous vector-loading
	// CG = column-stored, non-contiguous gather-loading

	// RD = row-stored, dot-based
	// CD = col-stored, dot-based

	// RC = row-stored, column-times-column
	// CR = column-stored, row-times-row

	// GX = general-stored generic implementation

	BLIS_GEMMSUP_RV_UKR = 0,
	BLIS_GEMMSUP_RG_UKR,
	BLIS_GEMMSUP_CV_UKR,
	BLIS_GEMMSUP_CG_UKR,

	BLIS_GEMMSUP_RD_UKR,
	BLIS_GEMMSUP_CD_UKR,

	BLIS_GEMMSUP_RC_UKR,
	BLIS_GEMMSUP_CR_UKR,

	BLIS_GEMMSUP_GX_UKR,
} l3sup_t;

#define BLIS_NUM_LEVEL3_SUP_UKRS 9
#endif


typedef enum
{
	// 3-operand storage combinations
	BLIS_RRR = 0,
	BLIS_RRC, // 1
	BLIS_RCR, // 2
	BLIS_RCC, // 3
	BLIS_CRR, // 4
	BLIS_CRC, // 5
	BLIS_CCR, // 6
	BLIS_CCC, // 7
	BLIS_XXX, // 8

#if 0
	BLIS_RRG,
	BLIS_RCG,
	BLIS_RGR,
	BLIS_RGC,
	BLIS_RGG,
	BLIS_CRG,
	BLIS_CCG,
	BLIS_CGR,
	BLIS_CGC,
	BLIS_CGG,
	BLIS_GRR,
	BLIS_GRC,
	BLIS_GRG,
	BLIS_GCR,
	BLIS_GCC,
	BLIS_GCG,
	BLIS_GGR,
	BLIS_GGC,
	BLIS_GGG,
#endif
} stor3_t;

#define BLIS_NUM_3OP_RC_COMBOS 9
//#define BLIS_NUM_3OP_RCG_COMBOS 27


#if 0
typedef enum
{
	BLIS_JC_IDX = 0,
	BLIS_PC_IDX,
	BLIS_IC_IDX,
	BLIS_JR_IDX,
	BLIS_IR_IDX,
	BLIS_PR_IDX
} thridx_t;
#endif

#define BLIS_NUM_LOOPS 6


// -- Operation ID type --

typedef enum
{
//
// NOTE: If/when additional type values are added to this enum,
// you must either:
// - keep the level-3 values (starting with _GEMM) beginning at
//   index 0; or
// - if the value range is moved such that it does not begin at
//   index 0, implement something like a BLIS_OPID_LEVEL3_RANGE_START
//   value that can be subtracted from the opid_t value to map it
//   to a zero-based range.
// This is needed because these level-3 opid_t values are used in
// bli_l3_ind.c to index into arrays.
//
	BLIS_GEMM = 0,
	BLIS_HEMM,
	BLIS_HERK,
	BLIS_HER2K,
	BLIS_SYMM,
	BLIS_SYRK,
	BLIS_SYR2K,
	BLIS_TRMM3,
	BLIS_TRMM,
	BLIS_TRSM,
	BLIS_GEMMT,
	BLIS_NOID
} opid_t;

#define BLIS_NUM_LEVEL3_OPS 11


// -- Blocksize ID type --

typedef enum
{
	// NOTE: the level-3 blocksizes MUST be indexed starting at zero.
	// At one point, we made this assumption in bli_cntx_set_blkszs()
	// and friends.

	BLIS_KR = 0,
	BLIS_MR,
	BLIS_NR,
	BLIS_MC,
	BLIS_KC,
	BLIS_NC,

	BLIS_M2, // level-2 blocksize in m dimension
	BLIS_N2, // level-2 blocksize in n dimension

	BLIS_AF, // level-1f axpyf fusing factor
	BLIS_DF, // level-1f dotxf fusing factor
	BLIS_XF, // level-1f dotxaxpyf fusing factor

	BLIS_NO_PART  // used as a placeholder when blocksizes are not applicable.
} bszid_t;

#define BLIS_NUM_BLKSZS 11


// -- Threshold ID type --

typedef enum
{
	BLIS_MT = 0, // level-3 small/unpacked matrix threshold in m dimension
	BLIS_NT,     // level-3 small/unpacked matrix threshold in n dimension
	BLIS_KT      // level-3 small/unpacked matrix threshold in k dimension

} threshid_t;

#define BLIS_NUM_THRESH 3


// -- Architecture ID type --

// NOTE: This typedef enum must be kept up-to-date with the arch_t
// string array in bli_arch.c. Whenever values are added/inserted
// OR if values are rearranged, be sure to update the string array
// in bli_arch.c.

typedef enum
{
	// Intel
	BLIS_ARCH_SKX = 0,
	BLIS_ARCH_KNL,
	BLIS_ARCH_KNC,
	BLIS_ARCH_HASWELL,
	BLIS_ARCH_SANDYBRIDGE,
	BLIS_ARCH_PENRYN,

	// AMD
	BLIS_ARCH_ZEN3,
	BLIS_ARCH_ZEN2,
	BLIS_ARCH_ZEN,
	BLIS_ARCH_EXCAVATOR,
	BLIS_ARCH_STEAMROLLER,
	BLIS_ARCH_PILEDRIVER,
	BLIS_ARCH_BULLDOZER,

	// ARM
	BLIS_ARCH_THUNDERX2,
	BLIS_ARCH_CORTEXA57,
	BLIS_ARCH_CORTEXA53,
	BLIS_ARCH_CORTEXA15,
	BLIS_ARCH_CORTEXA9,

	// IBM/Power
	BLIS_ARCH_POWER9,
	BLIS_ARCH_POWER7,
	BLIS_ARCH_BGQ,

	// Generic architecture/configuration
	BLIS_ARCH_GENERIC

} arch_t;

#define BLIS_NUM_ARCHS (BLIS_ARCH_GENERIC + 1)


//
// -- BLIS misc. structure types -----------------------------------------------
//

// These headers must be included here (or earlier) because definitions they
// provide are needed in the pool_t and related structs.
// begin bli_pthread.h


#ifndef BLIS_PTHREAD_H
#define BLIS_PTHREAD_H

#if defined(_MSC_VER)

// This branch defines a pthread-like API, bli_pthread_*(), and implements it
// in terms of Windows API calls.

// -- pthread_mutex_*() --

typedef SRWLOCK bli_pthread_mutex_t;
typedef void bli_pthread_mutexattr_t;

#define BLIS_PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

BLIS_EXPORT_BLIS int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     );

// -- pthread_once_*() --

typedef INIT_ONCE bli_pthread_once_t;

#define BLIS_PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

BLIS_EXPORT_BLIS void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     );

// -- pthread_cond_*() --

typedef CONDITION_VARIABLE bli_pthread_cond_t;
typedef void bli_pthread_condattr_t;

#define BLIS_PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

BLIS_EXPORT_BLIS int bli_pthread_cond_init
     (
       bli_pthread_cond_t*           cond,
       const bli_pthread_condattr_t* attr
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_destroy
     (
       bli_pthread_cond_t* cond
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_wait
     (
       bli_pthread_cond_t*  cond,
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_broadcast
     (
       bli_pthread_cond_t* cond
     );

// -- pthread_create(), pthread_join() --

typedef struct
{
    HANDLE handle;
    void* retval;
} bli_pthread_t;

typedef void bli_pthread_attr_t;

BLIS_EXPORT_BLIS int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     );

BLIS_EXPORT_BLIS int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     );

// -- pthread_barrier_*() --

typedef void bli_pthread_barrierattr_t;

typedef struct
{
    bli_pthread_mutex_t mutex;
    bli_pthread_cond_t  cond;
    int                 count;
    int                 tripCount;
} bli_pthread_barrier_t;

BLIS_EXPORT_BLIS int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count
     );

BLIS_EXPORT_BLIS int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t* barrier
     );

BLIS_EXPORT_BLIS int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t* barrier
     );

#else // !defined(_MSC_VER)

#include <pthread.h> // skipped

// This branch defines a pthreads-like API, bli_pthreads_*(), and implements it
// in terms of the corresponding pthreads_*() types, macros, and function calls. 

// -- pthread types --

typedef pthread_t              bli_pthread_t;
typedef pthread_attr_t         bli_pthread_attr_t;
typedef pthread_mutex_t        bli_pthread_mutex_t;
typedef pthread_mutexattr_t    bli_pthread_mutexattr_t;
typedef pthread_cond_t         bli_pthread_cond_t;
typedef pthread_condattr_t     bli_pthread_condattr_t;
typedef pthread_once_t         bli_pthread_once_t;

#if defined(__APPLE__)

// For OS X, we must define the barrier types ourselves since Apple does
// not implement barriers in their variant of pthreads.

typedef void bli_pthread_barrierattr_t;

typedef struct
{
    bli_pthread_mutex_t mutex;
    bli_pthread_cond_t  cond;
    int                 count;
    int                 tripCount;
} bli_pthread_barrier_t;

#else

// For other non-Windows OSes (primarily Linux), we can define the barrier
// types in terms of existing pthreads barrier types since we expect they
// will be provided by the pthreads implementation.

typedef pthread_barrier_t      bli_pthread_barrier_t;
typedef pthread_barrierattr_t  bli_pthread_barrierattr_t;

#endif

// -- pthreads macros --

#define BLIS_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define BLIS_PTHREAD_COND_INITIALIZER  PTHREAD_COND_INITIALIZER
#define BLIS_PTHREAD_ONCE_INIT         PTHREAD_ONCE_INIT

// -- pthread_create(), pthread_join() --

BLIS_EXPORT_BLIS int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     );

BLIS_EXPORT_BLIS int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     );

// -- pthread_mutex_*() --

BLIS_EXPORT_BLIS int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     );

// -- pthread_cond_*() --

BLIS_EXPORT_BLIS int bli_pthread_cond_init
     (
       bli_pthread_cond_t*           cond,
       const bli_pthread_condattr_t* attr
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_destroy
     (
       bli_pthread_cond_t* cond
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_wait
     (
       bli_pthread_cond_t*  cond,
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_broadcast
     (
       bli_pthread_cond_t* cond
     );

// -- pthread_once_*() --

BLIS_EXPORT_BLIS void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     );

// -- pthread_barrier_*() --

BLIS_EXPORT_BLIS int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count
     );

BLIS_EXPORT_BLIS int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t* barrier
     );

BLIS_EXPORT_BLIS int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t* barrier
     );

#endif // _MSC_VER

#endif // BLIS_PTHREAD_H
// end bli_pthread.h
// begin bli_malloc.h


// Typedef function pointer types for malloc() and free() substitutes.
typedef void* (*malloc_ft) ( size_t size );
typedef void  (*free_ft)   ( void*  p    );

// -----------------------------------------------------------------------------

#if 0
BLIS_EXPORT_BLIS void* bli_malloc_pool( size_t size );
BLIS_EXPORT_BLIS void   bli_free_pool( void* p );
#endif

void* bli_malloc_intl( size_t size );
void* bli_calloc_intl( size_t size );
void  bli_free_intl( void* p );

BLIS_EXPORT_BLIS void* bli_malloc_user( size_t size );
BLIS_EXPORT_BLIS void  bli_free_user( void* p );

// -----------------------------------------------------------------------------

void* bli_fmalloc_align( malloc_ft f, size_t size, size_t align_size );
void  bli_ffree_align( free_ft f, void* p );

void* bli_fmalloc_noalign( malloc_ft f, size_t size );
void  bli_ffree_noalign( free_ft f, void* p );

void  bli_fmalloc_align_check( malloc_ft f, size_t size, size_t align_size );
void  bli_fmalloc_post_check( void* p );

// end bli_malloc.h

// -- Pool block type --

typedef struct
{
	void*     buf;
	siz_t     block_size;

} pblk_t;


// -- Pool type --

typedef struct
{
	void*     block_ptrs;
	dim_t     block_ptrs_len;

	dim_t     top_index;
	dim_t     num_blocks;

	siz_t     block_size;
	siz_t     align_size;
	siz_t     offset_size;

	malloc_ft malloc_fp;
	free_ft   free_fp;

} pool_t;


// -- Array type --

typedef struct
{
	void*     buf;

	siz_t     num_elem;
	siz_t     elem_size;

} array_t;


// -- Locked pool-of-arrays-of-pools type --

typedef struct
{
	bli_pthread_mutex_t mutex;
	pool_t              pool;

	siz_t               def_array_len;

} apool_t;


// -- packing block allocator: Locked set of pools type --

typedef struct membrk_s
{
	pool_t              pools[3];
	bli_pthread_mutex_t mutex;

	// These fields are used for general-purpose allocation.
	siz_t               align_size;
	malloc_ft           malloc_fp;
	free_ft             free_fp;

} membrk_t;


// -- Memory object type --

typedef struct mem_s
{
	pblk_t    pblk;
	packbuf_t buf_type;
	pool_t*   pool;
	siz_t     size;
} mem_t;


// -- Control tree node type --

struct cntl_s
{
	// Basic fields (usually required).
	opid_t         family;
	bszid_t        bszid;
	void_fp        var_func;
	struct cntl_s* sub_prenode;
	struct cntl_s* sub_node;

	// Optional fields (needed only by some operations such as packm).
	// NOTE: first field of params must be a uint64_t containing the size
	// of the struct.
	void*          params;

	// Internal fields that track "cached" data.
	mem_t          pack_mem;
};
typedef struct cntl_s cntl_t;


// -- Blocksize object type --

typedef struct blksz_s
{
	// Primary blocksize values.
	dim_t  v[BLIS_NUM_FP_TYPES];

	// Blocksize extensions.
	dim_t  e[BLIS_NUM_FP_TYPES];

} blksz_t;


// -- Function pointer object type --

typedef struct func_s
{
	// Kernel function address.
	void_fp ptr[BLIS_NUM_FP_TYPES];

} func_t;


// -- Multi-boolean object type --

typedef struct mbool_s
{
	bool v[BLIS_NUM_FP_TYPES];

} mbool_t;


// -- Auxiliary kernel info type --

// Note: This struct is used by macro-kernels to package together extra
// parameter values that may be of use to the micro-kernel without
// cluttering up the micro-kernel interface itself.

typedef struct
{
	// The pack schemas of A and B.
	pack_t schema_a;
	pack_t schema_b;

	// Pointers to the micro-panels of A and B which will be used by the
	// next call to the micro-kernel.
	void*  a_next;
	void*  b_next;

	// The imaginary strides of A and B.
	inc_t  is_a;
	inc_t  is_b;

	// The panel strides of A and B.
	// NOTE: These are only used in situations where iteration over the
	// micropanels takes place in part within the kernel code (e.g. sup
	// millikernels).
	inc_t  ps_a;
	inc_t  ps_b;

	// The type to convert to on output.
	//num_t  dt_on_output;

} auxinfo_t;


// -- Global scalar constant data struct --

// Note: This struct is used only when statically initializing the
// global scalar constants in bli_const.c.
typedef struct constdata_s
{
	float    s;
	double   d;
	scomplex c;
	dcomplex z;
	gint_t   i;

} constdata_t;


//
// -- BLIS object type definitions ---------------------------------------------
//

typedef struct obj_s
{
	// Basic fields
	struct obj_s* root;

	dim_t         off[2];
	dim_t         dim[2];
	doff_t        diag_off;

	objbits_t     info;
	objbits_t     info2;
	siz_t         elem_size;

	void*         buffer;
	inc_t         rs;
	inc_t         cs;
	inc_t         is;

	// Bufferless scalar storage
	atom_t        scalar;

	// Pack-related fields
	dim_t         m_padded; // m dimension of matrix, including any padding
	dim_t         n_padded; // n dimension of matrix, including any padding
	inc_t         ps;       // panel stride (distance to next panel)
	inc_t         pd;       // panel dimension (the "width" of a panel:
	                        // usually MR or NR)
	dim_t         m_panel;  // m dimension of a "full" panel
	dim_t         n_panel;  // n dimension of a "full" panel
} obj_t;

// Pre-initializors. Things that must be set afterwards:
// - root object pointer
// - info bitfields: dt, target_dt, exec_dt, comp_dt
// - info2 bitfields: scalar_dt
// - elem_size
// - dims, strides
// - buffer
// - internal scalar buffer (must always set imaginary component)

#define BLIS_OBJECT_INITIALIZER \
{ \
	.root      = NULL, \
\
	.off       = { 0, 0 }, \
	.dim       = { 0, 0 }, \
	.diag_off  = 0, \
\
	.info      = 0x0 | BLIS_BITVAL_DENSE      | \
	                   BLIS_BITVAL_GENERAL, \
	.info2     = 0x0, \
	.elem_size = sizeof( float ),  \
\
	.buffer    = NULL, \
	.rs        = 0, \
	.cs        = 0, \
	.is        = 1,  \
\
	.scalar    = { 0.0, 0.0 }, \
\
	.m_padded  = 0, \
	.n_padded  = 0, \
	.ps        = 0, \
	.pd        = 0, \
	.m_panel   = 0, \
	.n_panel   = 0  \
}

#define BLIS_OBJECT_INITIALIZER_1X1 \
{ \
	.root      = NULL, \
\
	.off       = { 0, 0 }, \
	.dim       = { 1, 1 }, \
	.diag_off  = 0, \
\
	.info      = 0x0 | BLIS_BITVAL_DENSE      | \
	                   BLIS_BITVAL_GENERAL, \
	.info2     = 0x0, \
	.elem_size = sizeof( float ),  \
\
	.buffer    = NULL, \
	.rs        = 0, \
	.cs        = 0, \
	.is        = 1,  \
\
	.scalar    = { 0.0, 0.0 }, \
\
	.m_padded  = 0, \
	.n_padded  = 0, \
	.ps        = 0, \
	.pd        = 0, \
	.m_panel   = 0, \
	.n_panel   = 0  \
}

// Define these macros here since they must be updated if contents of
// obj_t changes.

BLIS_INLINE void bli_obj_init_full_shallow_copy_of( obj_t* a, obj_t* b )
{
	b->root      = a->root;

	b->off[0]    = a->off[0];
	b->off[1]    = a->off[1];
	b->dim[0]    = a->dim[0];
	b->dim[1]    = a->dim[1];
	b->diag_off  = a->diag_off;

	b->info      = a->info;
	b->info2     = a->info2;
	b->elem_size = a->elem_size;

	b->buffer    = a->buffer;
	b->rs        = a->rs;
	b->cs        = a->cs;
	b->is        = a->is;

	b->scalar    = a->scalar;

	//b->pack_mem  = a->pack_mem;
	b->m_padded  = a->m_padded;
	b->n_padded  = a->n_padded;
	b->ps        = a->ps;
	b->pd        = a->pd;
	b->m_panel   = a->m_panel;
	b->n_panel   = a->n_panel;
}

BLIS_INLINE void bli_obj_init_subpart_from( obj_t* a, obj_t* b )
{
	b->root      = a->root;

	b->off[0]    = a->off[0];
	b->off[1]    = a->off[1];
	// Avoid copying m and n since they will be overwritten.
	//b->dim[0]    = a->dim[0];
	//b->dim[1]    = a->dim[1];
	b->diag_off  = a->diag_off;

	b->info      = a->info;
	b->info2     = a->info2;
	b->elem_size = a->elem_size;

	b->buffer    = a->buffer;
	b->rs        = a->rs;
	b->cs        = a->cs;
	b->is        = a->is;

	b->scalar    = a->scalar;

	// Avoid copying pack_mem entry.
	// FGVZ: You should probably make sure this is right.
	//b->pack_mem  = a->pack_mem;
	b->m_padded  = a->m_padded;
	b->n_padded  = a->n_padded;
	b->ps        = a->ps;
	b->pd        = a->pd;
	b->m_panel   = a->m_panel;
	b->n_panel   = a->n_panel;
}

// Initializors for global scalar constants.
// NOTE: These must remain cpp macros since they are initializor
// expressions, not functions.

#define bli_obj_init_const( buffer0 ) \
{ \
	.root      = NULL, \
\
	.off       = { 0, 0 }, \
	.dim       = { 1, 1 }, \
	.diag_off  = 0, \
\
	.info      = 0x0 | BLIS_BITVAL_CONST_TYPE | \
	                   BLIS_BITVAL_DENSE      | \
	                   BLIS_BITVAL_GENERAL, \
	.info2     = 0x0, \
	.elem_size = sizeof( constdata_t ), \
\
	.buffer    = buffer0, \
	.rs        = 1, \
	.cs        = 1, \
	.is        = 1  \
}

#define bli_obj_init_constdata( val ) \
{ \
	.s =           ( float  )val, \
	.d =           ( double )val, \
	.c = { .real = ( float  )val, .imag = 0.0f }, \
	.z = { .real = ( double )val, .imag = 0.0 }, \
	.i =           ( gint_t )val, \
}


// -- Context type --

typedef struct cntx_s
{
	blksz_t   blkszs[ BLIS_NUM_BLKSZS ];
	bszid_t   bmults[ BLIS_NUM_BLKSZS ];

	blksz_t   trsm_blkszs[ BLIS_NUM_BLKSZS ];

	func_t    l3_vir_ukrs[ BLIS_NUM_LEVEL3_UKRS ];
	func_t    l3_nat_ukrs[ BLIS_NUM_LEVEL3_UKRS ];
	mbool_t   l3_nat_ukrs_prefs[ BLIS_NUM_LEVEL3_UKRS ];

	blksz_t   l3_sup_thresh[ BLIS_NUM_THRESH ];
	void*     l3_sup_handlers[ BLIS_NUM_LEVEL3_OPS ];
	blksz_t   l3_sup_blkszs[ BLIS_NUM_BLKSZS ];
	func_t    l3_sup_kers[ BLIS_NUM_3OP_RC_COMBOS ];
	mbool_t   l3_sup_kers_prefs[ BLIS_NUM_3OP_RC_COMBOS ];

	func_t    l1f_kers[ BLIS_NUM_LEVEL1F_KERS ];
	func_t    l1v_kers[ BLIS_NUM_LEVEL1V_KERS ];

	func_t    packm_kers[ BLIS_NUM_PACKM_KERS ];
	func_t    unpackm_kers[ BLIS_NUM_UNPACKM_KERS ];

	ind_t     method;
	pack_t    schema_a_block;
	pack_t    schema_b_panel;
	pack_t    schema_c_panel;

} cntx_t;


// -- Runtime type --

// NOTE: The order of these fields must be kept consistent with the definition
// of the BLIS_RNTM_INITIALIZER macro in bli_rntm.h.

typedef struct rntm_s
{
	// "External" fields: these may be queried by the end-user.
	bool      auto_factor;

	dim_t     num_threads;
	dim_t     thrloop[ BLIS_NUM_LOOPS ];
	bool      pack_a; // enable/disable packing of left-hand matrix A.
	bool      pack_b; // enable/disable packing of right-hand matrix B.
	bool      l3_sup; // enable/disable small matrix handling in level-3 ops.

	// "Internal" fields: these should not be exposed to the end-user.

	// The small block pool, which is attached in the l3 thread decorator.
	pool_t*   sba_pool;

	// The packing block allocator, which is attached in the l3 thread decorator.
	membrk_t* membrk;

} rntm_t;


// -- Error types --

typedef enum
{
	BLIS_NO_ERROR_CHECKING = 0,
	BLIS_FULL_ERROR_CHECKING
} errlev_t;

typedef enum
{
	// Generic error codes
	BLIS_SUCCESS                               = (  -1),
	BLIS_FAILURE                               = (  -2),

	BLIS_ERROR_CODE_MIN                        = (  -9),

	// General errors
	BLIS_INVALID_ERROR_CHECKING_LEVEL          = ( -10),
	BLIS_UNDEFINED_ERROR_CODE                  = ( -11),
	BLIS_NULL_POINTER                          = ( -12),
	BLIS_NOT_YET_IMPLEMENTED                   = ( -13),

	// Parameter-specific errors
	BLIS_INVALID_SIDE                          = ( -20),
	BLIS_INVALID_UPLO                          = ( -21),
	BLIS_INVALID_TRANS                         = ( -22),
	BLIS_INVALID_CONJ                          = ( -23),
	BLIS_INVALID_DIAG                          = ( -24),
	BLIS_INVALID_MACHVAL                       = ( -25),
	BLIS_EXPECTED_NONUNIT_DIAG                 = ( -26),

	// Datatype-specific errors
	BLIS_INVALID_DATATYPE                      = ( -30),
	BLIS_EXPECTED_FLOATING_POINT_DATATYPE      = ( -31),
	BLIS_EXPECTED_NONINTEGER_DATATYPE          = ( -32),
	BLIS_EXPECTED_NONCONSTANT_DATATYPE         = ( -33),
	BLIS_EXPECTED_REAL_DATATYPE                = ( -34),
	BLIS_EXPECTED_INTEGER_DATATYPE             = ( -35),
	BLIS_INCONSISTENT_DATATYPES                = ( -36),
	BLIS_EXPECTED_REAL_PROJ_OF                 = ( -37),
	BLIS_EXPECTED_REAL_VALUED_OBJECT           = ( -38),
	BLIS_INCONSISTENT_PRECISIONS               = ( -39),

	// Dimension-specific errors
	BLIS_NONCONFORMAL_DIMENSIONS               = ( -40),
	BLIS_EXPECTED_SCALAR_OBJECT                = ( -41),
	BLIS_EXPECTED_VECTOR_OBJECT                = ( -42),
	BLIS_UNEQUAL_VECTOR_LENGTHS                = ( -43),
	BLIS_EXPECTED_SQUARE_OBJECT                = ( -44),
	BLIS_UNEXPECTED_OBJECT_LENGTH              = ( -45),
	BLIS_UNEXPECTED_OBJECT_WIDTH               = ( -46),
	BLIS_UNEXPECTED_VECTOR_DIM                 = ( -47),
	BLIS_UNEXPECTED_DIAG_OFFSET                = ( -48),
	BLIS_NEGATIVE_DIMENSION                    = ( -49),

	// Stride-specific errors
	BLIS_INVALID_ROW_STRIDE                    = ( -50),
	BLIS_INVALID_COL_STRIDE                    = ( -51),
	BLIS_INVALID_DIM_STRIDE_COMBINATION        = ( -52),

	// Structure-specific errors
	BLIS_EXPECTED_GENERAL_OBJECT               = ( -60),
	BLIS_EXPECTED_HERMITIAN_OBJECT             = ( -61),
	BLIS_EXPECTED_SYMMETRIC_OBJECT             = ( -62),
	BLIS_EXPECTED_TRIANGULAR_OBJECT            = ( -63),

	// Storage-specific errors
	BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT        = ( -70),

	// Partitioning-specific errors
	BLIS_INVALID_3x1_SUBPART                   = ( -80),
	BLIS_INVALID_1x3_SUBPART                   = ( -81),
	BLIS_INVALID_3x3_SUBPART                   = ( -82),

	// Control tree-specific errors
	BLIS_UNEXPECTED_NULL_CONTROL_TREE          = ( -90),

	// Packing-specific errors
	BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK  = (-100),

	// Buffer-specific errors
	BLIS_EXPECTED_NONNULL_OBJECT_BUFFER        = (-110),

	// Memory errors
	BLIS_MALLOC_RETURNED_NULL                  = (-120),

	// Internal memory pool errors
	BLIS_INVALID_PACKBUF                       = (-130),
	BLIS_EXHAUSTED_CONTIG_MEMORY_POOL          = (-131),
	BLIS_INSUFFICIENT_STACK_BUF_SIZE           = (-132),
	BLIS_ALIGNMENT_NOT_POWER_OF_TWO            = (-133),
	BLIS_ALIGNMENT_NOT_MULT_OF_PTR_SIZE        = (-134),

	// Object-related errors
	BLIS_EXPECTED_OBJECT_ALIAS                 = (-140),

	// Architecture-related errors
	BLIS_INVALID_ARCH_ID                       = (-150),

	// Blocksize-related errors
	BLIS_MC_DEF_NONMULTIPLE_OF_MR              = (-160),
	BLIS_MC_MAX_NONMULTIPLE_OF_MR              = (-161),
	BLIS_NC_DEF_NONMULTIPLE_OF_NR              = (-162),
	BLIS_NC_MAX_NONMULTIPLE_OF_NR              = (-163),
	BLIS_KC_DEF_NONMULTIPLE_OF_KR              = (-164),
	BLIS_KC_MAX_NONMULTIPLE_OF_KR              = (-165),

	BLIS_ERROR_CODE_MAX                        = (-170)
} err_t;

#endif
// end bli_type_defs.h


enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#ifdef __cplusplus
extern "C" {
#endif


BLIS_EXPORT_BLAS float  cblas_sdsdot(f77_int N, float alpha, const float *X,
                    f77_int incX, const float *Y, f77_int incY);
BLIS_EXPORT_BLAS double cblas_dsdot(f77_int N, const float *X, f77_int incX, const float *Y,
                   f77_int incY);
BLIS_EXPORT_BLAS float  cblas_sdot(f77_int N, const float  *X, f77_int incX,
                  const float  *Y, f77_int incY);
BLIS_EXPORT_BLAS double cblas_ddot(f77_int N, const double *X, f77_int incX,
                  const double *Y, f77_int incY);


BLIS_EXPORT_BLAS void   cblas_cdotu_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotu);
BLIS_EXPORT_BLAS void   cblas_cdotc_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotc);

BLIS_EXPORT_BLAS void   cblas_zdotu_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotu);
BLIS_EXPORT_BLAS void   cblas_zdotc_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotc);



BLIS_EXPORT_BLAS float  cblas_snrm2(f77_int N, const float *X, f77_int incX);
BLIS_EXPORT_BLAS float  cblas_sasum(f77_int N, const float *X, f77_int incX);

BLIS_EXPORT_BLAS double cblas_dnrm2(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS double cblas_dasum(f77_int N, const double *X, f77_int incX);

BLIS_EXPORT_BLAS float  cblas_scnrm2(f77_int N, const void *X, f77_int incX);
BLIS_EXPORT_BLAS float  cblas_scasum(f77_int N, const void *X, f77_int incX);

BLIS_EXPORT_BLAS double cblas_dznrm2(f77_int N, const void *X, f77_int incX);
BLIS_EXPORT_BLAS double cblas_dzasum(f77_int N, const void *X, f77_int incX);



BLIS_EXPORT_BLAS f77_int cblas_isamax(f77_int N, const float  *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_idamax(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_icamax(f77_int N, const void   *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_izamax(f77_int N, const void   *X, f77_int incX);




void BLIS_EXPORT_BLAS cblas_sswap(f77_int N, float *X, f77_int incX,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_scopy(f77_int N, const float *X, f77_int incX,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_saxpy(f77_int N, float alpha, const float *X,
                 f77_int incX, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_saxpby(f77_int N, float alpha, const float *X,
                 f77_int incX, float beta, float *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_dswap(f77_int N, double *X, f77_int incX,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dcopy(f77_int N, const double *X, f77_int incX,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_daxpy(f77_int N, double alpha, const double *X,
                 f77_int incX, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_daxpby(f77_int N, double alpha, const double *X,
                 f77_int incX, double beta, double *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_cswap(f77_int N, void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ccopy(f77_int N, const void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_caxpy(f77_int N, const void *alpha, const void *X,
                 f77_int incX, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_caxpby(f77_int N, const void *alpha,
                const void *X, f77_int incX, const void* beta,
                void *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_zswap(f77_int N, void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zcopy(f77_int N, const void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zaxpy(f77_int N, const void *alpha, const void *X,
                 f77_int incX, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zaxpby(f77_int N, const void *alpha,
                const void *X, f77_int incX, const void *beta,
                void *Y, f77_int incY);



void BLIS_EXPORT_BLAS cblas_srotg(float *a, float *b, float *c, float *s);
void BLIS_EXPORT_BLAS cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void BLIS_EXPORT_BLAS cblas_srot(f77_int N, float *X, f77_int incX,
                float *Y, f77_int incY, const float c, const float s);
void BLIS_EXPORT_BLAS cblas_srotm(f77_int N, float *X, f77_int incX,
                float *Y, f77_int incY, const float *P);

void BLIS_EXPORT_BLAS cblas_drotg(double *a, double *b, double *c, double *s);
void BLIS_EXPORT_BLAS cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void BLIS_EXPORT_BLAS cblas_drot(f77_int N, double *X, f77_int incX,
                double *Y, f77_int incY, const double c, const double  s);
void BLIS_EXPORT_BLAS cblas_drotm(f77_int N, double *X, f77_int incX,
                double *Y, f77_int incY, const double *P);



void BLIS_EXPORT_BLAS cblas_sscal(f77_int N, float alpha, float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dscal(f77_int N, double alpha, double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_cscal(f77_int N, const void *alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_zscal(f77_int N, const void *alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_csscal(f77_int N, float alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_zdscal(f77_int N, double alpha, void *X, f77_int incX);




void BLIS_EXPORT_BLAS cblas_sgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 const float *X, f77_int incX, float beta,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, float alpha,
                 const float *A, f77_int lda, const float *X,
                 f77_int incX, float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_strmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *Ap, float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_strsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *A, f77_int lda, float *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_stbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *Ap, float *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_dgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 const double *X, f77_int incX, double beta,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, double alpha,
                 const double *A, f77_int lda, const double *X,
                 f77_int incX, double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dtrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *Ap, double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *A, f77_int lda, double *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *Ap, double *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_cgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *X, f77_int incX, const void *beta,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_cgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, const void *alpha,
                 const void *A, f77_int lda, const void *X,
                 f77_int incX, const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ctrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda, void *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_zgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *X, f77_int incX, const void *beta,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, const void *alpha,
                 const void *A, f77_int lda, const void *X,
                 f77_int incX, const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ztrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda, void *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);



void BLIS_EXPORT_BLAS cblas_ssymv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, float alpha, const float *A,
                 f77_int lda, const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ssbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, float alpha, const float *Ap,
                 const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sger(enum CBLAS_ORDER order, f77_int M, f77_int N,
                float alpha, const float *X, f77_int incX,
                const float *Y, f77_int incY, float *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_ssyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, float *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_sspr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, float *Ap);
void BLIS_EXPORT_BLAS cblas_ssyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, const float *Y, f77_int incY, float *A,
                f77_int lda);
void BLIS_EXPORT_BLAS cblas_sspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, const float *Y, f77_int incY, float *A);

void BLIS_EXPORT_BLAS cblas_dsymv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, double alpha, const double *A,
                 f77_int lda, const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dsbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, double alpha, const double *Ap,
                 const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dger(enum CBLAS_ORDER order, f77_int M, f77_int N,
                double alpha, const double *X, f77_int incX,
                const double *Y, f77_int incY, double *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_dsyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, double *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_dspr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, double *Ap);
void BLIS_EXPORT_BLAS cblas_dsyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, const double *Y, f77_int incY, double *A,
                f77_int lda);
void BLIS_EXPORT_BLAS cblas_dspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, const double *Y, f77_int incY, double *A);



void BLIS_EXPORT_BLAS cblas_chemv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_chbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_chpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *Ap,
                 const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_cgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_cgerc(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_cher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X, f77_int incX,
                void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_chpr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X,
                f77_int incX, void *A);
void BLIS_EXPORT_BLAS cblas_cher2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_chpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *Ap);

void BLIS_EXPORT_BLAS cblas_zhemv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zhbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zhpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *Ap,
                 const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zgerc(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const void *X, f77_int incX,
                void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zhpr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const void *X,
                f77_int incX, void *A);
void BLIS_EXPORT_BLAS cblas_zher2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zhpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *Ap);




void BLIS_EXPORT_BLAS cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 const float *B, f77_int ldb, float beta,
                 float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const float *A, f77_int lda,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  float alpha, const float *A, f77_int lda,
                  const float *B, f77_int ldb, float beta,
                  float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_strmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 float *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 float *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_sgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 const double *B, f77_int ldb, double beta,
                 double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 double alpha, const double *A, f77_int lda,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  double alpha, const double *A, f77_int lda,
                  const double *B, f77_int ldb, double beta,
                  double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dtrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 double *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_dtrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 double *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_dgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_cgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 const void *alpha, const void *A, f77_int lda,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, const void *beta,
                  void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ctrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_ctrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_cgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_zgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 const void *alpha, const void *A, f77_int lda,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, const void *beta,
                  void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ztrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_ztrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_zgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);



void BLIS_EXPORT_BLAS cblas_chemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const void *A, f77_int lda,
                 float beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, float beta,
                  void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_zhemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 double alpha, const void *A, f77_int lda,
                 double beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, double beta,
                  void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_xerbla(f77_int p, const char *rout, const char *form, ...);



BLIS_EXPORT_BLAS float  cblas_scabs1( const void *z);
BLIS_EXPORT_BLAS double  cblas_dcabs1( const void *z);




// -- Batch APIs -------
void BLIS_EXPORT_BLAS cblas_sgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const float *alpha_array, const float **A,
                 f77_int *lda_array, const float **B, f77_int *ldb_array,
                 const float *beta_array, float **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_dgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const double *alpha_array,
                 const double **A,f77_int *lda_array,
                 const double **B, f77_int *ldb_array,
                 const double *beta_array, double **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_cgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const void *alpha_array, const void **A,
                 f77_int *lda_array, const void **B, f77_int *ldb_array,
                 const void *beta_array, void **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_zgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const void *alpha_array, const void **A,
                 f77_int *lda_array, const void **B, f77_int *ldb_array,
                 const void *beta_array, void **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_cgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);

// -- AMIN APIs -------
BLIS_EXPORT_BLAS f77_int cblas_isamin(f77_int N, const float  *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_idamin(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_icamin(f77_int N, const void   *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_izamin(f77_int N, const void   *X, f77_int incX);

#ifdef __cplusplus
}
#endif
#endif
