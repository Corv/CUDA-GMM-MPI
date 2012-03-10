/*
 * Parameters file for gaussian mixture model based clustering application
 */

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

// Maxinum number of threads per block is 512, so that limits us to 512 clusters
// Probably will run out of memory and make the computation intractable far before 512 clusters though
#define MAX_CLUSTERS 512
#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E3
#define	NUM_BLOCKS 16
#define NUM_THREADS_ESTEP 512 // should be a power of 2 for parallel reductions to work
#define NUM_THREADS_MSTEP 256 // should be a power of 2 for parallel reductions to work
#define NUM_DIMENSIONS 32

// Which GPU to use, if more than 1
#define DEVICE 0
//#define DEVICE 1

// Using only diagonal covariance matrix, thus all dimensions are considered independent
#define DIAG_ONLY 0

// Maximum number of iterations for the EM convergence loop
#define MAX_ITERS 100
#define MIN_ITERS 100

// Prints verbose output during the algorithm
// Enables the DEBUG macro
#define ENABLE_DEBUG 0

// Used to enable regular print outs (such as the Rissanen scores, clustering results)
// This should be enabled for general use and disabled for performance evaluations only
#define ENABLE_PRINT 0

// Used to enable output of cluster results to .results and .summary files
#define ENABLE_OUTPUT 0

// Used to enable EMUPRINT macro, this can only be used when compiled for
// in emulation mode. It is used to print out during cuda kernels
#define EMU 0

#if ENABLE_DEBUG
#define DEBUG(fmt,args...) printf(fmt, ##args)
#else
#define DEBUG(fmt,args...)
#endif

#if ENABLE_PRINT
#define PRINT(fmt,args...) printf(fmt, ##args)
#else
#define PRINT(fmt,args...)
#endif

#ifdef EMU
#define EMUPRINT(fmt,args...) printf(fmt, ##args)
#else
#define EMUPRINT(fmt,args...)
#endif

typedef struct 
{
    // Key for array lengths
    //  N = number of events
    //  M = number of clusters
    //  D = number of dimensions
    float* N;        // expected # of pixels in cluster: [M]
    float* pi;       // probability of cluster in GMM: [M]
    float* constant; // Normalizing constant [M]
    float* avgvar;    // average variance [M]
    float* means;   // Spectral mean for the cluster: [M*D]
    float* R;      // Covariance matrix: [M*D*D]
    float* Rinv;   // Inverse of covariance matrix: [M*D*D]
    float* memberships; // Fuzzy memberships: [N*M]
} clusters_t;

int validateArguments(int argc, char** argv, int* num_clusters, FILE** infile, FILE** outfile);
void printUsage(char** argv);
#endif

