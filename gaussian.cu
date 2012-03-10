// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> // for clock(), clock_t, CLOCKS_PER_SEC
#include <omp.h>
#include <mpi.h>

// includes, project
#include <cutil.h>
#include "gaussian.h"
#include "invert_matrix.h"

// includes, kernels
#include <gaussian_kernel.cu>

#define LIKELIHOOD_TAG 1
#define MSTEP_N_TAG 2
#define MSTEP_MEAN_TAG 3
#define MSTEP_COVAR_TAG 4

// Function prototypes
extern "C" float* readData(char* f, int* ndims, int*nevents);
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters);
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);
float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);
void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions);
void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);

// Since cutil timers aren't thread safe, we do it manually with cuda events
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    float* et;
} cudaTimer_t;

void createTimer(cudaTimer_t* timer) {
    #pragma omp critical (create_timer) 
    {
        cudaEventCreate(&(timer->start));
        cudaEventCreate(&(timer->stop));
        timer->et = (float*) malloc(sizeof(float));
        *(timer->et) = 0.0f;
    }
}

void deleteTimer(cudaTimer_t timer) {
    #pragma omp critical (delete_timer) 
    {
        cudaEventDestroy(timer.start);
        cudaEventDestroy(timer.stop);
        free(timer.et);
    }
}

void startTimer(cudaTimer_t timer) {
    cudaEventRecord(timer.start,0);
}

void stopTimer(cudaTimer_t timer) {
    cudaEventRecord(timer.stop,0);
    cudaEventSynchronize(timer.stop);
    float tmp;
    cudaEventElapsedTime(&tmp,timer.start,timer.stop);
    *(timer.et) += tmp;
}

float getTimerValue(cudaTimer_t timer) {
    return *(timer.et);
}

// Structure to hold the timers for the different kernel.
//  One of these structs per GPU for profiling.
typedef struct {
    cudaTimer_t e_step;
    cudaTimer_t m_step;
    cudaTimer_t constants;
    cudaTimer_t reduce;
    cudaTimer_t memcpy;
    cudaTimer_t cpu;
    cudaTimer_t mpi;
} profile_t;

// Creates the CUDA timers inside the profile_t struct
void init_profile_t(profile_t* p) {
    createTimer(&(p->e_step));
    createTimer(&(p->m_step));
    createTimer(&(p->constants));
    createTimer(&(p->reduce));
    createTimer(&(p->memcpy));
    createTimer(&(p->cpu));
    createTimer(&(p->mpi));
}

// Deletes the timers in the profile_t struct
void cleanup_profile_t(profile_t* p) {
    deleteTimer(p->e_step);
    deleteTimer(p->m_step);
    deleteTimer(p->constants);
    deleteTimer(p->reduce);
    deleteTimer(p->memcpy);
    deleteTimer(p->cpu);
    deleteTimer(p->mpi);
}

void seed_clusters(clusters_t* clusters, float* fcs_data, int num_clusters, int num_dimensions, int num_events) {
    
    float seed;
    if(num_clusters > 1) {
        seed = (num_events-1.0f)/(num_clusters-1.0f);
    } else {
        seed = 0.0;
    }
    // Sets the means from evenly distributed points in the input data
    for(int c=0; c < num_clusters; c++) {
        clusters->N[c] = num_events / num_clusters;
        for(int d=0; d < num_dimensions; d++) {
            clusters->means[c*num_dimensions+d] = fcs_data[((int)(c*seed))*num_dimensions+d];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    int rank, num_nodes, len, provided;
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
    //MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    //printf("Tried: %d, Got %d\n\n",MPI_THREAD_FUNNELED,provided);

    MPI_Comm_size(MPI_COMM_WORLD,&num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Get_processor_name(name, &len);
    printf("Hello world from node %d of %d on %s\n",rank,num_nodes,name);
    
    double mstart, mfinish = 0.0;
    double mstart2, mfinish2 = 0.0;
    mstart = MPI_Wtime();


    int original_num_clusters, desired_num_clusters, ideal_num_clusters, stop_number;
    
    int regroup_iterations = 0;
    int params_iterations = 0;
    int constants_iterations = 0;
    int reduce_iterations = 0;
    
    // Keep track of total time
    unsigned int timer_total;
    cutCreateTimer( &timer_total);
    cutStartTimer( timer_total);
   
    // Keep track of I/O time
    unsigned int timer_io;
    cutCreateTimer( &timer_io);
    cutStartTimer( timer_io);
    
    // Keep track of Master thread CPU Time (outside #omp parallel)
    unsigned int timer_cpu;
    cutCreateTimer( &timer_cpu);
    
    // Validate the command-line arguments, parse # of clusters, etc 
    int error = validateArguments(argc,argv,&original_num_clusters,&desired_num_clusters);
    
    // Don't continue if we had a problem with the program arguments
    if(error) {
        return 1;
    }
    
    // Number of clusters to stop iterating at.
    if(desired_num_clusters == 0) {
        stop_number = 1;
    } else {
        stop_number = desired_num_clusters;
    }

    int num_dimensions;
    int num_events;
    
    // Read FCS data   
    PRINT("Parsing input file...");
    // This stores the data in a 1-D array with consecutive values being the dimensions from a single event
    // (num_events by num_dimensions matrix)
    float* fcs_data_by_event;
    if(rank == 0) {
        fcs_data_by_event = readData(argv[2],&num_dimensions,&num_events);    
        MPI_Bcast(&num_events,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(&num_dimensions,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(fcs_data_by_event,num_events*num_dimensions,MPI_FLOAT,0,MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&num_events,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(&num_dimensions,1,MPI_INT,0,MPI_COMM_WORLD);
        fcs_data_by_event = (float*) malloc(sizeof(float)*num_dimensions*num_events);
        MPI_Bcast(fcs_data_by_event,num_events*num_dimensions,MPI_FLOAT,0,MPI_COMM_WORLD);
    }

    if(!fcs_data_by_event) {
        printf("Error parsing input file. This could be due to an empty file ");
        printf("or an inconsistent number of dimensions. Aborting.\n");
        return 1;
    }
    
    // Transpose the event data (allows coalesced access pattern in E-step kernel)
    // This has consecutive values being from the same dimension of the data 
    // (num_dimensions by num_events matrix)
    float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);
    
    for(int e=0; e<num_events; e++) {
        for(int d=0; d<num_dimensions; d++) {
            fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
        }
    }    
    MPI_Barrier(MPI_COMM_WORLD);
    cutStopTimer(timer_io);
    cutStartTimer(timer_cpu);
   
    PRINT("Number of events: %d\n",num_events);
    PRINT("Number of dimensions: %d\n\n",num_dimensions);
    
    PRINT("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);
   
    int total_num_gpus, num_gpus;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&num_gpus));
    if (num_gpus < 1) {
        printf("ERROR: No CUDA capable GPUs detected.\n");
        return -1;
    } else if(num_gpus == 1) {
        printf("Warning: Only 1 CUDA GPU detected. Running single GPU version would be more efficient.\n");
    } else {
        PRINT("Using %d Host-threads with %d GPUs\n",num_gpus,num_gpus);
    }
    
    total_num_gpus = num_gpus*num_nodes;
    
    // Setup the cluster data structures on host
    // This the shared memory space between the GPUs
    clusters_t* clusters = (clusters_t*) malloc(sizeof(clusters_t)*total_num_gpus);
    for(int g=0; g < num_gpus; g++) {
        clusters[g].N = (float*) malloc(sizeof(float)*original_num_clusters);
        clusters[g].pi = (float*) malloc(sizeof(float)*original_num_clusters);
        clusters[g].constant = (float*) malloc(sizeof(float)*original_num_clusters);
        clusters[g].avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
        clusters[g].means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
        clusters[g].R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
        clusters[g].Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
        if(!clusters[g].means || !clusters[g].R || !clusters[g].Rinv) { 
        }
    }
    clusters[0].memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
    if(!clusters[0].memberships) {
        printf("ERROR: Could not allocate memory for clusters.\n"); 
        return 1; 
    }

    
    // Declare another set of clusters for saving the results of the best configuration
    clusters_t saved_clusters;
    saved_clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
    saved_clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    saved_clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    saved_clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
    if(!saved_clusters.means || !saved_clusters.R || !saved_clusters.Rinv || !saved_clusters.memberships) { 
        printf("ERROR: Could not allocate memory for clusters.\n"); 
        return 1; 
    }
    DEBUG("Finished allocating shared cluster structures on host\n");
        
    // Used to hold the result from regroup kernel
    float* shared_likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS*num_gpus);
    float likelihood, old_likelihood;
    float min_rissanen;
    
    cutStopTimer(timer_cpu);

    //omp_set_num_threads(num_gpus);

    mfinish = MPI_Wtime();
    printf("Time until OMP Parallel: %f\n",mfinish-mstart);
    #pragma omp parallel shared(clusters,fcs_data_by_event,fcs_data_by_dimension,shared_likelihoods,likelihood,old_likelihood,ideal_num_clusters,min_rissanen,regroup_iterations) 
    {
        // Set the device for this thread
        unsigned int tid  = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        if(num_cpu_threads != num_gpus) {
            printf("ERROR: Number of threads did not match number of GPUs. Perhaps not enough CPU cores?");
            exit(1);
        }
        cudaSetDevice(tid);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, tid);
        printf("Node %d (of %d), CPU thread %d (of %d) on %s using device %d: %s\n", rank, num_nodes, tid, num_gpus, name, tid, prop.name);
       
        // Timers for profiling
        //  timers use cuda events (which require a cuda context),
        //  cannot initialize them until after cudaSetDevice(...)
        profile_t timers;
        init_profile_t(&timers);
        
        // Used as a temporary cluster for combining clusters in "distance" computations
        startTimer(timers.cpu);
        clusters_t scratch_cluster;
        scratch_cluster.N = (float*) malloc(sizeof(float));
        scratch_cluster.pi = (float*) malloc(sizeof(float));
        scratch_cluster.constant = (float*) malloc(sizeof(float));
        scratch_cluster.avgvar = (float*) malloc(sizeof(float));
        scratch_cluster.means = (float*) malloc(sizeof(float)*num_dimensions);
        scratch_cluster.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        scratch_cluster.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        scratch_cluster.memberships = (float*) malloc(sizeof(float)*num_events);

        DEBUG("Finished allocating memory on host for clusters.\n");
        stopTimer(timers.cpu);
        
        // Setup the cluster data structures on device
        // First allocate structures on the host, CUDA malloc the arrays
        // Then CUDA malloc structures on the device and copy them over
        startTimer(timers.memcpy);
        clusters_t temp_clusters;
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.N),sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.pi),sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.constant),sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.avgvar),sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.means),sizeof(float)*num_dimensions*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.R),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.memberships),sizeof(float)*num_events*original_num_clusters));
       
        // Allocate a struct on the device 
        clusters_t* d_clusters;
        CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(clusters_t)));
        DEBUG("Finished allocating memory on device for clusters.\n");
        
        // Copy Cluster data to device
        CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));
        DEBUG("Finished copying cluster data to device.\n");
        
        // determine how many events this gpu will handle
        int events_per_gpu = num_events / total_num_gpus;
        int my_num_events = events_per_gpu;
        if(rank == total_num_gpus-1) {
            my_num_events += num_events % total_num_gpus; // last gpu has to handle the remaining uneven events
        }

        DEBUG("GPU %d:%d will handle %d events\n",rank,tid,my_num_events);

        // Temporary array, holds memberships from device before putting it in the shared clusters.memberships        
        float* temp_memberships = (float*) malloc(sizeof(float)*my_num_events*original_num_clusters);

        // allocate device memory for FCS data
        float* d_fcs_data_by_event;
        float* d_fcs_data_by_dimension;
        
        // allocate and copy relavant FCS data to device.
        int mem_size = num_dimensions * my_num_events * sizeof(float);
        CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
        CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));

        int gpu_num = rank*num_gpus+tid;

        CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, &fcs_data_by_event[num_dimensions*events_per_gpu*gpu_num], mem_size,cudaMemcpyHostToDevice) );

        // Copying the transposed data is trickier since it's not all contigious for the relavant events
        float* temp_fcs_data = (float*) malloc(mem_size);
        for(int d=0; d < num_dimensions; d++) {
            memcpy(&temp_fcs_data[d*my_num_events],&fcs_data_by_dimension[d*num_events + gpu_num*events_per_gpu],sizeof(float)*my_num_events);
        }
        CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, temp_fcs_data, mem_size,cudaMemcpyHostToDevice) );
        cudaThreadSynchronize();
        free(temp_fcs_data);
 
        DEBUG("GPU %d:%d on  Finished copying FCS data to device.\n",rank,tid);
        stopTimer(timers.memcpy);
        
        //////////////// Initialization done, starting kernels //////////////// 
        DEBUG("Invoking seed_clusters kernel...");

        // seed_clusters sets initial pi values, 
        // finds the means / covariances and copies it to all the clusters
        // TODO: Does it make any sense to use multiple blocks for this?
        #pragma omp master 
        {
            // TODO: seed_clusters can't be done on gpu since it doesnt have all the events
            //  Just have host pick random events for the means and use identity matrix for covariance
            //  Only tricky part is how to do average variance? 
            //   Make a kernel for that and reduce on host like the means/covariance?
            startTimer(timers.constants);
            seed_clusters<<< 1, NUM_THREADS_MSTEP >>>( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, my_num_events);
            cudaThreadSynchronize();
            DEBUG("done.\n"); 
            CUT_CHECK_ERROR("Seed Kernel execution failed: ");
            
            DEBUG("Invoking constants kernel...",NUM_THREADS);
            // Computes the R matrix inverses, and the gaussian constant
            constants_kernel<<<original_num_clusters, NUM_THREADS_MSTEP >>>(d_clusters,original_num_clusters,num_dimensions);
            constants_iterations++;
            cudaThreadSynchronize();
            CUT_CHECK_ERROR("Constants Kernel execution failed: ");
            DEBUG("done.\n");
            stopTimer(timers.constants);
        
            startTimer(timers.memcpy);
            // copy clusters from the device
            CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
            // copy all of the arrays from the structs
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].N, temp_clusters.N, sizeof(float)*original_num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].pi, temp_clusters.pi, sizeof(float)*original_num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].constant, temp_clusters.constant, sizeof(float)*original_num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].avgvar, temp_clusters.avgvar, sizeof(float)*original_num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].means, temp_clusters.means, sizeof(float)*num_dimensions*original_num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[0].Rinv, temp_clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters,cudaMemcpyDeviceToHost));
            //CUDA_SAFE_CALL(cudaMemcpy(clusters[0].p, temp_clusters[i].p, sizeof(float)*num_events,cudaMemcpyDeviceToHost));
            stopTimer(timers.memcpy);

            startTimer(timers.cpu);
            seed_clusters(&clusters[0],fcs_data_by_event,original_num_clusters,num_dimensions,num_events);
            DEBUG("Starting Clusters\n");
            for(int c=0; c < original_num_clusters; c++) {
                DEBUG("Cluster #%d\n",c);

                DEBUG("\tN: %f\n\tpi: %f\n\tconstant: %f\n\tavgvar: %f\n",clusters[0].N[c],clusters[0].pi[c],clusters[0].constant[c],clusters[0].avgvar[c]); 

                // means
                DEBUG("\tMeans: ");
                for(int d=0; d < num_dimensions; d++) {
                    DEBUG("%.2f ",clusters[0].means[c*num_dimensions+d]);
                }
                DEBUG("\n");
            }
            stopTimer(timers.cpu);
        }

        // synchronize after first gpu does the seeding, copy result to all gpus
        #pragma omp barrier
        startTimer(timers.memcpy);
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.N, clusters[0].N, sizeof(float)*original_num_clusters,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.pi, clusters[0].pi, sizeof(float)*original_num_clusters,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.constant, clusters[0].constant, sizeof(float)*original_num_clusters,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.avgvar, clusters[0].avgvar, sizeof(float)*original_num_clusters,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means, clusters[0].means, sizeof(float)*num_dimensions*original_num_clusters,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.R, clusters[0].R, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.Rinv, clusters[0].Rinv, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters,cudaMemcpyHostToDevice));
        stopTimer(timers.memcpy);
       
        startTimer(timers.cpu); 
        // Calculate an epsilon value
        //int ndata_points = num_events*num_dimensions;
        float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.01;
        int iters;
        
        //epsilon = 1e-6;
        PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

        float* d_likelihoods;
        CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*NUM_BLOCKS));
        
        // Variables for GMM reduce order
        float distance, min_distance = 0.0;
        float rissanen;
        int min_c1, min_c2;
        float* d_c;
        stopTimer(timers.cpu); 
        startTimer(timers.memcpy);
        CUDA_SAFE_CALL(cudaMalloc((void**) &d_c, sizeof(float)));
        stopTimer(timers.memcpy);
        
        double mstart3, mfinish3;
 
        for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
            /*************** EM ALGORITHM *****************************/
            
            // do initial E-step
            // Calculates a cluster membership probability
            // for each event and each cluster.
            DEBUG("Invoking E-step kernels...");
            startTimer(timers.e_step);
            estep1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,my_num_events);
            estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,my_num_events,d_likelihoods);
            cudaThreadSynchronize();
            #pragma omp master 
            {
                regroup_iterations++;
            }
            DEBUG("done.\n");
            // check if kernel execution generated an error
            CUT_CHECK_ERROR("Kernel execution failed");
            stopTimer(timers.e_step);

            // Copy the likelihood totals from each block, sum them up to get a total
            startTimer(timers.memcpy);
            CUDA_SAFE_CALL(cudaMemcpy(&shared_likelihoods[tid*NUM_BLOCKS],d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
            stopTimer(timers.memcpy);
            #pragma omp barrier
            #pragma omp master 
            {
                startTimer(timers.cpu); 
                // Gather likelihoods from every thread on this node
                likelihood = 0.0;
                for(int i=0;i<NUM_BLOCKS*num_gpus;i++) {
                    likelihood += shared_likelihoods[i]; 
                }
                DEBUG("Node %d's Likelihood: %e\n",rank,likelihood);
                stopTimer(timers.cpu); 
                startTimer(timers.mpi); 
                // Master node gathers likelihoods from every node
                MPI_Allreduce(MPI_IN_PLACE,&likelihood,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
                DEBUG("Node %d Total Likelihood: %e\n",rank,likelihood);
                stopTimer(timers.mpi); 
            }
            startTimer(timers.cpu); 
            #pragma omp barrier
            stopTimer(timers.cpu); 

            float change = epsilon*2;
            
            PRINT("Performing EM algorithm on %d clusters.\n",num_clusters);
            iters = 0;
            // This is the iterative loop for the EM algorithm.
            // It re-estimates parameters, re-computes constants, and then regroups the events
            // These steps keep repeating until the change in likelihood is less than some epsilon        
            while(iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
                #pragma omp master 
                {
                    old_likelihood = likelihood;
                }
                
                DEBUG("Invoking reestimate_parameters (M-step) kernel...",NUM_THREADS);
                startTimer(timers.m_step);
                // This kernel computes a new N, pi isn't updated until compute_constants though
                mstep_N<<<num_clusters, NUM_THREADS_MSTEP>>>(d_clusters,num_dimensions,num_clusters,my_num_events);
                cudaThreadSynchronize();
                stopTimer(timers.m_step);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].N,temp_clusters.N,sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
                stopTimer(timers.memcpy);
                
                // TODO: figure out the omp reduction pragma...
                // Reduce N for all clusters, copy back to device
                #pragma omp barrier
                #pragma omp master 
                {
                    startTimer(timers.cpu);
                    // Reduce results on this node from each thread;
                    for(int g=1; g < num_gpus; g++) {
                        for(int c=0; c < num_clusters; c++) {
                            clusters[0].N[c] += clusters[g].N[c];
                        }
                    }
                    for(int c=0; c < num_clusters; c++) {
                        DEBUG("Node %d's Cluster %d: N = %f\n",rank,c,clusters[0].N[c]);
                    }
                    stopTimer(timers.cpu);
                    // Reduce results globally from each node and broadcast
                    startTimer(timers.mpi);
                    MPI_Allreduce(MPI_IN_PLACE,clusters[0].N,num_clusters,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
                    MPI_Barrier(MPI_COMM_WORLD);
                    for(int c=0; c < num_clusters; c++) {
                        DEBUG("Node %d's Cluster %d after MPI reduce: N = %f\n",rank,c,clusters[0].N[c]);
                    }
                    stopTimer(timers.mpi);
                }
                startTimer(timers.cpu);
                #pragma omp barrier
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.N,clusters[0].N,sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
                stopTimer(timers.memcpy);

                startTimer(timers.m_step);
                dim3 gridDim1(num_clusters,num_dimensions);
                mstep_means<<<gridDim1, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,my_num_events);
                cudaThreadSynchronize();
                stopTimer(timers.m_step);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].means,temp_clusters.means,sizeof(float)*num_clusters*num_dimensions,cudaMemcpyDeviceToHost));
                stopTimer(timers.memcpy);
                // Reduce means for all clusters, copy back to device
                #pragma omp barrier
                #pragma omp master 
                {
                    startTimer(timers.cpu);
                    // Reduce from threads within the node
                    for(int g=1; g < num_gpus; g++) {
                        for(int c=0; c < num_clusters; c++) {
                            for(int d=0; d < num_dimensions; d++) {
                                clusters[0].means[c*num_dimensions+d] += clusters[g].means[c*num_dimensions+d];
                            }
                        }
                    }
                    stopTimer(timers.cpu);
                    
                    // Reduce globally from each node and broadcast
                    startTimer(timers.mpi);
                    MPI_Allreduce(MPI_IN_PLACE,clusters[0].means,num_clusters*num_dimensions,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
                    MPI_Barrier(MPI_COMM_WORLD);
                    stopTimer(timers.mpi);
                    
                    startTimer(timers.cpu);
                    // Compute final mean by dividing by cluster size
                    for(int c=0; c < num_clusters; c++) {
                        DEBUG("Node %d Cluster %d  Means:",rank,c,clusters[0].N[c]);
                        for(int d=0; d < num_dimensions; d++) {
                            if(clusters[0].N[c] > 0.5f) {
                                clusters[0].means[c*num_dimensions+d] /= clusters[0].N[c];
                            } else {
                                clusters[0].means[c*num_dimensions+d] = 0.0f;
                            }
                            DEBUG(" %f",clusters[0].means[c*num_dimensions+d]);
                        }
                        DEBUG("\n");
                    }
                    stopTimer(timers.cpu);
                }
                startTimer(timers.cpu);
                #pragma omp barrier
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means,clusters[0].means,sizeof(float)*num_clusters*num_dimensions,cudaMemcpyHostToDevice));
                stopTimer(timers.memcpy);

                startTimer(timers.m_step);
                // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
                dim3 gridDim2(num_clusters,num_dimensions*(num_dimensions+1)/2);
                mstep_covariance1<<<gridDim2, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,my_num_events);
                cudaThreadSynchronize();
                stopTimer(timers.m_step);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].R,temp_clusters.R,sizeof(float)*num_clusters*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
                stopTimer(timers.memcpy);
                // Reduce R for all clusters, copy back to device
                #pragma omp barrier
                #pragma omp master 
                {
                    startTimer(timers.cpu);
                    // Reduce from threads within the node
                    for(int g=1; g < num_gpus; g++) {
                        for(int c=0; c < num_clusters; c++) {
                            for(int d=0; d < num_dimensions*num_dimensions; d++) {
                                clusters[0].R[c*num_dimensions*num_dimensions+d] += clusters[g].R[c*num_dimensions*num_dimensions+d];
                            }
                        }
                    }
                    stopTimer(timers.cpu);
                    
                    // Reduce globally from each node and broadcast
                    startTimer(timers.mpi);
                    MPI_Allreduce(MPI_IN_PLACE,clusters[0].R,num_clusters*num_dimensions*num_dimensions,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
                    MPI_Barrier(MPI_COMM_WORLD);
                    stopTimer(timers.mpi);
                    
                    startTimer(timers.cpu);
                    for(int c=0; c < num_clusters; c++) {
                        if(clusters[0].N[c] > 0.5f) {
                            for(int d=0; d < num_dimensions*num_dimensions; d++) {
                                clusters[0].R[c*num_dimensions*num_dimensions+d] /= clusters[0].N[c];
                            }
                        } else {
                            for(int i=0; i < num_dimensions; i++) {
                                for(int j=0; j < num_dimensions; j++) {
                                    if(i == j) {
                                        clusters[0].R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 1.0;
                                    } else {
                                        clusters[0].R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                    stopTimer(timers.cpu);
                }
                startTimer(timers.cpu);
                #pragma omp barrier
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.R,clusters[0].R,sizeof(float)*num_clusters*num_dimensions*num_dimensions,cudaMemcpyHostToDevice));
                stopTimer(timers.memcpy);
                
                cudaThreadSynchronize();
                CUT_CHECK_ERROR("M-step Kernel execution failed: ");
                #pragma omp master 
                {
                    params_iterations++;
                }
                DEBUG("done.\n");
                //return 0; // RETURN FOR FASTER PROFILING
                
                DEBUG("Invoking constants kernel...",NUM_THREADS);
                // Inverts the R matrices, computes the constant, normalizes cluster probabilities
                startTimer(timers.constants);
                constants_kernel<<<num_clusters, NUM_THREADS_MSTEP>>>(d_clusters,num_clusters,num_dimensions);
                cudaThreadSynchronize();
                stopTimer(timers.constants);
                CUT_CHECK_ERROR("Constants Kernel execution failed: ");
                if(tid ==0){
                    constants_iterations++;
                }
                DEBUG("done.\n");

                DEBUG("Invoking regroup (E-step) kernel with %d blocks...",NUM_BLOCKS);
                startTimer(timers.e_step);
                // Compute new cluster membership probabilities for all the events
                estep1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,my_num_events);
                estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,my_num_events,d_likelihoods);
                cudaThreadSynchronize();
                CUT_CHECK_ERROR("E-step Kernel execution failed: ");
                stopTimer(timers.e_step);
                #pragma omp master 
                {
                    regroup_iterations++;
                }
                DEBUG("done.\n");
            
                // check if kernel execution generated an error
                CUT_CHECK_ERROR("Kernel execution failed");
            
                // Copy the likelihood totals from each block, sum them up to get a total
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(&shared_likelihoods[tid*NUM_BLOCKS],d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
                stopTimer(timers.memcpy);
                #pragma omp barrier
                startTimer(timers.cpu);
                #pragma omp master 
                {
                    // Gather likelihoods from every thread on this node
                    likelihood = 0.0;
                    for(int i=0;i<NUM_BLOCKS*num_gpus;i++) {
                        likelihood += shared_likelihoods[i]; 
                    }
                    DEBUG("Node %d's Likelihood: %e\n",rank,likelihood);
                    MPI_Allreduce(MPI_IN_PLACE ,&likelihood,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
                    DEBUG("Node %d Total Likelihood: %e\n",rank,likelihood);
                }
                // Master node gathers likelihoods from every node
                stopTimer(timers.cpu);
                #pragma omp barrier // synchronize for likelihood
                
                change = likelihood - old_likelihood;
                DEBUG("Node %d GPU %d, Change in likelihood: %e\n",rank,tid,change);

                iters++;
                #pragma omp barrier // synchronize loop iteration
                mfinish3 = MPI_Wtime();
                DEBUG("EM Loop iteration time: %f\n",mfinish3-mstart3);
            }

            DEBUG("GPU %d done with EM loop\n",tid);
            
            startTimer(timers.memcpy);
            // copy all of the arrays from the device
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].N, temp_clusters.N, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].pi, temp_clusters.pi, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].constant, temp_clusters.constant, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].avgvar, temp_clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].means, temp_clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[tid].Rinv, temp_clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(temp_memberships, temp_clusters.memberships, sizeof(float)*my_num_events*num_clusters,cudaMemcpyDeviceToHost));
            stopTimer(timers.memcpy);
            // Reduce membership values from each thread
            startTimer(timers.cpu);
            for(int c=0; c < num_clusters; c++) {
                memcpy(&(clusters[0].memberships[c*num_events+gpu_num*events_per_gpu]),&(temp_memberships[c*my_num_events]),sizeof(float)*my_num_events);
            }
            #pragma omp barrier
            DEBUG("GPU %d done with copying cluster data from device\n",tid); 
            
            // Gather membership values from each node
            // This isn't strictly neccesary unless we're saving the result to saved_clusters
            int memberships_per_node = my_num_events*num_gpus*num_clusters;
            stopTimer(timers.cpu);
 
            if(num_clusters > stop_number || ENABLE_OUTPUT) {
                #pragma omp master
                {
                    startTimer(timers.cpu);
                    mstart3 = MPI_Wtime();
                    float* temp = (float*) malloc(sizeof(float)*num_events*num_clusters);
                    memset(temp,0,sizeof(float)*num_events*num_clusters);
                    for(int e=0; e <num_events; e++) {
                        for(int c=0; c<num_clusters; c++) {
                            temp[e*num_clusters+c] = clusters[0].memberships[c*num_events+e];
                        }
                    }
                    memcpy(clusters[0].memberships,temp,sizeof(float)*num_events*num_clusters);
                    stopTimer(timers.cpu);
                    
                    startTimer(timers.mpi);
                    //MPI_Gather(clusters[0].memberships,memberships_per_node,MPI_FLOAT,temp,memberships_per_node,MPI_FLOAT,0,MPI_COMM_WORLD);
                    if(rank == 0) {
                        for(int i=1; i < num_nodes; i++) {
                            MPI_Status s;
                            MPI_Recv(&(temp[memberships_per_node*i]),memberships_per_node,MPI_FLOAT,i,1,MPI_COMM_WORLD,&s);
                        }
                    } else {
                        MPI_Send(&(clusters[0].memberships[memberships_per_node*rank]),memberships_per_node,MPI_FLOAT,0,1,MPI_COMM_WORLD);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    stopTimer(timers.mpi);
                    startTimer(timers.cpu);
                    if(rank == 0) {
                        for(int e=0; e < num_events; e++) {
                            for(int c=0; c<num_clusters; c++) {
                                clusters[0].memberships[c*num_events+e] = temp[e*num_clusters+c];
                            }
                        }    
                    }
                    free(temp);
                    mfinish3 = MPI_Wtime();
                    printf("Membership gathering time: %f\n",mfinish3-mstart3);
                    stopTimer(timers.cpu);
                }
            }

            // Calculate Rissanen Score
            rissanen = -likelihood + 0.5*(num_clusters*(1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)-1)*logf((float)num_events*num_dimensions);
            PRINT("\nRissanen Score: %e\n",rissanen);
            
            mfinish = MPI_Wtime();
            
            #pragma omp barrier
            if(rank == 0) {
                #pragma omp master 
                { 
                    startTimer(timers.cpu);
                    // Save the cluster data the first time through, so we have a base rissanen score and result
                    // Save the cluster data if the solution is better and the user didn't specify a desired number
                    // If the num_clusters equals the desired number, stop
                    if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
                        min_rissanen = rissanen;
                        ideal_num_clusters = num_clusters;
                        // Save the cluster configuration somewhere
                        memcpy(saved_clusters.N,clusters[0].N,sizeof(float)*num_clusters);
                        memcpy(saved_clusters.pi,clusters[0].pi,sizeof(float)*num_clusters);
                        memcpy(saved_clusters.constant,clusters[0].constant,sizeof(float)*num_clusters);
                        memcpy(saved_clusters.avgvar,clusters[0].avgvar,sizeof(float)*num_clusters);
                        memcpy(saved_clusters.means,clusters[0].means,sizeof(float)*num_dimensions*num_clusters);
                        memcpy(saved_clusters.R,clusters[0].R,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
                        memcpy(saved_clusters.Rinv,clusters[0].Rinv,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
                        memcpy(saved_clusters.memberships,clusters[0].memberships,sizeof(float)*num_events*num_clusters);
                    }
                    stopTimer(timers.cpu);
                }
            }
            #pragma omp barrier

            /**************** Reduce GMM Order ********************/
            startTimer(timers.reduce);
            // Don't want to reduce order on the last iteration
            if(num_clusters > stop_number) {
                if(rank == 0) {
                    #pragma omp master 
                    {
                        startTimer(timers.cpu);
                        // First eliminate any "empty" clusters 
                        for(int i=num_clusters-1; i >= 0; i--) {
                            if(clusters[0].N[i] < 0.5) {
                                DEBUG("Cluster #%d has less than 1 data point in it.\n",i);
                                for(int j=i; j < num_clusters-1; j++) {
                                    copy_cluster(clusters[0],j,clusters[0],j+1,num_dimensions);
                                }
                                num_clusters--;
                            }
                        }
                        
                        min_c1 = 0;
                        min_c2 = 1;
                        DEBUG("Number of non-empty clusters: %d\n",num_clusters); 
                        // For all combinations of subclasses...
                        // If the number of clusters got really big might need to do a non-exhaustive search
                        // Even with 100*99/2 combinations this doesn't seem to take too long
                        for(int c1=0; c1<num_clusters;c1++) {
                            for(int c2=c1+1; c2<num_clusters;c2++) {
                                // compute distance function between the 2 clusters
                                distance = cluster_distance(clusters[0],c1,c2,scratch_cluster,num_dimensions);
                                
                                // Keep track of minimum distance
                                if((c1 ==0 && c2 == 1) || distance < min_distance) {
                                    min_distance = distance;
                                    min_c1 = c1;
                                    min_c2 = c2;
                                }
                            }
                        }

                        PRINT("\nMinimum distance between (%d,%d). Combining clusters\n",min_c1,min_c2);
                        // Add the two clusters with min distance together
                        //add_clusters(&(clusters[min_c1]),&(clusters[min_c2]),scratch_cluster,num_dimensions);
                        add_clusters(clusters[0],min_c1,min_c2,scratch_cluster,num_dimensions);
                        // Copy new combined cluster into the main group of clusters, compact them
                        //copy_cluster(&(clusters[min_c1]),scratch_cluster,num_dimensions);
                        copy_cluster(clusters[0],min_c1,scratch_cluster,0,num_dimensions);
                        for(int i=min_c2; i < num_clusters-1; i++) {
                            //printf("Copying cluster %d to cluster %d\n",i+1,i);
                            //copy_cluster(&(clusters[i]),&(clusters[i+1]),num_dimensions);
                            copy_cluster(clusters[0],i,clusters[0],i+1,num_dimensions);
                        }
                        stopTimer(timers.cpu);
                    } // end master section
                } // end root section

                #pragma omp master
                {
                    startTimer(timers.cpu);
                    // Broadcast all the clusters params to the other nodes
                    MPI_Barrier(MPI_COMM_WORLD);
                    DEBUG("Broadcasting combined clusters\n");
                    MPI_Bcast(clusters[0].N,num_clusters,MPI_FLOAT,0,MPI_COMM_WORLD);
                    MPI_Bcast(clusters[0].pi,num_clusters,MPI_FLOAT,0,MPI_COMM_WORLD);
                    MPI_Bcast(clusters[0].constant,num_clusters,MPI_FLOAT,0,MPI_COMM_WORLD);
                    MPI_Bcast(clusters[0].avgvar,num_clusters,MPI_FLOAT,0,MPI_COMM_WORLD);
                    MPI_Bcast(clusters[0].means,num_clusters*num_dimensions,MPI_FLOAT,0,MPI_COMM_WORLD);
                    MPI_Bcast(clusters[0].R,num_clusters*num_dimensions*num_dimensions,MPI_FLOAT,0,MPI_COMM_WORLD);
                    MPI_Bcast(clusters[0].Rinv,num_clusters*num_dimensions*num_dimensions,MPI_FLOAT,0,MPI_COMM_WORLD);
                    DEBUG("Done\n");
                    MPI_Barrier(MPI_COMM_WORLD);
                    stopTimer(timers.cpu);
                }
                
                #pragma omp barrier

                startTimer(timers.memcpy);
                // Copy the clusters back to the device
                DEBUG("Copying clusters to device\n");
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.N, clusters[0].N, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.pi, clusters[0].pi, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.constant, clusters[0].constant, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.avgvar, clusters[0].avgvar, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means, clusters[0].means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.R, clusters[0].R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.Rinv, clusters[0].Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
                stopTimer(timers.memcpy);

                startTimer(timers.cpu);
                for(int c=0; c < num_clusters; c++) {
                    memcpy(&temp_memberships[c*my_num_events],&(clusters[0].memberships[c*num_events+gpu_num*(num_events/num_gpus)]),sizeof(float)*my_num_events);
                }
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.memberships,temp_memberships, sizeof(float)*my_num_events*num_clusters,cudaMemcpyHostToDevice));
                stopTimer(timers.memcpy);
            } // GMM reduction block 
            stopTimer(timers.reduce);
            #pragma omp master 
            {
                reduce_iterations++;
            }

            #pragma omp barrier
        } // outer loop from M to 1 clusters
        if(rank == 0) {
            PRINT("\nFinal rissanen score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);
        }
        #pragma omp barrier 
    
        // Print some profiling information
        printf("Node %02d GPU %d:\n\tE-step Kernel:\t%7.4f\t%d\t%7.4f\n\tM-step Kernel:\t%7.4f\t%d\t%7.4f\n\tConsts Kernel:\t%7.4f\t%d\t%7.4f\n\tOrder Reduce:\t%7.4f\t%d\t%7.4f\n\tGPU Memcpy:\t%7.4f\n\tCPU:\t\t%7.4f\n\tMPI:\t\t%7.4f\n",rank,tid,getTimerValue(timers.e_step) / 1000.0,regroup_iterations, (double) getTimerValue(timers.e_step) / (double) regroup_iterations / 1000.0,getTimerValue(timers.m_step) / 1000.0,params_iterations, (double) getTimerValue(timers.m_step) / (double) params_iterations / 1000.0,getTimerValue(timers.constants) / 1000.0,constants_iterations, (double) getTimerValue(timers.constants) / (double) constants_iterations / 1000.0, getTimerValue(timers.reduce) / 1000.0,reduce_iterations, (double) getTimerValue(timers.reduce) / (double) reduce_iterations / 1000.0, getTimerValue(timers.memcpy) / 1000.0, getTimerValue(timers.cpu) / 1000.0, getTimerValue(timers.mpi) / 1000.0);

        cleanup_profile_t(&timers);

        free(scratch_cluster.N);
        free(scratch_cluster.pi);
        free(scratch_cluster.constant);
        free(scratch_cluster.avgvar);
        free(scratch_cluster.means);
        free(scratch_cluster.R);
        free(scratch_cluster.Rinv);
        free(scratch_cluster.memberships);
        free(temp_memberships);   
     
        // cleanup GPU memory
        CUDA_SAFE_CALL(cudaFree(d_likelihoods));
     
        CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
        CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));

        CUDA_SAFE_CALL(cudaFree(temp_clusters.N));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.pi));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.constant));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.avgvar));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.means));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.R));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.Rinv));
        CUDA_SAFE_CALL(cudaFree(temp_clusters.memberships));
        CUDA_SAFE_CALL(cudaFree(d_clusters));
    } // end of parallel block

    if(rank == 0) { 
        cutStartTimer(timer_io);
        char* result_suffix = ".results";
        char* summary_suffix = ".summary";
        int filenamesize1 = strlen(argv[3]) + strlen(result_suffix) + 1;
        int filenamesize2 = strlen(argv[3]) + strlen(summary_suffix) + 1;
        char* result_filename = (char*) malloc(filenamesize1);
        char* summary_filename = (char*) malloc(filenamesize2);
        strcpy(result_filename,argv[3]);
        strcpy(summary_filename,argv[3]);
        strcat(result_filename,result_suffix);
        strcat(summary_filename,summary_suffix);
        
        PRINT("Summary filename: %s\n",summary_filename);
        PRINT("Results filename: %s\n",result_filename);
        
        // Open up the output file for cluster summary
        FILE* outf = fopen(summary_filename,"w");
        if(!outf) {
            printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
            return -1;
        }

        // Print the clusters with the lowest rissanen score to the console and output file
        for(int c=0; c<ideal_num_clusters; c++) {
            //if(saved_clusters.N[c] == 0.0) {
            //    continue;
            //}
            if(ENABLE_PRINT) {
                // Output the final cluster stats to the console
                PRINT("Cluster #%d\n",c);
                printCluster(saved_clusters,c,num_dimensions);
                PRINT("\n\n");
            }

            if(ENABLE_OUTPUT) {
                // Output the final cluster stats to the output file        
                fprintf(outf,"Cluster #%d\n",c);
                writeCluster(outf,saved_clusters,c,num_dimensions);
                fprintf(outf,"\n\n");
            }
        }
        fclose(outf);
       
        if(ENABLE_OUTPUT) { 
            // Open another output file for the event level clustering results
            FILE* fresults = fopen(result_filename,"w");
            
            for(int i=0; i<num_events; i++) {
                for(int d=0; d<num_dimensions-1; d++) {
                    fprintf(fresults,"%f,",fcs_data_by_event[i*num_dimensions+d]);
                }
                fprintf(fresults,"%f",fcs_data_by_event[i*num_dimensions+num_dimensions-1]);
                fprintf(fresults,"\t");
                for(int c=0; c<ideal_num_clusters-1; c++) {
                    fprintf(fresults,"%f,",saved_clusters.memberships[c*num_events+i]);
                }
                fprintf(fresults,"%f",saved_clusters.memberships[(ideal_num_clusters-1)*num_events+i]);
                fprintf(fresults,"\n");
            }
            fclose(fresults);
        }
        cutStopTimer(timer_io);
    }
 
    cutStartTimer(timer_cpu);
    
    // cleanup host memory
    free(fcs_data_by_event);
    free(fcs_data_by_dimension);
    
    for(int g=0; g < num_gpus; g++) {
        free(clusters[g].N);
        free(clusters[g].pi);
        free(clusters[g].constant);
        free(clusters[g].avgvar);
        free(clusters[g].means);
        free(clusters[g].R);
        free(clusters[g].Rinv);
        free(clusters[g].memberships);
    }

    free(saved_clusters.N);
    free(saved_clusters.pi);
    free(saved_clusters.constant);
    free(saved_clusters.avgvar);
    free(saved_clusters.means);
    free(saved_clusters.R);
    free(saved_clusters.Rinv);
    free(saved_clusters.memberships);
    
    free(shared_likelihoods);
    
    cutStopTimer(timer_cpu);
    
    printf( "Node %02d I/O time: %f (ms)\n", rank, cutGetTimerValue(timer_io));
    cutDeleteTimer(timer_io);
    
    printf( "Node %02d Main Thread CPU time: %f (ms)\n", rank, cutGetTimerValue(timer_cpu));
    cutDeleteTimer(timer_cpu);

    cutStopTimer(timer_total);
    printf( "Node %02d Total time: %f (ms)\n", rank, cutGetTimerValue(timer_total));
    cutDeleteTimer(timer_total);

    MPI_Finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Validate command line arguments
///////////////////////////////////////////////////////////////////////////////
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters) {
    if(argc <= 5 && argc >= 4) {
        // parse num_clusters
        if(!sscanf(argv[1],"%d",num_clusters)) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        } 
        
        // Check bounds for num_clusters
        if(*num_clusters < 1 || *num_clusters > MAX_CLUSTERS) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        }
        
        // parse infile
        FILE* infile = fopen(argv[2],"r");
        if(!infile) {
            printf("Invalid infile.\n\n");
            printUsage(argv);
            return 2;
        } 
        
        // parse outfile
        //FILE* outfile = fopen(argv[3],"w");
        //if(!outfile) {
        //    printf("Unable to create output file.\n\n");
        //    printUsage(argv);
        //    return 3;
        //}        
        // parse target_num_clusters
        if(argc == 5) {
            if(!sscanf(argv[4],"%d",target_num_clusters)) {
                printf("Invalid number of desired clusters.\n\n");
                printUsage(argv);
                return 4;
            }
            if(*target_num_clusters > *num_clusters) {
                printf("target_num_clusters must be less than equal to num_clusters\n\n");
                printUsage(argv);
                return 4;
            }
        } else {
            *target_num_clusters = 0;
        }
        
        // Clean up so the EPA is happy
        fclose(infile);
        //fclose(outfile);
        return 0;
    } else {
        printUsage(argv);
        return 1;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Print usage statement
///////////////////////////////////////////////////////////////////////////////
void printUsage(char** argv)
{
   printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n",argv[0]);
   printf("\t num_clusters: The number of starting clusters\n");
   printf("\t infile: ASCII space-delimited FCS data file\n");
   printf("\t outfile: Clustering results output file\n");
   printf("\t target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters\n");
}

void writeCluster(FILE* f, clusters_t clusters, int c, int num_dimensions) {
    fprintf(f,"Probability: %f\n", clusters.pi[c]);
    fprintf(f,"N: %f\n",clusters.N[c]);
    fprintf(f,"Means: ");
    for(int i=0; i<num_dimensions; i++){
        fprintf(f,"%.3f ",clusters.means[c*num_dimensions+i]);
    }
    fprintf(f,"\n");

    fprintf(f,"\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%.3f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    }
    fflush(f);   
}

void printCluster(clusters_t clusters, int c, int num_dimensions) {
    writeCluster(stdout,clusters,c,num_dimensions);
}

float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
    // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
    add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);
    
    return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] - temp_cluster.N[0]*temp_cluster.constant[0];
}

void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
    float wt1,wt2;
 
    wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
    wt2 = 1.0f - wt1;
    
    // Compute new weighted means
    for(int i=0; i<num_dimensions;i++) {
        temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
    }
    
    // Compute new weighted covariance
    for(int i=0; i<num_dimensions; i++) {
        for(int j=i; j<num_dimensions; j++) {
            // Compute R contribution from cluster1
            temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
                                                *(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
                                                +clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
            // Add R contribution from cluster2
            temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
                                                    *(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
                                                    +clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
            // Because its symmetric...
            temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
        }
    }
    
    // Compute pi
    temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];
    
    // compute N
    temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

    float log_determinant;
    // Copy R to Rinv matrix
    memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
    // Invert the matrix
    invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
    // Compute the constant
    temp_cluster.constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
    // avgvar same for all clusters
    temp_cluster.avgvar[0] = clusters.avgvar[0];
}

void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions) {
    dest.N[c_dest] = src.N[c_src];
    dest.pi[c_dest] = src.pi[c_src];
    dest.constant[c_dest] = src.constant[c_src];
    dest.avgvar[c_dest] = src.avgvar[c_src];
    memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
    memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
    memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
    // do we need to copy memberships?
}
