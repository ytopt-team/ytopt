#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 19;
	int mype = 0;
	double omp_start, omp_end;
	int nprocs = 1;
	unsigned long long verification;

	#ifdef MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif

	// Process CLI Fields -- store in "Inputs" structure
	Inputs in = read_CLI( argc, argv );

	// Set number of OpenMP Threads
	#ifdef OPENMP
	omp_set_num_threads(in.nthreads); 
	#endif

	// Print-out of Input Summary
	if( mype == 0 )
		print_inputs( in, nprocs, version );

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// This is not reflective of a real Monte Carlo simulation workload,
	// therefore, do not profile this region!
	// =====================================================================
	
	SimulationData SD;

	// If read from file mode is selected, skip initialization and load
	// all simulation data structures from file instead
	if( in.binary_mode == READ )
		SD = binary_read(in);
	else
		SD = grid_init_do_not_profile( in, mype );

	// If writing from file mode is selected, write all simulation data
	// structures to file
	if( in.binary_mode == WRITE && mype == 0 )
		binary_write(in, SD);


	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation
	// This is the section that should be profiled, as it reflects a 
	// realistic continuous energy Monte Carlo macroscopic cross section
	// lookup kernel.
	// =====================================================================

	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	// Start Simulation Timer
	omp_start = get_time();

	// Run simulation
	if( in.simulation_method == EVENT_BASED )
	{
		if( in.kernel_id == 0 )
			verification = run_event_based_simulation(in, SD, mype);
		else if( in.kernel_id == 1 )
			verification = run_event_based_simulation_optimization_1(in, SD, mype);
		else
		{
			printf("Error: No kernel ID %d found!\n", in.kernel_id);
			exit(1);
		}
	}
	else
		verification = run_history_based_simulation(in, SD, mype);

	if( mype == 0)	
	{	
		printf("\n" );
		printf("Simulation complete.\n" );
	}

	// End Simulation Timer
	omp_end = get_time();

	// =====================================================================
	// Output Results & Finalize
	// =====================================================================

	// Final Hash Step
	verification = verification % 999983;

	// Print / Save Results and Exit
	int is_invalid_result = print_results( in, mype, omp_end-omp_start, nprocs, verification );

	#ifdef MPI
	MPI_Finalize();
	#endif

	return is_invalid_result;
}

//io.c

// Prints program logo
void logo(int version)
{
	border_print();
	printf(
	"                   __   __ ___________                 _                        \n"
	"                   \\ \\ / //  ___| ___ \\               | |                       \n"
	"                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
	"                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
	"                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
	"                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n"
    );
	border_print();
	center_print("Developed at Argonne National Laboratory", 79);
	char v[100];
	sprintf(v, "Version: %d", version);
	center_print(v, 79);
	border_print();
}

// Prints Section titles in center of 80 char terminal
void center_print(const char *s, int width)
{
	int length = strlen(s);
	int i;
	for (i=0; i<=(width-length)/2; i++) {
		fputs(" ", stdout);
	}
	fputs(s, stdout);
	fputs("\n", stdout);
}

int print_results( Inputs in, int mype, double runtime, int nprocs,
	unsigned long long vhash )
{
	// Calculate Lookups per sec
	int lookups = 0;
	if( in.simulation_method == HISTORY_BASED )
		lookups = in.lookups * in.particles;
	else if( in.simulation_method == EVENT_BASED )
		lookups = in.lookups;
	int lookups_per_sec = (int) ((double) lookups / runtime);
	
	// If running in MPI, reduce timing statistics and calculate average
	#ifdef MPI
	int total_lookups = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&lookups_per_sec, &total_lookups, 1, MPI_INT,
	           MPI_SUM, 0, MPI_COMM_WORLD);
	#endif

	int is_invalid_result = 1;
	
	// Print output
	if( mype == 0 )
	{
		border_print();
		center_print("RESULTS", 79);
		border_print();

		// Print the results
		printf("Threads:     %d\n", in.nthreads);
		#ifdef MPI
		printf("MPI ranks:   %d\n", nprocs);
		#endif
		#ifdef MPI
		printf("Total Lookups/s:            ");
		fancy_int(total_lookups);
		printf("Avg Lookups/s per MPI rank: ");
		fancy_int(total_lookups / nprocs);
		#else
		printf("Runtime:     %.3lf seconds\n", runtime);
		printf("Lookups:     "); fancy_int(lookups);
		printf("Lookups/s:   ");
		fancy_int(lookups_per_sec);
		#endif
	}

	unsigned long long large = 0;
	unsigned long long small = 0; 
	if( in.simulation_method == EVENT_BASED )
	{
		small = 945990;
		large = 952131;
	}
	else if( in.simulation_method == HISTORY_BASED )
	{
		small = 941535;
		large = 954318; 
	}
	if( strcmp(in.HM, "large") == 0 )
	{
		if( vhash == large )
			is_invalid_result = 0;
	}
	else if( strcmp(in.HM, "small") == 0 )
	{
		if( vhash == small )
			is_invalid_result = 0;
	}

	if(mype == 0 )
	{
		if( is_invalid_result )
			printf("Verification checksum: %llu (WARNING - INAVALID CHECKSUM!)\n", vhash);
		else
			printf("Verification checksum: %llu (Valid)\n", vhash);
		border_print();
	}

	return is_invalid_result;
}

void print_inputs(Inputs in, int nprocs, int version )
{
	// Calculate Estimate of Memory Usage
	int mem_tot = estimate_mem_usage( in );
	logo(version);
	center_print("INPUT SUMMARY", 79);
	border_print();
	if( in.simulation_method == EVENT_BASED )
		printf("Simulation Method:            Event Based\n");
	else
		printf("Simulation Method:            History Based\n");
	if( in.grid_type == NUCLIDE )
		printf("Grid Type:                    Nuclide Grid\n");
	else if( in.grid_type == UNIONIZED )
		printf("Grid Type:                    Unionized Grid\n");
	else
		printf("Grid Type:                    Hash\n");

	printf("Materials:                    %d\n", 12);
	printf("H-M Benchmark Size:           %s\n", in.HM);
	printf("Total Nuclides:               %ld\n", in.n_isotopes);
	printf("Gridpoints (per Nuclide):     ");
	fancy_int(in.n_gridpoints);
	if( in.grid_type == HASH )
	{
		printf("Hash Bins:                    ");
		fancy_int(in.hash_bins);
	}
	if( in.grid_type == UNIONIZED )
	{
		printf("Unionized Energy Gridpoints:  ");
		fancy_int(in.n_isotopes*in.n_gridpoints);
	}
	if( in.simulation_method == HISTORY_BASED )
	{
		printf("Particle Histories:           "); fancy_int(in.particles);
		printf("XS Lookups per Particle:      "); fancy_int(in.lookups);
	}
	printf("Total XS Lookups:             "); fancy_int(in.lookups);
	#ifdef MPI
	printf("MPI Ranks:                    %d\n", nprocs);
	printf("OMP Threads per MPI Rank:     %d\n", in.nthreads);
	printf("Mem Usage per MPI Rank (MB):  "); fancy_int(mem_tot);
	#else
	printf("Threads:                      %d\n", in.nthreads);
	printf("Est. Memory Usage (MB):       "); fancy_int(mem_tot);
	#endif
	printf("Binary File Mode:             ");
	if( in.binary_mode == NONE )
		printf("Off\n");
	else if( in.binary_mode == READ)
		printf("Read\n");
	else
		printf("Write\n");
	border_print();
	center_print("INITIALIZATION - DO NOT PROFILE", 79);
	border_print();
}

void border_print(void)
{
	printf(
	"==================================================================="
	"=============\n");
}

// Prints comma separated integers - for ease of reading
void fancy_int( long a )
{
    if( a < 1000 )
        printf("%ld\n",a);

    else if( a >= 1000 && a < 1000000 )
        printf("%ld,%03ld\n", a / 1000, a % 1000);

    else if( a >= 1000000 && a < 1000000000 )
        printf("%ld,%03ld,%03ld\n",a / 1000000,(a % 1000000) / 1000,a % 1000 );

    else if( a >= 1000000000 )
        printf("%ld,%03ld,%03ld,%03ld\n",
               a / 1000000000,
               (a % 1000000000) / 1000000,
               (a % 1000000) / 1000,
               a % 1000 );
    else
        printf("%ld\n",a);
}

void print_CLI_error(void)
{
	printf("Usage: ./XSBench <options>\n");
	printf("Options include:\n");
	printf("  -m <simulation method>   Simulation method (history, event)\n");
	printf("  -t <threads>             Number of OpenMP threads to run\n");
	printf("  -s <size>                Size of H-M Benchmark to run (small, large, XL, XXL)\n");
	printf("  -g <gridpoints>          Number of gridpoints per nuclide (overrides -s defaults)\n");
	printf("  -G <grid type>           Grid search type (unionized, nuclide, hash). Defaults to unionized.\n");
	printf("  -p <particles>           Number of particle histories\n");
	printf("  -l <lookups>             History Based: Number of Cross-section (XS) lookups per particle. Event Based: Total number of XS lookups.\n");
	printf("  -h <hash bins>           Number of hash bins (only relevant when used with \"-G hash\")\n");
	printf("  -b <binary mode>         Read or write all data structures to file. If reading, this will skip initialization phase. (read, write)\n");
	printf("  -k <kernel ID>           Specifies which kernel to run. 0 is baseline, 1, 2, etc are optimized variants. (0 is default.)\n");
	printf("Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized\n");
	printf("See readme for full description of default run values\n");
	exit(4);
}

Inputs read_CLI( int argc, char * argv[] )
{
	Inputs input;

	// defaults to the history based simulation method
	input.simulation_method = HISTORY_BASED;
	
	// defaults to max threads on the system	
	#ifdef OPENMP
	//input.nthreads = omp_get_num_procs();
	input.nthreads = #P0;
	#else
	input.nthreads = 1;
	#endif
	
	// defaults to 355 (corresponding to H-M Large benchmark)
	input.n_isotopes = 355;
	
	// defaults to 11303 (corresponding to H-M Large benchmark)
	input.n_gridpoints = 11303;

	// defaults to 500,000
	input.particles = 500000;
	
	// defaults to 34
	input.lookups = 34;
	
	// default to unionized grid
	input.grid_type = UNIONIZED;

	// default to unionized grid
	input.hash_bins = 10000;

	// default to no binary read/write
	input.binary_mode = NONE;
	
	// defaults to baseline kernel
	input.kernel_id = 0;
	
	// defaults to H-M Large benchmark
	input.HM = (char *) malloc( 6 * sizeof(char) );
	input.HM[0] = 'l' ; 
	input.HM[1] = 'a' ; 
	input.HM[2] = 'r' ; 
	input.HM[3] = 'g' ; 
	input.HM[4] = 'e' ; 
	input.HM[5] = '\0';
	
	// Check if user sets these
	int user_g = 0;

	int default_lookups = 1;
	int default_particles = 1;
	
	// Collect Raw Input
	for( int i = 1; i < argc; i++ )
	{
		char * arg = argv[i];

		// nthreads (-t)
		if( strcmp(arg, "-t") == 0 )
		{
			if( ++i < argc )
				input.nthreads = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// n_gridpoints (-g)
		else if( strcmp(arg, "-g") == 0 )
		{	
			if( ++i < argc )
			{
				user_g = 1;
				input.n_gridpoints = atol(argv[i]);
			}
			else
				print_CLI_error();
		}
		// Simulation Method (-m)
		else if( strcmp(arg, "-m") == 0 )
		{
			char * sim_type;
			if( ++i < argc )
				sim_type = argv[i];
			else
				print_CLI_error();

			if( strcmp(sim_type, "history") == 0 )
				input.simulation_method = HISTORY_BASED;
			else if( strcmp(sim_type, "event") == 0 )
			{
				input.simulation_method = EVENT_BASED;
				// Also resets default # of lookups
				if( default_lookups && default_particles )
				{
					input.lookups =  input.lookups * input.particles;
					input.particles = 0;
				}
			}
			else
				print_CLI_error();
		}
		// lookups (-l)
		else if( strcmp(arg, "-l") == 0 )
		{
			if( ++i < argc )
			{
				input.lookups = atoi(argv[i]);
				default_lookups = 0;
			}
			else
				print_CLI_error();
		}
		// hash bins (-h)
		else if( strcmp(arg, "-h") == 0 )
		{
			if( ++i < argc )
				input.hash_bins = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// particles (-p)
		else if( strcmp(arg, "-p") == 0 )
		{
			if( ++i < argc )
			{
				input.particles = atoi(argv[i]);
				default_particles = 0;
			}
			else
				print_CLI_error();
		}
		// HM (-s)
		else if( strcmp(arg, "-s") == 0 )
		{	
			if( ++i < argc )
				input.HM = argv[i];
			else
				print_CLI_error();
		}
		// grid type (-G)
		else if( strcmp(arg, "-G") == 0 )
		{
			char * grid_type;
			if( ++i < argc )
				grid_type = argv[i];
			else
				print_CLI_error();

			if( strcmp(grid_type, "unionized") == 0 )
				input.grid_type = UNIONIZED;
			else if( strcmp(grid_type, "nuclide") == 0 )
				input.grid_type = NUCLIDE;
			else if( strcmp(grid_type, "hash") == 0 )
				input.grid_type = HASH;
			else
				print_CLI_error();
		}
		// binary mode (-b)
		else if( strcmp(arg, "-b") == 0 )
		{
			char * binary_mode;
			if( ++i < argc )
				binary_mode = argv[i];
			else
				print_CLI_error();

			if( strcmp(binary_mode, "read") == 0 )
				input.binary_mode = READ;
			else if( strcmp(binary_mode, "write") == 0 )
				input.binary_mode = WRITE;
			else
				print_CLI_error();
		}
		// kernel optimization selection (-k)
		else if( strcmp(arg, "-k") == 0 )
		{
			if( ++i < argc )
			{
				input.kernel_id = atoi(argv[i]);
			}
			else
				print_CLI_error();
		}
		else
			print_CLI_error();
	}

	// Validate Input
	
	// Validate nthreads
	if( input.nthreads < 1 )
		print_CLI_error();
	
	// Validate n_isotopes
	if( input.n_isotopes < 1 )
		print_CLI_error();
	
	// Validate n_gridpoints
	if( input.n_gridpoints < 1 )
		print_CLI_error();

	// Validate lookups
	if( input.lookups < 1 )
		print_CLI_error();

	// Validate Hash Bins 
	if( input.hash_bins < 1 )
		print_CLI_error();
	
	// Validate HM size
	if( strcasecmp(input.HM, "small") != 0 &&
		strcasecmp(input.HM, "large") != 0 &&
		strcasecmp(input.HM, "XL") != 0 &&
		strcasecmp(input.HM, "XXL") != 0 )
		print_CLI_error();
	
	// Set HM size specific parameters
	// (defaults to large)
	if( strcasecmp(input.HM, "small") == 0 )
		input.n_isotopes = 68;
	else if( strcasecmp(input.HM, "XL") == 0 && user_g == 0 )
		input.n_gridpoints = 238847; // sized to make 120 GB XS data
	else if( strcasecmp(input.HM, "XXL") == 0 && user_g == 0 )
		input.n_gridpoints = 238847 * 2.1; // 252 GB XS data

	// Return input struct
	return input;
}

void binary_write( Inputs in, SimulationData SD )
{
	char * fname = "XS_data.dat";
	printf("Writing all data structures to binary file %s...\n", fname);
	FILE * fp = fopen(fname, "w");

	// Write SimulationData Object. Include pointers, even though we won't be using them.
	fwrite(&SD, sizeof(SimulationData), 1, fp);

	// Write heap arrays in SimulationData Object
	fwrite(SD.num_nucs,       sizeof(int), SD.length_num_nucs, fp);
	fwrite(SD.concs,          sizeof(double), SD.length_concs, fp);
	fwrite(SD.mats,           sizeof(int), SD.length_mats, fp);
	fwrite(SD.nuclide_grid,   sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp); 
	fwrite(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
	fwrite(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);

	fclose(fp);
}

SimulationData binary_read( Inputs in )
{
	SimulationData SD;
	
	char * fname = "XS_data.dat";
	printf("Reading all data structures from binary file %s...\n", fname);

	FILE * fp = fopen(fname, "r");
	assert(fp != NULL);

	// Read SimulationData Object. Include pointers, even though we won't be using them.
	fread(&SD, sizeof(SimulationData), 1, fp);

	// Allocate space for arrays on heap
	SD.num_nucs = (int *) malloc(SD.length_num_nucs * sizeof(int));
	SD.concs = (double *) malloc(SD.length_concs * sizeof(double));
	SD.mats = (int *) malloc(SD.length_mats * sizeof(int));
	SD.nuclide_grid = (NuclideGridPoint *) malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	SD.index_grid = (int *) malloc( SD.length_index_grid * sizeof(int));
	SD.unionized_energy_array = (double *) malloc( SD.length_unionized_energy_array * sizeof(double));

	// Read heap arrays into SimulationData Object
	fread(SD.num_nucs,       sizeof(int), SD.length_num_nucs, fp);
	fread(SD.concs,          sizeof(double), SD.length_concs, fp);
	fread(SD.mats,           sizeof(int), SD.length_mats, fp);
	fread(SD.nuclide_grid,   sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp); 
	fread(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
	fread(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);

	fclose(fp);

	return SD;
}
//Simulation.c
////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, XSBench will only run the baseline implementation. Optimized variants
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype)
{
	if( mype == 0)	
		printf("Beginning event based simulation...\n");
	
	////////////////////////////////////////////////////////////////////////////////
	// SUMMARY: Simulation Data Structure Manifest for "SD" Object
	// Here we list all heap arrays (and lengths) in SD that would need to be
	// offloaded manually if using an accelerator with a seperate memory space
	////////////////////////////////////////////////////////////////////////////////
	// int * num_nucs;                     // Length = length_num_nucs;
	// double * concs;                     // Length = length_concs
	// int * mats;                         // Length = length_mats
	// double * unionized_energy_array;    // Length = length_unionized_energy_array
	// int * index_grid;                   // Length = length_index_grid
	// NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
	// 
	// Note: "unionized_energy_array" and "index_grid" can be of zero length
	//        depending on lookup method.
	//
	// Note: "Lengths" are given as the number of objects in the array, not the
	//       number of bytes.
	////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////////////////////////
	// Begin Actual Simulation Loop 
	////////////////////////////////////////////////////////////////////////////////
	unsigned long long verification = 0;
	#pragma omp parallel for schedule(dynamic,#P1) reduction(+:verification)
	for( int i = 0; i < in.lookups; i++ )
	{
		// Set the initial seed value
		uint64_t seed = STARTING_SEED;	

		// Forward seed to lookup index (we need 2 samples per lookup)
		seed = fast_forward_LCG(seed, 2*i);

		// Randomly pick an energy and material for the particle
		double p_energy = LCG_random_double(&seed);
		int mat         = pick_mat(&seed); 

		double macro_xs_vector[5] = {0};

		// Perform macroscopic Cross Section Lookup
		calculate_macro_xs(
				p_energy,        // Sampled neutron energy (in lethargy)
				mat,             // Sampled material type index neutron is in
				in.n_isotopes,   // Total number of isotopes in simulation
				in.n_gridpoints, // Number of gridpoints per isotope in simulation
				SD.num_nucs,     // 1-D array with number of nuclides per material
				SD.concs,        // Flattened 2-D array with concentration of each nuclide in each material
				SD.unionized_energy_array, // 1-D Unionized energy array
				SD.index_grid,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
				SD.nuclide_grid, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
				SD.mats,         // Flattened 2-D array with nuclide indices defining composition of each type of material
				macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
				in.grid_type,    // Lookup type (nuclide, hash, or unionized)
				in.hash_bins,    // Number of hash bins used (if using hash lookup type)
				SD.max_num_nucs  // Maximum number of nuclides present in any material
				);

		// For verification, and to prevent the compiler from optimizing
		// all work out, we interrogate the returned macro_xs_vector array
		// to find its maximum value index, then increment the verification
		// value by that index. In this implementation, we prevent thread
		// contention by using an OMP reduction on the verification value.
		// For accelerators, a different approach might be required
		// (e.g., atomics, reduction of thread-specific values in large
		// array via CUDA thrust, etc).
		double max = -1.0;
		int max_idx = 0;
		for(int j = 0; j < 5; j++ )
		{
			if( macro_xs_vector[j] > max )
			{
				max = macro_xs_vector[j];
				max_idx = j;
			}
		}
		verification += max_idx+1;
	}

	return verification;
}

unsigned long long run_history_based_simulation(Inputs in, SimulationData SD, int mype)
{
	if( mype == 0)	
		printf("Beginning history based simulation...\n");

	
	////////////////////////////////////////////////////////////////////////////////
	// SUMMARY: Simulation Data Structure Manifest for "SD" Object
	// Here we list all heap arrays (and lengths) in SD that would need to be
	// offloaded manually if using an accelerator with a seperate memory space
	////////////////////////////////////////////////////////////////////////////////
	// int * num_nucs;                     // Length = length_num_nucs;
	// double * concs;                     // Length = length_concs
	// int * mats;                         // Length = length_mats
	// double * unionized_energy_array;    // Length = length_unionized_energy_array
	// int * index_grid;                   // Length = length_index_grid
	// NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
	// 
	// Note: "unionized_energy_array" and "index_grid" can be of zero length
	//        depending on lookup method.
	//
	// Note: "Lengths" are given as the number of objects in the array, not the
	//       number of bytes.
	////////////////////////////////////////////////////////////////////////////////

	unsigned long long verification = 0;

	// Begin outer lookup loop over particles. This loop is independent.
	#pragma omp parallel for schedule(dynamic, #P1) reduction(+:verification)
	for( int p = 0; p < in.particles; p++ )
	{
		// Set the initial seed value
		uint64_t seed = STARTING_SEED;	

		// Forward seed to lookup index (we need 2 samples per lookup, and
		// we may fast forward up to 5 times after each lookup)
		seed = fast_forward_LCG(seed, p*in.lookups*2*5);

		// Randomly pick an energy and material for the particle
		double p_energy = LCG_random_double(&seed);
		int mat         = pick_mat(&seed); 

		// Inner XS Lookup Loop
		// This loop is dependent!
		// i.e., Next iteration uses data computed in previous iter.
		for( int i = 0; i < in.lookups; i++ )
		{
			double macro_xs_vector[5] = {0};

			// Perform macroscopic Cross Section Lookup
			calculate_macro_xs(
					p_energy,        // Sampled neutron energy (in lethargy)
					mat,             // Sampled material type neutron is in
					in.n_isotopes,   // Total number of isotopes in simulation
					in.n_gridpoints, // Number of gridpoints per isotope in simulation
					SD.num_nucs,     // 1-D array with number of nuclides per material
					SD.concs,        // Flattened 2-D array with concentration of each nuclide in each material
					SD.unionized_energy_array, // 1-D Unionized energy array
					SD.index_grid,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
					SD.nuclide_grid, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
					SD.mats,         // Flattened 2-D array with nuclide indices for each type of material
					macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
					in.grid_type,    // Lookup type (nuclide, hash, or unionized)
					in.hash_bins,    // Number of hash bins used (if using hash lookups)
					SD.max_num_nucs  // Maximum number of nuclides present in any material
					);

		
			// For verification, and to prevent the compiler from optimizing
			// all work out, we interrogate the returned macro_xs_vector array
			// to find its maximum value index, then increment the verification
			// value by that index. In this implementation, we prevent thread
			// contention by using an OMP reduction on it. For other accelerators,
			// a different approach might be required (e.g., atomics, reduction
			// of thread-specific values in large array via CUDA thrust, etc)
			double max = -1.0;
			int max_idx = 0;
			for(int j = 0; j < 5; j++ )
			{
				if( macro_xs_vector[j] > max )
				{
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification += max_idx+1;

			// Randomly pick next energy and material for the particle
			// Also incorporates results from macro_xs lookup to
			// enforce loop dependency.
			// In a real MC app, this dependency is expressed in terms
			// of branching physics sampling, whereas here we are just
			// artificially enforcing this dependence based on fast
			// forwarding the LCG state
			uint64_t n_forward = 0;
			for( int j = 0; j < 5; j++ )
				if( macro_xs_vector[j] > 1.0 )
					n_forward++;
			if( n_forward > 0 )
				seed = fast_forward_LCG(seed, n_forward);

			p_energy = LCG_random_double(&seed);
			mat      = pick_mat(&seed); 
		}

	}
	return verification;
}

// Calculates the microscopic cross section for a given nuclide & energy
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           double * restrict egrid, int * restrict index_data,
                           NuclideGridPoint * restrict nuclide_grids,
                           long idx, double * restrict xs_vector, int grid_type, int hash_bins ){
	// Variables
	double f;
	NuclideGridPoint * low, * high;

	// If using only the nuclide grid, we must perform a binary search
	// to find the energy location in this particular nuclide's grid.
	if( grid_type == NUCLIDE )
	{
		// Perform binary search on the Nuclide Grid to find the index
		idx = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], 0, n_gridpoints-1);

		// pull ptr from nuclide grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( idx == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + idx - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + idx];
	}
	else if( grid_type == UNIONIZED) // Unionized Energy Grid - we already know the index, no binary search needed.
	{
		// pull ptr from energy grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc]];
	}
	else // Hash grid
	{
		// load lower bounding index
		int u_low = index_data[idx * n_isotopes + nuc];

		// Determine higher bounding index
		int u_high;
		if( idx == hash_bins - 1 )
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

		// Check edge cases to make sure energy is actually between these
		// Then, if things look good, search for gridpoint in the nuclide grid
		// within the lower and higher limits we've calculated.
		double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
		double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
		int lower;
		if( p_energy <= e_low )
			lower = 0;
		else if( p_energy >= e_high )
			lower = n_gridpoints - 1;
		else
			lower = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], u_low, u_high);

		if( lower == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + lower - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + lower];
	}
	
	high = low + 1;
	
	// calculate the re-useable interpolation factor
	f = (high->energy - p_energy) / (high->energy - low->energy);

	// Total XS
	xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
	
	// Elastic XS
	xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
	
	// Absorbtion XS
	xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
	
	// Fission XS
	xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
	
	// Nu Fission XS
	xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

// Calculates macroscopic cross section based on a given material & energy 
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int * restrict num_nucs,
                         double * restrict concs,
                         double * restrict egrid, int * restrict index_data,
                         NuclideGridPoint * restrict nuclide_grids,
                         int * restrict mats,
                         double * restrict macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
	int p_nuc; // the nuclide we are looking up
	long idx = -1;	
	double conc; // the concentration of the nuclide in the material

	// cleans out macro_xs_vector
	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	// If we are using the unionized energy grid (UEG), we only
	// need to perform 1 binary search per macroscopic lookup.
	// If we are using the nuclide grid search, it will have to be
	// done inside of the "calculate_micro_xs" function for each different
	// nuclide in the material.
	if( grid_type == UNIONIZED )
		idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);	
	else if( grid_type == HASH )
	{
		double du = 1.0 / hash_bins;
		idx = p_energy / du;
	}
	
	// Once we find the pointer array on the UEG, we can pull the data
	// from the respective nuclide grids, as well as the nuclide
	// concentration data for the material
	// Each nuclide from the material needs to have its micro-XS array
	// looked up & interpolatied (via calculate_micro_xs). Then, the
	// micro XS is multiplied by the concentration of that nuclide
	// in the material, and added to the total macro XS array.
	// (Independent -- though if parallelizing, must use atomic operations
	//  or otherwise control access to the xs_vector and macro_xs_vector to
	//  avoid simulataneous writing to the same data structure)
	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		double xs_vector[5];
		p_nuc = mats[mat*max_num_nucs + j];
		conc = concs[mat*max_num_nucs + j];
		calculate_micro_xs( p_energy, p_nuc, n_isotopes,
		                    n_gridpoints, egrid, index_data,
		                    nuclide_grids, idx, xs_vector, grid_type, hash_bins );
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}


// binary search for energy on unionized energy grid
// returns lower index
long grid_search( long n, double quarry, double * restrict A)
{
	long lowerLimit = 0;
	long upperLimit = n-1;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );
		
		if( A[examinationPoint] > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;
		
		length = upperLimit - lowerLimit;
	}
	
	return lowerLimit;
}

// binary search for energy on nuclide energy grid
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high)
{
	long lowerLimit = low;
	long upperLimit = high;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );
		
		if( A[examinationPoint].energy > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;
		
		length = upperLimit - lowerLimit;
	}
	
	return lowerLimit;
}

// picks a material based on a probabilistic distribution
int pick_mat( uint64_t * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a 
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	double dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies
	
	double roll = LCG_random_double(seed);

	// makes a pick based on the distro
	for( int i = 0; i < 12; i++ )
	{
		double running = 0;
		for( int j = i; j > 0; j-- )
			running += dist[j];
		if( roll < running )
			return i;
	}

	return 0;
}

double LCG_random_double(uint64_t * seed)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}	

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0) 
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;

}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED VARIANT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// This section contains a number of optimized variants of some of the above
// functions, which each deploy a different combination of optimizations strategies.
// By default, XSBench will not run any of these variants. They
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
//
// As fast parallel sorting will be required for these optimizations, we will
// first define a set of key-value parallel quicksort routines.
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////
// Parallel Quicksort Key-Value Sorting Algorithms
////////////////////////////////////////////////////////////////////////////////////
//
// These algorithms are based on the parallel quicksort implementation by
// Eduard Lopez published at https://github.com/eduardlopez/quicksort-parallel
//
// Eduard's original version was for an integer type quicksort, but I have modified
// it to form two different versions that can sort key-value pairs together without
// having to bundle them into a separate object. Additionally, I have modified the
// optimal chunk sizes and restricted the number of threads for the array sizing
// that XSBench will be using by default.
//
// Eduard's original implementation carries the following license, which applies to
// the following functions only:
//
//	void quickSort_parallel_internal_i_d(int* key,double * value, int left, int right, int cutoff) 
//  void quickSort_parallel_i_d(int* key,double * value, int lenArray, int numThreads)
//  void quickSort_parallel_internal_d_i(double* key,int * value, int left, int right, int cutoff)
//  void quickSort_parallel_d_i(double* key,int * value, int lenArray, int numThreads)
//
// The MIT License (MIT)
//
// Copyright (c) 2016 Eduard López
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////////
void quickSort_parallel_internal_i_d(int* key,double * value, int left, int right, int cutoff) 
{
	int i = left, j = right;
	int tmp;
	int pivot = key[(left + right) / 2];
	
	{
		while (i <= j) {
			while (key[i] < pivot)
				i++;
			while (key[j] > pivot)
				j--;
			if (i <= j) {
				tmp = key[i];
				key[i] = key[j];
				key[j] = tmp;
				double tmp_v = value[i];
				value[i] = value[j];
				value[j] = tmp_v;
				i++;
				j--;
			}
		}

	}

	if ( ((right-left)<cutoff) ){
		if (left < j){ quickSort_parallel_internal_i_d(key, value, left, j, cutoff); }			
		if (i < right){ quickSort_parallel_internal_i_d(key, value, i, right, cutoff); }

	}else{
		#pragma omp task 	
		{ quickSort_parallel_internal_i_d(key, value, left, j, cutoff); }
		#pragma omp task 	
		{ quickSort_parallel_internal_i_d(key, value, i, right, cutoff); }		
	}

}

void quickSort_parallel_i_d(int* key,double * value, int lenArray, int numThreads){

	// Set minumum problem size to still spawn threads for
	int cutoff = 10000;

	// For this problem size, more than 16 threads on CPU is not helpful
	if( numThreads > 16 )
		numThreads = 16;

	#pragma omp parallel num_threads(numThreads)
	{	
		#pragma omp single nowait
		{
			quickSort_parallel_internal_i_d(key,value, 0, lenArray-1, cutoff);	
		}
	}	

}

void quickSort_parallel_internal_d_i(double* key,int * value, int left, int right, int cutoff) 
{
	int i = left, j = right;
	double tmp;
	double pivot = key[(left + right) / 2];
	
	{
		while (i <= j) {
			while (key[i] < pivot)
				i++;
			while (key[j] > pivot)
				j--;
			if (i <= j) {
				tmp = key[i];
				key[i] = key[j];
				key[j] = tmp;
				int tmp_v = value[i];
				value[i] = value[j];
				value[j] = tmp_v;
				i++;
				j--;
			}
		}

	}

	if ( ((right-left)<cutoff) ){
		if (left < j){ quickSort_parallel_internal_d_i(key, value, left, j, cutoff); }			
		if (i < right){ quickSort_parallel_internal_d_i(key, value, i, right, cutoff); }

	}else{
		#pragma omp task 	
		{ quickSort_parallel_internal_d_i(key, value, left, j, cutoff); }
		#pragma omp task 	
		{ quickSort_parallel_internal_d_i(key, value, i, right, cutoff); }		
	}

}

void quickSort_parallel_d_i(double* key,int * value, int lenArray, int numThreads){

	// Set minumum problem size to still spawn threads for
	int cutoff = 10000;

	// For this problem size, more than 16 threads on CPU is not helpful
	if( numThreads > 16 )
		numThreads = 16;

	#pragma omp parallel num_threads(numThreads)
	{	
		#pragma omp single nowait
		{
			quickSort_parallel_internal_d_i(key,value, 0, lenArray-1, cutoff);	
		}
	}	

}

////////////////////////////////////////////////////////////////////////////////////
// Optimization 1 -- Event-based Sample/XS Lookup kernel splitting + Sorting
//                   lookups by material and energy
////////////////////////////////////////////////////////////////////////////////////
// This kernel separates out the sampling and lookup regions of the event-based
// model, and then sorts the lookups by material type and energy. The goal of this
// optimization is to allow for greatly improved cache locality, and XS indices
// loaded from memory may be re-used for multiple lookups.
//
// As efficienct sorting is key for performance, we also must implement an
// efficient key-value parallel sorting algorithm. We also experimented with using
// the C++ version of thrust for these purposes, but found that our own implemtation
// was slightly faster than the thrust library version, so for speed and
// simplicity we will do not add the thrust dependency.
////////////////////////////////////////////////////////////////////////////////////


unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype)
{
	char * optimization_name = "Optimization 1 - Kernel splitting + full material & energy sort";
	
	if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);
	
	////////////////////////////////////////////////////////////////////////////////
	// Allocate Additional Data Structures Needed by Optimized Kernel
	////////////////////////////////////////////////////////////////////////////////
	if( mype == 0)	printf("Allocating additional data required by optimized kernel...\n");
	size_t sz;
	size_t total_sz = 0;
	double start, stop;

	sz = in.lookups * sizeof(double);
	SD.p_energy_samples = (double *) malloc(sz);
	total_sz += sz;
	SD.length_p_energy_samples = in.lookups;

	sz = in.lookups * sizeof(int);
	SD.mat_samples = (int *) malloc(sz);
	total_sz += sz;
	SD.length_mat_samples = in.lookups;
	
	if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);
	
	////////////////////////////////////////////////////////////////////////////////
	// Begin Actual Simulation 
	////////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////////
	// Sample Materials and Energies
	////////////////////////////////////////////////////////////////////////////////
	#pragma omp parallel for schedule(dynamic, #P1)
	for( int i = 0; i < in.lookups; i++ )
	{
		// Set the initial seed value
		uint64_t seed = STARTING_SEED;	

		// Forward seed to lookup index (we need 2 samples per lookup)
		seed = fast_forward_LCG(seed, 2*i);

		// Randomly pick an energy and material for the particle
		double p_energy = LCG_random_double(&seed);
		int mat         = pick_mat(&seed); 

		SD.p_energy_samples[i] = p_energy;
		SD.mat_samples[i] = mat;
	}
	if(mype == 0) printf("finished sampling...\n");
	
	////////////////////////////////////////////////////////////////////////////////
	// Sort by Material
	////////////////////////////////////////////////////////////////////////////////
	
	start = get_time();

	quickSort_parallel_i_d(SD.mat_samples, SD.p_energy_samples, in.lookups, in.nthreads);

	stop = get_time();

	if(mype == 0) printf("Material sort took %.3lf seconds\n", stop-start);
	
	////////////////////////////////////////////////////////////////////////////////
	// Sort by Energy
	////////////////////////////////////////////////////////////////////////////////
	
	start = get_time();
	
	// Count up number of each type of sample. 
	int num_samples_per_mat[12] = {0};
	for( int l = 0; l < in.lookups; l++ )
		num_samples_per_mat[ SD.mat_samples[l] ]++;

	// Determine offsets
	int offsets[12] = {0};
	for( int m = 1; m < 12; m++ )
		offsets[m] = offsets[m-1] + num_samples_per_mat[m-1];
	
	stop = get_time();
	if(mype == 0) printf("Counting samples and offsets took %.3lf seconds\n", stop-start);
	start = stop;

	// Sort each material type by energy level
	int offset = 0;
	for( int m = 0; m < 12; m++ )
		quickSort_parallel_d_i(SD.p_energy_samples + offsets[m],SD.mat_samples + offsets[m], num_samples_per_mat[m], in.nthreads);

	stop = get_time();
	if(mype == 0) printf("Energy Sorts took %.3lf seconds\n", stop-start);
	
	////////////////////////////////////////////////////////////////////////////////
	// Perform lookups for each material separately
	////////////////////////////////////////////////////////////////////////////////
	start = get_time();

	unsigned long long verification = 0;

	// Individual Materials
	offset = 0;
	#P2
	for( int m = 0; m < 12; m++ )
	{
		#pragma omp parallel for schedule(dynamic,#P1) reduction(+:verification)
		for( int i = offset; i < offset + num_samples_per_mat[m]; i++)
		{
			// load pre-sampled energy and material for the particle
			double p_energy = SD.p_energy_samples[i];
			int mat         = SD.mat_samples[i]; 

			double macro_xs_vector[5] = {0};

			// Perform macroscopic Cross Section Lookup
			calculate_macro_xs(
					p_energy,        // Sampled neutron energy (in lethargy)
					mat,             // Sampled material type index neutron is in
					in.n_isotopes,   // Total number of isotopes in simulation
					in.n_gridpoints, // Number of gridpoints per isotope in simulation
					SD.num_nucs,     // 1-D array with number of nuclides per material
					SD.concs,        // Flattened 2-D array with concentration of each nuclide in each material
					SD.unionized_energy_array, // 1-D Unionized energy array
					SD.index_grid,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
					SD.nuclide_grid, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
					SD.mats,         // Flattened 2-D array with nuclide indices defining composition of each type of material
					macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
					in.grid_type,    // Lookup type (nuclide, hash, or unionized)
					in.hash_bins,    // Number of hash bins used (if using hash lookup type)
					SD.max_num_nucs  // Maximum number of nuclides present in any material
					);

			// For verification, and to prevent the compiler from optimizing
			// all work out, we interrogate the returned macro_xs_vector array
			// to find its maximum value index, then increment the verification
			// value by that index. In this implementation, we prevent thread
			// contention by using an OMP reduction on the verification value.
			// For accelerators, a different approach might be required
			// (e.g., atomics, reduction of thread-specific values in large
			// array via CUDA thrust, etc).
			double max = -1.0;
			int max_idx = 0;
			for(int j = 0; j < 5; j++ )
			{
				if( macro_xs_vector[j] > max )
				{
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification += max_idx+1;
		}
		offset += num_samples_per_mat[m];
	}
	
	stop = get_time();
	if(mype == 0) printf("XS Lookups took %.3lf seconds\n", stop-start);
	return verification;
}

//GridInit.c

SimulationData grid_init_do_not_profile( Inputs in, int mype )
{
	// Structure to hold all allocated simuluation data arrays
	SimulationData SD;

	// Keep track of how much data we're allocating
	size_t nbytes = 0;

	// Set the initial seed value
	uint64_t seed = 42;	

	////////////////////////////////////////////////////////////////////
	// Initialize Nuclide Grids
	////////////////////////////////////////////////////////////////////
	
	if(mype == 0) printf("Intializing nuclide grids...\n");

	// First, we need to initialize our nuclide grid. This comes in the form
	// of a flattened 2D array that hold all the information we need to define
	// the cross sections for all isotopes in the simulation. 
	// The grid is composed of "NuclideGridPoint" structures, which hold the
	// energy level of the grid point and all associated XS data at that level.
	// An array of structures (AOS) is used instead of
	// a structure of arrays, as the grid points themselves are accessed in 
	// a random order, but all cross section interaction channels and the
	// energy level are read whenever the gridpoint is accessed, meaning the
	// AOS is more cache efficient.
	
	// Initialize Nuclide Grid
	SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
	SD.nuclide_grid     = (NuclideGridPoint *) malloc( SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	assert(SD.nuclide_grid != NULL);
	nbytes += SD.length_nuclide_grid * sizeof(NuclideGridPoint);
	for( int i = 0; i < SD.length_nuclide_grid; i++ )
	{
		SD.nuclide_grid[i].energy        = LCG_random_double(&seed);
		SD.nuclide_grid[i].total_xs      = LCG_random_double(&seed);
		SD.nuclide_grid[i].elastic_xs    = LCG_random_double(&seed);
		SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
		SD.nuclide_grid[i].fission_xs    = LCG_random_double(&seed);
		SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
	}

	// Sort so that each nuclide has data stored in ascending energy order.
	#P3
	for( int i = 0; i < in.n_isotopes; i++ )
		qsort( &SD.nuclide_grid[i*in.n_gridpoints], in.n_gridpoints, sizeof(NuclideGridPoint), NGP_compare);
	
	// error debug check
	/*
	#P3
	for( int i = 0; i < in.n_isotopes; i++ )
	{
		printf("NUCLIDE %d ==============================\n", i);
		for( int j = 0; j < in.n_gridpoints; j++ )
			printf("E%d = %lf\n", j, SD.nuclide_grid[i * in.n_gridpoints + j].energy);
	}
	*/
	

	////////////////////////////////////////////////////////////////////
	// Initialize Acceleration Structure
	////////////////////////////////////////////////////////////////////
	
	if( in.grid_type == NUCLIDE )
	{
		SD.length_unionized_energy_array = 0;
		SD.length_index_grid = 0;
	}
	
	if( in.grid_type == UNIONIZED )
	{
		if(mype == 0) printf("Intializing unionized grid...\n");

		// Allocate space to hold the union of all nuclide energy data
		SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
		SD.unionized_energy_array = (double *) malloc( SD.length_unionized_energy_array * sizeof(double));
		assert(SD.unionized_energy_array != NULL );
		nbytes += SD.length_unionized_energy_array * sizeof(double);

		// Copy energy data over from the nuclide energy grid
		#P3
		for( int i = 0; i < SD.length_unionized_energy_array; i++ )
			SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;

		// Sort unionized energy array
		qsort( SD.unionized_energy_array, SD.length_unionized_energy_array, sizeof(double), double_compare);

		// Allocate space to hold the acceleration grid indices
		SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
		SD.index_grid = (int *) malloc( SD.length_index_grid * sizeof(int));
		assert(SD.index_grid != NULL);
		nbytes += SD.length_index_grid * sizeof(int);

		// Generates the double indexing grid
		int * idx_low = (int *) calloc( in.n_isotopes, sizeof(int));
		assert(idx_low != NULL );
		double * energy_high = (double *) malloc( in.n_isotopes * sizeof(double));
		assert(energy_high != NULL );

		#P3
		for( int i = 0; i < in.n_isotopes; i++ )
			energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;

		#pragma clang loop(e,i) tile sizes(#P4,#P5)
                #pragma clang loop id(e)
		for( long e = 0; e < SD.length_unionized_energy_array; e++ )
		{
			#pragma clang loop id(i)
			for( long i = 0; i < in.n_isotopes; i++ )
			{
			double unionized_energy = SD.unionized_energy_array[e];
				if( unionized_energy < energy_high[i]  )
					SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
				else if( idx_low[i] == in.n_gridpoints - 2 )
					SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
				else
				{
					idx_low[i]++;
					SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
					energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + idx_low[i] + 1].energy;	
				}
			}
		}

		free(idx_low);
		free(energy_high);
	}

	if( in.grid_type == HASH )
	{
		if(mype == 0) printf("Intializing hash grid...\n");
		SD.length_unionized_energy_array = 0;
		SD.length_index_grid  = in.hash_bins * in.n_isotopes;
		SD.index_grid = (int *) malloc( SD.length_index_grid * sizeof(int)); 
		assert(SD.index_grid != NULL);
		nbytes += SD.length_index_grid * sizeof(int);

		double du = 1.0 / in.hash_bins;

		// For each energy level in the hash table
		#pragma omp parallel for
		for( long e = 0; e < in.hash_bins; e++ )
		{
			double energy = e * du;

			// We need to determine the bounding energy levels for all isotopes
			for( long i = 0; i < in.n_isotopes; i++ )
			{
				SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide( in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints-1);
			}
		}
	}

	////////////////////////////////////////////////////////////////////
	// Initialize Materials and Concentrations
	////////////////////////////////////////////////////////////////////
	if(mype == 0) printf("Intializing material data...\n");
	
	// Set the number of nuclides in each material
	SD.num_nucs  = load_num_nucs(in.n_isotopes);
	SD.length_num_nucs = 12; // There are always 12 materials in XSBench

	// Intialize the flattened 2D grid of material data. The grid holds
	// a list of nuclide indices for each of the 12 material types. The
	// grid is allocated as a full square grid, even though not all
	// materials have the same number of nuclides.
	SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
	SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;

	// Intialize the flattened 2D grid of nuclide concentration data. The grid holds
	// a list of nuclide concentrations for each of the 12 material types. The
	// grid is allocated as a full square grid, even though not all
	// materials have the same number of nuclides.
	SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
	SD.length_concs = SD.length_mats;

	if(mype == 0) printf("Intialization complete. Allocated %.0lf MB of data.\n", nbytes/1024.0/1024.0 );

	return SD;

}

