#include "rsbench.h"

int main(int argc, char * argv[])
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================

	int version = 12;
	double start, stop;
	
	// set the env variables for thread affinity
	setenv("OMP_PLACES","cores",1);
	system("echo $OMP_PLACES");
	setenv("KMP_AFFINITY","compact",1);
	system("echo $KMP_AFFINITY");

	// Process CLI Fields
	Input input = read_CLI( argc, argv );

	// Set number of OpenMP Threads
	omp_set_num_threads(input.nthreads); 
	
	// =====================================================================
	// Print-out of Input Summary
	// =====================================================================
	logo(version);
	center_print("INPUT SUMMARY", 79);
	border_print();
	print_input_summary(input);

	// =====================================================================
	// Intialize Simulation Data Structures
	// =====================================================================
	border_print();
	center_print("INITIALIZATION", 79);
	border_print();
	
	start = get_time();
	
	SimulationData SD = initialize_simulation( input );

	stop = get_time();

	printf("Initialization Complete. (%.2lf seconds)\n", stop-start);
	
	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================
	border_print();
	center_print("SIMULATION", 79);
	border_print();

	unsigned long vhash = 0;

	// Run Simulation
	start = get_time();

	// Run simulation
	if( input.simulation_method == EVENT_BASED )
	{
		if( input.kernel_id == 0 )
			run_event_based_simulation(input, SD, &vhash );
		else if( input.kernel_id == 1 )
			run_event_based_simulation_optimization_1(input, SD, &vhash );
		else
		{
			printf("Error: No kernel ID %d found!\n", input.kernel_id);
			exit(1);
		}
	}
	else if( input.simulation_method == HISTORY_BASED )
		run_history_based_simulation(input, SD, &vhash );

	stop = get_time();

	// Final hash step
	vhash = vhash % 999983;

	printf("Simulation Complete.\n");

	// =====================================================================
	// Print / Save Results and Exit
	// =====================================================================
	border_print();
	center_print("RESULTS", 79);
	border_print();

	int is_invalid = validate_and_print_results(input, stop-start, vhash);

	border_print();

	return is_invalid;
}

// init.c

SimulationData initialize_simulation( Input input )
{
	uint64_t seed = INITIALIZATION_SEED;
	
	// Get material data
	printf("Loading Hoogenboom-Martin material data...\n");
	SimulationData SD = get_materials( input, &seed ); 
	
	// Allocate & fill energy grids
	printf("Generating resonance distributions...\n");
	SD.n_poles = generate_n_poles( input, &seed );
	SD.length_n_poles = input.n_nuclides;

	// Allocate & fill Window grids
	printf("Generating window distributions...\n");
	SD.n_windows = generate_n_windows( input, &seed );
	SD.length_n_windows = input.n_nuclides;

	// Prepare full resonance grid
	printf("Generating resonance parameter grid...\n");
	SD.poles = generate_poles( input, SD.n_poles, &seed, &SD.max_num_poles );
	SD.length_poles = input.n_nuclides * SD.max_num_poles;

	// Prepare full Window grid
	printf("Generating window parameter grid...\n");
	SD.windows = generate_window_params( input, SD.n_windows, SD.n_poles, &seed, &SD.max_num_windows);
	SD.length_windows = input.n_nuclides * SD.max_num_windows;

	// Prepare 0K Resonances
	printf("Generating 0K l_value data...\n");
	SD.pseudo_K0RS = generate_pseudo_K0RS( input, &seed );
	SD.length_pseudo_K0RS = input.n_nuclides * input.numL;

	return SD;
}

int * generate_n_poles( Input input, uint64_t * seed )
{
	int total_resonances = input.avg_n_poles * input.n_nuclides;

	int * R = (int *) malloc( input.n_nuclides * sizeof(int));
	
	// Ensure all nuclides have at least 1 resonance
	for( int i = 0; i < input.n_nuclides; i++ )
		R[i] = 1;

	// Sample the rest
	for( int i = 0; i < total_resonances - input.n_nuclides; i++ )
		R[LCG_random_int(seed) % input.n_nuclides]++;
	
	/* Debug	
	for( int i = 0; i < input.n_nuclides; i++ )
		printf("R[%d] = %d\n", i, R[i]);
	*/
	
	return R;
}

int * generate_n_windows( Input input, uint64_t * seed )
{
	int total_resonances = input.avg_n_windows * input.n_nuclides;

	int * R = (int *) malloc( input.n_nuclides * sizeof(int));
	
	// Ensure all nuclides have at least 1 resonance
	for( int i = 0; i < input.n_nuclides; i++ )
		R[i] = 1;

	// Sample the rest
	for( int i = 0; i < total_resonances - input.n_nuclides; i++ )
		R[LCG_random_int(seed) % input.n_nuclides]++;
	
	/* Debug	
	for( int i = 0; i < input.n_nuclides; i++ )
		printf("R[%d] = %d\n", i, R[i]);
	*/
	
	return R;
}

Pole * generate_poles( Input input, int * n_poles, uint64_t * seed, int * max_num_poles )
{
	// Pole Scaling Factor -- Used to bias hitting of the fast Faddeeva
	// region to approximately 99.5% (i.e., only 0.5% of lookups should
	// require the full eval).
	double f = 152.5;
	RSComplex f_c = {f, 0};

	int max_poles = -1;
	
	for( int i = 0; i < input.n_nuclides; i++ )
	{
		if( n_poles[i] > max_poles)
			max_poles = n_poles[i];
	}
	*max_num_poles = max_poles;

	// Allocating 2D matrix as a 1D contiguous vector
	Pole * R = (Pole *) malloc( input.n_nuclides * max_poles * sizeof(Pole));
	
	// fill with data
	for( int i = 0; i < input.n_nuclides; i++ )
		for( int j = 0; j < n_poles[i]; j++ )
		{
			double r = LCG_random_double(seed);
			double im = LCG_random_double(seed);
			RSComplex t1 = {r, im};
			R[i * max_poles + j].MP_EA = c_mul(f_c,t1);
			r = LCG_random_double(seed);
			im = LCG_random_double(seed);
			RSComplex t2 = {f*r, im};
			R[i * max_poles + j].MP_RT = t2;
			r = LCG_random_double(seed);
			im = LCG_random_double(seed);
			RSComplex t3 = {f*r, im};
			R[i * max_poles + j].MP_RA = t3;
			r = LCG_random_double(seed);
			im = LCG_random_double(seed);
			RSComplex t4 = {f*r, im};
			R[i * max_poles + j].MP_RF = t4;
			R[i * max_poles + j].l_value = LCG_random_int(seed) % input.numL;
		}
	
	/* Debug
	for( int i = 0; i < input.n_nuclides; i++ )
		for( int j = 0; j < n_poles[i]; j++ )
			printf("R[%d][%d]: Eo = %lf lambda_o = %lf Tn = %lf Tg = %lf Tf = %lf\n", i, j, R[i * max_poles + j].Eo, R[i * max_poles + j].lambda_o, R[i * max_poles + j].Tn, R[i * max_poles + j].Tg, R[i * max_poles + j].Tf);
	*/

	return R;
}

Window * generate_window_params( Input input, int * n_windows, int * n_poles, uint64_t * seed, int * max_num_windows )
{
	int max_windows = -1;
	
	for( int i = 0; i < input.n_nuclides; i++ )
	{
		if( n_windows[i] > max_windows)
			max_windows = n_windows[i];
	}
	*max_num_windows = max_windows;

	// Allocating 2D contiguous matrix
	Window * R = (Window *) malloc( input.n_nuclides * max_windows * sizeof(Window));
	
	// fill with data
	for( int i = 0; i < input.n_nuclides; i++ )
	{
		int space = n_poles[i] / n_windows[i];
		int remainder = n_poles[i] - space * n_windows[i];
		int ctr = 0;
		for( int j = 0; j < n_windows[i]; j++ )
		{
			R[i * max_windows + j].T = LCG_random_double(seed);
			R[i * max_windows + j].A = LCG_random_double(seed);
			R[i * max_windows + j].F = LCG_random_double(seed);
			R[i * max_windows + j].start = ctr; 
			R[i * max_windows + j].end = ctr + space - 1;

			ctr += space;

			if ( j < remainder )
			{
				ctr++;
				R[i * max_windows + j].end++;
			}
		}
	}

	return R;
}

double * generate_pseudo_K0RS( Input input, uint64_t * seed )
{
	double * R = (double *) malloc( input.n_nuclides * input.numL * sizeof(double));

	for( int i = 0; i < input.n_nuclides; i++)
		for( int j = 0; j < input.numL; j++ )
			R[i * input.numL + j] = LCG_random_double(seed);

	return R;
}


// io.c

// Prints program logo
void logo(int version)
{
	border_print();
	printf(
"                    _____   _____ ____                  _     \n"
"                   |  __ \\ / ____|  _ \\                | |    \n"
"                   | |__) | (___ | |_) | ___ _ __   ___| |__  \n"
"                   |  _  / \\___ \\|  _ < / _ \\ '_ \\ / __| '_ \\ \n"
"                   | | \\ \\ ____) | |_) |  __/ | | | (__| | | |\n"
"                   |_|  \\_\\_____/|____/ \\___|_| |_|\\___|_| |_|\n\n"
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

void border_print(void)
{
	printf(
	"==================================================================="
	"=============\n");
}

// Prints comma separated integers - for ease of reading
void fancy_int( int a )
{
    if( a < 1000 )
        printf("%d\n",a);

    else if( a >= 1000 && a < 1000000 )
        printf("%d,%03d\n", a / 1000, a % 1000);

    else if( a >= 1000000 && a < 1000000000 )
        printf("%d,%03d,%03d\n", a / 1000000, (a % 1000000) / 1000, a % 1000 );

    else if( a >= 1000000000 )
        printf("%d,%03d,%03d,%03d\n",
               a / 1000000000,
               (a % 1000000000) / 1000000,
               (a % 1000000) / 1000,
               a % 1000 );
    else
        printf("%d\n",a);
}

Input read_CLI( int argc, char * argv[] )
{
	Input input;

	// defaults to the history based simulation method
	input.simulation_method = HISTORY_BASED;
	// defaults to max threads on the system	
	input.nthreads = omp_get_num_procs();
	// defaults to 355 (corresponding to H-M Large benchmark)
	input.n_nuclides = 355;
	// defaults to 300,000
	input.particles = 300000;
	// defaults to 34
	input.lookups = 34;
	// defaults to H-M Large benchmark
	input.HM = LARGE;
	// defaults to 3000 resonancs (avg) per nuclide
	input.avg_n_poles = 1000;
	// defaults to 100
	input.avg_n_windows = 100;
	// defaults to 4;
	input.numL = 4;
	// defaults to no temperature dependence (Doppler broadening)
	input.doppler = 1;
	// defaults to baseline simulation kernel
	input.kernel_id = 0;
	
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
		// nuclides (-n)
		else if( strcmp(arg, "-n") == 0 )
		{
			if( ++i < argc )
				input.n_nuclides = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// HM (-s)
		else if( strcmp(arg, "-s") == 0 )
		{	
			if( ++i < argc )
			{
				if( strcmp(argv[i], "small") == 0 )
					input.HM = SMALL;
				else if ( strcmp(argv[i], "large") == 0 )
					input.HM = LARGE;
				else
					print_CLI_error();
			}
			else
				print_CLI_error();
		}
		// Doppler Broadening (Temperature Dependence)
		else if( strcmp(arg, "-d") == 0 )
		{	
			input.doppler = 0;
		}
		// Avg number of windows per nuclide (-w)
		else if( strcmp(arg, "-W") == 0 )
		{
			if( ++i < argc )
				input.avg_n_windows = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// Avg number of poles per nuclide (-p)
		else if( strcmp(arg, "-P") == 0 )
		{
			if( ++i < argc )
				input.avg_n_poles = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// Kernel ID (-k)
		else if( strcmp(arg, "-k") == 0 )
		{
			if( ++i < argc )
				input.kernel_id = atoi(argv[i]);
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
	if( input.n_nuclides < 1 )
		print_CLI_error();
	
	// Validate lookups
	if( input.lookups < 1 )
		print_CLI_error();
	
	// Validate lookups
	if( input.avg_n_poles < 1 )
		print_CLI_error();
	
	// Validate lookups
	if( input.avg_n_windows < 1 )
		print_CLI_error();
	
	// Set HM size specific parameters
	// (defaults to large)
	if( input.HM == SMALL )
		input.n_nuclides = 68;

	// Return input struct
	return input;
}

void print_CLI_error(void)
{
	printf("Usage: ./multibench <options>\n");
	printf("Options include:\n");
	printf("  -t <threads>            Number of OpenMP threads to run\n");
	printf("  -m <simulation method>  Simulation method (history, event)\n");
	printf("  -s <size>               Size of H-M Benchmark to run (small, large)\n");
	printf("  -l <lookups>            Number of Cross-section (XS) lookups per particle history\n");
	printf("  -p <particles>          Number of particle histories\n");
	printf("  -P <poles>              Average Number of Poles per Nuclide\n");
	printf("  -W <poles>              Average Number of Windows per Nuclide\n");
	printf("  -d                      Disables Temperature Dependence (Doppler Broadening)\n");
	printf("Default is equivalent to: -s large -m history -l 34 -p 300000 -P 1000 -W 100\n");
	printf("See readme for full description of default run values\n");
	exit(4);
}

void print_input_summary(Input input)
{
	// Calculate Estimate of Memory Usage
	size_t mem = get_mem_estimate(input);

	if( input.simulation_method == EVENT_BASED )
		printf("Simulation Method:           Event Based\n");
	else
		printf("Simulation Method:           History Based\n");
	printf("Materials:                   12\n");
	printf("H-M Benchmark Size:          ");
	if( input.HM == 0 )
		printf("Small\n");
	else
		printf("Large\n");
	if( input.doppler == 1 )
		printf("Temperature Dependence:      ON\n");
	else
		printf("Temperature Dependence:      OFF\n");
	printf("Total Nuclides:              %d\n", input.n_nuclides);
	printf("Avg Poles per Nuclide:       "); fancy_int(input.avg_n_poles);
	printf("Avg Windows per Nuclide:     "); fancy_int(input.avg_n_windows);

	int lookups = input.lookups;
	if( input.simulation_method == HISTORY_BASED )
	{
		printf("Particles:                   "); fancy_int(input.particles);
		printf("XS Lookups per Particle:     "); fancy_int(input.lookups);
		lookups *= input.particles;
	}
	printf("Total XS Lookups:            "); fancy_int(lookups);
	printf("Threads:                     %d\n", input.nthreads);
	printf("Est. Memory Usage (MB):      %.1lf\n", mem / 1024.0 / 1024.0);
}

int validate_and_print_results(Input input, double runtime, unsigned long vhash)
{
	printf("Threads:               %d\n", input.nthreads);
	printf("Runtime:               %.3lf seconds\n", runtime);
	int lookups = 0;
	if( input.simulation_method == HISTORY_BASED )
		lookups = input.lookups*input.particles;
	else
		lookups = input.lookups;
	printf("Lookups:               "); fancy_int(lookups);
	printf("Lookups/s:             "); fancy_int((double) lookups / (runtime));

	int is_invalid = 1;

	unsigned long long large = 0;
	unsigned long long small = 0;
	if(input.simulation_method == HISTORY_BASED )
	{
		large = 351485;
		small = 879693;
	}
	else if( input.simulation_method == EVENT_BASED )
	{
		large = 358389;
		small = 880018;
	}

	if( input.HM  == LARGE )
	{
		if( vhash == large )
		{
			printf("Verification checksum: %lu (Valid)\n", vhash);
			is_invalid = 0;
		}
		else
			printf("Verification checksum: %lu (WARNING - INAVALID CHECKSUM!)\n", vhash);
	}
	else if( input.HM  == SMALL )
	{
		if( vhash == small )
		{
			printf("Verification checksum: %lu (Valid)\n", vhash);
			is_invalid = 0;
		}
		else
			printf("Verification checksum: %lu (WARNING - INAVALID CHECKSUM!)\n", vhash);
	}

	return is_invalid;
}

// simulation.c

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, RSBench will only run the baseline implementation. Optimized variants
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////

void run_event_based_simulation(Input input, SimulationData data, unsigned long * vhash_result )
{
	printf("Beginning baseline event based simulation...\n");
	unsigned long verification = 0;

	// Main simulation loop over macroscopic cross section lookups
	#pragma omp parallel for schedule(dynamic, 1000) default(none) shared(input, data) reduction(+:verification)
	for( int i = 0; i < input.lookups; i++ )
	{
		// Set the initial seed value
		uint64_t seed = STARTING_SEED;	

		// Forward seed to lookup index (we need 2 samples per lookup)
		seed = fast_forward_LCG(seed, 2*i);

		// Randomly pick an energy and material for the particle
		double E = LCG_random_double(&seed);
		int mat  = pick_mat(&seed);

		double macro_xs[4] = {0};

		calculate_macro_xs( macro_xs, mat, E, input, data ); 

		// For verification, and to prevent the compiler from optimizing
		// all work out, we interrogate the returned macro_xs_vector array
		// to find its maximum value index, then increment the verification
		// value by that index. In this implementation, we prevent thread
		// contention by using an OMP reduction on it. For other accelerators,
		// a different approach might be required (e.g., atomics, reduction
		// of thread-specific values in large array via CUDA thrust, etc)
		double max = -DBL_MAX;
		int max_idx = 0;
		for(int x = 0; x < 4; x++ )
		{
			if( macro_xs[x] > max )
			{
				max = macro_xs[x];
				max_idx = x;
			}
		}
		verification += max_idx+1;
	}

	*vhash_result = verification;
}

void run_history_based_simulation(Input input, SimulationData data, unsigned long * vhash_result )
{
	printf("Beginning history based simulation...\n");
	unsigned long verification = 0;

	// Main simulation loop over particle histories
	#pragma omp parallel for schedule(dynamic, 1000) default(none) shared(input, data) reduction(+:verification)
	for( int p = 0; p < input.particles; p++ )
	{
		// Set the initial seed value
		uint64_t seed = STARTING_SEED;	

		// Forward seed to lookup index (we need 2 samples per lookup)
		seed = fast_forward_LCG(seed, p * input.lookups * 2 * 4);

		// Randomly pick an energy and material for the particle
		double E = LCG_random_double(&seed);
		int mat  = pick_mat(&seed);

		// Loop over macroscopic cross section events. This loop is dependent!
		// I.e., This loop must be executed sequentially,
		// as each lookup depends on results from the previous lookup.
		for( int i = 0; i < input.lookups; i++ )
		{
			double macro_xs[4] = {0};

			calculate_macro_xs( macro_xs, mat, E, input, data ); 

			// For verification, and to prevent the compiler from optimizing
			// all work out, we interrogate the returned macro_xs_vector array
			// to find its maximum value index, then increment the verification
			// value by that index. In this implementation, we prevent thread
			// contention by using an OMP reduction on it. For other accelerators,
			// a different approach might be required (e.g., atomics, reduction
			// of thread-specific values in large array via CUDA thrust, etc)
			double max = -DBL_MAX;
			int max_idx = 0;
			for(int x = 0; x < 4; x++ )
			{
				if( macro_xs[x] > max )
				{
					max = macro_xs[x];
					max_idx = x;
				}
			}
			verification += max_idx+1;

			// Randomly pick next energy and material for the particle
			// Also incorporates results from macro_xs lookup to
			// enforce loop dependency.
			// In a real MC app, this dependency is expressed in terms
			// of branching physics sampling, whereas here we are just
			// artificially enforcing this dependence based on altering
			// the seed
			uint64_t n_forward = 0;
			for( int x = 0; x < 4; x++ )
				if( macro_xs[x] > 1.0 )
					n_forward++;
			if( n_forward > 0 )
				seed = fast_forward_LCG(seed, n_forward);

			E   = LCG_random_double(&seed);
			mat = pick_mat(&seed);
		}
	}
	*vhash_result = verification;
}


void calculate_macro_xs( double * macro_xs, int mat, double E, Input input, SimulationData data ) 
{
	// zero out macro vector
	for( int i = 0; i < 4; i++ )
		macro_xs[i] = 0;

	// for nuclide in mat
	for( int i = 0; i < data.num_nucs[mat]; i++ )
	{
		double micro_xs[4];
		int nuc = data.mats[mat * data.max_num_nucs + i];

		if( input.doppler == 1 )
			calculate_micro_xs_doppler( micro_xs, nuc, E, input, data );
		else
			calculate_micro_xs( micro_xs, nuc, E, input, data);

		for( int j = 0; j < 4; j++ )
		{
			macro_xs[j] += micro_xs[j] * data.concs[mat * data.max_num_nucs + i];
		}
		// Debug
		/*
		printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
		E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
		*/
	}

	// Debug
	/*
	printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
	E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
	*/
}

// No Temperature dependence (i.e., 0K evaluation)
void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, SimulationData data)
{
	// MicroScopic XS's to Calculate
	double sigT;
	double sigA;
	double sigF;
	double sigE;

	// Calculate Window Index
	double spacing = 1.0 / data.n_windows[nuc];
	int window = (int) ( E / spacing );
	if( window == data.n_windows[nuc] )
		window--;

	// Calculate sigTfactors
	RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
	calculate_sig_T(nuc, E, input, data, sigTfactors );

	// Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
	Window w = data.windows[nuc * data.max_num_windows + window];
	sigT = E * w.T;
	sigA = E * w.A;
	sigF = E * w.F;

	// Loop over Poles within window, add contributions
	for( int i = w.start; i < w.end; i++ )
	{
		RSComplex PSIIKI;
		RSComplex CDUM;
		Pole pole = data.poles[nuc * data.max_num_poles + i];
		RSComplex t1 = {0, 1};
		RSComplex t2 = {sqrt(E), 0 };
		PSIIKI = c_div( t1 , c_sub(pole.MP_EA,t2) );
		RSComplex E_c = {E, 0};
		CDUM = c_div(PSIIKI, E_c);
		sigT += (c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])) ).r;
		sigA += (c_mul( pole.MP_RA, CDUM)).r;
		sigF += (c_mul(pole.MP_RF, CDUM)).r;
	}

	sigE = sigT - sigA;

	micro_xs[0] = sigT;
	micro_xs[1] = sigA;
	micro_xs[2] = sigF;
	micro_xs[3] = sigE;
}

// Temperature Dependent Variation of Kernel
// (This involves using the Complex Faddeeva function to
// Doppler broaden the poles within the window)
void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, SimulationData data )
{
	// MicroScopic XS's to Calculate
	double sigT;
	double sigA;
	double sigF;
	double sigE;

	// Calculate Window Index
	double spacing = 1.0 / data.n_windows[nuc];
	int window = (int) ( E / spacing );
	if( window == data.n_windows[nuc] )
		window--;

	// Calculate sigTfactors
	RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
	calculate_sig_T(nuc, E, input, data, sigTfactors );

	// Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
	Window w = data.windows[nuc * data.max_num_windows + window];
	sigT = E * w.T;
	sigA = E * w.A;
	sigF = E * w.F;

	double dopp = 0.5;

	// Loop over Poles within window, add contributions
	for( int i = w.start; i < w.end; i++ )
	{
		Pole pole = data.poles[nuc * data.max_num_poles + i];

		// Prep Z
		RSComplex E_c = {E, 0};
		RSComplex dopp_c = {dopp, 0};
		RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);

		// Evaluate Fadeeva Function
		RSComplex faddeeva = fast_nuclear_W( Z );

		// Update W
		sigT += (c_mul( pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]) )).r;
		sigA += (c_mul( pole.MP_RA , faddeeva)).r;
		sigF += (c_mul( pole.MP_RF , faddeeva)).r;
	}

	sigE = sigT - sigA;

	micro_xs[0] = sigT;
	micro_xs[1] = sigA;
	micro_xs[2] = sigF;
	micro_xs[3] = sigE;
}

void calculate_sig_T( int nuc, double E, Input input, SimulationData data, RSComplex * sigTfactors )
{
	double phi;

	for( int i = 0; i < 4; i++ )
	{
		phi = data.pseudo_K0RS[nuc * input.numL + i] * sqrt(E);

		if( i == 1 )
			phi -= - atan( phi );
		else if( i == 2 )
			phi -= atan( 3.0 * phi / (3.0 - phi*phi));
		else if( i == 3 )
			phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));

		phi *= 2.0;

		sigTfactors[i].r = cos(phi);
		sigTfactors[i].i = -sin(phi);
	}
}

// This function uses a combination of the Abrarov Approximation
// and the QUICK_W three term asymptotic expansion.
// Only expected to use Abrarov ~0.5% of the time.
RSComplex fast_nuclear_W( RSComplex Z )
{
	// Abrarov 
	if( c_abs(Z) < 6.0 )
	{
		// Precomputed parts for speeding things up
		// (N = 10, Tm = 12.0)
		RSComplex prefactor = {0, 8.124330e+01};
		double an[10] = {
			2.758402e-01,
			2.245740e-01,
			1.594149e-01,
			9.866577e-02,
			5.324414e-02,
			2.505215e-02,
			1.027747e-02,
			3.676164e-03,
			1.146494e-03,
			3.117570e-04
		};
		double neg_1n[10] = {
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0
		};

		double denominator_left[10] = {
			9.869604e+00,
			3.947842e+01,
			8.882644e+01,
			1.579137e+02,
			2.467401e+02,
			3.553058e+02,
			4.836106e+02,
			6.316547e+02,
			7.994380e+02,
			9.869604e+02
		};

		RSComplex t1 = {0, 12};
		RSComplex t2 = {12, 0};
		RSComplex i = {0,1};
		RSComplex one = {1, 0};
		RSComplex W = c_div(c_mul(i, ( c_sub(one, fast_cexp(c_mul(t1, Z))) )) , c_mul(t2, Z));
		RSComplex sum = {0,0};
		for( int n = 0; n < 10; n++ )
		{
			RSComplex t3 = {neg_1n[n], 0};
			RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
			RSComplex t4 = {denominator_left[n], 0};
			RSComplex t5 = {144, 0};
			RSComplex bot = c_sub(t4, c_mul(t5,c_mul(Z,Z)));
			RSComplex t6 = {an[n], 0};
			sum = c_add(sum, c_mul(t6, c_div(top,bot)));
		}
		W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
		return W;
	}
	else
	{
		// QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).
		// Pre-computed parameters
		RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182,0};
		RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0};
		RSComplex c = {0.051765358792987823963876628425793170829107067780337219430904, 0};
		RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0};

		RSComplex i = {0,1};
		RSComplex Z2 = c_mul(Z, Z);
		// Three Term Asymptotic Expansion
		RSComplex W = c_mul(c_mul(Z,i), (c_add(c_div(a,(c_sub(Z2, b))) , c_div(c,(c_sub(Z2, d))))));

		return W;
	}
}

double LCG_random_double(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}	

uint64_t LCG_random_int(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return *seed;
}	

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
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

// Complex arithmetic functions

RSComplex c_add( RSComplex A, RSComplex B)
{
	RSComplex C;
	C.r = A.r + B.r;
	C.i = A.i + B.i;
	return C;
}

RSComplex c_sub( RSComplex A, RSComplex B)
{
	RSComplex C;
	C.r = A.r - B.r;
	C.i = A.i - B.i;
	return C;
}

RSComplex c_mul( RSComplex A, RSComplex B)
{
	double a = A.r;
	double b = A.i;
	double c = B.r;
	double d = B.i;
	RSComplex C;
	C.r = (a*c) - (b*d);
	C.i = (a*d) + (b*c);
	return C;
}

RSComplex c_div( RSComplex A, RSComplex B)
{
	double a = A.r;
	double b = A.i;
	double c = B.r;
	double d = B.i;
	RSComplex C;
	double denom = c*c + d*d;
	C.r = ( (a*c) + (b*d) ) / denom;
	C.i = ( (b*c) - (a*d) ) / denom;
	return C;
}

double c_abs( RSComplex A)
{
	return sqrt(A.r*A.r + A.i * A.i);
}

// Fast (but inaccurate) exponential function
// Written By "ACMer":
// https://codingforspeed.com/using-faster-exponential-approximation/
// We use our own to avoid small differences in compiler specific
// exp() intrinsic implementations that make it difficult to verify
// if the code is working correctly or not.
double fast_exp(double x)
{
  x = 1.0 + x * 0.000244140625;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

// Implementation based on:
// z = x + iy
// cexp(z) = e^x * (cos(y) + i * sin(y))
RSComplex fast_cexp( RSComplex z )
{
	double x = z.r;
	double y = z.i;

	// For consistency across architectures, we
	// will use our own exponetial implementation
	//double t1 = exp(x);
	double t1 = fast_exp(x);
	double t2 = cos(y);
	double t3 = sin(y);
	RSComplex t4 = {t2, t3};
	RSComplex t5 = {t1, 0};
	RSComplex result = c_mul(t5, (t4));
	return result;
}	

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

void run_event_based_simulation_optimization_1(Input in, SimulationData SD, unsigned long * vhash_result )
{
	char * optimization_name = "Optimization 1 - Kernel splitting + full material & energy sort";
	
	printf("Simulation Kernel:\"%s\"\n", optimization_name);
	
	////////////////////////////////////////////////////////////////////////////////
	// Allocate Additional Data Structures Needed by Optimized Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Allocating additional data required by optimized kernel...\n");
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
	
	printf("Allocated an additional %.0lf MB of data on CPU.\n", total_sz/1024.0/1024.0);
	
	////////////////////////////////////////////////////////////////////////////////
	// Begin Actual Simulation 
	////////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////////
	// Sample Materials and Energies
	////////////////////////////////////////////////////////////////////////////////
	printf("Sampling event data...\n");
	#pragma omp parallel for schedule(dynamic, 1000)
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
	printf("Finished sampling.\n");
	
	////////////////////////////////////////////////////////////////////////////////
	// Sort by Material
	////////////////////////////////////////////////////////////////////////////////
	
	start = get_time();

	quickSort_parallel_i_d(SD.mat_samples, SD.p_energy_samples, in.lookups, in.nthreads);

	stop = get_time();

	printf("Material sort took %.3lf seconds\n", stop-start);
	
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
	printf("Counting samples and offsets took %.3lf seconds\n", stop-start);
	start = stop;

	// Sort each material type by energy level
	int offset = 0;
	for( int m = 0; m < 12; m++ )
		quickSort_parallel_d_i(SD.p_energy_samples + offsets[m],SD.mat_samples + offsets[m], num_samples_per_mat[m], in.nthreads);

	stop = get_time();
	printf("Energy Sorts took %.3lf seconds\n", stop-start);
	
	////////////////////////////////////////////////////////////////////////////////
	// Perform lookups for each material separately
	////////////////////////////////////////////////////////////////////////////////
	start = get_time();

	unsigned long long verification = 0;

	// Individual Materials
	offset = 0;
	for( int m = 0; m < 12; m++ )
	{
		#pragma omp parallel for schedule(dynamic,100) reduction(+:verification)
		for( int i = offset; i < offset + num_samples_per_mat[m]; i++)
		{
			// load pre-sampled energy and material for the particle
			double E = SD.p_energy_samples[i];
			int mat  = SD.mat_samples[i]; 

			double macro_xs_vector[4] = {0};

			// Perform macroscopic Cross Section Lookup
			calculate_macro_xs( macro_xs_vector, mat, E, in, SD ); 

			// For verification, and to prevent the compiler from optimizing
			// all work out, we interrogate the returned macro_xs_vector array
			// to find its maximum value index, then increment the verification
			// value by that index. In this implementation, we prevent thread
			// contention by using an OMP reduction on the verification value.
			// For accelerators, a different approach might be required
			// (e.g., atomics, reduction of thread-specific values in large
			// array via CUDA thrust, etc).
			double max = -DBL_MAX;
			int max_idx = 0;
			for(int j = 0; j < 4; j++ )
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
	printf("XS Lookups took %.3lf seconds\n", stop-start);
	*vhash_result = verification;
}
