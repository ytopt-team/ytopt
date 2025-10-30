import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.io as scio
import time

def load_simulation_log(file_path: str = 'simulation_log.json') -> Dict:
    """Load the simulation log containing metadata about all scenarios."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_state_metadata(file_path: str = 'state_metadata.json') -> Dict:
    """Load the state metadata that describes all state variables."""
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_scenarios(
    simulation_log: Dict, 
    sample_idx: Optional[int] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = None
) -> Dict:
    """Filter scenarios based on criteria."""
    filtered_log = {}
    
    for scenario_id, data in simulation_log.items():
        match = True
        
        if sample_idx is not None and data.get('sample_idx') != sample_idx:
            match = False
        if fault_location is not None and data['fault_location'] != fault_location:
            match = False
        if fault_impedance is not None and data['fault_impedance'] != fault_impedance:
            match = False
        if diverged is not None and data['diverged'] != diverged:
            match = False
            
        if match:
            filtered_log[scenario_id] = data
            
    return filtered_log

def find_state_index(
    state_metadata: Dict, 
    model: Optional[str] = None, 
    device_number: Optional[str] = None,
    bus_num: Optional[int] = None,
    state_name: Optional[str] = None
) -> List[int]:
    """Find indices of states matching the criteria."""
    indices = []
    
    for idx, (state_idx, data) in enumerate(state_metadata.items()):
        match = True
        
        if model is not None and data.get('model') != model:
            match = False
        if device_number is not None and str(data.get('device_number')) != str(device_number):
            match = False
        if bus_num is not None and data.get('bus_num') != bus_num:
            match = False
        if state_name is not None and data.get('state_name') != state_name:
            match = False
            
        if match:
            indices.append(int(state_idx))
            
    return indices

def load_scenario_data(scenario_id: str, simulation_log: Dict) -> Dict:
    """Load data for a specific scenario."""
    file_path = simulation_log[scenario_id]['file']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scenario data file not found: {file_path}")
    
    data = np.load(file_path, mmap_mode='r')
    return {
        'history': data['history'],
        'tvec': data['tvec'],
        'metadata': simulation_log[scenario_id],
        'p_gen_scaled': data['p_gen_scaled'],
        'p_load_scaled': data['p_load_scaled'],
        'q_load_scaled': data['q_load_scaled'],
    }

def get_state_timeseries(
    scenario_data: Dict, 
    state_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a specific state variable from a scenario."""
    tvec = scenario_data['tvec']
    history = scenario_data['history']
    
    if history is None:
        return np.array([]), np.array([])
    
    state_values = history[state_idx, :].copy()
    return tvec, state_values

def get_state_timeseries_all(
    simulation_log: Dict,
    state_metadata: Dict,
    model: Optional[str] = None,
    device_number: Optional[str] = None,
    bus_num: Optional[int] = None,
    state_name: Optional[str] = None,
    sample_idx: Optional[int] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = False
) -> Dict[str, Dict[str, Union[np.ndarray, Any]]]:
    """Extract a state variable from multiple scenarios with filtering."""
    # Filter scenarios
    filtered_scenarios = filter_scenarios(
        simulation_log, 
        sample_idx,
        fault_location,
        fault_impedance,
        diverged
    )
    
    # Find state indices
    state_indices = find_state_index(
        state_metadata, 
        model, 
        device_number, 
        bus_num,
        state_name
    )
    
    if not state_indices:
        raise ValueError(f"No states found matching criteria: model={model}, device_number={device_number}, state_name={state_name}")
    
    state_idx = state_indices[0]  # Take the first match if multiple
    
    # Extract data for each scenario
    results = {}
    for scenario_id, scenario_info in filtered_scenarios.items():
        try:
            scenario_data = load_scenario_data(scenario_id, simulation_log)
            tvec, values = get_state_timeseries(scenario_data, state_idx)
            
            results[scenario_id] = {
                'tvec': tvec,
                'values': values,
                'metadata': scenario_info
            }
        except Exception as e:
            print(f"Error loading scenario {scenario_id}: {e}")
    
    return results

def plot_state_comparison(
    results: Dict[str, Dict[str, Union[np.ndarray, Any]]],
    title: str = None,
    xlabel: str = 'Time (s)',
    ylabel: str = None,
    legend_key: str = 'fault_location'
):
    """Plot comparison of state variables from multiple scenarios."""
    plt.figure(figsize=(10, 6))
    
    for scenario_id, data in results.items():
        tvec = data['tvec']
        values = data['values']
        metadata = data['metadata']
        
        # Skip empty results (e.g., from diverged simulations)
        if len(tvec) == 0 or len(values) == 0:
            continue
        
        plt.plot(tvec, values, color='gray', alpha=0.5)
    
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    
    return plt

# Example usage
def example_gen1_speed_deviation():
    """Example that extracts generator 1 speed deviation for all scenarios."""
    # Load metadata
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # Get all generator 1 speed deviations from non-diverged simulations
    results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='1',
        state_name='w',
        diverged=False
    )
    
    # Plot results grouped by base load
    plt = plot_state_comparison(
        results,
        title='Generator 1 Speed Deviation',
        ylabel='Speed Deviation (pu)',
        legend_key='base_load'
    )
    
    # Save figure
    plt.savefig('gen1_speed_comparison.png')
    plt.close()
    
    # Now filter by fault location and show for a specific load level
    results_filtered = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='1',
        state_name='w',
        sample_idx=1,
        diverged=False
    )
    
    plt = plot_state_comparison(
        results_filtered,
        title='Generator 1 Speed Deviation (Sample Index = 1)',
        ylabel='Speed Deviation (pu)',
        legend_key='fault_location'
    )
    
    plt.savefig('gen1_speed_comparison_sample1.png')
    
    print(f"Found {len(results)} non-diverged scenarios")
    print(f"Found {len(results_filtered)} non-diverged scenarios at sample_idx=1")
    
    return results

def ComputeTSI():
    # Load metadata
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # 1) find all (device_number, bus_num) pairs for GenGENROU δ-states
    gen_pairs = {
        (str(data['device_number']), data['bus_num'])
        for data in state_metadata.values()
        if data.get('model') == 'GenGENROU'
        and data.get('state_name') == 'delta'
    }

    # sort so results are deterministic
    gen_list = sorted(gen_pairs, key=lambda x: (x[1], x[0]))

    # 2) Load each generator's data ONCE (like original), but don't store all in 3D array
    print("⟳ Loading generator delta data...")
    delta_dicts = {}
    for device_number, bus_num in gen_list:
        print(f"⟳ loading δ for GenGENROU device {device_number} on bus {bus_num}")
        d = get_state_timeseries_all(
            simulation_log,
            state_metadata,
            model='GenGENROU',
            device_number=device_number,
            bus_num=bus_num,
            state_name='delta',
            diverged=False
        )
        delta_dicts[(bus_num, device_number)] = d

    if not delta_dicts:
        raise RuntimeError("No generator deltas were loaded!")

    # find common scenarios
    scenario_sets = [set(d.keys()) for d in delta_dicts.values()]
    common_scenarios = sorted(set.intersection(*scenario_sets))
    if not common_scenarios:
        raise RuntimeError("No scenario is common to all generators!")

    print(f"Found {len(common_scenarios)} common scenarios")

    # Get dimensions
    first_key = next(iter(delta_dicts))
    first_scenario = next(iter(delta_dicts[first_key]))
    tvec = delta_dicts[first_key][first_scenario]['tvec']
    T = len(tvec)
    G = len(delta_dicts)

    # Initialize result dictionaries
    tsi_per_scenario = {}
    tsi_ts_per_scenario = {}
    pg_per_scenario = {}
    pl_per_scenario = {}
    ql_per_scenario = {}

    # 3) Process scenarios one by one (memory efficient)
    print("⟳ Processing scenarios...")
    for s_idx, scenario_id in enumerate(common_scenarios):
        if s_idx % 100 == 0:  # Progress indicator
            print(f"   Processing scenario {s_idx + 1}/{len(common_scenarios)}")
        
        # Load scenario data once for pg, pl, ql
        try:
            scenario_data = load_scenario_data(scenario_id, simulation_log)
            pg_per_scenario[scenario_id] = scenario_data['p_gen_scaled']
            pl_per_scenario[scenario_id] = scenario_data['p_load_scaled']
            ql_per_scenario[scenario_id] = scenario_data['q_load_scaled']
            del scenario_data  # Free immediately
        except Exception as e:
            print(f"Error loading scenario {scenario_id}: {e}")
            continue

        # Extract delta values for this scenario from pre-loaded data
        delta_values = np.zeros((G, T))
        for g_idx, key in enumerate(delta_dicts):
            delta_values[g_idx, :] = delta_dicts[key][scenario_id]['values']

        # Compute TSI for this scenario only
        spread_ts = delta_values.max(axis=0) - delta_values.min(axis=0)
        Delta_max = spread_ts.max()
        tsi_scalar = (2*np.pi - Delta_max) / (2*np.pi + Delta_max) * 100
        tsi_ts = (2*np.pi - spread_ts) / (2*np.pi + spread_ts) * 100

        tsi_per_scenario[scenario_id] = tsi_scalar
        tsi_ts_per_scenario[scenario_id] = tsi_ts

        # Free memory for this scenario
        del delta_values, spread_ts, tsi_ts

    # Clear the delta_dicts to free memory before creating final arrays
    del delta_dicts

    # 4) Create final arrays  
    tsi_all = np.array([tsi_per_scenario[sc] for sc in common_scenarios])
    tsi_all_time = np.vstack([tsi_ts_per_scenario[sc] for sc in common_scenarios])

    # Package results
    post_data = {}
    post_data['tsi_per_scenario'] = tsi_per_scenario
    post_data['tsi_all'] = tsi_all
    post_data['tsi_ts_per_scenario'] = tsi_ts_per_scenario
    post_data['tsi_all_time'] = tsi_all_time
    post_data['pg_per_scenario'] = pg_per_scenario
    post_data['pl_per_scenario'] = pl_per_scenario
    post_data['ql_per_scenario'] = ql_per_scenario

    print(f'TSI for all scenarios: {tsi_all.shape}')
    print(f"TSI per scenario", len(tsi_per_scenario))
    print(f'TSI for all time scenarios: {tsi_all_time.shape}')
    # print(f'TSI ts per scenario (all cases - diverged and converged): {tsi_ts_per_scenario.shape}')
    
    # Plotting code (unchanged)
    try:
        import seaborn as sns
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.histplot(tsi_all, bins=20, stat='density', kde=True)
        plt.xlabel('TSI at all times')
        plt.ylabel('Density')

        plt.subplot(1,2,2)
        sns.histplot(np.squeeze(tsi_all_time[:,-1]), bins=20, stat='density', kde=True)
        plt.xlabel('TSI at final time')

        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.axvline(0, color='k', ls='--', lw=1.5)  
            plt.text(-1, plt.ylim()[1] * 0.95, 'unstable', ha='right', va='top', fontsize=10)
            plt.text(1, plt.ylim()[1] * 0.95, 'stable', ha='left', va='top', fontsize=10)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Seaborn is not installed. Please install it if you want to see cool stuff with `pip install seaborn`.")
        
    return post_data

def create_training_samples(post_data: Dict):
    tsi_all_time = post_data['tsi_ts_per_scenario']
    tsi_dict = post_data['tsi_per_scenario']
    # tsi_dict = post_data['tsi_all']
    print(f"tsi_all_time", tsi_all_time)
    print(f"tsi_all_time first element's value", tsi_all_time[list(tsi_all_time.keys())[0]])
    pg_dict  = post_data['pg_per_scenario']
    pl_dict  = post_data['pl_per_scenario']
    ql_dict  = post_data['ql_per_scenario']

    scenario_ids = sorted(tsi_dict.keys())

    first_sid = scenario_ids[0]
    pg_len = len(pg_dict[first_sid])
    pl_len = len(pl_dict[first_sid])
    ql_len = len(ql_dict[first_sid])

    col_name = (    

        [f'pg_{i+1}' for i in range(pg_len)] +
        [f'pl_{i+1}' for i in range(pl_len)] +
        [f'ql_{i+1}' for i in range(ql_len)] +
        ['tsi'] +
        [f'tsi_t{i+1}' for i in range(len(tsi_all_time[list(tsi_all_time.keys())[0]]))]
    )

    rows = []
    for sid in scenario_ids:
        pg = pg_dict[sid]
        pl = pl_dict[sid]
        ql = ql_dict[sid]
        tsi = np.array([tsi_dict[sid]])
        print(f"tsi_all_time[sid]", tsi_all_time[sid])
        tsi_ts = tsi_all_time[sid]
    
        row = np.hstack((pg, pl, ql, tsi, tsi_ts))
        rows.append(row)

    Data = np.vstack(rows)

    # save to .mat file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"Save samples to IEEE500_{Data.shape[0]}_samples_{timestamp}.mat")
    scio.savemat(f'IEEE500_{Data.shape[0]}_samples_{timestamp}.mat', {'Data': Data, 'col_name': col_name})

if __name__ == "__main__":
    # (original) compute generator speeds (ω)
    #example_gen1_speed_deviation()

    post_data = ComputeTSI()
    create_training_samples(post_data)
