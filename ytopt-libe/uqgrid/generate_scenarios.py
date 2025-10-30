import itertools
import numpy as np
import uuid
import json
import os
import copy
from joblib import Parallel, delayed
import argparse
import yaml
from pathlib import Path

from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config   import IntegrationConfig
from uqgrid.io.parse            import load_psse, add_dyr


def generate_perturbations(base_p, base_q,
                                *,
                                noise_type="normal", var=0.1,
                                rng=None, return_noise=False):
    """
    Apply per-bus noise -> return scaled loads.
    If return_noise=True, also return (p_noise, q_noise)

        P_scaled = base_p * (1 + p_noise)
        Q_scaled = base_q * (1 + q_noise)
    TODO: may need to change it so it's more flexible.
    """
    rng = np.random.default_rng() if rng is None else rng

    if noise_type == "normal":
        p_noise = rng.normal(0.0, var, size=base_p.shape)
        q_noise = rng.normal(0.0, var, size=base_q.shape)
    elif noise_type == "uniform":
        half = np.sqrt(3 * var)              # Var(U[-a,a]) = var
        p_noise = rng.uniform(-half, half, size=base_p.shape)
        q_noise = rng.uniform(-half, half, size=base_q.shape)
    elif noise_type == "none":
        p_noise = q_noise = np.zeros_like(base_p)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    p_scaled = base_p * (1.0 + p_noise)
    q_scaled = base_q * (1.0 + q_noise)

    if return_noise:
        return p_scaled, q_scaled, p_noise, q_noise
    return p_scaled, q_scaled


def sample_scenarios(n_samples, fault_locations, fault_impedances):
    return list(itertools.product(range(n_samples), fault_locations, fault_impedances))


def generate_metadata(scenarios):
    """
    Creates one UUID per scenario.  Metadata no longer contains 'base_load'.
    """
    metadata = {}
    for sample_idx, floc, fz in scenarios:
        sid = str(uuid.uuid4())
        metadata[sid] = {
            "sample_idx"     : sample_idx,
            "fault_location" : floc,
            "fault_impedance": fz,
        }
    with open("scenario_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


def run_single_scenario(
        base_psys, scenario, scenario_id,
        base_p_load, base_q_load,
        base_p_gen,  base_q_gen,
        noise_type="normal", noise_var=0.1,
        balance_generation=False):

    psys = copy.deepcopy(base_psys)

    #  Draw noise and obtain positive, scaled loads
    pL_scaled, qL_scaled, pL_noise, qL_noise = generate_perturbations(
        base_p_load, base_q_load,
        noise_type=noise_type, var=noise_var,
        return_noise=True)

    pG_scaled, qG_scaled, pG_noise, qG_noise = generate_perturbations(
        base_p_gen, base_q_gen,
        noise_type=noise_type, var=noise_var,
        return_noise=True)

    if balance_generation:
        sum_pL = np.sum(pL_scaled)
        sum_qL = np.sum(qL_scaled)
        sum_pG = np.sum(pG_scaled)
        sum_qG = np.sum(qG_scaled)

        if sum_pG != 0: pG_scaled *= (sum_pL / sum_pG)
        if sum_qG != 0: qG_scaled *= (sum_qL / sum_qG)

    psys.set_load_pq(pL_scaled, qL_scaled)
    psys.set_gen_pq(pG_scaled, qG_scaled)

    psys.add_busfault(scenario["fault_location"],
                      scenario["fault_impedance"], 0.25)
    psys.createYbusComplex()

    cfg = IntegrationConfig(
        tend=10.0, dt=1/120.0, power_injection=False,
        ton=0.25, toff=0.4, verbose=False, petsc=True
    )

    try:
        sim       = integrate_system(psys, cfg)
        diverged  = False
    except Exception:
        sim       = {"history": None, "tvec": None}
        diverged  = True

    os.makedirs("simulation_data", exist_ok=True)
    fn = f"simulation_data/scenario_{scenario_id}.npz"
    np.savez_compressed(
        fn,
        history=sim["history"],
        tvec=sim["tvec"],
        #  loads
        p_load_scaled=pL_scaled, q_load_scaled=qL_scaled,
        p_load_noise =pL_noise,  q_load_noise =qL_noise,
        #  generators
        p_gen_scaled =pG_scaled, q_gen_scaled =qG_scaled,
        p_gen_noise  =pG_noise,  q_gen_noise  =qG_noise,
    )
    return {"file": fn, "diverged": diverged}


def run_simulation_driver_batched(
        raw, dyr, scenarios_metadata,
        *, noise_type="normal", noise_var=0.1,
        balance_generation=True, 
        n_jobs=-1, batch_size=10):

    scenario_ids   = list(scenarios_metadata.keys())
    simulation_log = {}

    for batch_start in range(0, len(scenario_ids), batch_size):
        batch_ids = scenario_ids[batch_start : batch_start+batch_size]
        print(f"Processing batch {batch_start//batch_size + 1}"
              f" / {int(np.ceil(len(scenario_ids)/batch_size))}")

        base_psys = load_psse(raw)
        add_dyr(base_psys, dyr)
        base_psys.export_state_metadata()

        base_p, base_q = base_psys.get_load_pq()
        base_pG, base_qG = base_psys.get_gen_pq()

        batch_args = [
            (base_psys, scenarios_metadata[sid], sid,
             base_p, base_q, base_pG, base_qG, noise_type, noise_var, 
             balance_generation)
            for sid in batch_ids
        ]

        batch_out = Parallel(n_jobs=n_jobs)(
            delayed(run_single_scenario)(*args) for args in batch_args)

        for sid, out in zip(batch_ids, batch_out):
            simulation_log[sid] = {**scenarios_metadata[sid], **out}

        with open("simulation_log.json", "w") as f:
            json.dump(simulation_log, f, indent=4)

        del base_psys

    return simulation_log

def main():

    # default config
    DEFAULT_CONFIG = {
       'PowerGridModel': 'IEEE-9',
       'SAMPLES_PER_FAULT_LOCATION': 1,
       'FAULT_IMPEDANCES': [0.0001, 0.01],
       'NOISE_TYPE': 'uniform',
       'NOISE_VAR': 1.0,
       'BALANCE_GENERATION': True,
       'N_JOBS': 10,
       'BATCH_SIZE': 10
    }  

    # parse command-line flag --config
    parser = argparse.ArgumentParser(
        description="Scenario generation + TSI analysis pipeline")
    parser.add_argument(
        "--config",
        help="Path to YAML file with hyper-parameters (overrides defaults)")
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VAL",
        help="Override individual keys, e.g.  --set NOISE_VAR=0.2 BATCH_SIZE=4")
    cli_args = parser.parse_args()

    #begin with defaults → YAML 
    CONFIG = dict(DEFAULT_CONFIG)          # start with defaults

    # YAML file, if provided, update the config
    if cli_args.config:
        cfg_path = Path(cli_args.config).expanduser()
        with cfg_path.open() as f:
            yaml_cfg = yaml.safe_load(f) or {}
        CONFIG.update(yaml_cfg)

    #PowerGridModel = "IEEE-9" #"ACTIVSg200" #  "IEEE-39" #  "ACTIVSg500" #
    PowerGridModel = CONFIG['PowerGridModel']
    # Scenario sampling configuration
    #SAMPLES_PER_FAULT_LOCATION = 1     # Samples per fault location
    SAMPLES_PER_FAULT_LOCATION = CONFIG['SAMPLES_PER_FAULT_LOCATION']
    #FAULT_IMPEDANCES = [0.0001]         # Fault impedance values [p.u]
    FAULT_IMPEDANCES = CONFIG['FAULT_IMPEDANCES']
    #N_JOBS = 10
    N_JOBS = CONFIG['N_JOBS']
    #BATCH_SIZE = 10
    BATCH_SIZE = CONFIG['BATCH_SIZE']

    #PATH = "/usr/workspace/hiop/dane/project/scidac_2025/cheng38/uqgrid/bin/data"
    PATH = "../../data"

    if PowerGridModel == "IEEE-9":
        raw = f"{PATH}/ieee9_v33.raw"
        dyr = f"{PATH}/ieee9bus_gov.dyr"
        n_bus = 9
    elif PowerGridModel == "IEEE-39":
        raw = f"{PATH}/IEEE39_v33.raw"
        dyr = f"{PATH}/IEEE39_gov.dyr"
        n_bus = 39
    elif PowerGridModel == "ACTIVSg200":
        raw = f"{PATH}/ACTIVSG/ACTIVSg200.raw"
        dyr = f"{PATH}/ACTIVSG/ACTIVSg200.dyr"
        n_bus = 49
    elif PowerGridModel == "ACTIVSg500":
        raw = f"{PATH}/ACTIVSG/ACTIVSg500.raw"
        dyr = f"{PATH}/ACTIVSG/ACTIVSg500.dyr"
        n_bus = 90
    else:
        raise RuntimeError(f"{PowerGridModel} is an invalid model!")

    fault_locations   = list(range(1, n_bus + 1))

    # Calculate total scenarios
    total_scenarios = SAMPLES_PER_FAULT_LOCATION * len(fault_locations) * len(FAULT_IMPEDANCES)
    print(f"Configuration: {total_scenarios} total scenarios")
    print(f"  - {SAMPLES_PER_FAULT_LOCATION} noise samples per fault location")
    print(f"  - {len(fault_locations)} fault locations.")
    print(f"  - {len(FAULT_IMPEDANCES)} fault impedances: {FAULT_IMPEDANCES}")

    scenarios = sample_scenarios(
        SAMPLES_PER_FAULT_LOCATION, fault_locations, FAULT_IMPEDANCES)
    metadata  = generate_metadata(scenarios)

    # noise settings
    #TODO Separate the noise in two parts, one for generators and one for loads
    noise_type = "normal"   # "normal", "uniform", "none", 
    noise_var  = 0.10       # variance of the chosen distribution TODO: need to change this to be more flexible

    balance_generation = False

    run_simulation_driver_batched(
        raw, dyr, metadata,
        noise_type=noise_type, noise_var=noise_var,
        balance_generation=balance_generation,
        n_jobs=N_JOBS, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
