from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace

from langchain.agents import create_agent
#from langchain.tools import Tool
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from academy.agent import action
from academy.agent import Agent
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.manager import Manager
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.logging.recommended import recommended_logging

#for applications
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from scipy.special import jn, yn  # Bessel functions
from sklearn.metrics import mean_squared_error
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer

logger = logging.getLogger(__name__)

def is_auto_mode() -> bool:
    """Return True when examples should bypass interactive prompts."""
    return os.environ.get("EXAMPLES_INTERACTIVE_MODE", "").lower() == "auto"


# An Academy agent that wraps computational tools: in this case,
# a single function that runs locally.
#
# A more sophisticated version might:
#  -- Wrap multiple tools
#  -- Dispatch tool calls to an HPC system
#
# Note that the agent and individual tools have doc strings,
# these are used by the LLM when generating tool calls.

# parameter space
def construct_config(random_seed: int):
    cs = ConfigurationSpace(seed=random_seed)
    # m: M_GAUSSIANS
    p0 = Integer('p0', bounds=(0, 10), default=6)
    # n: N_SIGMOIDS
    p1 = Integer('p1', bounds=(0, 10), default=6)
    # smoothing: SPLINE_SMOOTH
    p2 = Float('p2', bounds=(0.0001, 0.01), default=0.001)
    cs.add([p0, p1, p2])
    configs = cs.sample_configuration()

    m =configs['p0']   # @param {type:"integer"} range: (0, 10)
    n =configs['p1']    # @param {type:"integer"} range: (0, 10)
    if m == 0 and n == 0:
        configs = cs.sample_configuration()
        m =configs['p0']
        n =configs['p1']

    smoothing =configs['p2'] # @param {type:"float"} Controls spline aggressiveness range: (0.0001, 0.1)

    return m, n, smoothing

# --- 1. Define Comprehensive Function Set ---
# Format: "Name": (Function, (min_x, max_x))
test_functions = {
    # --- Classic Feynman Shapes ---
    "Feynman I.6.20 (Gaussian)": (lambda x: np.exp(-(x**2)/2), (-3, 3)),
    "Feynman I.10.7 (Lorentzian)": (lambda x: 1.0 / (1 + x**2), (-3, 3)),
    "Feynman I.12.5 (Quadratic)": (lambda x: x**2, (-2, 2)),
    "Feynman II.35.21 (Tanh)": (lambda x: np.tanh(x), (-3, 3)),
    "Feynman I.15.10 (Inv Root)": (lambda x: 1.0 / np.sqrt(1 - x**2 + 1e-6), (-0.9, 0.9)), # Relativistic p
    "Feynman I.41.16 (Rad. Density)": (lambda x: (x**3)/(np.exp(x)-1), (0.1, 5)), # Blackbody
    "Feynman I.39.1 (Inverse)": (lambda x: 1.0/x, (0.5, 5)), # PV=const

    # --- Special Functions ---
    "Bessel J0 (1st Kind)": (lambda x: jn(0, x), (0, 10)),
    "Bessel J1 (1st Kind)": (lambda x: jn(1, x), (0, 10)),
    "Bessel Y0 (2nd Kind)": (lambda x: yn(0, x), (0.2, 10)), # Diverges at 0
    "Bessel Y1 (2nd Kind)": (lambda x: yn(1, x), (0.2, 10)),
}

# --- 2. Define Mixture Model ---
def sigmoid(x, height, center, width):
    return height / (1 + np.exp(-np.clip((x - center) / width, -100, 100)))

def gaussian(x, height, mean, width):
    return height * np.exp(-((x - mean)**2) / (2 * (width + 1e-6)**2))

def gaussian(x, height, mean, width):
    return height * np.exp(-((x - mean)**2) / (2 * (width + 1e-6)**2))

def mixture_model(params, x, m, n):
    y_pred = np.zeros_like(x)
    # Gaussians
    for i in range(m):
        idx = i * 3
        y_pred += gaussian(x, params[idx], params[idx+1], params[idx+2])
    # Sigmoids
    offset = m * 3
    for j in range(n):
        idx = offset + j * 3
        y_pred += sigmoid(x, params[idx], params[idx+1], params[idx+2])
    return y_pred

def mixture_residuals(params, x, y_true, m, n):
    return mixture_model(params, x, m, n) - y_true

def fit_mixture(x, y, m, n):
    # Robust Heuristic Initialization
    initial_params = []

    # Spread centers evenly across the domain
    centers = np.linspace(np.min(x), np.max(x), max(m, n))

    for i in range(m):
        initial_params.extend([np.std(y), centers[i % len(centers)], 1.0])
    for i in range(n):
        initial_params.extend([np.max(y)-np.min(y), centers[i % len(centers)], 1.0])

    res = least_squares(mixture_residuals, initial_params, args=(x, y, m, n),
                        method='lm', max_nfev=6000)
    return res.x

class MySimAgent(Agent):
    """Agent for running tools to characterize mse_diff."""

    @action
    async def compute_mse_diff(self, smiles: str) -> float:
        """Compute the mse_diff."""
        random_seed = np.random.randint(1,10000)
        m, n, smoothing = construct_config(random_seed)

        print(f"--- Benchmarking: {m}G + {n}S vs Splines (Smoothing={smoothing}) ---")

        results = []

        for i, (name, (func, domain)) in enumerate(test_functions.items()):
            # Generate Data specific to domain
            X = np.linspace(domain[0], domain[1], 150)
            y_true = func(X)

            # --- A. Mixture Model ---
            mix_param_count = 3 * (m + n)
            try:
                mix_params = fit_mixture(X, y_true, m, n)
                y_mix = mixture_model(mix_params, X, m, n)
                mse_mix = mean_squared_error(y_true, y_mix)
            except Exception as e:
                y_mix = np.zeros_like(X)
                mse_mix = float('inf')
                print(f"Fit failed for {name}: {e}")

            # --- B. Spline Model ---
            # k=3 is cubic spline. s controls smoothing (number of knots)
            spl = UnivariateSpline(X, y_true, k=3, s=smoothing)
            y_spline = spl(X)
            mse_spline = mean_squared_error(y_true, y_spline)

            # Get Spline Complexity
            spline_coeffs = spl.get_coeffs()
            spline_param_count = len(spline_coeffs)
            results.append({
                "Name": name,
                "MSE_Mix": mse_mix,
                "P_Mix": mix_param_count,
                "MSE_Spl": mse_spline,
                "P_Spl": spline_param_count
            })

        # --- Summary Table ---
        # Header
        print(f"{'Equation':<35} | {'Mix MSE':<20} | {'Spline MSE':<20} | {'Winner'}")
        print("-" * 95)
    
        mix_avg = 0.0
        spl_avg = 0.0
        avg = 0.0
        for r in results:
            # Determine winner based on MSE
            winner = "Mix" if r['MSE_Mix'] < r['MSE_Spl'] else "Spline"
    
            mix_str = f"{r['MSE_Mix']:.1e}"
            spl_str = f"{r['MSE_Spl']:.1e}"
    
            print(f"{r['Name']:<35} | {mix_str:<20} | {spl_str:<20} | {winner}")
            mix_avg += r['MSE_Mix']
            spl_avg += r['MSE_Spl']
    
        avg = mix_avg - spl_avg
        mse_diff = abs(avg)
        print(mse_diff)
        return mse_diff

def make_sim_tool(handle: Handle[MySimAgent]) -> Tool:
    """Wraps an academy handle in a langchain tool.

    Note: Since the documentation of the tool is used by the language
    model, a specific wrapper method may need to be written per agent.
    """

    @tool
    async def compute_mse_diff(smiles: str) -> float:
        """Compute the mse_diff."""
        return await handle.compute_mse_diff(smiles)

    return compute_mse_diff


# An Academy agent that creates a LangChain agent that will respond to
# questions about mse_diff by running a ReACT loop
class Orchestrator(Agent):
    """Orchestrate a scientific workflow."""

    def __init__(
        self,
        model: str,
        access_token: str,
        simulators: list[Handle[MySimAgent]],
        base_url: str | None = None,
    ):
        self.model = model
        self.access_token = access_token
        self.base_url = base_url
        self.simulators = simulators

    async def agent_on_startup(self) -> None:
        llm = ChatOpenAI(
            model=self.model,
            api_key=self.access_token,
            base_url=self.base_url,
        )

        tools = [make_sim_tool(agent) for agent in self.simulators]
        # The following call creates the LangChain agent
        self.react_loop = create_agent(llm, tools=tools)

    @action
    async def answer(self, goal: str) -> str:
        """Use other agents to answer questions about mse_diff."""

        # This call runs the ReACT loop, in which:
        #   1) the LLM is used to determine which tool to call,
        #   2) the tool is called (by messaging the Academy agent)
        response= await self.react_loop.ainvoke(
            {'messages': [{'role': 'user', 'content': goal}]},
        )
        #last_ai_message = next(m for m in reversed(response["messages"]) if getattr(m, "type", None) == "ai")
        last_ai_message = next(m for m in reversed(response["messages"]) if getattr(m, "type", None) == "tool")


        # Print or return only the content
        print(last_ai_message.content)
        return last_ai_message.content

# The main program creates the two Academy agents, SIMULATOR and ORCHESTRATOR
async def main() -> int:
    model = await asyncio.to_thread(input, 'Please input a model name: ')
    token = await asyncio.to_thread(input, 'Please input an access token: ')
    url_input = await asyncio.to_thread(
        input,
        '(Optionally) Input a model api url: ',
    )
    url = url_input if len(url_input) > 0 else None

    mp_context = multiprocessing.get_context('spawn')
    executor = ProcessPoolExecutor(
    #executor = ThreadPoolExecutor(
        max_workers=3,
        mp_context=mp_context,
    )

    async with await Manager.from_exchange_factory(
        #factory=LocalExchangeFactory(),
        factory=HttpExchangeFactory(),
        # Agents are run by the manager in the processes of this
        # process pool executor.
        executors=executor,
        log_config=recommended_logging(),
    ) as manager:
        simulator = await manager.launch(MySimAgent)
        orchestrator = await manager.launch(
            Orchestrator,
            kwargs={
                'model': model,
                'access_token': token,
                'simulators': [simulator],
                'base_url': url,
            },
        )

        msg = 'given smiles=construct_config,Execute the function compute_mse_diff, minimize the metric mse_diff'
        #print(msg)
        logger.info(
            'Invoking process("%s") on %s',
            msg,
            orchestrator.agent_id,
        )

        #auto_mode = is_auto_mode()
        auto_mode = True
        max_rounds = 3 if auto_mode else None
        rounds = 0

        # We'll run the entire workflow in a single trace
        with trace("LLM as a judge"):
         # We'll run the entire workflow in a single trace
            while True:  
                result = float(await orchestrator.answer(msg))
                # The best: 6.1e-07
                if result <= 6.1e-04:
                    print("Result is equal or less than the constraint 6.1e-07 for stopping.")
                    break
                if auto_mode:
                    rounds += 1
                    if max_rounds is not None and rounds >= max_rounds:
                        print("Stopping after limited rounds.")
                        break
         
            logger.info('Received result: "%s"', result)

    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
