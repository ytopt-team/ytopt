from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import csv, time
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

logger = logging.getLogger(__name__)

# An Academy agent that wraps computational tools: in this case,
# a single function that runs locally.
#
# A more sophisticated version might:
#  -- Wrap multiple tools
#  -- Dispatch tool calls to an HPC system
#
# Note that the agent and individual tools have doc strings,
# these are used by the LLM when generating tool calls.

#mkh.py

import csv
import logging
from os import path, walk
from typing import List, TypedDict

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from ConfigSpace import ConfigurationSpace, Float


CLASSES = ["N", "L", "R", "A", "V", "/"]


class ExperimentalKernel:
    """Custom kernel functions for SVM classification."""

    def sigmoid_kernel(
        self,
        x: NDArray[np.int64],
        y: NDArray[np.int64],
        ratio: float,
        coef0_param: float,
    ) -> NDArray[np.float64]:
        return np.asarray(
            np.tanh(ratio * np.dot(x, y.T) + coef0_param)
        )

    def euclidean_dist_matrix(
        self,
        data_1: NDArray[np.int64],
        data_2: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        norms_1 = (data_1**2).sum(axis=1)
        norms_2 = (data_2**2).sum(axis=1)

        return np.asarray(
            np.abs(
                norms_1.reshape(-1, 1)
                + norms_2
                - 2 * np.dot(data_1, data_2.T)
            )
        )

    def gaussian_kernel(
        self,
        x: NDArray[np.int64],
        y: NDArray[np.int64],
        ratio: float,
    ) -> NDArray[np.float64]:
        dists_sq = self.euclidean_dist_matrix(x, y)
        return np.asarray(np.exp(-ratio * dists_sq))

    def mixed_kernel(
        self,
        x: NDArray[np.int64],
        y: NDArray[np.int64],
        mixing_ratio: float,
        sigmoid_ratio: float,
        coef0_param: float,
        gaussian_ratio: float,
    ) -> NDArray[np.float64]:
        sigmoid = self.sigmoid_kernel(
            x, y, sigmoid_ratio, coef0_param
        )
        gaussian = self.gaussian_kernel(
            x, y, gaussian_ratio
        )

        return np.asarray(
            (1 - mixing_ratio) * sigmoid
            + mixing_ratio * gaussian
        )


class TrainData(TypedDict):
    x_train: NDArray[np.int64]
    x_test: NDArray[np.int64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]


class Utils:
    """Read ECG data and prepare training and testing datasets."""

    window_size = 160
    maximum_counting = 10000

    classes = CLASSES
    n_classes = len(classes)

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.count_classes = [0] * self.n_classes
        self.data: List[List[int]] = []

    def read_data(self) -> List[List[int]]:
        x: List[List[int]] = []
        y: List[int] = []

        records: List[str] = []
        annotations: List[str] = []
        all_file_paths: List[str] = []

        for dirpath, _, filenames in walk(self.data_path):
            for filename in filenames:
                all_file_paths.append(
                    path.join(dirpath, filename)
                )

        all_file_paths.sort()

        logging.info("Reading data...")

        for file_path in all_file_paths:
            _, file_extension = path.splitext(file_path)

            if file_extension == ".txt":
                annotations.append(file_path)
            elif file_extension == ".csv":
                records.append(file_path)

        for record, annotation in zip(records, annotations):
            signals: List[int] = []

            with open(record, "r", newline="") as csvfile:
                reader = csv.reader(
                    csvfile,
                    quotechar="|",
                    delimiter=",",
                )
                next(reader, None)

                for row in reader:
                    signals.append(int(row[1]))

            with open(annotation, "r") as txtfile:
                content = txtfile.readlines()

            for index in range(1, len(content)):
                splitted = [
                    value
                    for value in content[index].strip().split(" ")
                    if value != ""
                ]

                pos = int(splitted[1])
                arrhythmia_type = splitted[2]

                if arrhythmia_type not in self.classes:
                    continue

                arrhythmia_index = self.classes.index(
                    arrhythmia_type
                )

                # Limit the number of examples per class.
                if (
                    self.count_classes[arrhythmia_index]
                    > self.maximum_counting
                ):
                    continue

                self.count_classes[arrhythmia_index] += 1

                if (
                    self.window_size < pos
                    < len(signals) - self.window_size
                ):
                    beat = signals[
                        pos - self.window_size + 1:
                        pos + self.window_size
                    ]

                    x.append(beat)
                    y.append(arrhythmia_index)

        self.data = [
            features + [label]
            for features, label in zip(x, y)
        ]

        return self.data

    def get_train_data(
        self,
        sample_size: int = 5000,
    ) -> TrainData:
        if not self.data:
            self.read_data()

        if not self.data:
            raise ValueError(
                f"No valid ECG samples were found in {self.data_path}"
            )

        df = DataFrame(self.data)
        label_column = df.shape[1] - 1

        class_frames = {}

        # Preserve the original class-specific sampling strategy.
        for class_index in range(self.n_classes):
            class_df = df[df[label_column] == class_index]

            if class_df.empty:
                raise ValueError(
                    f"Class {class_index} ({self.classes[class_index]}) "
                    "has no samples. Cannot construct the training dataset."
                )

            if class_index == 0:
                if len(class_df) < sample_size:
                    raise ValueError(
                        f"Class 0 has {len(class_df)} samples, "
                        f"but {sample_size} are required for sampling "
                        "without replacement."
                    )

                class_frames[class_index] = class_df.sample(
                    n=sample_size,
                    random_state=42,
                )
            else:
                class_frames[class_index] = DataFrame(
                    resample(
                        class_df,
                        replace=True,
                        n_samples=sample_size,
                        random_state=121 + class_index,
                    )
                )

        train_df = concat(
            [
                class_frames[i]
                for i in range(self.n_classes)
            ],
            ignore_index=True,
        )

        train, test = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
        )

        x_train = train.iloc[:, :-1].to_numpy()
        x_test = test.iloc[:, :-1].to_numpy()
        y_train = train[label_column].to_numpy()
        y_test = test[label_column].to_numpy()

        logging.info(
            "Train data size: x %s | y %s",
            x_train.shape,
            y_train.shape,
        )
        logging.info(
            "Test data size: x %s | y %s",
            x_test.shape,
            y_test.shape,
        )

        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }


def construct_config(random_seed: int):
    """Sample the five mixed-kernel SVM hyperparameters."""

    cs = ConfigurationSpace(seed=random_seed)

    p0 = Float(
        "p0",
        bounds=(0.0, 1.0),
    )
    p1 = Float(
        "p1",
        bounds=(0.00005, 0.0003),
        default=0.0001,
    )
    p2 = Float(
        "p2",
        bounds=(0.8, 1.0),
        default=1.0,
    )
    p3 = Float(
        "p3",
        bounds=(-1.0, 1.0),
        default=0.0,
    )
    p4 = Float(
        "p4",
        bounds=(0.37, 10.0),
        default=1.0,
    )

    cs.add([p0, p1, p2, p3, p4])
    config = cs.sample_configuration()

    mixing_ratio = config["p0"]
    sigmoid_ratio = config["p1"]
    gaussian_ratio = config["p2"]
    coef0_param = config["p3"]
    C_param = config["p4"]

    return (
        mixing_ratio,
        sigmoid_ratio,
        gaussian_ratio,
        coef0_param,
        C_param,
    )


def calculate_class_scores(
    y_test: NDArray[np.int64],
    y_predict: NDArray[np.int64],
) -> List[float]:
    """Calculate per-class accuracy."""

    predict_count = [0] * len(CLASSES)
    true_count = [0] * len(CLASSES)

    for actual, predicted in zip(y_test, y_predict):
        true_count[int(actual)] += 1

        if actual == predicted:
            predict_count[int(actual)] += 1

    return [
        predict_count[i] / true_count[i]
        if true_count[i] > 0
        else 0.0
        for i in range(len(CLASSES))
    ]


def evaluate_model(
    x_train: NDArray[np.int64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.int64],
    y_test: NDArray[np.int64],
    smiles,
) -> NDArray[np.float64]:
    """
    Train the mixed-kernel SVM and evaluate its predictions.

    The function replaces Classification.run() without using
    a Classification class.
    """

    result: List[List[float]] = []

    # Accept either a tuple of parameters or a string representation.
    if isinstance(smiles, tuple):
        parameters = [float(value) for value in smiles]
    elif isinstance(smiles, str):
        parameters = [
            float(value.strip())
            for value in smiles.replace("(", "").replace(")", "").split(",")
        ]
    else:
        parameters = [float(value) for value in smiles]

    if len(parameters) != 5:
        raise ValueError(
            "Expected five hyperparameters: "
            "mixing_ratio, sigmoid_ratio, gaussian_ratio, "
            "coef0_param, and C_param."
        )

    (
        mixing_ratio,
        sigmoid_ratio,
        gaussian_ratio,
        coef0_param,
        C_param,
    ) = parameters

    logging.info(
        "Mixed-kernel SVM with mixing ratio %s begins",
        mixing_ratio,
    )

    kernel = ExperimentalKernel()

    clf = make_pipeline(
        StandardScaler(),
        SVC(
            C=C_param,
            kernel=lambda x, y: kernel.mixed_kernel(
                x,
                y,
                mixing_ratio,
                sigmoid_ratio,
                coef0_param,
                gaussian_ratio,
            ),
        ),
    )

    logging.info("Training...")
    clf.fit(x_train, y_train)

    logging.info("Predicting...")
    y_predict = np.asarray(
        clf.predict(x_test)
    ).astype(np.int64)

    score = calculate_class_scores(y_test, y_predict)

    logging.info("Per-class accuracy: %s", score)

    overall_accuracy = float(
        clf.score(x_test, y_test)
    )

    score.append(overall_accuracy)
    result.append(score)

    logging.info(
        "Mixed-kernel SVM with mixing ratio %s completes",
        mixing_ratio,
    )

    return np.array(result)

class MySimAgent(Agent):
    """Agent for running tools to characterize avg_accuracy."""

    @action
    async def compute_avg_accuracy(self, smiles: str) -> float:
        """Compute the avg_accuracy."""

        # Start the timer
        start_time = time.perf_counter()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

        data_path = path.join(path.dirname(__file__), "./", "data")

        utils = Utils(data_path)
        train_data = utils.get_train_data()

       # random_seed = np.random.randint(1,10000)
       # smiles = construct_config(random_seed)
        result = evaluate_model(train_data["x_train"], train_data["y_train"], 
                          train_data["x_test"], train_data["y_test"], smiles)

        avg_accuracy = np.mean(result)

        # End the timer
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        #print(elapsed_time)

        if isinstance(smiles, tuple):
            smiles = ",".join(map(str, smiles))
        smiles = smiles.replace("(", "").replace(")", "").split(",")
        mixing_ratio = float(smiles[0].strip())
        sigmoid_ratio = float(smiles[1].strip())
        gaussian_ratio = float(smiles[2].strip())
        coef0_param = float(smiles[3].strip())
        C_param = float(smiles[4].strip())

        #write the configuration and metric to CSV
        with open('./results.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([mixing_ratio, sigmoid_ratio, gaussian_ratio, coef0_param, C_param, avg_accuracy, elapsed_time])

        print("Result: ", avg_accuracy)

        return avg_accuracy

def make_sim_tool(handle: Handle[MySimAgent]) -> Tool:
    """Wraps an academy handle in a langchain tool.

    Note: Since the documentation of the tool is used by the language
    model, a specific wrapper method may need to be written per agent.
    """

    @tool
    async def compute_avg_accuracy(smiles: str) -> float:
        """Compute the avg_accuracy."""
        return await handle.compute_avg_accuracy(smiles)

    return compute_avg_accuracy

class Configuration(Agent):
    @action
    async def construct_config(self, random_seed: int) -> str:
        return construct_config(random_seed)

# An Academy agent that creates a LangChain agent that will respond to
# questions about avg_accuracy by running a ReACT loop
class Orchestrator(Agent):
    """Orchestrate a scientific workflow."""

    def __init__(
        self,
        model: str,
        access_token: str,
        config: Handle[Configuration],
        simulators: list[Handle[MySimAgent]],
        base_url: str | None = None,
    ):
        self.model = model
        self.access_token = access_token
        self.base_url = base_url
        self.config = config
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
        """Use other agents to answer questions about avg_accuracy."""

        random_seed = np.random.randint(1,10000)
        config = await self.config.construct_config(random_seed)
        goal = "You are a hyperparameter optimization expert. Search the parameter space cs, where mixing_ratio is a float number between 0.00009 and 0.0009, sigmoid_ratio is a float number between 0.00009 and 0.0009, gaussian_ratio is a float number between 0.96 and 1.5, coef0_param is a float number between -1.0 and 1.0, C_param is a float number between 0.37 and 10.0, execute the function compute_avg_accuracy to maximize the metric avg_accuracy. Given smiles= " + str(config) + goal
        print(goal)

        # This call runs the ReACT loop, in which:
        #   1) the LLM is used to determine which tool to call,
        #   2) the tool is called (by messaging the Academy agent)

        response= await self.react_loop.ainvoke(
            {'messages': [{'role': 'user', 'content': goal}]},
        )
        #last_ai_message = next(m for m in reversed(response["messages"]) if getattr(m, "type", None) == "ai")
        last_ai_message = next(m for m in reversed(response["messages"]) if getattr(m, "type", None) == "tool")

        #print(last_ai_message.content)
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

  #  mp_context = multiprocessing.get_context('spawn')
  #  executor = ProcessPoolExecutor(
  #      max_workers=3,
  #      mp_context=mp_context,
  #  )

    async with await Manager.from_exchange_factory(
  #      factory=HttpExchangeFactory(),
        # Agents are run by the manager in the processes of this
        # process pool executor.
  #      executors=executor,
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
        log_config=recommended_logging(),
    ) as manager:
        configuration = await manager.launch(Configuration)
        simulator = await manager.launch(MySimAgent)
        orchestrator = await manager.launch(
            Orchestrator,
            kwargs={
                'model': model,
                'access_token': token,
                'config': configuration, 
                'simulators': [simulator],
                'base_url': url,
            },
        )

        msg = ' as a starting point, use Bayesian Optimization to search the parameter space cs to choose the values for mixing_ratio, sigmoid_ratio, gaussian_ratio, coef0_param, C_param, execute the function compute_avg_accuracy to maximize the metric avg_accuracy. avg_accuracy is equal or bigger than the constraint 0.97 for stopping. At the end of all the runs, present the maximum of all avg_accuracy values with the configuration of mixing_ratio, sigmoid_ratio, gaussian_ratio, coef0_param, C_param'
        #print(msg)
        logger.info(
            'Invoking process("%s") on %s',
            msg,
            orchestrator.agent_id,
        )

        auto_mode = True
        max_rounds = 2 if auto_mode else None
        rounds = 0
        best = 0.0

        # We'll run the entire workflow in a single trace
        with trace("LLM as a judge"):
         # We'll run the entire workflow in a single trace
            while True:
                result = float(await orchestrator.answer(msg))
                if result >= best:
                    best = result
                print('Best: ',best)
                # The possible best: 0.973
                if best >= 0.97:
                    print("Result is equal or bigger than the constraint 0.97 for stopping.")
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
