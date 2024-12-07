import configparser
import itertools
import json
import os
import re
from json import JSONDecodeError
from pathlib import Path
from typing import List

import numpy as np

import conf
from conf import MAB_UCB_EXPLORATION_FACTOR, MAB_UCB2_ALPHA, MAB_KL_UCB_C
from main import main
from . import consts


def write_custom_configfile(expname: str, strategy: str, axis_pre: str, axis_post: str, params_names: List[str],
                            params_values: List[float]):
    outfile_stats = generate_outfile_name(consts.PREFIX_STATSFILE, strategy, axis_pre, axis_post, params_names,
                                          params_values) + consts.SUFFIX_STATSFILE
    outfile_mabstats = generate_outfile_name(consts.PREFIX_MABSTATSFILE, strategy, axis_pre, axis_post, params_names,
                                             params_values) + consts.SUFFIX_MABSTATSFILE

    outconfig = configparser.ConfigParser()

    # other
    outconfig.add_section(conf.SEC_SIM)
    outconfig.set(conf.SEC_SIM, conf.SPEC_FILE, "spec.yml")
    outconfig.set(conf.SEC_SIM, conf.STAT_PRINT_INTERVAL, str(360))
    outconfig.set(conf.SEC_SIM, conf.STAT_PRINT_FILE, outfile_stats)
    outconfig.set(conf.SEC_SIM, "mab-stats-print-file", outfile_mabstats)
    outconfig.set(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, str(28800))
    outconfig.set(conf.SEC_SIM, conf.PLOT_RESP_TIMES, "false")
    outconfig.set(conf.SEC_SIM, conf.SEED, str(123))
    outconfig.set(conf.SEC_SIM, conf.EDGE_EXPOSED_FRACTION, str(0.25))
    outconfig.add_section(conf.SEC_POLICY)
    outconfig.set(conf.SEC_POLICY, conf.POLICY_NAME, "basic")
    outconfig.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, str(0.3))
    outconfig.add_section(conf.SEC_LB)
    outconfig.set(conf.SEC_LB, conf.LB_POLICY, "random-lb")

    outconfig.add_section(conf.SEC_MAB)
    outconfig.set(conf.SEC_MAB, conf.MAB_UPDATE_INTERVAL, str(300))
    outconfig.set(conf.SEC_MAB, conf.MAB_NON_STATIONARY_ENABLED, "false" if axis_pre == axis_post else "true")
    outconfig.set(conf.SEC_MAB, conf.MAB_LB_POLICIES,
                  "random-lb, round-robin-lb, mama-lb, const-hash-lb, wrr-speedup-lb, wrr-memory-lb, wrr-cost-lb")
    outconfig.set(conf.SEC_MAB, conf.MAB_STRATEGY, strategy)

    for i, param_name in enumerate(params_names):
        outconfig.set(conf.SEC_MAB, param_name, str(params_values[i]))

    # stationary
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_ALPHA, str(1) if axis_pre == consts.RewardFnAxis.LOADIMB.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_BETA,
                  str(1) if axis_pre == consts.RewardFnAxis.RESPONSETIME.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_GAMMA, str(1) if axis_pre == consts.RewardFnAxis.COST.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_DELTA, str(1) if axis_pre == consts.RewardFnAxis.UTILITY.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_ZETA, str(1) if axis_pre == consts.RewardFnAxis.VIOLATIONS.value else str(0))

    # non-stationary
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_ALPHA_POST,
                  str(1) if axis_post == consts.RewardFnAxis.LOADIMB.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_BETA_POST,
                  str(1) if axis_post == consts.RewardFnAxis.RESPONSETIME.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_GAMMA_POST, str(1) if axis_post == consts.RewardFnAxis.COST.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_DELTA_POST,
                  str(1) if axis_post == consts.RewardFnAxis.UTILITY.value else str(0))
    outconfig.set(conf.SEC_MAB, conf.MAB_REWARD_ZETA_POST,
                  str(1) if axis_post == consts.RewardFnAxis.VIOLATIONS.value else str(0))
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", expname, "results", consts.CONFIG_FILE))

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as of:
        outconfig.write(of)
    return config_path


class MABExperiment_Param:
    def __init__(self, name: str, start: float, step: float, end: float):
        self.name = name
        self.start = start
        self.step = step
        self.end = end


def get_param_simple_name(full_name: str) -> str:
    if full_name == MAB_UCB_EXPLORATION_FACTOR:
        return "ef"
    elif full_name == MAB_UCB2_ALPHA:
        return "alpha"
    elif full_name == MAB_KL_UCB_C:
        return "c"
    return full_name


def generate_outfile_name(prefix, strategy, axis_pre, axis_post, params_names, params_values):
    output_suffix = consts.DELIMITER_HYPHEN.join([prefix, strategy, consts.DELIMITER_AXIS.join([axis_pre, axis_post])])
    for i, param_name in enumerate(params_names):
        output_suffix = consts.DELIMITER_HYPHEN.join(
            [output_suffix, consts.DELIMITER_PARAMS.join([get_param_simple_name(param_name), str(params_values[i])])])
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../_stats", output_suffix))
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    return config_path


def filter_params_for_strategy(params: List[MABExperiment_Param], strategy: str, include_ef=True) -> List[
    MABExperiment_Param]:
    parammap = {
        "UCBTuned": [conf.MAB_UCB_EXPLORATION_FACTOR],
        "UCB2": [conf.MAB_UCB_EXPLORATION_FACTOR, conf.MAB_UCB2_ALPHA],
        "KL-UCB": [conf.MAB_UCB_EXPLORATION_FACTOR, conf.MAB_KL_UCB_C]
    }

    required_params = parammap.get(strategy, [])

    return [param for param in params if
            (param.name in required_params) or (include_ef and param.name == MAB_UCB_EXPLORATION_FACTOR)]


def extract_params(config):
    pararr = []
    parameters_sect = config["parameters"]
    param_names = set()
    for key in parameters_sect.keys():
        match = re.match(r"^(.*?)-(start|step|end)$", key)
        if match:
            param_names.add(match.group(1))

    param_values = {}
    for param_name in sorted(param_names):
        start = float(parameters_sect.get(f"{param_name}-start"))
        step = float(parameters_sect.get(f"{param_name}-step"))
        end = float(parameters_sect.get(f"{param_name}-end"))
        param_values[param_name] = (start, step, end)
        par = MABExperiment_Param(param_name, start, step, end)
        pararr.append(par)
    print(pararr)
    return pararr


class MABExperiment:
    def __init__(self, name: str, strategies: List[str], axis_pre: List[str] = None, axis_post: List[str] = None,
                 params: MABExperiment_Param = None, graphs: List[str] = None,
                 rundup: str = consts.RundupBehavior.SKIP_EXISTENT.value):
        self.name = name
        self.strategies = strategies
        self.axis_pre = axis_pre
        self.axis_post = axis_post
        self.params = params
        self.graphs = graphs
        self.rundup = rundup

    def _generate_config(self, strategy: str, axis_pre: str, axis_post: str, param_names: List[str],
                         param_values: List[float]):
        return write_custom_configfile(self.name, strategy, axis_pre, axis_post, param_names, param_values)

    def _filter_params(self, strategy: str) -> List[MABExperiment_Param]:
        return filter_params_for_strategy(self.params, strategy)

    def run(self):
        rundup = self.rundup
        print(f"Starting experiment {self.name}...")
        for strategy in self.strategies:
            for ax_pre in self.axis_pre:
                for ax_post in self.axis_post:
                    filtered_params = self._filter_params(strategy)
                    ranges = [np.arange(param.start, param.end + param.step, param.step) for param in filtered_params]
                    param_combinations = itertools.product(*ranges)
                    for combination in param_combinations:
                        rounded_combination = [round(value, 2) for value in combination]
                        param_names = [p.name for p in filtered_params]
                        current_values = [p for p in rounded_combination]

                        print()
                        print("============================================")
                        print(f"Running experiment \"{self.name}\"")
                        print(f"with the following configuration")
                        print(f"\tStrategy:\t{strategy}")
                        print(f"\tAxis:\t\t{ax_pre} -> {ax_post}")
                        print(f"\tParameters:")
                        for i, _ in enumerate(param_names):
                            print(f"\t\t> {param_names[i]} = {current_values[i]}")
                        print("--------------------------------------------")

                        path = self._generate_config(strategy, ax_pre, ax_post, param_names, current_values)

                        statsfile = generate_outfile_name(
                            consts.PREFIX_STATSFILE, strategy, ax_pre, ax_post, param_names, current_values
                        ) + consts.SUFFIX_STATSFILE

                        mabfile = generate_outfile_name(
                            consts.PREFIX_MABSTATSFILE, strategy, ax_pre, ax_post, param_names, current_values
                        ) + consts.SUFFIX_MABSTATSFILE

                        run_simulation = None
                        if rundup == consts.RundupBehavior.ALWAYS.value:
                            run_simulation = True
                        elif rundup == consts.RundupBehavior.NO.value:
                            run_simulation = False
                        elif rundup == consts.RundupBehavior.SKIP_EXISTENT.value:
                            if Path(statsfile).exists():
                                # make sure that the file is not only existent, but also not incomplete
                                # (i.e. correctly JSON-readable until the EOF)
                                with open(mabfile, 'r', encoding='utf-8') as r:
                                    try:
                                        json.load(r)
                                    except JSONDecodeError:
                                        print(
                                            "mab-stats file non existent or JSON parsing error, running simulation...")
                                        run_simulation = True
                                    else:
                                        print("parseable stats- and mab-stats file found, skipping simulation.")
                                        run_simulation = False
                            else:
                                print("stats-file non not found, running simulation...")
                                run_simulation = True
                        if run_simulation is None:
                            print("Something is really odd...")
                            exit(1)

                        if run_simulation:
                            main(path)

                        print("end.")
