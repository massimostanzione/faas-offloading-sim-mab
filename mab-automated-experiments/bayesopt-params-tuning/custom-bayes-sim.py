import datetime
import json
import os
import sys
from json import JSONDecodeError
from pathlib import Path

from bayes_opt import BayesianOptimization

import conf
from conf import MAB_UCB_EXPLORATION_FACTOR, MAB_UCB2_ALPHA, MAB_KL_UCB_C

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from _internal.experiment import generate_outfile_name
from _internal.experiment import write_custom_configfile
from _internal import consts
from main import main


def obj_ucbtuned(ef):
    for _ in range(num_simulations):
        write_custom_configfile(EXPNAME, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR], [ef])

        statsfile = generate_outfile_name(
            consts.PREFIX_STATSFILE, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR], [ef]
        ) + consts.SUFFIX_STATSFILE
        mabfile = generate_outfile_name(
            consts.PREFIX_MABSTATSFILE, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR], [ef]
        ) + consts.SUFFIX_MABSTATSFILE

    return compute_total_reward(mabfile, statsfile) / num_simulations


def obj_ucb2(ef, alpha):
    for _ in range(num_simulations):
        write_custom_configfile(EXPNAME, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR, MAB_UCB2_ALPHA],
                                [ef, alpha])

        statsfile = generate_outfile_name(
            consts.PREFIX_STATSFILE, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR, MAB_UCB2_ALPHA], [ef, alpha]
        ) + consts.SUFFIX_STATSFILE
        mabfile = generate_outfile_name(
            consts.PREFIX_MABSTATSFILE, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR, MAB_UCB2_ALPHA],
            [ef, alpha]
        ) + consts.SUFFIX_MABSTATSFILE

        return compute_total_reward(mabfile, statsfile) / num_simulations


def obj_klucb(ef, c):
    for _ in range(num_simulations):
        path = write_custom_configfile(EXPNAME, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR, MAB_KL_UCB_C],
                                       [ef, c])

        statsfile = generate_outfile_name(
            consts.PREFIX_STATSFILE, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR, MAB_KL_UCB_C], [ef, c]
        ) + consts.SUFFIX_STATSFILE
        mabfile = generate_outfile_name(
            consts.PREFIX_MABSTATSFILE, strat, ax_pre, ax_post, [MAB_UCB_EXPLORATION_FACTOR, MAB_KL_UCB_C], [ef, c]
        ) + consts.SUFFIX_MABSTATSFILE

    return compute_total_reward(mabfile, statsfile) / num_simulations


def compute_total_reward(mabfile, statsfile):
    total_reward = 0
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
                    print("mab-stats file non existent or JSON parsing error, running simulation...")
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
        main(EXPNAME + "/results/" + consts.CONFIG_FILE)
    with open(mabfile, 'r') as f:
        data = json.load(f)
    rewards = []
    for d in data:
        rewards.append(d['reward'])
    f.close()
    total_reward += sum(rewards) / len(rewards)
    return total_reward


if __name__ == "__main__":
    timestamp = datetime.datetime.now().replace(microsecond=0)
    path = os.path.abspath(".")
    config = conf.parse_config_file(path + "/bayesopt-params-tuning/" + consts.EXPCONF_FILE)
    rundup = config["output"]["run-duplicates"]
    EXPNAME = config["experiment"]["name"]
    strategies = config["strategies"]["strategies"].split(consts.DELIMITER_COMMA)
    axis_pre = config["reward_fn"]["axis_pre"].split(consts.DELIMITER_COMMA)
    axis_post = config["reward_fn"]["axis_post"].split(consts.DELIMITER_COMMA)
    ef_lower = config["parameters"]["ef-lower"]
    ef_upper = config["parameters"]["ef-upper"]
    num_simulations = config.getint("parameters", "objfn-stabilizations-iterations")

    selected_obj_fn = None
    for strat in strategies:
        if strat == "UCBTuned":
            pbounds = {'ef': (ef_lower, ef_upper)}
            selected_obj_fn = obj_ucbtuned
        elif strat == "UCB2":
            pbounds = {'ef': (ef_lower, ef_upper),
                       'alpha': (config["parameters"]["ucb2-alpha-lower"], config["parameters"]["ucb2-alpha-upper"])}
            selected_obj_fn = obj_ucb2
        elif strat == "KL-UCB":
            pbounds = {'ef': (ef_lower, ef_upper),
                       'c': (config["parameters"]["klucb-c-lower"], config["parameters"]["klucb-c-upper"])}
            selected_obj_fn = obj_klucb
        else:
            print("what?")
            exit(1)

        for ax_pre in axis_pre:
            for ax_post in axis_post:
                optimizer = BayesianOptimization(
                    f=selected_obj_fn,
                    pbounds=pbounds,
                    verbose=2,
                    random_state=1,
                )
                optimizer.maximize(
                    init_points=config.getint("parameters", "rand-points"),
                    n_iter=config.getint("parameters", "iterations"),
                )

                output_file = (EXPNAME + "/results/OUTPUT")

                with open(output_file, "a") as file:
                    file.write('{0:19} {{avgst={5:2}, init={6:2}, iter={7:2}}} {1:8} {2:10} > {3:10} {4}\n'
                               .format(str(timestamp), strat, ax_pre, ax_post,
                                       optimizer.max['params'].__str__(),
                                       config.getint("parameters", "objfn-stabilizations-iterations"),
                                       config.getint("parameters", "rand-points"),
                                       config.getint("parameters", "iterations")))
