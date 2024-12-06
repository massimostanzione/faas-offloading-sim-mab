import json
import os
import sys
from datetime import datetime

import numpy as np

import conf
from conf import MAB_KL_UCB_C, MAB_UCB2_ALPHA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from _internal.experiment import generate_outfile_name
from _internal.experiment import extract_params
from _internal.experiment import filter_params_for_strategy
from _internal.experiment import get_param_simple_name
from _internal import consts
from _internal.graphs import line_graph

EXPNAME = "arms-disparity"
EXPCONF_PATH = os.path.join(EXPNAME, consts.EXPCONF_FILE)


def main():
    timestamp = datetime.now().replace(microsecond=0)
    config = conf.parse_config_file(EXPCONF_PATH)

    strategies = config["strategies"]["strategies"].split(consts.DELIMITER_COMMA)

    axis_pre = config["reward_fn"]["axis_pre"].split(consts.DELIMITER_COMMA)
    axis_post = config["reward_fn"]["axis_post"].split(consts.DELIMITER_COMMA)

    efstart = config.getfloat("parameters", "mab-ucb-exploration-factor-start")
    efstep = config.getfloat("parameters", "mab-ucb-exploration-factor-step")
    efend = config.getfloat("parameters", "mab-ucb-exploration-factor-end")

    graphs_ctr = -1

    for strat in strategies:
        if strat == "KL-UCB":
            paramstart = config.getfloat("parameters", "mab-kl-ucb-c-start")
            paramend = config.getfloat("parameters", "mab-kl-ucb-c-end")
            paramstep = config.getfloat("parameters", "mab-kl-ucb-c-step")
            paramname = get_param_simple_name(MAB_KL_UCB_C)

        elif strat == "UCB2":
            paramstart = config.getfloat("parameters", "mab-ucb2-alpha-start")
            paramend = config.getfloat("parameters", "mab-ucb2-alpha-end")
            paramstep = config.getfloat("parameters", "mab-ucb2-alpha-step")
            paramname = get_param_simple_name(MAB_UCB2_ALPHA)

        else:
            paramstart = 1
            paramend = 1
            paramstep = 1
            paramname = ""

        params = extract_params(config)
        filtered_params = filter_params_for_strategy(params, strat, False)
        ranges = [np.arange(param.start, param.end + param.step, param.step) for param in filtered_params]
        for ax_pre in axis_pre:
            for ax_post in axis_post:
                glob_var_coeff = {}
                policies = []
                for param_iter in np.arange(paramstart, paramend + paramstep, paramstep):
                    local_var_coeff = []
                    for espl_fact in np.arange(efstart, efend + efstep, efstep):
                        param_iter = round(param_iter, 2)
                        time_frames = []
                        espl_fact = np.round(espl_fact, 2)
                        if (paramname != ""):
                            mabfile = consts.DELIMITER_HYPHEN.join(
                                [generate_outfile_name(consts.PREFIX_MABSTATSFILE, strat, ax_pre, ax_pre, [], []),
                                 "ef" + "=" + str(espl_fact), paramname + "=" + str(
                                    param_iter)]) + consts.SUFFIX_MABSTATSFILE
                        else:
                            mabfile = consts.DELIMITER_HYPHEN.join(
                                [generate_outfile_name(consts.PREFIX_MABSTATSFILE, strat, ax_pre, ax_pre, [], []),
                                 "ef" + "=" + str(
                                     espl_fact)]) + consts.SUFFIX_MABSTATSFILE

                        with open(mabfile, 'r') as f:
                            data = json.load(f)

                        # todo grafico pre/post
                        pol_fr_prev = {"random-lb": 0, "round-robin-lb": 0, "mama-lb": 0, "const-hash-lb": 0,
                                       "wrr-speedup-lb": 0,
                                       "wrr-memory-lb": 0, "wrr-cost-lb": 0}
                        for d in data:
                            # todo pre/post if d['time'] < 9000:
                            pol_fr_prev[d['policy']] += 1
                            time_frames.append(d['time'])
                            policies.append(d['policy'])

                        listr = []
                        for value in pol_fr_prev.values():
                            listr.append(value)

                        var_coeff = np.std(listr) / np.average(listr)
                        local_var_coeff.append(var_coeff)
                glob_var_coeff["param=" + str(param_iter)] = {"x": np.arange(efstart, efend + efstep, efstep),
                                                              "y": local_var_coeff}

                fig = line_graph(glob_var_coeff, title=strat + " " + ax_pre + " -> " + ax_post)

                graphs_ctr += 1
                fig.savefig(os.path.join(SCRIPT_DIR, "results",
                                         consts.DELIMITER_HYPHEN.join([str(timestamp), str(graphs_ctr)]).replace(' ',
                                                                                                                 '-') + ".svg"))
                fig.clf()


if __name__ == "__main__":
    main()
