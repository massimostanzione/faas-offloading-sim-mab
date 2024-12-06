import os

import conf
from _internal import consts
from _internal.experiment import MABExperiment, extract_params

if __name__ == "__main__":
    # execute all the esperiments according to the pipeline file
    with open(consts.PIPELINE_FILE, "r") as pipeline_file:
        for experiment_name in pipeline_file:
            experiment_confpath = os.path.join(experiment_name.strip(), consts.EXPCONF_FILE)

            if os.path.exists(experiment_confpath):
                with open(experiment_confpath, "r") as expconf_file:
                    config = conf.parse_config_file(experiment_confpath)

                    mode_simulations = config["experiment"]["mode-simulations"]
                    mode_postprocessing = config["experiment"]["mode-postprocessing"]

                    # =========================================================
                    # 1. Simulations
                    # ---------------------------------------------------------

                    if mode_simulations == consts.ExecMode.NONE.value:
                        print("Simulations skipped")

                    elif mode_simulations == consts.ExecMode.AUTOMATED.value:
                        exp = MABExperiment(
                            config["experiment"]["name"],
                            config["strategies"]["strategies"].split(consts.DELIMITER_COMMA),
                            config["reward_fn"]["axis_pre"].split(consts.DELIMITER_COMMA),
                            config["reward_fn"]["axis_post"].split(consts.DELIMITER_COMMA),
                            extract_params(config),
                            [],
                            config["output"]["run-duplicates"]
                        )
                        exp.run()

                    else:
                        name = config["experiment"]["name"]
                        print()
                        print("============================================")
                        print(f"Running experiment \"{name}\"")
                        print(f"with custom simulations")
                        print("--------------------------------------------")
                        os.system("python3.8 " + os.path.join(name, mode_simulations))

                    # =========================================================
                    # 2. Post-processing
                    # ---------------------------------------------------------

                    if mode_postprocessing == consts.ExecMode.NONE.value:
                        print("Postprocessing skipped")

                    elif mode_postprocessing == consts.ExecMode.AUTOMATED.value:
                        print("No automated postprocessing defined")

                    else:
                        name = config["experiment"]["name"]
                        os.system("python3.8 " + os.path.join(name, mode_postprocessing))
