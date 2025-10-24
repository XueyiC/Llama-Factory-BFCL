# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from copy import deepcopy


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli bfcl -h: run BFCL function calling evaluation   |\n"  # BFCL Evaluation Command 
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli env: show environment info                      |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)


def launch():
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_ray

    logger = logging.get_logger(__name__)
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to LLaMA Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
        + "-" * 58
    )

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "train" and (is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray())):
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        if rdzv_id is not None:
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            process = subprocess.run(
                (
                    "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                    "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                    "--max-restarts {max_restarts} {file_name} {args}"
                )
                .format(
                    rdzv_nnodes=rdzv_nnodes,
                    nproc_per_node=nproc_per_node,
                    rdzv_id=rdzv_id,
                    master_addr=master_addr,
                    master_port=master_port,
                    max_restarts=max_restarts,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )
        else:
            # NOTE: DO NOT USE shell=True to avoid security risk
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=nnodes,
                    node_rank=node_rank,
                    nproc_per_node=nproc_per_node,
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )

        sys.exit(process.returncode)

    elif command == "api":
        from .api.app import run_api

        run_api()

    elif command == "chat":
        from .chat.chat_model import run_chat

        run_chat()

    elif command == "eval":
        raise NotImplementedError("Evaluation will be deprecated in the future.")
    
# ==================== BFCL Evaluation Command ====================
    elif command == "bfcl":
        logger.info("Starting BFCL (Berkeley Function Calling Leaderboard) Evaluation")
        from .eval.bfcl_evaluator import run_bfcl_eval
        
        try:
            import argparse
            
            # 创建 BFCL 参数解析器
            bfcl_parser = argparse.ArgumentParser(
                prog="llamafactory-cli bfcl",
                description="BFCL (Berkeley Function Calling Leaderboard) Evaluation"
            )
            
            # 必需参数
            bfcl_parser.add_argument(
                "--model_name_or_path",
                type=str,
                required=True,
                help="Path to the model or model name"
            )
            
            # 执行阶段参数
            bfcl_parser.add_argument(
                "--bfcl_stage",
                type=str,
                default="all",
                choices=["all", "generate", "evaluate", "scores"],
                help="Execution stage: all (default), generate, evaluate, or scores"
            )
            
            # 可选参数
            bfcl_parser.add_argument(
                "--bfcl_category",
                type=str,
                default="multiple",
                help="BFCL test category (default: multiple)"
            )
            bfcl_parser.add_argument(
                "--bfcl_port",
                type=int,
                default=8000,
                help="API server port (default: 8000)"
            )
            bfcl_parser.add_argument(
                "--bfcl_api_base",
                type=str,
                default="http://localhost",
                help="API base URL (default: http://localhost)"
            )
            bfcl_parser.add_argument(
                "--bfcl_num_threads",
                type=int,
                default=1,
                help="Number of parallel threads (default: 1)"
            )
            bfcl_parser.add_argument(
                "--bfcl_include_input_log",
                action="store_true",
                help="Include detailed input logs"
            )
            bfcl_parser.add_argument(
                "--bfcl_run_ids",
                action="store_true",
                help="Use test_case_ids_to_generate.json"
            )
            bfcl_parser.add_argument(
                "--bfcl_partial_eval",
                action="store_true",
                help="Partial evaluation mode"
            )
            bfcl_parser.add_argument(
                "--bfcl_result_dir",
                type=str,
                default="",
                help="Custom result directory"
            )
            bfcl_parser.add_argument(
                "--bfcl_score_dir",
                type=str,
                default="",
                help="Custom score directory"
            )
            
            # 解析参数
            bfcl_args = bfcl_parser.parse_args(sys.argv[1:])
            
            # 运行评估
            run_bfcl_eval(vars(bfcl_args))
            
        except ImportError as e:
            logger.error(f"Failed to import BFCL evaluator: {e}")
            logger.error("Please ensure BFCL dependencies are installed.")
            sys.exit(1)
        except SystemExit:
            # argparse 会在参数错误或 -h 时调用 sys.exit()
            # 这里捕获它以避免程序异常退出
            raise
        except Exception as e:
            logger.error(f"BFCL evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
# ========================================================

    elif command == "export":
        from .train.tuner import export_model

        export_model()

    elif command == "train":
        from .train.tuner import run_exp

        run_exp()

    elif command == "webchat":
        from .webui.interface import run_web_demo

        run_web_demo()

    elif command == "webui":
        from .webui.interface import run_web_ui

        run_web_ui()

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from llamafactory.train.tuner import run_exp  # use absolute import

    run_exp()
