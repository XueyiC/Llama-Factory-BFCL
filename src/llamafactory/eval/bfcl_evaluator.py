"""
BFCL Evaluator for LlamaFactory
简单包装原生 BFCL 命令
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class BFCLEvaluator:
    """
    LlamaFactory 的 BFCL 评估器
    直接调用原生 BFCL 命令
    """
    
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        self.args = args or {}
        
        # BFCL 根目录
        self.bfcl_root = Path(__file__).parent / "bfcl" / "berkeley-function-call-leaderboard"
        
        # 提取参数
        self.model_name = self.args.get("model_name_or_path", "")
        self.test_category = self.args.get("bfcl_category", "multiple")
        self.port = self.args.get("bfcl_port", 8000)
        self.api_base = self.args.get("bfcl_api_base", "http://localhost")
        self.num_threads = self.args.get("bfcl_num_threads", 1)
        self.include_input_log = self.args.get("bfcl_include_input_log", False)
        self.run_ids = self.args.get("bfcl_run_ids", False)
        self.partial_eval = self.args.get("bfcl_partial_eval", False)
        self.stage = self.args.get("bfcl_stage", "all")
        
        custom_result_dir = self.args.get("bfcl_result_dir", "")  
        custom_score_dir = self.args.get("bfcl_score_dir", "")
        
        if custom_result_dir:
            self.result_dir = Path(custom_result_dir)
        else:
            # 使用 BFCL 默认路径：不指定，让 BFCL 自己处理
            self.result_dir = None
        
        if custom_score_dir:
            self.score_dir = Path(custom_score_dir)
        else:
            # 使用 BFCL 默认路径：不指定，让 BFCL 自己处理
            self.score_dir = None
        
        # 结果目录
        # model_safe_name = os.path.basename(self.model_name).replace("/", "_")
        # self.result_dir = Path.cwd() / "bfcl_results" / model_safe_name / "result"
        # self.score_dir = Path.cwd() / "bfcl_results" / model_safe_name / "score"
        
        print(f"\n{'='*70}")
        print(f"🔧 BFCL Evaluator Initialized")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Stage: {self.stage}")
        print(f"Test Category: {self.test_category}")
        print(f"API: {self.api_base}:{self.port}")
        print(f"Result Dir: {self.result_dir}")
        print(f"Score Dir: {self.score_dir}")
        print(f"{'='*70}\n")
    
    def _setup_environment(self) -> Dict[str, str]:
        """设置环境变量"""
        env = os.environ.copy()
        base_url = f"{self.api_base}:{self.port}/v1"
        
        # BFCL 需要的环境变量
        env["OPENAI_API_BASE"] = base_url
        env["OPENAI_API_KEY"] = "EMPTY"
        
        env["LLAMAFACTORY_BASE_URL"] = base_url
        env["LLAMAFACTORY_API_KEY"] = "EMPTY"
        env["LLAMAFACTORY_PORT"] = str(self.port)
        
        # 项目根目录
        env["BFCL_PROJECT_ROOT"] = str(self.bfcl_root)
        
        return env
    
    def _run_bfcl_command(self, command: list, description: str) -> bool:
        """
        运行原生 BFCL 命令
        
        Args:
            command: BFCL 命令参数列表
            description: 命令描述
        
        Returns:
            是否成功
        """
        print(f"\n{'='*70}")
        print(f"🚀 {description}")
        print(f"{'='*70}\n")
        
        env = self._setup_environment()
        
        # ✅ 直接调用 bfcl 命令
        full_command = ["bfcl"] + command
        
        print(f"Command: {' '.join(full_command)}\n")
        
        try:
            result = subprocess.run(
                full_command,
                env=env,
                cwd=str(self.bfcl_root),
                check=False  # 不自动抛出异常
            )
            
            if result.returncode == 0:
                print(f"\n✅ {description} completed successfully")
                return True
            else:
                print(f"\n❌ {description} failed with return code {result.returncode}")
                return False
        
        except FileNotFoundError:
            print(f"\n❌ 'bfcl' command not found")
            print("Please ensure BFCL is installed:")
            print(f"  cd {self.bfcl_root}")
            print("  pip install -e .")
            return False
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(self) -> bool:
        """生成模型响应"""
        # 确保结果目录存在
        if self.result_dir is not None:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            
        # 构建命令
        command = [
            "generate",
            "--model", self.model_name,  
            "--test-category", self.test_category,
            "--num-threads", str(self.num_threads),
        ]
        
        if self.result_dir:
            command.append("--result-dir")
            command.append(str(self.result_dir))
        
        if self.include_input_log:
            command.append("--include-input-log")
        
        if self.run_ids:
            command.append("--run-ids")
        
        return self._run_bfcl_command(command, "Generate")
    
    def evaluate(self) -> bool:
        """评估生成的响应"""
        # 确保评分目录存在
        if self.score_dir is not None:
            self.score_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        command = [
            "evaluate",
            "--model", self.model_name,
            "--test-category", self.test_category,
        ]
        
        if self.result_dir:
            command.append("--result-dir")
            command.append(str(self.result_dir))
        
        if self.score_dir:
            command.append("--score-dir")
            command.append(str(self.score_dir))
        
        if self.partial_eval:
            command.append("--partial-eval")
        
        return self._run_bfcl_command(command, "Evaluate")
    
    def show_scores(self) -> bool:
        """显示评分"""
        command = [
            "scores",
        ]
        
        return self._run_bfcl_command(command, "Scores")
    
    def run_stage(self) -> Dict[str, Any]:
        """根据 stage 参数执行对应阶段"""
        
        if self.stage == "generate":
            print(f"\n{'='*70}")
            print(f"🎯 Running Stage: Generate Only")
            print(f"{'='*70}\n")
            
            if not self.generate():
                return {"status": "failed", "stage": "generate"}
            
            return {"status": "success", "stage": "generate"}
        
        elif self.stage == "evaluate":
            print(f"\n{'='*70}")
            print(f"🎯 Running Stage: Evaluate Only")
            print(f"{'='*70}\n")
            
            if not self.evaluate():
                return {"status": "failed", "stage": "evaluate"}
            
            return {"status": "success", "stage": "evaluate"}
        
        elif self.stage == "scores":
            print(f"\n{'='*70}")
            print(f"🎯 Running Stage: Show Scores Only")
            print(f"{'='*70}\n")
            
            if not self.show_scores():
                return {"status": "failed", "stage": "scores"}
            
            return {"status": "success", "stage": "scores"}
        
        elif self.stage == "all":
            return self.run_full_evaluation()
        
        else:
            print(f"❌ Unknown stage: {self.stage}")
            return {"status": "failed", "error": f"Unknown stage: {self.stage}"}
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """运行完整的评估流程"""
        print(f"\n{'='*70}")
        print(f"🎯 Running Full BFCL Evaluation (All Stages)")
        print(f"{'='*70}\n")
        
        # 1. Generate
        if not self.generate():
            return {"status": "failed", "stage": "generate", "error": "Generate failed"}
        
        # 2. Evaluate
        if not self.evaluate():
            return {"status": "failed", "stage": "evaluate", "error": "Evaluate failed"}
        
        # 3. Show scores
        # self.show_scores()
        
        # 计算实际路径
        model_safe_name = self.model_name.replace("/", "_").replace("\\", "_")
        
        if self.result_dir is not None:
            actual_result_dir = self.result_dir
        else:
            actual_result_dir = self.bfcl_root / "result" / model_safe_name
        
        if self.score_dir is not None:
            actual_score_dir = self.score_dir
        else:
            actual_score_dir = self.bfcl_root / "score" / model_safe_name
        
        # 显示结果
        print(f"\n{'='*70}")
        print(f"✅ Full Evaluation Completed")
        print(f"{'='*70}")
        print(f"📁 Results: {actual_result_dir}")
        print(f"📊 Scores: {actual_score_dir}")
        
        # 列出生成的文件
        if actual_result_dir.exists():
            result_files = list(actual_result_dir.glob("*.json"))
            print(f"   Generated {len(result_files)} result file(s)")
        
        if actual_score_dir.exists():
            score_files = list(actual_score_dir.glob("*.json"))
            print(f"   Generated {len(score_files)} score file(s)")
        
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "result_dir": str(actual_result_dir),
            "score_dir": str(actual_score_dir),
        }



def run_bfcl_eval(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    运行 BFCL 评估的入口函数
    """
    evaluator = BFCLEvaluator(args)
    return evaluator.run_stage()