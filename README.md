## 项目简介
本项目将 Berkeley Function Calling Leaderboard (BFCL) 评测工具集成到 LlamaFactory 中，实现了从模型训练到函数调用能力评估的完整工作流。用户可以直接通过 llamafactory-cli 命令运行 BFCL 评测，无需手动切换工具或环境。

## 核心设计理念

原生命令调用：直接调用 BFCL 原生命令，保证功能完整性和稳定性

最小侵入式集成：只修改必要的文件，保持 LlamaFactory 原有架构

灵活的阶段控制：支持分阶段执行（generate、evaluate、scores）或完整流程

参数透传机制：LlamaFactory 的参数直接映射到 BFCL 命令


## 使用指南

安装环境

1. 创建环境 
``` conda create -n llamafactory_bfcl python=3.10 
conda activate llamafactory_bfcl  
```

2. 克隆仓库（如果还没有）
``` 
git clone https://github.com/XueyiC/Llama-Factory-BFCL.git 
```

3. 安装环境
``` 
pip install -r requirements_all.txt
```


完整工作流程

1. 用原生llama factory 方式启动模型 API 服务器

```
API_PORT=4444 llamafactory-cli api \
    --model_name_or_path /path/to/model \
    --template qwen3 \
    --infer_backend vllm
```

或者使用配置文件

```API_PORT=4444 llamafactory-cli api examples/inference/qwen3_thinking.yaml```


2. 运行 BFCL 评测
   
2.1 完整流程（Generate + Evaluate）

目前需手动将BFCL框架中 model_config.py文件 912行和913行的 model name 改为和上一步骤启动Llama Factory 模型的路径，如果模型支持Function Calling模式，需在末尾加上 -FC （文件位置：Llama-Factory-BFCL/src/llamafactory/eval/bfcl/berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py）

```
llamafactory-cli bfcl \
    --model_name_or_path /path/to/model \
    --bfcl_category multiple \
    --bfcl_port 4444 \
    --bfcl_stage all  # 可选，默认就是 all
```

2.2 分阶段运行
第一步：生成响应
```
llamafactory-cli bfcl \
    --model_name_or_path /path/to/model \
    --bfcl_category multiple \
    --bfcl_port 4444 \
    --bfcl_stage generate
```

第二步：评估结果（API 服务器可以关闭）

```
llamafactory-cli bfcl \
    --model_name_or_path /path/to/model \
    --bfcl_category multiple \
    --bfcl_stage evaluate
```


## 参数详解
必需参数
参数  --model_name_or_path   模型路径或名称/path/to/model 

可选参数
--bfcl_stagestrall执行阶段：all, generate, evaluate, scores 

--bfcl_category  str  multiple 测试类别

--bfcl_port  int 8000 API 服务器端口

--bfcl_api_base  str  http://localhostAPI 基础 URL

--bfcl_num_threads int  1  并行线程数

--bfcl_include_input_logflag False 包含详细输入日志

--bfcl_run_idsflag  False 使用 test_case_ids_to_generate.json

--bfcl_partial_evalflag  False  部分评估模式

--bfcl_result_dirstr  ``   自定义结果目录

--bfcl_score_dirstr  ``  自定义评分目录

<img width="749" height="653" alt="image" src="https://github.com/user-attachments/assets/c2257697-d7b5-4ade-8e69-b15b85fbf97c" />

## 新增文件
1. src/llamafactory/eval/bfcl_evaluator.py ⭐️ 核心文件

作用: LlamaFactory 和 BFCL 之间的桥接层

主要功能:

BFCLEvaluator 类：封装所有 BFCL 操作

generate(): 调用 bfcl generate 命令

evaluate(): 调用 bfcl evaluate 命令

show_scores(): 调用 bfcl scores 命令

run_stage(): 根据 stage 参数执行对应阶段

run_full_evaluation(): 执行完整流程

2. src/llamafactory/eval/bfcl/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/llamafactory_qwen.py ⭐️ Handler 文件

作用: BFCL 的模型处理器，专门处理通过 LlamaFactory 部署的 Qwen Thinking 模型

主要功能:

继承 OpenAIResponseHandler

连接到 LlamaFactory API 服务器

提取和处理 <think> 标签内容

支持 function calling
