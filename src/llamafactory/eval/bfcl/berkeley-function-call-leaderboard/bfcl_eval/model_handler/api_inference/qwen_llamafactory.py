import os
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from openai import OpenAI
from overrides import override
from qwen_agent.llm import get_chat_model
import time

class QwenLlamaFactoryAPIHandler(OpenAICompletionsHandler):
    """
    This is the OpenAI-compatible API handler for LLaMA Factory served models.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        # 使用LLaMA Factory服务地址
        self.client = OpenAI(
            base_url="http://10.0.5.74:8000/v1",
            api_key="EMPTY",  # LLaMA Factory默认不需要API密钥
        )

    #### FC methods ####

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # 使用正确的参数调用LLaMA Factory API，启用思考过程
        return self.generate_with_backoff(
            messages=message,
            model=self.model_name.replace("-FC", ""),
            tools=tools,
            parallel_tool_calls=True,
            extra_body={
                "enable_thinking": True
            },
            stream=False,
            # stream=True,
            # stream_options={
            #     "include_usage": True
            # },  # retrieving token usage for stream response
        )

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        # 解析LLaMA Factory的响应格式
        # api_response是一个元组(api_response_object, latency)
        # 我们只需要第一个元素
        if isinstance(api_response, tuple):
            response_data = api_response[0]
        else:
            response_data = api_response
            
        try:
            if hasattr(response_data, 'choices') and len(response_data.choices) > 0:
                choice = response_data.choices[0]
                message = choice.message
                
                # 提取reasoning_content（如果存在）
                reasoning_content = ""
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                
                # 检查是否有工具调用
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = message.tool_calls
                    model_response = []
                    tool_call_ids = []
                    
                    for tool_call in tool_calls:
                        # 提取工具调用信息
                        func_name = tool_call.function.name
                        func_args = tool_call.function.arguments
                        model_response.append({func_name: func_args})
                        tool_call_ids.append(tool_call.id if hasattr(tool_call, 'id') else func_name)
                    
                    model_response_message_for_chat_history = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    }
                else:
                    # 普通文本响应
                    content = message.content if hasattr(message, 'content') else ""
                    model_response = content
                    tool_call_ids = []
                    model_response_message_for_chat_history = {
                        "role": "assistant",
                        "content": content
                    }
                
                # 如果有reasoning_content，添加到聊天历史中
                if reasoning_content:
                    model_response_message_for_chat_history["reasoning_content"] = reasoning_content
                
                # 获取token使用情况
                input_token = response_data.usage.prompt_tokens if hasattr(response_data, 'usage') else 0
                output_token = response_data.usage.completion_tokens if hasattr(response_data, 'usage') else 0
                
                result_data = {
                    "model_responses": model_response,
                    "model_responses_message_for_chat_history": model_response_message_for_chat_history,
                    "tool_call_ids": tool_call_ids,
                    "input_token": input_token,
                    "output_token": output_token,
                }
                
                # 如果有reasoning_content，也添加到结果中
                if reasoning_content:
                    result_data["reasoning_content"] = reasoning_content
                
                return result_data
            else:
                # 如果响应格式不正确，返回默认值
                return {
                    "model_responses": "",
                    "model_responses_message_for_chat_history": {"role": "assistant", "content": ""},
                    "tool_call_ids": [],
                    "input_token": 0,
                    "output_token": 0,
                }
        except Exception as e:
            # 出现异常时返回默认值
            print(f"Error parsing response: {e}")
            return {
                "model_responses": "",
                "model_responses_message_for_chat_history": {"role": "assistant", "content": ""},
                "tool_call_ids": [],
                "input_token": 0,
                "output_token": 0,
            }

    #### Prompting methods ####

    @override
    def _query_prompting(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        return self.generate_with_backoff(
            messages=message,
            model=self.model_name,
            extra_body={
                "enable_thinking": True
            },
            stream=False,
            # stream=True,
            # stream_options={
            #     "include_usage": True
            # },  # retrieving token usage for stream response
        )

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        # api_response是一个元组(api_response_object, latency)
        # 我们只需要第一个元素
        if isinstance(api_response, tuple):
            response_data = api_response[0]
        else:
            response_data = api_response
            
        try:
            if hasattr(response_data, 'choices') and len(response_data.choices) > 0:
                message = response_data.choices[0].message
                content = message.content if hasattr(message, 'content') else ""
                
                # 提取reasoning_content（如果存在）
                reasoning_content = ""
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                
                input_token = response_data.usage.prompt_tokens if hasattr(response_data, 'usage') else 0
                output_token = response_data.usage.completion_tokens if hasattr(response_data, 'usage') else 0
                
                result_data = {
                    "model_responses": content,
                    "model_responses_message_for_chat_history": {
                        "role": "assistant",
                        "content": content,
                    },
                    "input_token": input_token,
                    "output_token": output_token,
                }
                
                # 如果有reasoning_content，也添加到结果中
                if reasoning_content:
                    result_data["reasoning_content"] = reasoning_content
                    result_data["model_responses_message_for_chat_history"]["reasoning_content"] = reasoning_content
                
                return result_data
            else:
                return {
                    "model_responses": "",
                    "model_responses_message_for_chat_history": {
                        "role": "assistant",
                        "content": "",
                    },
                    "input_token": 0,
                    "output_token": 0,
                }
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {
                "model_responses": "",
                "model_responses_message_for_chat_history": {
                    "role": "assistant",
                    "content": "",
                },
                "input_token": 0,
                "output_token": 0,
            }



class QwenAgentThinkHandler(OpenAICompletionsHandler):

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        """
        Note: Need to start vllm server first with command:
        vllm serve xxx \
            --served-model-name xxx \
            --port 8000 \
            --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
            --max-model-len 65536
        """
        
        self.llm = get_chat_model({
        'model': model_name,  # name of the model served by vllm server
        'model_type': 'oai',
        'model_server':'http://localhost:8000/v1', # can be replaced with server host
        'api_key': "none",
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'extra_body': {
                'chat_template_kwargs': {
                    'enable_thinking': True
                }
            },
            "thought_in_content": True,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,
            'max_input_tokens': 58000,
            'timeout': 1000,
            'max_tokens': 16384
        }
    })

    #### FC methods ####
    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        start_time = time.time()
        if len(tools) > 0:
            responses = None
            for resp in self.llm.quick_chat_oai(message, tools):
                responses = resp  # 保留最后一个完整响应
                
        else:
            responses = None
            for resp in self.llm.quick_chat_oai(message):
                responses = resp
        end_time = time.time()
        
        return responses, end_time-start_time
    
    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        try:
            model_responses = [
                {func_call['function']['name']: func_call['function']['arguments']}
                for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
            tool_call_ids = [
                func_call['function']['name'] for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
        except:
            model_responses = api_response["choices"][0]["message"]["content"]
            tool_call_ids = []
        
        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response["choices"][0]["message"],
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.get("usage", {}).get("prompt_tokens", 0),
            "output_token": api_response.get("usage", {}).get("completion_tokens", 0),
        }
        return response_data
        

    
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        
        
        if isinstance(model_response_data["model_responses_message_for_chat_history"], list):
            inference_data["message"]+=model_response_data["model_responses_message_for_chat_history"]
        else:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"]
            )

        return inference_data


class QwenAgentNoThinkHandler(QwenAgentThinkHandler):

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        """
        Note: Need to start vllm server first with command:
        vllm serve xxx \
            --served-model-name xxx \
            --port 8000 \
            --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
            --max-model-len 65536
        """
        
        self.llm = get_chat_model({
        'model': model_name, # name of the model served by vllm server
        'model_type': 'oai',
        'model_server':'http://localhost:8000/v1', # can be replaced with server host
        'api_key': "none",
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'extra_body': {
                'chat_template_kwargs': {
                    'enable_thinking': False
                }
            },
            "thought_in_content": False,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'repetition_penalty': 1.0,
            'presence_penalty': 1.5,
            'max_input_tokens': 58000,
            'timeout': 1000,
            'max_tokens': 16384
        }
    })