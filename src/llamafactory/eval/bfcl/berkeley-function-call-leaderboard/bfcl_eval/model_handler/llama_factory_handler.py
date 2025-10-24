import os
import json
import re
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_tool, convert_to_function_call
from openai import OpenAI
from overrides import override


class LlamaFactoryHandler(OpenAICompletionsHandler):
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

    def decode_ast(self, result, language, has_tool_call_tag):
        # if self.is_fc_model:
        #     decoded_output = []
        #     for invoked_function in result:
        #         name = list(invoked_function.keys())[0]
        #         params = json.loads(invoked_function[name])
        #         decoded_output.append({name: params})
        #     return decoded_output
        # else:
        #     # 参考QwenHandler的实现
        #     tool_calls = self._extract_tool_calls(result)
        #     if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
        #         raise ValueError(f"Model did not return a list of function calls: {result}")
        #     return [
        #         {call["name"]: {k: v for k, v in call["arguments"].items()}}
        #         for call in tool_calls
        #     ]
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    def decode_execute(self, result, has_tool_call_tag):
        # if self.is_fc_model:
        #     return convert_to_function_call(result)
        # else:
        #     # 参考QwenHandler的实现
        #     tool_calls = self._extract_tool_calls(result)
        #     if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
        #         raise ValueError(f"Model did not return a list of function calls: {result}")
        #     decoded_result = []
        #     for item in tool_calls:
        #         if type(item) == str:
        #             item = eval(item)
        #         decoded_result.append({item["name"]: item["arguments"]})
        #     return convert_to_function_call(decoded_result)
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    #### FC methods ####

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # 使用正确的参数调用LLaMA Factory API，启用思考过程
        return self.generate_with_backoff(
            messages=message,
            model=self.model_name,
            tools=tools,
            stream=False,  # LLaMA Factory可能不支持流式响应
            extra_body={
                "enable_thinking": True
            },
        )

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
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
                
                # 获取token使用情况
                input_token = response_data.usage.prompt_tokens if hasattr(response_data, 'usage') else 0
                output_token = response_data.usage.completion_tokens if hasattr(response_data, 'usage') else 0
                
                result_data = {
                    "model_responses": model_response,
                    "model_responses_message_for_chat_history": model_response_message_for_chat_history,
                    "reasoning_content": reasoning_content,
                    "tool_call_ids": tool_call_ids,
                    "input_token": input_token,
                    "output_token": output_token,
                }
                
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
            stream=False,  # LLaMA Factory可能不支持流式响应
            extra_body={
                "enable_thinking": True
            },
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

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception as e:
                pass
        return result