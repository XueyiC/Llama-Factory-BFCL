import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from openai import Stream
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from overrides import override


class LlamaFactoryQwenHandler(OpenAICompletionsHandler):
    """
    Handler for Qwen Thinking models deployed via LlamaFactory.
    """
    
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        base_url: str = None,
        api_key: str = "dummy",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        
        self.base_url = base_url or os.getenv("LLAMAFACTORY_BASE_URL", "http://localhost:4444/v1")
        self.api_key = api_key or os.getenv("LLAMAFACTORY_API_KEY", "dummy")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _extract_thinking_content(self, content: str) -> Tuple[str, str]:
        """
        Extract thinking content from 
        """
        if not content:
            return "", ""
        
        # Pattern to match thinking content between  tags
        # pattern = r"(.*?)"
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # reasoning_content = match.group(1).strip()
            # # Remove the thinking content from the main content
            # clean_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
            # return clean_content, reasoning_content
            
            # 提取所有 reasoning 内容
            reasoning_content = "\n\n".join(match.strip() for match in matches)
            
            # 从原内容中移除所有 <think>...</think> 块
            clean_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
            
            return clean_content, reasoning_content
        
        return content, ""

    @override
    def _query_FC(self, inference_data: dict) -> Tuple:
        """
        Override to handle LlamaFactory's limitations with streaming + tools.
        """
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self.model_name,
            # "model": "/data/model/modelweight/Qwen3-4B-Thinking-2507-FC",

            "temperature": self.temperature,
            "store": False,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools
            # Cannot use streaming with tools in LlamaFactory
            kwargs["stream"] = False
        
        # Always enable thinking for thinking models
        if "thinking" in self.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": True}

        return self.generate_with_backoff(**kwargs)

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parse the API response and extract reasoning content.
        """
        # Let parent class handle the standard parsing
        parent_result = super()._parse_query_response_FC(api_response)
        
        # Extract reasoning content from the message content
        reasoning_content = ""
        
        # Check message content
        message = api_response.choices[0].message
        if message.content:
            content, reasoning_content = self._extract_thinking_content(message.content)
            # Update the content in parent result to remove thinking part
            if reasoning_content:
                parent_result["model_responses_message_for_chat_history"].content = content
        
        # Add reasoning if found
        if reasoning_content:
            parent_result["reasoning_content"] = reasoning_content
        
        return parent_result