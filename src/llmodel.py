from openai import OpenAI
from typing import Callable, List, Dict
import base64

class LLModel:
    def __init__(self, base_url:str, api_key:str, 
                 model_name:str, sys_prompt: Callable[..., str], 
                 max_round_dialog:int = 10, min_round_dialog:int = 1, 
                 stream = False, max_tokens = 2000, **kwargs):
        # instance of OpenAI
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # api params
        self.model_name = model_name
        self.stream = stream
        self.max_tokens = max_tokens
        self.sys_prompt_fun = sys_prompt
        self.sys_prompt = []
        self.messages = []

        # dialog
        self.max_round_dialog = max_round_dialog
        self.min_round_dialog = min_round_dialog
        self.dialog_history = []

    def temp_add_user_message(self, user_message:str, photo:base64 = None):
        if not photo:
            user_massage = {"role": "user", "content": user_message}
        else:
            user_massage = {
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{photo}"}, 
                },
                {"type": "text", "text": user_message}]
            }
        self.dialog_history.append(user_massage)
    
    def remove_last_user_message(self):
        self.dialog_history.pop()
    
    def add_user_message(self, user_message:str, photo:base64 = None):
        self.temp_add_user_message(user_message, photo)
        self._maybe_summarize_and_trim()
    
    def add_assistant_message(self, assistant_message:str):
        self.messages.append({"role": "assistant", "content": assistant_message})
        self._maybe_summarize_and_trim()   

    def clean_dialog(self):
        self.dialog_history = []     
    
    def _maybe_summarize_and_trim(self) -> None:
        """
        Summarize and trim the conversation buffer if necessary.
        """
        current_rounds = len(self.dialog_history) // 2
        
        if current_rounds > self.max_round_dialog:
            self.dialog_history = self.dialog_history[-self.min_round_dialog * 2:]
    
    def _get_reponse(self):
        self.messages = self.sys_prompt + self.dialog_history

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            stream=self.stream,
            max_tokens=self.max_tokens,
        )

        return response

    def get_response(self, print_response:bool = True, has_photo=None) -> str:
        if has_photo:
            self.sys_prompt = [{"role": "system", "content": self.sys_prompt_fun(True)}]
        else:
            self.sys_prompt = [{"role": "system", "content": self.sys_prompt_fun(False)}]
        response = self._get_reponse()
        return self._process_response(response, print_response)
    
    def _process_response(self, response, print_response:bool) -> str:        
        if not self.stream:
            content = response.choices[0].message.content
            if print_response:
                print(f"Assistant: {content}")
            return content
        
        if print_response:
            print("\nAssistant: ", end="")
            full_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                full_content.append(content)
            print()
        else:
            full_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_content.append(content)
        return "".join(full_content)
    
    def print_messages(self):
        print(f"-------------------------------")
        for message in self.messages:
            print(f"\n{message['role']}: {message['content']}")
        print(f"-------------------------------")