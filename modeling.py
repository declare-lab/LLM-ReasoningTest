import multiprocessing
import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import google.generativeai as genai
import openai
import vllm
from fire import Fire
from openai import AzureOpenAI
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import LLM, SamplingParams


class Generate(ABC):
    @abstractmethod
    def generate(self):
        """Method documentation."""
        pass

    def fail_safe_generate(self, prompt):
        for _ in range(3):
            output = self.generate(prompt)
            if output != "not available":
                return output
            time.sleep(20)

    def generate_batch(self, prompts, threads=128):
        with multiprocessing.Pool(threads) as pool:
            results = list(
                tqdm(pool.imap(self.fail_safe_generate, prompts), total=len(prompts))
            )
        return results


class GPT_o1(BaseModel, Generate, arbitrary_types_allowed=True, extra="allow"):
    azure_endpoint = "https://declaregpt4.openai.azure.com/"
    api_key: Optional[str] = None
    api_version = "2024-02-01"
    loaded = False

    def load(self):
        api_key = os.environ["o1_preview_api"]
        self.client = AzureOpenAI(
            azure_endpoint="https://declaregpt4.openai.azure.com/",
            api_key=api_key,
            api_version="2024-02-01",
        )

    def generate(self, prompt):
        if not self.loaded:
            self.load()
            self.loaded = True
        if prompt == "NA":
            return "NA"
        try:
            response = self.client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content
        except Exception:
            return "not available"


class GPT_4o(BaseModel, Generate, arbitrary_types_allowed=True, extra="allow"):
    azure_endpoint = "https://declaregpt4.openai.azure.com/"
    api_key: Optional[str] = None
    api_version = "2024-02-01"
    loaded = False

    def load(self):
        api_key = os.environ["gpt4o_api"]
        self.client = AzureOpenAI(
            azure_endpoint="https://declaregpt4.openai.azure.com/",
            api_key=api_key,
            api_version="2024-02-01",
        )

    def generate(self, prompt):
        if not self.loaded:
            self.load()
            self.loaded = True
        if prompt == "NA":
            return "NA"
        try:
            response = self.client.chat.completions.create(
                model="GPT4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content
        except Exception:
            return "not available"


class ZeroShotChatTemplate:
    # This is the default template used in llama-factory for training
    texts: List[str] = []

    @staticmethod
    def make_prompt(prompt: str) -> str:
        return f"Human: {prompt}\nAssistant: "

    @staticmethod
    def get_stopping_words() -> List[str]:
        return ["Human:"]

    @staticmethod
    def extract_answer(text: str) -> str:
        filtered = "".join([char for char in text if char.isdigit() or char == " "])
        if not filtered.strip():
            return text
        return re.findall(pattern=r"\d+", string=filtered)[-1]


class VLLMModel(BaseModel, arbitrary_types_allowed=True):
    path_model: str
    model: vllm.LLM = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    max_input_length: int = 512
    max_output_length: int = 512
    stopping_words: Optional[List[str]] = None

    def load(self):
        if self.model is None:
            self.model = vllm.LLM(model=self.path_model, trust_remote_code=True)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model)

    def format_prompt(self, prompt: str) -> str:
        self.load()
        prompt = prompt.rstrip(" ")  # Llama is sensitive (eg "Answer:" vs "Answer: ")
        return prompt

    def make_kwargs(self, do_sample: bool, **kwargs) -> dict:
        if self.stopping_words:
            kwargs.update(stop=self.stopping_words)
        params = vllm.SamplingParams(
            temperature=0.5 if do_sample else 0.0,
            max_tokens=self.max_output_length,
            **kwargs,
        )

        outputs = dict(sampling_params=params, use_tqdm=False)
        return outputs

    def generate(self, prompt: str) -> str:
        prompt = f"Human: {prompt}\nAssistant: "
        prompt = self.format_prompt(prompt)
        outputs = self.model.generate([prompt], **self.make_kwargs(do_sample=False))
        pred = outputs[0].outputs[0].text
        pred = pred.split("<|endoftext|>")[0]
        return pred


def select_model(model_name, **kwargs):
    if model_name == "o1":
        model = GPT_o1()
    elif model_name == "4o":
        model = GPT_4o()
    else:
        model = VLLMModel(path_model=model_name, **kwargs)

    return model


def test_model(
    model_name: str = "flan",
    **kwargs,
):
    prompt = "John has 3 boxes, each with external dimensions of 5 inches by 6 inches by 4 inches and a wall thickness of 1 inch. If the total internal volume of all three boxes increases by a certain amount,  X , while the number of boxes and external dimensions remain unchanged, by how much,  y , will the wall thickness increase? Derive the equation relating  X  and  y ."
    model = select_model(model_name, **kwargs)
    print(locals())
    print(prompt)
    print(model.generate(prompt))


if __name__ == "__main__":
    Fire(test_model)
