import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class LLMClient:
    def complete(self, prompt, **kwargs):
        raise NotImplementedError

class OpenAILLM(LLMClient):
    def __init__(self, api_key, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    def complete(self, prompt, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.7),
        )
        return response.choices[0].message["content"]

class LlamaLLM(LLMClient):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def complete(self, prompt, **kwargs):
        outputs = self.pipeline(prompt, max_new_tokens=256)
        return outputs[0]["generated_text"]

