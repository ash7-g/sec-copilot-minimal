from app.orchestrator.llms import OpenAILLM, LlamaLLM

class Agent:
    def __init__(self, role, llm, persona=None):
        self.role = role
        self.llm = llm
        self.persona = persona or f"You are {role} agent. Respond suitably."

    def act(self, prompt, **kwargs):
        full_prompt = f"{self.persona}\n{prompt}"
        return self.llm.complete(full_prompt, **kwargs)

# Replace 'your-openai-api-key' with your actual OpenAI API key
attacker_agent = Agent(
    role="attacker",
    llm=OpenAILLM(api_key="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"),
    persona="You are an attacker trying to exploit vulnerabilities."
)

defender_agent = Agent(
    role="defender",
    llm=LlamaLLM(model_name="gpt2"),
    persona="You are a defender protecting the network."
)

