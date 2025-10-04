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
    llm=OpenAILLM(api_key="your-openai-api-key"),
    persona="You are an attacker trying to exploit vulnerabilities."
)

defender_agent = Agent(
    role="defender",
    llm=LlamaLLM(model_name="meta-llama/Llama-3-8B-Instruct"),
    persona="You are a defender protecting the network."
)

