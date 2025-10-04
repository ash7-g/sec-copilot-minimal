from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Path to your downloaded GGUF model
CLAUDE_MODEL_PATH = r"C:\Users\Aashirvaad\Downloads\claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf"

# Instantiate common LlamaCpp model
llm_common = LlamaCpp(model_path=CLAUDE_MODEL_PATH)

# --- Custom Tools ---
class ExploitTool(BaseTool):
    name: str = "exploit"
    description: str = "Simulate multi-step exploit logic"

    def _run(self, target: str) -> str:
        return f"Simulated exploit against {target}"

    async def _arun(self, target: str) -> str:
        return self._run(target)

class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "A placeholder dummy tool"

    def _run(self, input_text: str) -> str:
        return "Dummy response"

    async def _arun(self, input_text: str) -> str:
        return self._run(input_text)

dummy_tool = DummyTool()

# Attacker agent
attacker_tools = [ExploitTool()]
attacker_agent = initialize_agent(attacker_tools, llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Defender tools and agent
def log_timeline_func(query: str) -> str:
    return f"Timeline analyzed for: {query}"

def playbook_executor_func(response: str) -> str:
    return f"Executed playbook: {response}"

log_timeline_tool = Tool.from_function(log_timeline_func, name="LogTimeline", description="Analyze timeline of events")
playbook_executor_tool = Tool.from_function(playbook_executor_func, name="PlaybookExecutor", description="Execute a given playbook response")

defender_tools = [log_timeline_tool, playbook_executor_tool]
defender_agent = initialize_agent(defender_tools, llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Decider agent
class DeciderTool(BaseTool):
    name: str = "decide"
    description: str = "Adjudicate disputes using prompt"

    def _run(self, prompt: str) -> str:
        return llm_common(prompt)

    async def _arun(self, prompt: str) -> str:
        return self._run(prompt)

decider_tools = [DeciderTool()]
decider_agent = initialize_agent(decider_tools, llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Narrator agent with dummy tool
narrator_agent = initialize_agent([dummy_tool], llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Analyst tools and agent
def report_builder_func(data: str) -> str:
    return f"Post-mortem report generated from: {data}"

report_builder_tool = Tool.from_function(report_builder_func, name="ReportBuilder", description="Generate post-mortem reports")
analyst_tools = [report_builder_tool]
analyst_agent = initialize_agent(analyst_tools, llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Detection tools and agent
def yara_generator_func(signature: str) -> str:
    try:
        return llm_common(f"Generate YARA/Sigma rules for: {signature}")
    except Exception as e:
        return f"Error: {str(e)}"

yara_generator_tool = Tool.from_function(yara_generator_func, name="YARAGenerator", description="Generate YARA/Sigma rules")
detection_tools = [yara_generator_tool]
detection_agent = initialize_agent(detection_tools, llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Threat-Intel agent setup
embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key_here")  # Replace with your key
faiss_db = FAISS.load_local("path/to/vector/index", embeddings, allow_dangerous_deserialization=True)
retriever = faiss_db.as_retriever()

def ttp_mapper_func(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Map ATT&CK tactics/techniques for: {query}\nContext:\n{context}"
    try:
        return llm_common(prompt)
    except Exception as e:
        return f"Error: {str(e)}"

ttp_mapper_tool = Tool.from_function(ttp_mapper_func, name="TTMapper", description="Map tactics/techniques for a query")
threatintel_tools = [ttp_mapper_tool]
threatintel_agent = initialize_agent(threatintel_tools, llm_common, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Export agents dictionary
agents = {
    "attacker": attacker_agent,
    "defender": defender_agent,
    "decider": decider_agent,
    "narrator": narrator_agent,
    "analyst": analyst_agent,
    "detection": detection_agent,
    "threatintel": threatintel_agent,
}

if __name__ == "__main__":
    result = agents["decider"].run("Who is the winner in a tie between attacker and defender? Justify.")
    print("Result:", result)
