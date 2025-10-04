from fastapi import APIRouter, Body
from app.orchestrator.agents import attacker_agent, defender_agent

router = APIRouter()

AGENT_MAP = {
    "attacker": attacker_agent,
    "defender": defender_agent,
}

@router.post("/chat/{role}")
def chat_agent(role: str, prompt: str = Body(...)):
    agent = AGENT_MAP.get(role)
    if not agent:
        return {"error": f"Unknown agent role '{role}'"}
    response = agent.act(prompt)
    return {"role": role, "response": response}

