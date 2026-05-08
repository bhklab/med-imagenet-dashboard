from google.adk.agents import Agent

from voxel.prompts import QUERY_AGENT_INSTRUCTION, ROOT_INSTRUCTION
from voxel.tools import execute_query, list_collections

query_agent = Agent(
    model="gemini-2.5-flash",
    name="query_agent",
    description=(
        "Translates natural-language dataset requests into ValidQuery JSON, "
        "validates and executes them against Med-ImageNet, and returns the "
        "matching series.  Delegate to this agent whenever the user wants to "
        "search, filter, or select specific imaging series."
    ),
    instruction=QUERY_AGENT_INSTRUCTION,
    tools=[execute_query],
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="voxel",
    description="Voxel -- conversational assistant for exploring Med-ImageNet datasets.",
    instruction=ROOT_INSTRUCTION,
    tools=[list_collections],
    sub_agents=[query_agent],
)
