import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain / Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# MCP
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# Your existing tool-calling agent with MCP execution
from analytics.data_analyst.lang_agent import DataAnalyst

# ----------------------- Config -----------------------

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT / "data"

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "claude-3-5-sonnet-latest")  # planning (stronger)
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "claude-3-5-haiku-latest")  # execution (cheaper)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ----------------------- Planning schema -----------------------

class Plan(BaseModel):
    objective: str = Field(..., description="One-sentence objective.")
    steps: List[str] = Field(..., description="Minimal ordered steps, each referencing a specific tool when relevant.")
    risks_or_ambiguities: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(..., description="Observable criteria of correctness.")
    tool_agent_prompt: str = Field(..., description="Purpose-built prompt for the tool-calling agent to execute.")

# ----------------------- MCP tool discovery -----------------------

async def _resolve_pandas_server_path() -> List[str]:
    """Prefer running the pandas server with absolute path."""
    server_path = ROOT / "mcp_servers" / "pandas-mcp-server" / "server.py"
    return ["python", str(server_path)]

async def list_mcp_tools() -> List[Dict[str, Any]]:
    """Start filesystem + pandas MCP servers and list their tools (compact summary)."""
    pandas_cmd = await _resolve_pandas_server_path()

    filesystem_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(DATA_DIR)],
    )
    pandas_params = StdioServerParameters(command=pandas_cmd[0], args=pandas_cmd[1:])

    async with stdio_client(filesystem_params) as (fs_read, fs_write), \
               stdio_client(pandas_params) as (pd_read, pd_write):
        async with ClientSession(fs_read, fs_write) as fs_session, \
                   ClientSession(pd_read, pd_write) as pd_session:
            await fs_session.initialize()
            await pd_session.initialize()

            fs_tools = await fs_session.list_tools()
            pd_tools = await pd_session.list_tools()

            def _compress(t) -> Dict[str, Any]:
                props = []
                try:
                    props = list((t.inputSchema or {}).get("properties", {}).keys())[:12]
                except Exception:
                    pass
                return {"name": t.name, "description": t.description or "", "params": props}

            return [_compress(t) for t in fs_tools.tools] + [_compress(t) for t in pd_tools.tools]

# ----------------------- Planner agent -----------------------

class PlannerAgent:
    def __init__(self, api_key: str, model: str = PLANNER_MODEL):
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing.")
        self.llm = ChatAnthropic(model=model, api_key=api_key, max_tokens=1800)

    async def make_plan(self, user_query: str, tools_summary: List[Dict[str, Any]]) -> Plan:
        system = (
            "You are a project planner for a tool-using data analyst agent.\n"
            "Constraints & context:\n"
            "- The execution agent can only read files from the project root './data' directory.\n"
            "- Available tools are summarized by name/description/top-level params.\n"
            "- Produce the SHORTEST correct plan, naming specific tools when applicable.\n"
            "- Avoid unnecessary listing/inspection if you can confidently select files.\n"
            "- Include crisp success criteria.\n"
            "- Generate a purpose-built 'tool_agent_prompt' that the executor will run verbatim.\n"
            "- The executor can call tools (filesystem + pandas MCP) and print outputs/dataframes."
        )
        human = (
            "User task:\n{task}\n\n"
            "Available tools (summary):\n{tools}\n\n"
            "Return JSON with keys: objective, steps, risks_or_ambiguities, success_criteria, tool_agent_prompt."
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        structured_llm = self.llm.with_structured_output(Plan)
        return await structured_llm.ainvoke(prompt.format_messages(task=user_query, tools=tools_summary))

# ----------------------- Orchestrator -----------------------

class PlannerOrchestrator:
    """Plan with Sonnet, then execute with your DataAnalyst (Haiku)."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.planner = PlannerAgent(api_key=api_key, model=PLANNER_MODEL)

    async def run(self, user_query: str):
        tools_summary = await list_mcp_tools()
        plan = await self.planner.make_plan(user_query, tools_summary)

        print("\n============================================================")
        print("TASK:", user_query)
        print("=== PLAN ===")
        print("Objective:", plan.objective)
        print("Steps:")
        for i, s in enumerate(plan.steps, 1):
            print(f"  {i}. {s}")
        if plan.risks_or_ambiguities:
            print("Risks/Ambiguities:")
            for r in plan.risks_or_ambiguities:
                print("  -", r)
        print("Success Criteria:")
        for c in plan.success_criteria:
            print("  -", c)
        print("\n=== TOOL AGENT PROMPT ===\n")
        print(plan.tool_agent_prompt)
        print("\n=== EXECUTION OUTPUT ===\n")

        executor = DataAnalyst(api_key=self.api_key, model=EXECUTOR_MODEL)
        await executor.run(plan.tool_agent_prompt)

# ----------------------- Main: run 3 built-in examples -----------------------

async def main():
    """
    Runs three example tasks against datasets in ./data:
    - orders+revenue by date (brand-level)
    - spend, impressions, clicks by date and platform_code
    """
    examples = [
        # 1) Data interpretation (statistics)
        (
            "Interpretation: Load the orders/revenue dataset from ./data. "
            "Compute basic statistics by date: total revenue, total orders (if available), "
            "mean/median/stdev of daily revenue, top 5 revenue dates, and any outliers you detect. "
            "Summarize seasonality by day-of-week if possible. Only use files in ./data."
        ),
        # 2) Data manipulation (filtering, grouping, adding features)
        (
            "Manipulation: From the orders/revenue dataset in ./data, filter to the last 90 days (if date range allows). "
            "Group by week (Mon-Sun or natural ISO week), compute weekly revenue and week-over-week percent change. "
            "Also add a 7-day rolling average of daily revenue. Return a tidy dataframe with columns: "
            "[date, revenue, rolling_7d, week, weekly_revenue, wow_pct]. Show the head and tail."
        ),
        # 3) Join (format each file, then join and return final dataframe)
        (
            "Joining: Use only ./data. Load the orders/revenue-by-date dataset and the spend/impressions/clicks-by-date dataset "
            "(which also contains platform_code). Clean/standardize date columns and align schemas. "
            "Aggregate to daily brand-level as needed. Join on date. Compute derived metrics: "
            "ROAS = revenue/spend, CTR = clicks/impressions, CPC = spend/clicks. "
            "Return the final merged dataframe sorted by date (show head), and provide a short note on any missing dates or mismatched keys."
        ),
    ]

    orch = PlannerOrchestrator(api_key=ANTHROPIC_API_KEY)
    for task in examples:
        await orch.run(task)

if __name__ == "__main__":
    asyncio.run(main())
