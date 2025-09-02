import asyncio
from anthropic import Anthropic
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from dotenv import load_dotenv
import os

load_dotenv()

class DataAnalyst:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    async def run(self, prompt: str):
        pandas_params = StdioServerParameters(
            command="python",
            args=[os.path.abspath("mcp_servers/pandas-mcp-server/server.py")]
        )
        filesystem_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", os.path.abspath("data")]
        )

        async with stdio_client(filesystem_params) as (fs_read, fs_write), \
                   stdio_client(pandas_params) as (pd_read, pd_write):
            
            async with ClientSession(fs_read, fs_write) as fs_session, \
                       ClientSession(pd_read, pd_write) as pd_session:
                
                # Initialize and get tools
                await fs_session.initialize()
                await pd_session.initialize()
                
                fs_tools = await fs_session.list_tools()
                pd_tools = await pd_session.list_tools()
                
                # Map tools to sessions
                sessions = {tool.name: fs_session for tool in fs_tools.tools}
                sessions.update({tool.name: pd_session for tool in pd_tools.tools})
                
                all_tools = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} 
                            for t in fs_tools.tools + pd_tools.tools]
                
                # Conversation loop
                messages = [{"role": "user", "content": prompt}]
                
                while True:
                    # Get Claude's response
                    response = self.client.messages.create(
                        model="claude-3-5-haiku-latest",
                        max_tokens=1000,
                        messages=messages,
                        tools=all_tools
                    )
                    
                    messages.append({"role": "assistant", "content": response.content})
                    
                    # Execute any tool calls
                    tool_calls = [b for b in response.content if b.type == "tool_use"]
                    if not tool_calls:
                        # Print final response and exit
                        print([b.text for b in response.content if b.type == "text"][0])
                        break
                    
                    # Execute tools and collect results
                    results = []
                    for call in tool_calls:
                        try:
                            result = await sessions[call.name].call_tool(call.name, call.input)
                            content = result.content[0].text if hasattr(result, 'content') else str(result)
                            
                            if call.name == "generate_chartjs_tool":
                                chart_data = self.extract_chart_data(content)
                                if chart_data:
                                    self.current_chart = chart_data  # Store for Streamlit
                            
                            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content})
                        except Exception as e:
                            results.append({"type": "tool_result", "tool_use_id": call.id, "content": f"Error: {e}"})

                    
                    messages.append({"role": "user", "content": results})

    def extract_chart_data(self, content):
        """Extract chart configuration for Streamlit"""
        import json
        import re
        
        config_match = re.search(r'const config = ({.*?});', content, re.DOTALL)
        if config_match:
            try:
                return json.loads(config_match.group(1))
            except:
                pass
        return None

async def main():
    analyst = DataAnalyst(os.getenv("ANTHROPIC_API_KEY"))
    # await analyst.run("Tell me what files you have available to load in the data directory.")
    # await analyst.run("of the files you listed load the one most likely associated with revenue and describe it to me")
    await analyst.run("using pandas can you plot the revenue over time for me? You will need to use your tools to determine which file to load that you have available and then use your tools to plot this")

if __name__ == '__main__':
    asyncio.run(main())


# query -> PlanningAgent (Detailed plan (steps, agents to call for tools)) -> ExecutionAgent (Execute plan, call tools, handle errors) -> Result
# -> (if error/failure) -> PlanningAgent (Replan) -> ExecutionAgent (Execute plan, call tools, handle errors) -> Result
# -> (if success) -> FinalAgent (Summarize results, next steps, etc.) -> Final Response