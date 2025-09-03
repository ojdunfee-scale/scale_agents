import asyncio
import json
import os
import re
from typing import Any, Dict, Callable

from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

load_dotenv()

from pydantic import field_validator
import json

# Constants
DATA_DIR = "/home/owen/dev/scale_agents/data"

# Individual tool schemas
class PandasCodeSchema(BaseModel):
    code: str = Field(description="Python code using pandas to execute")

class FilePathSchema(BaseModel):
    file_path: str = Field(description="Path to the file")
    
class DirectoryPathSchema(BaseModel):
    path: str = Field(description="Directory path to list")

class SearchFilesSchema(BaseModel):
    path: str = Field(description="Directory path to search")
    pattern: str = Field(description="File pattern to search for")

class ReadTextFileSchema(BaseModel):
    path: str = Field(description="Path to text file to read")

from pydantic import BaseModel, Field, field_validator
from typing import Any, Union

class GenerateChartSchema(BaseModel):
    input_data: Union[dict, str, Any] = Field(description="Chart data in any format - will be processed internally")
    
    @field_validator('input_data', mode='before')
    def validate_input_data(cls, v):
        # Accept any input type and let the tool handler process it
        return v

def make_mcp_tool(
      name: str,
      description: str,
      call_fn: Callable[[Dict[str, Any]], Any],
      input_schema: Dict[str, Any] | None  
) -> Tool:
    # Choose appropriate schema based on tool name
    if name == "run_pandas_code_tool":
        schema_class = PandasCodeSchema
    elif name in ["read_metadata_tool", "read_text_file"]:
        if name == "read_metadata_tool":
            schema_class = FilePathSchema
        else:
            schema_class = ReadTextFileSchema
    elif name == "list_directory":
        schema_class = DirectoryPathSchema
    elif name == "search_files":
        schema_class = SearchFilesSchema
    elif name == "generate_chartjs_tool":
        schema_class = GenerateChartSchema
    else:
        # Generic fallback
        schema_class = DirectoryPathSchema
    
    param_hint = ""
    if input_schema and isinstance(input_schema, dict):
        props = input_schema.get("properties", {})
        if isinstance(props, dict) and props:
            prop_list = ", ".join(list(props.keys())[:10])
            more = " (...)" if len(props) > 10 else "" 
            param_hint = f"\nParameters: {prop_list}{more}"
    
    def _run_tool(*args, **kwargs) -> str:
        raise NotImplementedError("Use the async coroutine path")
    
    async def _arun_tool(*args, **kwargs) -> str:
        try:
            # Handle arguments from LangChain pydantic model
            if args and len(args) == 1:
                arg_value = args[0]
                # Map the single argument to the appropriate parameter name
                if name == "run_pandas_code_tool":
                    tool_args = {"code": arg_value}
                elif name in ["read_metadata_tool"]:
                    tool_args = {"file_path": arg_value}
                elif name in ["list_directory"]:
                    tool_args = {"path": arg_value}
                elif name == "read_text_file":
                    tool_args = {"path": arg_value}
                elif name == "generate_chartjs_tool":
                    # For chart generation, map input_data from schema to proper MCP format
                    tool_args = {"data": {}, "chart_types": ["bar"], "title": "Chart"}
                    if isinstance(arg_value, dict):
                        # If it's already a dict, use it directly
                        if "data" in arg_value:
                            tool_args.update(arg_value)
                        else:
                            tool_args["data"] = arg_value
                    elif isinstance(arg_value, str):
                        try:
                            # Try to parse as JSON
                            parsed_data = json.loads(arg_value)
                            tool_args["data"] = parsed_data
                        except:
                            tool_args["data"] = {"raw": arg_value}
                    else:
                        tool_args["data"] = {"raw": str(arg_value)}
                else:
                    tool_args = {"path": arg_value}  # Default to path
            elif kwargs:
                tool_args = kwargs
            else:
                tool_args = {}
            
            # Special handling for chart generation tool with multiple args
            if name == "generate_chartjs_tool" and len(args) > 1:
                # Multiple arguments: data, chart_types, title
                tool_args = {}
                if len(args) >= 1:
                    if isinstance(args[0], str):
                        try:
                            tool_args["data"] = json.loads(args[0])
                        except:
                            tool_args["data"] = args[0]
                    else:
                        tool_args["data"] = args[0]
                if len(args) >= 2:
                    tool_args["chart_types"] = args[1]
                if len(args) >= 3:
                    tool_args["title"] = args[2]
            
            result = await call_fn(tool_args)
            if hasattr(result, "content") and isinstance(result.content, list):
                texts = [b.text for b in result.content if getattr(b, "type", None) == "text" and hasattr(b, "text")]
                if texts:
                    return texts[0]
            return str(result)
        except Exception as e:
            return f"Error calling MCP Tool `{name}`: {e}"
    
    return Tool(
        name=name,
        description=(description or "No Description") + param_hint,
        args_schema=schema_class,
        func=_run_tool,
        coroutine=_arun_tool
    )


class DataAnalyst:
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-latest"):
        self.llm = ChatAnthropic(model=model, api_key=api_key, max_tokens=1000)
        self.current_chart: Dict[str, Any] | None = None

    async def run(self, prompt: str):
        pandas_params = StdioServerParameters(
            command="python",
            args=[os.path.abspath("/home/owen/dev/scale_agents/mcp_servers/pandas-mcp-server/server.py")]
        )
        filesystem_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", DATA_DIR]
        )

        async with stdio_client(filesystem_params) as (fs_read, fs_write), stdio_client(pandas_params) as (pd_read, pd_write):
            async with ClientSession(fs_read, fs_write) as fs_session, ClientSession(pd_read, pd_write) as pd_session:
                await fs_session.initialize()
                await pd_session.initialize()

                fs_tools = await fs_session.list_tools()
                pd_tools = await pd_session.list_tools()

                session_for_tool: Dict[str, ClientSession] = {
                    t.name: fs_session for t in fs_tools.tools
                }
                session_for_tool.update({
                    t.name: pd_session for t in pd_tools.tools
                })

                async def make_call_fn(session: ClientSession, tool_name: str):
                    async def _call(args: Dict[str, Any]):
                        return await session.call_tool(tool_name, args or {})
                    return _call
                
                lc_tools = []
                mcp_tool_specs = []

                for t in list(fs_tools.tools) + list(pd_tools.tools):
                    call_fn = await make_call_fn(session_for_tool[t.name], t.name)
                    lc_tools.append(make_mcp_tool(
                        name=t.name,
                        description=t.description or "",
                        call_fn=call_fn,
                        input_schema=t.inputSchema if hasattr(t, "inputSchema") else None
                    ))
                    mcp_tool_specs.append({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": getattr(t, "inputSchema", None)
                    })
                
                system_msg = (
                    "You are a helpful data analyst. You can use tools to list files, "
                    "load data, analyze with pandas, and (optionally) generate Chart.js configs. "
                    f"All data files are located in: {DATA_DIR}\n"
                    "CRITICAL PANDAS CODE RULES:\n"
                    f"1. ALWAYS use the full path {DATA_DIR}/filename.csv when referencing files\n"
                    "2. Write simple, direct pandas code - avoid complex multiline strings or scripts\n" 
                    "3. ALWAYS assign your final DataFrame or result to 'result' variable\n"
                    "4. Use simple operations: df = pd.read_csv('path'); result = df.describe()\n"
                    "5. Avoid f-strings and complex formatting in pandas code\n"
                    "6. Keep code concise and avoid security triggers like eval, exec, etc.\n"
                    "7. For Chart.js: const config = {{ /* valid JSON config */ }};\n"
                    "8. Only work with files in the data directory\n"
                )

                prompt_tmpl = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_msg),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad")
                    ]
                )

                agent = create_tool_calling_agent(self.llm, lc_tools, prompt_tmpl)
                executor = AgentExecutor(
                    agent=agent,
                    tools=lc_tools,
                    verbose=True,
                    handle_parsing_errors=True
                )

                result = await executor.ainvoke({"input": prompt, "chat_history": []})
                output_text: str = result.get("output", "")

                chart = self.extract_chart_data(output_text)
                if chart:
                    self.current_chart = chart

                print(output_text)
    
    def extract_chart_data(self, text: str | None):
        if not text:
            return None
        try:
            match = re.search(r"const\s+config\s*=\s*({.*?});", text, re.DOTALL)
            if not match:
                return None
            return json.loads(match.group(1))
        except Exception:
            return None
        

async def main():
    analyst = DataAnalyst(os.getenv("ANTHROPIC_API_KEY"))

    test_cases = [
        f"Load {DATA_DIR}/lc_orders.csv using pandas and calculate basic statistics for the orders column. Use simple code.",
        
        f"Read {DATA_DIR}/lc_orders.csv and show the first 5 rows. Keep it simple.",
        
        f"Load {DATA_DIR}/lc_orders.csv and calculate the mean of the orders column.",
    ]
    

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {prompt[:50]}...")
        print(f"{'='*60}")
        
        try:
            await analyst.run(prompt)
        except Exception as e:
            print(f"Error in test case {i}: {e}")
        
        print(f"\nCompleted test case {i}")
        
        # Small delay between test cases
        import asyncio
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())