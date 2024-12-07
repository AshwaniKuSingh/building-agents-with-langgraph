from typing import Annotated

from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

import matplotlib
matplotlib.use('TKAgg')  # If 'MacOSX' doesn't work, try TKAgg



tavily_tool = TavilySearchResults(max_results=5)
# search_query = "latest AI advancements in 2024"
# search_results = tavily_tool.invoke({"query": search_query})
# print(search_results)


# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
    
# # Example of calling the python_repl_tool with invoke method
# example_code = """
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# plt.plot(x, y)
# plt.title('Example Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# """
# # Correctly using the invoke method
# execution_output = python_repl_tool.invoke(example_code)
# print(execution_output)

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )
    


from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# Research agent and node
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    state_modifier=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague."
    ),
)


def research_node(
    state: MessagesState,
) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


# Chart generator agent and node
# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    state_modifier=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague."
    ),
)


def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )
    
from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "researcher")
graph = workflow.compile()

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# try:
#     # Get the graph's mermaid PNG representation
#     graph_image = graph.get_graph().draw_mermaid_png()

#     # Save the image to a file
#     image_path = 'graph_output.png'
#     with open(image_path, "wb") as f:
#         f.write(graph_image)

#     # Display the image using matplotlib
#     img = mpimg.imread(image_path)
#     plt.imshow(img)
#     plt.axis('off')  # Hide axes
#     plt.show()

# except Exception as e:
#     print(f"Error: {str(e)}")


events = graph.stream(
    {
        "messages": [
            (
                "user",
                "First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, finish.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 50},
)
for s in events:
    print(s)
    print("----")

# If you are running this code on a MacBook, you are likely to encounter the error:
# "libc++abi: terminating due to uncaught exception of type NSException."
# To fix this, you should set the Matplotlib backend to "MacOSX" or "TKAgg" to avoid GUI issues.
# You can set the backend by adding the following:
# import matplotlib
# matplotlib.use("MacOSX")  # or matplotlib.use("TKAgg")

# Additionally, make sure to install/update tcl-tk for compatibility:
# brew install tcl-tk

# This should help resolve the error related to GUI handling on macOS.
