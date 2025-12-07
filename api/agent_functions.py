from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from api.search import search
from dotenv import load_dotenv
from langchain.tools import tool
import json
import os
from langchain.prompts import PromptTemplate
load_dotenv()

# Store chat histories per conversation ID
conversation_histories = {}
# Store agent executors per conversation ID for reuse
agent_executors = {}

def create_get_info_tool(conversation_id: str):
    """Creates a get_info tool bound to a specific conversation ID."""
    @tool
    def get_info(tool_input: str) -> str:
        """Searches for information in the uploaded document. 
        Input can be either:
        1. A JSON string with 'query' key (e.g., a JSON object with a query field)
        2. A plain string (will be treated as the query directly)
        
        The conversation ID is automatically used for the search."""
        query = None
        
        # Try to parse as JSON first
        try:
            parsed_input = json.loads(tool_input)
            if isinstance(parsed_input, dict) and "query" in parsed_input:
                query = parsed_input["query"]
            else:
                # If it's not a dict with query, treat the whole input as query
                query = tool_input
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, treat the whole input as the query
            query = tool_input
        
        if not query or not query.strip():
            return "Error: No query provided. Please provide a search query."
        
        if not conversation_id:
            return "Error: No conversation ID available. Cannot search."
        
        try:
            search_results = search(query, conversation_id)
            if not search_results.matches:
                return f"No results found for query: {query}"
            
            results = [match['metadata'] for match in search_results.matches]
            return f"Found {len(results)} result(s) for query '{query}': {str(results)}"
        except Exception as e:
            return f"Error searching: {str(e)}"
    
    return get_info

def create_prompt_template(conversation_id: str):
    """Creates a prompt template with the conversation ID included."""
    # Format conversation_id directly into the template since it's constant for this conversation
    template_str = f"""Answer the following questions as best you can. You have access to the following tools:
{{tools}}

Your purpose is to find information in the uploaded document and return it to the user.

IMPORTANT: The conversation ID for this session is: {conversation_id}
The conversation ID is automatically used when searching - you don't need to provide it.

When using the get_info tool:
- You can provide the query as a plain string (e.g., "what is machine learning?")
- Or as JSON with a query key (the tool accepts both formats)
- The tool will automatically search in the correct document using the conversation ID

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one or more of [{{tool_names}}]
Action Input: the input to the action (just the search query as a string or JSON)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Chat History:
{{chat_history}}
Question: {{input}}
{{agent_scratchpad}}
"""
    return PromptTemplate.from_template(template_str)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Use with chat history
from langchain_core.messages import AIMessage, HumanMessage

def _format_chat_history(chat_history: list) -> str:
    formatted_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"Assistant: {message.content}")
    return "\n".join(formatted_history)

def get_or_create_chat_history(conversation_id: str):
    """Gets or creates a chat history for a conversation ID."""
    if conversation_id not in conversation_histories:
        conversation_histories[conversation_id] = [
            AIMessage(content="Hello! How can I assist you today?")
        ]
    return conversation_histories[conversation_id]

'''
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    chat_history_list.append(HumanMessage(content=user_input))
    formatted_chat_history = _format_chat_history(chat_history_list)

    response = agent_executor.invoke(
        {
            "input": user_input,
            "chat_history": formatted_chat_history,
        }
    )
    agent_response = response["output"]
    print(f"Agent: {agent_response}")
    chat_history_list.append(AIMessage(content=agent_response))
'''

def get_or_create_agent_executor(conversation_id: str):
    """Gets or creates an agent executor for a conversation ID."""
    if conversation_id not in agent_executors:
        # Create tools and prompt bound to this conversation ID
        tools = [create_get_info_tool(conversation_id)]
        prompt_template = create_prompt_template(conversation_id)
        
        # Create agent with conversation-specific tools and prompt
        agent = create_react_agent(llm, tools, prompt=prompt_template)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            handle_parsing_errors=True, 
            max_execution_time=120,  # Increased from 60 seconds
            max_iterations=15  # Increased from 10 to allow more tool calls
        )
        agent_executors[conversation_id] = agent_executor
    
    return agent_executors[conversation_id]

def process_user_input(user_input: str, conversation_id: str):
    """
    Takes user input and conversation ID, saves it to chat history, invokes the agent, 
    saves agent output to history, and returns agent output.
    """
    # Get or create chat history for this conversation
    chat_history_list = get_or_create_chat_history(conversation_id)
    
    # Get or create agent executor for this conversation
    agent_executor = get_or_create_agent_executor(conversation_id)
    
    # Add user input to history
    chat_history_list.append(HumanMessage(content=user_input))
    formatted_chat_history = _format_chat_history(chat_history_list)

    # Invoke agent
    try:
        response = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": formatted_chat_history,
            }
        )
        agent_response = response["output"]
        chat_history_list.append(AIMessage(content=agent_response))
        return agent_response
    except Exception as e:
        error_msg = f"Agent execution error: {str(e)}"
        chat_history_list.append(AIMessage(content=error_msg))
        return error_msg
    
