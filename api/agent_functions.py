from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from api.search import search
from dotenv import load_dotenv
import json
import os
import logging
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store chat histories per conversation ID
conversation_histories = {}
# Store agents per conversation ID for reuse
agents = {}

def create_get_info_tool(conversation_id: str):
    """Creates a get_info tool bound to a specific conversation ID."""
    logger.info(f"Creating get_info tool for conversation_id: {conversation_id}")
    
    @tool
    def get_info(tool_input: str) -> str:
        """Searches for information in the uploaded document. 
        Input can be either:
        1. A JSON string with 'query' key (e.g., a JSON object with a query field)
        2. A plain string (will be treated as the query directly)
        
        The conversation ID is automatically used for the search."""
        logger.info(f"get_info tool called with conversation_id: {conversation_id}, tool_input: {tool_input[:100]}...")
        query = None
        
        # Try to parse as JSON first
        try:
            parsed_input = json.loads(tool_input)
            if isinstance(parsed_input, dict) and "query" in parsed_input:
                query = parsed_input["query"]
                logger.debug(f"Parsed JSON input, extracted query: {query[:100]}...")
            else:
                # If it's not a dict with query, treat the whole input as query
                query = tool_input
                logger.debug(f"Input is not a dict with 'query' key, using entire input as query")
        except (json.JSONDecodeError, TypeError) as e:
            # If not valid JSON, treat the whole input as the query
            query = tool_input
            logger.debug(f"Input is not valid JSON, using as plain string query. Error: {str(e)}")
        
        if not query or not query.strip():
            logger.warning(f"Empty query provided for conversation_id: {conversation_id}")
            return "Error: No query provided. Please provide a search query."
        
        if not conversation_id:
            logger.error("No conversation ID available for search")
            return "Error: No conversation ID available. Cannot search."
        
        try:
            logger.info(f"Searching with query: '{query[:100]}...' for conversation_id: {conversation_id}")
            search_results = search(query, conversation_id)
            if not search_results.matches:
                logger.info(f"No results found for query: '{query[:100]}...' in conversation_id: {conversation_id}")
                return f"No results found for query: {query}"
            
            results = [match['metadata'] for match in search_results.matches]
            logger.info(f"Found {len(results)} result(s) for query '{query[:100]}...' in conversation_id: {conversation_id}")
            return f"Found {len(results)} result(s) for query '{query}': {str(results)}"
        except Exception as e:
            logger.error(f"Error searching for query '{query[:100]}...' in conversation_id {conversation_id}: {str(e)}", exc_info=True)
            return f"Error searching: {str(e)}"
    
    return get_info

def create_system_prompt(conversation_id: str) -> str:
    """Creates a system prompt with the conversation ID included."""
    logger.debug(f"Creating system prompt for conversation_id: {conversation_id}")
    return f"""You are a helpful assistant that finds information in uploaded documents.

IMPORTANT: The conversation ID for this session is: {conversation_id}
The conversation ID is automatically used when searching - you don't need to provide it.

Your purpose is to find information in the uploaded document and return it to the user.

When using the get_info tool:
- You can provide the query as a plain string (e.g., "what is machine learning?")
- Or as JSON with a query key (the tool accepts both formats)
- The tool will automatically search in the correct document using the conversation ID

Always be helpful, accurate, and provide clear answers based on the information you find."""

# Chat history types already imported above

def get_or_create_chat_history(conversation_id: str):
    """Gets or creates a chat history for a conversation ID."""
    if conversation_id not in conversation_histories:
        logger.info(f"Creating new chat history for conversation_id: {conversation_id}")
        conversation_histories[conversation_id] = []
    else:
        logger.debug(f"Retrieving existing chat history for conversation_id: {conversation_id} (length: {len(conversation_histories[conversation_id])})")
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

def get_or_create_agent(conversation_id: str):
    """Gets or creates an agent for a conversation ID."""
    if conversation_id not in agents:
        logger.info(f"Creating new agent for conversation_id: {conversation_id}")
        # Create tools and system prompt bound to this conversation ID
        tools = [create_get_info_tool(conversation_id)]
        system_prompt = create_system_prompt(conversation_id)
        
        # Create agent with conversation-specific tools and system prompt
        logger.debug(f"Creating agent with {len(tools)} tool(s) for conversation_id: {conversation_id}")
        agent = create_agent(
            model="gpt-4o",
            tools=tools,
            system_prompt=system_prompt
        )
        agents[conversation_id] = agent
        logger.info(f"Agent created successfully for conversation_id: {conversation_id}")
    else:
        logger.debug(f"Reusing existing agent for conversation_id: {conversation_id}")
    
    return agents[conversation_id]

def process_user_input(user_input: str, conversation_id: str):
    """
    Takes user input and conversation ID, saves it to chat history, invokes the agent, 
    saves agent output to history, and returns agent output.
    """
    logger.info(f"Processing user input for conversation_id: {conversation_id}, input length: {len(user_input)}")
    logger.debug(f"User input (first 200 chars): {user_input[:200]}...")
    
    # Get or create chat history for this conversation
    chat_history_list = get_or_create_chat_history(conversation_id)
    logger.debug(f"Chat history length before adding user input: {len(chat_history_list)}")
    
    # Get or create agent for this conversation
    agent = get_or_create_agent(conversation_id)
    
    # Add user input to history
    chat_history_list.append(HumanMessage(content=user_input))
    
    # Build messages list for agent invocation (LangChain v1.x format)
    messages = chat_history_list.copy()
    logger.debug(f"Invoking agent with {len(messages)} message(s) in history")

    # Invoke agent
    try:
        logger.info(f"Invoking agent for conversation_id: {conversation_id}")
        response = agent.invoke({"messages": messages})
        
        # Extract the response - in LangChain v1.x, response contains messages
        # The response is a dict with "messages" key containing the full conversation
        if isinstance(response, dict) and "messages" in response:
            response_messages = response["messages"]
            # Find the last AIMessage in the response
            agent_response = None
            for msg in reversed(response_messages):
                if isinstance(msg, AIMessage):
                    agent_response = msg.content
                    break
            
            if agent_response is None:
                # Fallback: get the last message content
                if response_messages:
                    last_msg = response_messages[-1]
                    if hasattr(last_msg, 'content'):
                        agent_response = last_msg.content
                    else:
                        agent_response = str(last_msg)
                else:
                    agent_response = "No response generated"
        elif isinstance(response, dict) and "output" in response:
            # Fallback for older format
            agent_response = response["output"]
        elif isinstance(response, list) and len(response) > 0:
            # If response is a list of messages
            last_msg = response[-1]
            if hasattr(last_msg, 'content'):
                agent_response = last_msg.content
            else:
                agent_response = str(last_msg)
        else:
            agent_response = str(response)
        
        logger.info(f"Agent execution completed successfully for conversation_id: {conversation_id}")
        logger.debug(f"Agent response (first 200 chars): {agent_response[:200]}...")
        logger.debug(f"Full agent response length: {len(agent_response)} characters")
        
        # Add agent response to chat history
        chat_history_list.append(AIMessage(content=agent_response))
        logger.debug(f"Chat history length after adding agent response: {len(chat_history_list)}")
        return agent_response
    except Exception as e:
        logger.error(f"Agent execution error for conversation_id {conversation_id}: {str(e)}", exc_info=True)
        error_msg = f"Agent execution error: {str(e)}"
        chat_history_list.append(AIMessage(content=error_msg))
        return error_msg
    
