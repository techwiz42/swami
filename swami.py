from swarm import Swarm
from swarm.types import Agent, Response
from typing import List, Dict, Any, Optional
import logging
import openai
import os
from datetime import datetime
import json
import os

class MemoryStore:
    """Simple memory management system"""
    def __init__(self, storage_path: str = "agent_memories"):
        self.storage_path = storage_path
        self.memory_limit = 100  # Keep last 100 interactions by default
        os.makedirs(storage_path, exist_ok=True)
        
    def store(self, agent_id: str, memory: Dict[str, Any]):
        """Store a memory for an agent"""
        file_path = os.path.join(self.storage_path, f"{agent_id}.json")
        
        try:
            # Load existing memories
            memories = self.load(agent_id)
            
            # Add new memory
            memories.append({
                **memory,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only the last N memories
            if len(memories) > self.memory_limit:
                memories = memories[-self.memory_limit:]
                
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(memories, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error storing memory for agent {agent_id}: {e}")
    
    def load(self, agent_id: str) -> List[Dict[str, Any]]:
        """Load memories for an agent"""
        file_path = os.path.join(self.storage_path, f"{agent_id}.json")
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
            
        except Exception as e:
            logging.error(f"Error loading memories for agent {agent_id}: {e}")
            return []
    
    def search(self, agent_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple search for relevant memories"""
        memories = self.load(agent_id)
        
        # Basic keyword matching (could be enhanced with more sophisticated search)
        relevant = []
        query_terms = query.lower().split()
        
        for memory in memories:
            content = memory.get('content', '').lower()
            if any(term in content for term in query_terms):
                relevant.append(memory)
                
        return relevant[-limit:]  # Return most recent relevant memories

from pydantic import Field

class EnhancedAgent(Agent):
    """Agent with memory capabilities"""
    memory_store: MemoryStore = Field(default=None, exclude=True)
    agent_id: str = Field(default="", exclude=True)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"  # Allow extra attributes
    }
    
    def __init__(
        self,
        system: str,
        model: str = "gpt-4",
        functions: Optional[List[Dict[str, Any]]] = None,
        memory_path: str = "agent_memories"
    ):
        super().__init__(
            system=system,
            model=model,
            functions=functions if functions is not None else []
        )
        object.__setattr__(self, 'memory_store', MemoryStore(memory_path))
        object.__setattr__(self, 'agent_id', f"agent_{model}_{hash(system)}")
    
    async def think(self, query: str) -> str:
        """Process query with context from memory"""
        # Get relevant memories
        memories = self.memory_store.search(self.agent_id, query)
        
        # Format context from memories
        context = ""
        if memories:
            context = "Relevant context from memory:\n" + "\n".join([
                f"- {memory.get('content', '')}"
                for memory in memories
            ]) + "\n\n"
        
        # Store this interaction
        self.memory_store.store(self.agent_id, {
            "content": query,
            "type": "query"
        })
        
        return f"{context}Current query: {query}"

class EnhancedSwarm(Swarm):
    """Swarm with memory-enhanced agents"""
    
    async def run(
        self,
        agent: EnhancedAgent,
        messages: List[Dict[str, str]],
        context_variables: Dict[str, Any] = None,
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float('inf'),
        execute_tools: bool = True
    ) -> Response:
        """Run with memory enhancement"""
        # Initialize context variables if None
        if context_variables is None:
            context_variables = {}

        # Process messages with agent's memory
        enhanced_messages = []
        for message in messages:
            if message["role"] == "user":
                # Add context from memory
                thought = await agent.think(message["content"])
                enhanced_messages.extend([
                    {"role": "system", "content": thought},
                    message
                ])
            else:
                enhanced_messages.append(message)
        
        # Run normal Swarm execution
        response = await super().run(
            agent=agent,
            messages=enhanced_messages,
            context_variables=context_variables,
            model_override=model_override,
            stream=stream,
            debug=debug,
            max_turns=max_turns,
            execute_tools=execute_tools
        )
        
        # Store response in memory
        if response.message:
            agent.memory_store.store(agent.agent_id, {
                "content": response.message,
                "type": "response"
            })
        
        return response

def get_openai_api_key() -> str:
    """Get OpenAI API key from various sources."""
    # First try environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Try reading from shell rc files if env var not found
    if not api_key:
        shell_files = [
            os.path.expanduser('~/.bashrc'),
            os.path.expanduser('~/.bash_profile'),
            os.path.expanduser('~/.zshrc')
        ]
        
        for file_path in shell_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            if 'OPENAI_API_KEY' in line and '=' in line:
                                # Extract key from export or direct assignment
                                api_key = line.split('=')[1].strip().strip('"\'')
                                if api_key:
                                    return api_key
                except Exception as e:
                    logging.warning(f"Error reading {file_path}: {e}")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please ensure it's properly set in your environment."
        )
    
    return api_key

async def main():
    # Get API key
    api_key = get_openai_api_key()
    
    # Configure OpenAI client
    import openai
    openai.api_key = api_key
    # Create enhanced agent
    agent = EnhancedAgent(
        system="I am an AI assistant with memory capabilities.",
        model="gpt-4",
        memory_path="./agent_memories"
    )
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Create enhanced swarm
    swarm = EnhancedSwarm(api_key=api_key)
    
    # Example conversation
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = await swarm.run(agent, messages)
    print(f"First Response: {response.message}")
    
    # Ask about previous conversation
    messages = [
        {"role": "user", "content": "What did we just discuss?"}
    ]
    
    response = await swarm.run(agent, messages)
    print(f"\nSecond Response: {response.message}")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
