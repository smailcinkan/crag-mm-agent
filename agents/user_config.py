import sys
sys.path.insert(0, '..')
from my_rag_agent import MyRAGAgent

from agents.rag_agent import SimpleRAGAgent
from agents.random_agent import RandomAgent
from agents.vanilla_llama_vision_agent import LlamaVisionModel

UserAgent = MyRAGAgent

# UserAgent = SimpleRAGAgent
# UserAgent = LlamaVisionModel

