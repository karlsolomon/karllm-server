from exllamav2 import ExLlamaV2Cache_Q8

# Configuration constants for LLM model behavior and server operation
INSTRUCTION_LIMIT = 4096  # Max tokens allowed in the instruction context
CHAT_CONTEXT_LIMIT = 28672  # Max total tokens in context window (instruction + chat)
PROMPT_LIMIT = 2048  # Max input prompt tokens per interaction
RESPONSE_LIMIT = 4096  # Max output tokens generated per interaction
MODEL_MAX_SEQ_LEN = 32768

MODEL_DIR = "/home/ksolomon/git/quant"  # Path to quantized model directory
SAFETENSORS_FILE = MODEL_DIR + "/model.safetensors"  # Path to main model weights
SESSION_CACHE_FILE = "session_cache.pt"  # Path for saving/restoring KV cache
INTERACTION_DIR = "./users/ksolomon/sessions"  # Path for saving interaction traces
SESSION_DIR = "/home/ksolomon/git/karllm/karllm-server/users/"  # Path for saving interaction traces

CHUNK_SIZE = 4  # Streaming chunk size (tokens per SSE flush)

TENSOR_PARALLEL = True  # Enable multi-GPU model sharding
NO_GRAPHS = False
NO_FLASH_ATTN = False
GPU_SPLIT = "auto"  # Auto-assign memory split across devices

SAVE_INTERACTION = False

SERVER_TIMEOUT_MINUTES = 30

# Model Parameters
CACHE_QUANTIZATION = ExLlamaV2Cache_Q8
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
TOKEN_REPETITION_PENALTY = 1.1
EOS_TOKEN_ID = 151645
