from exllamav2 import ExLlamaV2Cache, ExLlamaV2Cache_Q4, ExLlamaV2Cache_Q8

MODEL_MAX_SEQ_LEN = 1 * 32768
# Configuration constants for LLM model behavior and server operation
INSTRUCTION_LIMIT = 2048
CHAT_CONTEXT_LIMIT = MODEL_MAX_SEQ_LEN - INSTRUCTION_LIMIT
PROMPT_LIMIT = 1024
RESPONSE_LIMIT = 8192

WORKING_DIR = "/home/ksolomon/git/karllm/karllm-server/"

MODEL_DIR = "/home/ksolomon/git/models/"  # Path to quantized model directory
ACTIVE_MODEL_FILE = WORKING_DIR + "active_model.json"
SAFETENSORS_FILE = MODEL_DIR + "/model.safetensors"  # Path to main model weights
SESSION_CACHE_FILE = "session_cache.pt"  # Path for saving/restoring KV cache
SESSION_DIR = WORKING_DIR + "users/"  # Path for saving interaction traces

CHUNK_SIZE = 4  # Streaming chunk size (tokens per SSE flush)

TENSOR_PARALLEL = True  # Enable multi-GPU model sharding
NO_GRAPHS = False
NO_FLASH_ATTN = False
GPU_SPLIT = "auto"  # Auto-assign memory split across devices

SAVE_INTERACTION = False

SERVER_TIMEOUT_MINUTES = 300

# Model Parameters
CACHE_QUANTIZATION = ExLlamaV2Cache_Q8
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
TOKEN_REPETITION_PENALTY = 1.1
EOS_TOKEN_ID = 151645
EOS_TOKEN_ID_BACKUP = 151643
