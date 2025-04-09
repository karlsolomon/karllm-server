import os

from config import INSTRUCTION_FILE, INSTRUCTION_LIMIT
from model.init import ModelState

instruction_context = []
chat_context = []

if os.path.exists(INSTRUCTION_FILE):
    with open(INSTRUCTION_FILE, "r") as f:
        instruction_context = f.readlines()


def trim_instruction_context():
    while (
        ModelState.tokenizer.num_tokens("\n".join(instruction_context))
        > INSTRUCTION_LIMIT
    ):
        instruction_context.pop(0)
    with open(INSTRUCTION_FILE, "w") as f:
        f.writelines(instruction_context)
