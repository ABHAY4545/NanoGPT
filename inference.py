import os
import torch
import torch.nn.functional as F
from NanoGPT import GPTConfig, GPT
import tiktoken

torch._dynamo.config.suppress_errors = True  # Windows Only
# torch._dynamo.config.optimize_ddp = False
torch.set_float32_matmul_precision("high")

checkpoint_path = "checkpoints/parameter_checkpoint_19072.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 100
temperature = 1.0
top_k = 40

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model_state_dict = checkpoint["model"]
config = checkpoint["config"]


model = GPT(config).to(device)
model = torch.compile(model)
model.load_state_dict(model_state_dict)
model.eval()

enc = tiktoken.get_encoding("gpt2")


def generate(
    prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
):

    input_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(input_ids)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

        if next_token.item() == enc.eot_token:
            break

    return enc.decode(input_ids[0].tolist())


prompt = "I'm language model and "
output = generate(prompt)
print("Prompt:", prompt)
print("Generated:", output)
