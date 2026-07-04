import torch, time
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import LlamaForCausalLM, AttentionInterface

M = "unsloth/Llama-3.2-1B"

def eff_gqa_attention(module, query, key, value, attention_mask=None,
                      scaling=None, dropout=0.0, **kwargs):
    # query [b, Hq, q, d]; key/value [b, Hkv, k, d]  (GQA: Hq > Hkv)
    Hq, Hkv = query.shape[1], key.shape[1]
    if Hkv != Hq:
        rep = Hq // Hkv
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    q_len, k_len = query.shape[2], key.shape[2]
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH]  # math kept only as last resort
    if q_len == k_len and q_len > 1:
        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, is_causal=True, scale=scaling)
    else:
        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, is_causal=False,
                scale=scaling)
    return out.transpose(1, 2).contiguous(), None

AttentionInterface.register("eff_gqa", eff_gqa_attention)

m = LlamaForCausalLM.from_pretrained(M, torch_dtype=torch.float16,
                                     attn_implementation="eff_gqa").cuda().eval()
print("attn impl:", m.config._attn_implementation)

# correctness vs default on a small input
ids_small = torch.randint(10, 100000, (1, 128), device="cuda")
with torch.no_grad():
    a = m.model(ids_small).last_hidden_state
m2 = LlamaForCausalLM.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()
with torch.no_grad():
    b = m2.model(ids_small).last_hidden_state
print("max abs diff vs default:", (a - b).abs().max().item())
del m2; torch.cuda.empty_cache()

for L in [4096, 8192, 16384]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(10, 100000, (1, L), device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    try:
        with torch.no_grad():
            _ = m.model(ids)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()/1024**2
        print(f"  L={L:5d}  {time.time()-t0:.2f}s  peak={peak:.0f}MB")
    except RuntimeError as e:
        print(f"  L={L:5d}  ERROR {str(e)[:80]}")
    del ids; torch.cuda.empty_cache()
