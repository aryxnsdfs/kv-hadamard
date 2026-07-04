import torch, time
from transformers import LlamaForCausalLM
M = "unsloth/Llama-3.2-1B"

for impl in ["sdpa", "eager"]:
    try:
        m = LlamaForCausalLM.from_pretrained(M, torch_dtype=torch.float16,
                                             attn_implementation=impl).cuda().eval()
    except Exception as e:
        print(impl, "load failed:", e); continue
    print("requested", impl, "| actual config:", m.config._attn_implementation)
    for L in [4096, 8192]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(10, 100000, (1, L), device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            _ = m.model(ids)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()/1024**2
        print(f"  L={L:5d}  {time.time()-t0:.2f}s  peak={peak:.0f}MB")
        del ids
    del m; torch.cuda.empty_cache()
