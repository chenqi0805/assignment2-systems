import argparse
from ast import parse
import logging
import timeit

import numpy as np
import pandas as pd
import torch
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

predefined_configs = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {"size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000,
                       help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=256,
                       help="Context length")
    parser.add_argument("--d_model", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=16,
                       help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344,
                       help="Feed-forward dimension")
    parser.add_argument("--theta", type=float, default=10000.0,
                       help="RoPE theta parameter")
    
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--mode", type=str, default="forward",
                       help="Benchmark mode: forward or forward_backward")
    parser.add_argument("--warmup_steps", type=int, default=5,
                       help="Number of warmup steps")
    parser.add_argument("--timing_steps", type=int, default=10,
                       help="Number of timing steps")
    parser.add_argument("--all", action="store_true", help="Run all predefined configurations")
    
    return parser.parse_args()

def main():
    args = parse_args()

    device="cuda" if torch.cuda.is_available() else "mps"

    if args.all:
        model_configs = predefined_configs
    else:
        model_configs = [{
            "size": "custom",
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }]

    results = []
    for model_config in model_configs:
        if device == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()
        print(f"Running benchmark for model config: {model_config}")

        vocab_size = args.vocab_size
        batch_size = args.batch_size
        context_length = args.context_length

        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            rope_theta=args.theta,
        ).to(device)

        optimizer = AdamW(model.parameters())

        model.train()

        # Create random input data  
        dataset = np.random.randint(
            low=0, 
            high=vocab_size, 
            size=(batch_size * context_length,)
        )

        x, y = get_batch(dataset, batch_size, context_length, device)

        def step_forward():
            with torch.no_grad():
                _ = model(x)

        def step_forward_backward():
            optimizer.zero_grad()
            out = model(x)
            loss = cross_entropy(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

        step_fn = step_forward if args.mode == "forward" else step_forward_backward

        for _ in range(args.warmup_steps):
            step_fn()

        times = []
        for _ in range(args.timing_steps):
            start = timeit.default_timer()
            step_fn()
            if device == "cuda":
                torch.cuda.synchronize()
            else:
                torch.mps.synchronize()
            end = timeit.default_timer()
            times.append(end - start)

        results.append(
            {
                "vocab_size": vocab_size,
                "batch_size": batch_size,
                "context_length": context_length,
                **model_config,
                "time": np.mean(times),
                "std": np.std(times),
            }
        )

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # Save to file
    with open("benchmark_results.md", "w") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    main()