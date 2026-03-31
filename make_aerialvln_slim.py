import argparse
import json
import os
import random


def load_episodes(split_path):
    with open(split_path, "r") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    return payload, episodes


def sample_episodes(episodes, sample_size, seed):
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if len(episodes) < sample_size:
        raise ValueError(
            f"Requested {sample_size} episodes, but only {len(episodes)} available."
        )
    rng = random.Random(seed)
    return rng.sample(episodes, sample_size)


def save_slim_split(output_path, payload, sampled_episodes):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload["episodes"] = sampled_episodes
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def build_slim_dataset(src_dir, dst_dir, split, sample_size, seed):
    src_split_path = os.path.join(src_dir, f"{split}.json")
    dst_split_path = os.path.join(dst_dir, f"{split}.json")

    payload, episodes = load_episodes(src_split_path)
    sampled = sample_episodes(episodes, sample_size=sample_size, seed=seed)
    save_slim_split(dst_split_path, payload, sampled)

    return dst_split_path, len(sampled), len(episodes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create aerialvln-slim by sampling episodes from aerialvln-s."
    )
    parser.add_argument(
        "--src_dir",
        default="/home/liusongbo/AirVLN_ws/DATA/data/aerialvln-s",
        help="Source aerialvln-s directory.",
    )
    parser.add_argument(
        "--dst_dir",
        default="/home/liusongbo/AirVLN_ws/DATA/data/aerialvln-slim",
        help="Output aerialvln-slim directory.",
    )
    parser.add_argument(
        "--split",
        default="val_seen",
        help="Split file name without .json.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=30,
        help="Number of episodes to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, kept, total = build_slim_dataset(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(
        f"Saved {kept}/{total} episodes to {output_path} "
        f"(split={args.split}, seed={args.seed})."
    )


if __name__ == "__main__":
    main()
