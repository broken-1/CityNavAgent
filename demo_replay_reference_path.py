import argparse
import json
import os
import time

import airsim

# python demo_replay_reference_path.py --scene_id 3 --list
# python demo_replay_reference_path.py --scene_id 3 --episode_index 0

DEFAULT_DATASET = "/home/liusongbo/AirVLN_ws/DATA/data/aerialvln-s/val_seen.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay one reference_path from the dataset in a manually opened AirSim scene."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset JSON path.")
    parser.add_argument("--scene_id", type=int, default=12, help="Scene id to filter.")
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Index within the filtered scene episodes.",
    )
    parser.add_argument(
        "--episode_id",
        default="",
        help="Optional episode id. Overrides episode_index if provided.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="AirSim host.")
    parser.add_argument("--port", type=int, default=41451, help="AirSim API port.")
    parser.add_argument(
        "--step_sleep",
        type=float,
        default=0.12,
        help="Sleep seconds between poses.",
    )
    parser.add_argument(
        "--vehicle_name",
        default="Drone_1",
        help="AirSim vehicle name used by the scene.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available episodes for the selected scene and exit.",
    )
    return parser.parse_args()


def load_episode(dataset_path, scene_id, episode_index=0, episode_id=""):
    with open(dataset_path, "r") as f:
        episodes = json.load(f)["episodes"]

    scene_episodes = [ep for ep in episodes if ep["scene_id"] == scene_id]
    if not scene_episodes:
        raise ValueError(f"No episodes found for scene_id={scene_id} in {dataset_path}")

    if episode_id:
        for ep in scene_episodes:
            if ep["episode_id"] == episode_id:
                return ep, scene_episodes
        raise ValueError(f"episode_id={episode_id} not found in scene_id={scene_id}")

    if not (0 <= episode_index < len(scene_episodes)):
        raise IndexError(
            f"episode_index={episode_index} out of range for scene_id={scene_id}, "
            f"available={len(scene_episodes)}"
        )
    return scene_episodes[episode_index], scene_episodes


def pose_from_item(pose_item):
    if len(pose_item) != 6:
        raise ValueError(f"Expected pose item length 6, got {len(pose_item)}")
    x, y, z, pitch, roll, yaw = pose_item
    q = airsim.to_quaternion(pitch, roll, yaw)
    return airsim.Pose(
        position_val=airsim.Vector3r(x, y, z),
        orientation_val=airsim.Quaternionr(q.x_val, q.y_val, q.z_val, q.w_val),
    )


def connect_client(host, port):
    client = airsim.VehicleClient(ip=host, port=port, timeout_value=3600)
    client.confirmConnection()
    return client


def resolve_vehicle_name(client, preferred_name):
    try:
        client.simGetVehiclePose(preferred_name)
        print(f"Using vehicle_name={preferred_name!r}")
        return preferred_name
    except Exception:
        pass

    try:
        client.simGetVehiclePose("")
        print("Using default vehicle_name=''")
        return ""
    except Exception as e:
        raise RuntimeError(
            "Unable to access vehicle pose with either the requested vehicle name "
            f"({preferred_name!r}) or the default unnamed vehicle."
        ) from e


def replay_reference_path(client, episode, vehicle_name, step_sleep):
    ref_path = episode["reference_path"]
    for step, pose_item in enumerate(ref_path):
        pose = pose_from_item(pose_item)
        client.simSetVehiclePose(
            pose=pose,
            ignore_collision=True,
            vehicle_name=vehicle_name,
        )
        print(f"[step {step + 1}/{len(ref_path)}] pose={pose_item[:3]}")
        time.sleep(step_sleep)


def main():
    args = parse_args()
    episode, scene_episodes = load_episode(
        args.dataset,
        scene_id=args.scene_id,
        episode_index=args.episode_index,
        episode_id=args.episode_id,
    )

    if args.list:
        print(f"Scene {args.scene_id} episodes: {len(scene_episodes)}")
        for idx, ep in enumerate(scene_episodes):
            print(f"[{idx}] {ep['episode_id']}")
        return

    print(f"Dataset: {args.dataset}")
    print(f"Scene {args.scene_id} episodes: {len(scene_episodes)}")
    print(f"Selected episode: {episode['episode_id']}")
    print(f"Instruction: {episode['instruction']['instruction_text']}")
    print(f"Reference path length: {len(episode['reference_path'])}")
    print(f"Connecting to AirSim at {args.host}:{args.port} ...")

    client = connect_client(args.host, args.port)
    vehicle_name = resolve_vehicle_name(client, args.vehicle_name)
    replay_reference_path(
        client,
        episode,
        vehicle_name=vehicle_name,
        step_sleep=args.step_sleep,
    )
    print("Replay complete.")


if __name__ == "__main__":
    main()
