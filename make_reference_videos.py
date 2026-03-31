import argparse
import json
import os
import sys
from collections import defaultdict

import airsim
import cv2
from tqdm import tqdm

def build_tool(scene_id):
    from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool

    machines_info = [
        {
            "MACHINE_IP": "127.0.0.1",
            "SOCKET_PORT": 30000,
            "MAX_SCENE_NUM": 8,
            "open_scenes": [scene_id],
        }
    ]
    tool = AirVLNSimulatorClientTool(machines_info=machines_info)
    tool.run_call()
    return tool


def parse_pose(pose_item):
    if len(pose_item) == 6:
        x, y, z, pitch, roll, yaw = pose_item
        q = airsim.to_quaternion(pitch, roll, yaw)
        return airsim.Pose(
            position_val=airsim.Vector3r(x, y, z),
            orientation_val=airsim.Quaternionr(q.x_val, q.y_val, q.z_val, q.w_val),
        )
    if len(pose_item) == 7:
        x, y, z, qx, qy, qz, qw = pose_item
        return airsim.Pose(
            position_val=airsim.Vector3r(x, y, z),
            orientation_val=airsim.Quaternionr(qx, qy, qz, qw),
        )
    raise ValueError(f"Unsupported pose length: {len(pose_item)}")


def write_video_from_frames(rgb_dir, output_video, text, fps):
    from utils.utils import append_text_to_image

    img_files = sorted(
        [name for name in os.listdir(rgb_dir) if name.endswith(".png")],
        key=lambda name: int(name.split("_")[-1].split(".")[0]),
    )
    if not img_files:
        return False

    first = cv2.imread(os.path.join(rgb_dir, img_files[0]))
    first = append_text_to_image(first, text)
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    out.write(first)

    for img_name in img_files[1:]:
        img = cv2.imread(os.path.join(rgb_dir, img_name))
        frame = append_text_to_image(img, text)
        out.write(frame)
    out.release()
    return True


def render_episode_reference_video(tool, episode, output_root, fps):
    from utils.env_utils import get_front_observations

    episode_id = episode["episode_id"]
    scene_id = episode["scene_id"]
    instruction = episode["instruction"]["instruction_text"]
    reference_path = episode["reference_path"]

    save_dir = os.path.join(output_root, str(scene_id), episode_id)
    rgb_dir = os.path.join(save_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    frame_count = 0
    for i, pose_item in enumerate(reference_path):
        pose = parse_pose(pose_item)
        if not tool.setPoses([[pose]]):
            raise RuntimeError(f"setPoses failed at frame {i}, episode {episode_id}")

        pano_obs, _ = get_front_observations(pose, tool, scene_id=scene_id)
        rgb = pano_obs[0][0]
        if rgb is None:
            continue
        cv2.imwrite(os.path.join(rgb_dir, f"rgb_obs_front_{i}.png"), rgb)
        frame_count += 1

    video_path = os.path.join(save_dir, "demo.avi")
    ok = write_video_from_frames(rgb_dir, video_path, instruction, fps=fps)
    return frame_count, video_path if ok else None


def group_by_scene(episodes):
    scene_groups = defaultdict(list)
    for ep in episodes:
        scene_groups[ep["scene_id"]].append(ep)
    return scene_groups


def parse_user_args():
    parser = argparse.ArgumentParser(description="Render videos from reference_path trajectories.")
    parser.add_argument(
        "--dataset",
        default="/home/liusongbo/AirVLN_ws/DATA/data/aerialvln-slim/val_seen.json",
        help="Input dataset JSON with episodes.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output/reference_videos",
        help="Output directory for rendered frames and videos.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video fps.")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Optional limit for quick testing. 0 means all episodes.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def with_project_defaults(argv):
    defaults = {
        "--run_type": "eval",
        "--collect_type": "TF",
        "--Image_Width_RGB": "512",
        "--Image_Height_RGB": "512",
        "--Image_Width_DEPTH": "512",
        "--Image_Height_DEPTH": "512",
    }
    out = list(argv)
    for key, value in defaults.items():
        if key not in out:
            out.extend([key, value])
    return out


def main():
    args, unknown = parse_user_args()
    # Keep this script args for us, forward the rest to project-level argparse.
    sys.argv = [sys.argv[0]] + with_project_defaults(unknown)

    with open(args.dataset, "r") as f:
        episodes = json.load(f)["episodes"]

    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    os.makedirs(args.output_dir, exist_ok=True)
    scene_groups = group_by_scene(episodes)

    total_ok = 0
    total_fail = 0

    for scene_id, scene_episodes in scene_groups.items():
        print(f"[scene {scene_id}] opening simulator for {len(scene_episodes)} episodes...")
        tool = build_tool(scene_id)

        try:
            for ep in tqdm(scene_episodes, desc=f"Scene {scene_id}", ncols=100):
                try:
                    frame_count, video_path = render_episode_reference_video(
                        tool, ep, args.output_dir, fps=args.fps
                    )
                    if video_path is None:
                        print(f"[skip] {ep['episode_id']}: no frames")
                        total_fail += 1
                    else:
                        print(f"[ok] {ep['episode_id']} frames={frame_count} video={video_path}")
                        total_ok += 1
                except Exception as e:
                    print(f"[fail] {ep['episode_id']}: {e}")
                    total_fail += 1
        finally:
            try:
                tool.closeScenes()
            except Exception:
                pass

    print(f"Done. success={total_ok}, failed={total_fail}")


if __name__ == "__main__":
    main()
