import json
import os
import re
import sys
import unicodedata
import time

import airsim
import cv2
import numpy as np

from tqdm import tqdm

from src.llm.query_llm import OpenAI_LLM_v2
from airsim_plugin.airsim_settings import (
    DefaultAirsimActionCodes,
    ObservationDirections,
)
from utils.env_utils import getPoseAfterMakeActions, get_pano_observations, get_front_observations
from utils.utils import append_text_to_image
from evaluator.nav_evaluator import CityNavEvaluator
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool


def load_navigation_tasks(split, data_dir):
    split_path = os.path.join(data_dir, f"{split}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cannot find navigation data: {split_path}")

    with open(split_path, 'r') as f:
        payload = json.load(f)

    episodes = payload.get('episodes')
    if episodes is None:
        raise KeyError(f"Missing 'episodes' in navigation data: {split_path}")

    for episode in episodes:
        if 'scene_id' not in episode:
            raise KeyError("Episode is missing scene_id while running in multi-scene mode")

    return episodes

def build_single_scene_machines_info(scene_id):
    return [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

def convert_airsim_pose(pose):
    assert len(pose) == 7, "The length of input pose must be 7"
    formatted_airsim_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            pose[0],
            pose[1],
            pose[2]
        ),
        orientation_val=airsim.Quaternionr(
            x_val=pose[3],
            y_val=pose[4],
            z_val=pose[5],
            w_val=pose[6],
        )
    )
    return formatted_airsim_pose

def parse_action_response(raw_text):
    cleaned_text = raw_text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    action_name = None
    reason = ""

    try:
        payload = json.loads(cleaned_text)
        action_text = payload.get("action")
        if action_text is not None:
            normalized = unicodedata.normalize("NFKC", str(action_text)).strip().upper()
            normalized = re.sub(r"[\s\-]+", "_", normalized)
            if normalized in DefaultAirsimActionCodes:
                action_name = normalized
        reason = str(payload.get("reason", "")).strip()
    except Exception:
        pass

    if not action_name:
        action_names = sorted(DefaultAirsimActionCodes.keys(), key=len, reverse=True)
        for candidate in action_names:
            pattern = r"\b" + re.escape(candidate).replace("_", r"[\s_\-]*") + r"\b"
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                action_name = candidate
                break

    if not action_name:
        action_name = "STOP"
        reason = f"[PARSE_FAILED] fallback to STOP. cleaned_response={cleaned_text}"
    elif not reason:
        reason = cleaned_text

    return action_name, reason, cleaned_text

def build_qwen_action_prompt(instruction_text):
    action_space = ", ".join(DefaultAirsimActionCodes.keys())

    prompt = f"""
You are a drone that is currently executing an aerial navigation task.
You will make one navigation decision at a time and continue iterating until the task is finished.

Navigation instruction:
{instruction_text}

Your visual observations from different viewpoints are provided above: left, slightly left, front, slightly right, right.
Based on the navigation instruction and multi-view observations, plan the single best next action.
Use the images to decide what helps the drone move closer to the target location.
Do not treat the instruction as a script to complete in one step.
Only output STOP when the destination is clearly reached and further movement would overshoot or be unnecessary.
If there is any reasonable next move that would improve alignment or get the drone closer, choose that move instead of STOP.

Choose exactly one next action from this action space:
{action_space}

Return JSON only:
{{"action":"ACTION_NAME","reason":"short reason"}}
""".strip()
    return prompt

def query_qwen_action(llm, instruction_text, viewpoint_img_path):
    prompt = build_qwen_action_prompt(instruction_text)
    raw_response = llm.query_viewpoint_api(
        prompt,
        viewpoint_img_path,
        show_response=False,
    )
    action_name, reason, cleaned_response = parse_action_response(raw_response)
    return action_name, reason, cleaned_response

def CityNavAgentMCTS(split, data_dir, max_step_size, record, model_name, max_episodes=0):
    os.makedirs("obs_imgs", exist_ok=True)

    predict_routes = []
    navi_tasks = load_navigation_tasks(split, data_dir)
    if max_episodes and max_episodes > 0:
        navi_tasks = navi_tasks[:max_episodes]

    nav_evaluator = CityNavEvaluator()

    llm = OpenAI_LLM_v2(
        max_tokens=10000,
        model_name=model_name,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        client_type="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking=False,
        cache_name="navigation_mcts",
    )

    tool = None
    active_scene_id = None

    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]
        episode_id = navi_task['episode_id']
        episode_scene_id = int(navi_task['scene_id'])
        curr_pose = convert_airsim_pose(
            navi_task["start_position"] + navi_task["start_rotation"][1:] + [navi_task["start_rotation"][0]]
        )

        if tool is None:
            tool = AirVLNSimulatorClientTool(machines_info=build_single_scene_machines_info(episode_scene_id))

        if active_scene_id != episode_scene_id:
            print(f"Switch simulator scene: {active_scene_id} -> {episode_scene_id} for episode {episode_id}")
            tool.machines_info = build_single_scene_machines_info(episode_scene_id)
            tool.run_call()
            active_scene_id = episode_scene_id

        print(
            f"================================ Start episode {episode_id} "
            f"(scene {episode_scene_id}) =================================="
        )
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']

        step_size = 0
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position'] + [0, 0, 0, 1])

        tool.setPoses([[curr_pose]])

        data_dict = {
            "episode_id": episode_id,
            "scene_id": episode_scene_id,
            "instruction": instruction,
            "gt_traj": [pose[:3] for pose in reference_path],
            "pred_traj": [],
            "pred_traj_explore": [list(curr_pose.position) + list(airsim.to_eularian_angles(curr_pose.orientation))]
        }

        while step_size < max_step_size:
            action_idx = step_size + 1
            try:
                pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=episode_scene_id)
                pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0]]
                pano_obs_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2]]

                pano_obs_imgs_path = [
                    "obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_"))
                    for view_drc in ObservationDirections
                ]
                for j in range(len(pano_obs_imgs_path)):
                    cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])
            except Exception as e:
                data_dict['pred_traj'].append(list(curr_pose.position))
                print(
                    f"Task idx: {i}. Action idx: {action_idx}. Step size: {step_size}. "
                    f"Success: False. Failed to get images. Exception: {e}"
                )
                break

            viewpoint_img_path = {
                "left": pano_obs_imgs_path[0],
                "slightly_left": pano_obs_imgs_path[1],
                "front": pano_obs_imgs_path[2],
                "slightly_right": pano_obs_imgs_path[3],
                "right": pano_obs_imgs_path[4],
            }

            action_name, reason, raw_response = query_qwen_action(
                llm=llm,
                instruction_text=instruction,
                viewpoint_img_path=viewpoint_img_path,
            )
            print(f"[Action {action_idx}] decision: {action_name}.")
            print(f"[Action {action_idx}] reason: {reason}")

            if action_name == "STOP":
                if len(data_dict["pred_traj"]) == 0:
                    data_dict["pred_traj"].append(list(curr_pose.position))
                break

            new_pose = getPoseAfterMakeActions(curr_pose, [DefaultAirsimActionCodes[action_name]])
            curr_pose = new_pose
            tool.setPoses([[curr_pose]])
            step_size += 1

            curr_pos = [curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val]
            curr_ori = list(airsim.to_eularian_angles(curr_pose.orientation))
            data_dict['pred_traj'].append(curr_pos)
            data_dict['pred_traj_explore'].append(curr_pos + curr_ori)

        stop_pos = np.array(list(curr_pose.position))
        target_pos = np.array(list(target_pose.position))
        ne = np.linalg.norm(np.array(target_pos) - np.array(stop_pos))

        if ne < 20:
            data_dict.update({"success": True})
            print(f"############## Episode {episode_id}: success, NE: {ne}. Step size: {step_size}")
        else:
            data_dict.update({"success": False})
            print(f"############## Episode {episode_id}: failed. NE: {ne}")

        if len(data_dict["pred_traj"]) == 0:
            data_dict["pred_traj"].append(list(curr_pose.position))

        nav_evaluator.update(data_dict)
        nav_evaluator.log_metrics()
        predict_routes.append(data_dict)

        if record:
            with open('output/output_data_mcts.json', 'w') as f:
                json.dump(predict_routes, f, indent=4)

    print("=" * 30 + " FINAL METRICS " + "=" * 30)
    nav_evaluator.log_metrics()
    print("=" * 75)

def replay_path(trajectory_files, img_type='rgb', save_failed_demo=False):
    tool = None
    active_scene_id = None

    with open(trajectory_files, 'r') as f:
        meta_data = json.load(f)

    for i, traj_info in enumerate(meta_data):
        episode_id = traj_info['episode_id']
        if not save_failed_demo and not traj_info['success']:
            continue
        traj_scene_id = traj_info.get('scene_id')
        if traj_scene_id is None:
            print(f"Missing scene_id for episode {episode_id}, skip replay")
            continue
        traj_scene_id = int(traj_scene_id)
        pred_traj = None

        if tool is None:
            tool = AirVLNSimulatorClientTool(machines_info=build_single_scene_machines_info(traj_scene_id))

        try:
            pred_traj = traj_info['pred_traj_explore']
        except Exception as e:
            print(e)
            continue

        if active_scene_id != traj_scene_id:
            print(f"Switch replay scene: {active_scene_id} -> {traj_scene_id} for episode {episode_id}")
            tool.machines_info = build_single_scene_machines_info(traj_scene_id)
            tool.run_call()
            active_scene_id = traj_scene_id

        if len(pred_traj) > 2000:
            continue

        save_dir_rgb = os.path.join(f"./output/video/{traj_scene_id}", episode_id, 'rgb')
        os.makedirs(save_dir_rgb, exist_ok=True)
        print(f"image saved in :{save_dir_rgb}")

        save_dir_dep = os.path.join(f"./output/video/{traj_scene_id}", episode_id, 'dep')
        os.makedirs(save_dir_dep, exist_ok=True)
        print(f"depth saved in :{save_dir_dep}")

        for j in tqdm(range(len(pred_traj))):
            pose = pred_traj[j]
            pos = pose[:3]
            p, r, y = pose[3:]
            ori = airsim.to_quaternion(p, r, y)

            curr_pose = convert_airsim_pose(pos+[ori.x_val, ori.y_val, ori.z_val, ori.w_val])
            tool.setPoses([[curr_pose]])

            try:
                pano_obs, pano_pose = get_front_observations(curr_pose, tool, scene_id=traj_scene_id)
                pano_obs_imgs = pano_obs[0][0]
                pano_obs_deps = pano_obs[0][1] * 300

                if img_type == 'rgb':
                    pano_obs_imgs_path = os.path.join(save_dir_rgb, f"rgb_obs_front_{j}.png")
                    cv2.imwrite(pano_obs_imgs_path, pano_obs_imgs)
                elif img_type == 'dep':
                    pano_obs_imgs_path = os.path.join(save_dir_dep, f"dep_obs_front_{j}.npy")
                    np.save(pano_obs_imgs_path, pano_obs_deps)
                elif img_type == 'all':
                    pano_obs_imgs_path = os.path.join(save_dir_rgb, f"rgb_obs_front_{j}.png")
                    cv2.imwrite(pano_obs_imgs_path, pano_obs_imgs)

                    pano_obs_imgs_path = os.path.join(save_dir_dep, f"dep_obs_front_{j}.npy")
                    np.save(pano_obs_imgs_path, pano_obs_deps)

            except Exception as e:
                print(f"{e}, skip {episode_id}")

def make_demo_video(data_root, episode_id):
    traj_data_path = os.path.join('output', 'output_data_mcts.json')

    tgt_traj = None
    with open(traj_data_path, 'r') as f:
        output_trajs = json.load(f)
    for out_traj in output_trajs:
        if out_traj['episode_id'] == episode_id:
            tgt_traj = out_traj
            break

    if tgt_traj is None:
        raise ValueError(f"Cannot find episode {episode_id} from {traj_data_path}")

    scene_id = int(tgt_traj['scene_id'])
    data_dir = f"{data_root}/{scene_id}/{episode_id}/rgb"
    save_dir = f"{data_root}/{scene_id}/{episode_id}"
    instruction = tgt_traj['instruction']
    img_files = os.listdir(data_dir)
    sorted_img_files = sorted(img_files, key=lambda name: int(name.split('_')[-1].split('.')[0]))

    frames = []
    for img_f in sorted_img_files:
        img = cv2.imread(os.path.join(data_dir, img_f))
        frame = append_text_to_image(img, instruction)
        frames.append(frame)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(
        os.path.join(save_dir, 'demo.avi'), fourcc, 10, (w, h))

    for frame in frames:
        out.write(frame)

    out.release()
    print("Video processing complete.")

def setup_auto_log(split, model_name):
    class _TeeStream:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()
            return len(data)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    os.makedirs("output/logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name)
    log_path = os.path.join(
        "output",
        "logs",
        f"mcts_{split}_{safe_model_name}_{timestamp}.log",
    )
    setup_auto_log.original_stdout = getattr(setup_auto_log, "original_stdout", sys.__stdout__)
    setup_auto_log.original_stderr = getattr(setup_auto_log, "original_stderr", sys.__stderr__)
    setup_auto_log.log_path = log_path
    setup_auto_log.log_file = open(log_path, "w", buffering=1)
    sys.stdout = _TeeStream(setup_auto_log.original_stdout, setup_auto_log.log_file)
    sys.stderr = _TeeStream(setup_auto_log.original_stderr, setup_auto_log.log_file)
    print(f"Auto log file: {log_path}")
    return log_path

if __name__ == '__main__':
    split = "val_seen"
    save_demo = True
    save_failed_demo = False
    data_dir = os.path.join("..", "DATA", "data", "aerialvln-slim")
    # model_name = "qwen3-max"
    model_name = "qwen3-vl-plus"
    # model_name = "qwen3.5-omni-plus"
    # model_name = "qwen3.5-plus"
    max_episodes = 0

    setup_auto_log(split, model_name)
    CityNavAgentMCTS(
        split,
        data_dir=data_dir,
        max_step_size=200,
        record=save_demo,
        model_name=model_name,
        max_episodes=max_episodes,
    )
    if save_demo:
        output_data_path = "./output/output_data_mcts.json"
        replay_path(output_data_path, img_type='rgb', save_failed_demo=save_failed_demo)

        with open(output_data_path, 'r') as f:
            output_trajs = json.load(f)

        for out_traj in output_trajs:
            if save_failed_demo or out_traj.get('success'):
                make_demo_video('./output/video', episode_id=out_traj['episode_id'])
