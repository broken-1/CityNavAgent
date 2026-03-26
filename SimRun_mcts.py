import json
import os
import re
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


ACTION_NAMES = list(DefaultAirsimActionCodes.keys())
STOP_ACTION = "STOP"
ARRIVAL_KEYWORDS = [
    "found",
    "reached",
    "arrived",
    "close",
    "visible",
    "next to",
    "at the",
]


def normalize_action_name(action_text):
    if action_text is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(action_text)).strip().upper()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    if normalized in DefaultAirsimActionCodes:
        return normalized
    return None


ACTION_PATTERNS = [
    (
        re.compile(
            r"\b" + re.escape(action_name).replace("_", r"[\s_\-]*") + r"\b",
            re.IGNORECASE,
        ),
        action_name,
    )
    for action_name in sorted(ACTION_NAMES, key=len, reverse=True)
]


def parse_action_response(raw_text):
    cleaned_text = raw_text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    action_name = None
    reason = ""
    try:
        payload = json.loads(cleaned_text)
        action_name = normalize_action_name(payload.get("action"))
        reason = str(payload.get("reason", "")).strip()
    except Exception:
        payload = None

    if not action_name:
        for pattern, candidate in ACTION_PATTERNS:
            if pattern.search(cleaned_text):
                action_name = candidate
                break

    if not action_name:
        action_name = STOP_ACTION
        reason = f"Failed to parse model response: {cleaned_text}"
    elif not reason:
        reason = cleaned_text

    return action_name, reason, cleaned_text


def should_advance_landmark(reason, current_landmark, landmark_step_count, max_landmark_steps=5):
    normalized_reason = unicodedata.normalize("NFKC", reason).lower()
    landmark_text = current_landmark.lower()
    if landmark_text in normalized_reason and any(keyword in normalized_reason for keyword in ARRIVAL_KEYWORDS):
        return True
    if landmark_step_count >= max_landmark_steps:
        return True
    return False


def build_qwen_action_prompt(
        instruction_text,
        landmarks,
        traversed_landmarks,
        next_landmark,
        current_pose,
        recent_actions,
):
    altitude = -float(current_pose.position.z_val)
    action_space = ", ".join(ACTION_NAMES)
    traversed_text = ", ".join(traversed_landmarks) if traversed_landmarks else "NONE"
    landmarks_text = ", ".join(landmarks)
    recent_action_text = ", ".join(recent_actions) if recent_actions else "NONE"

    prompt = f"""
You are an AI agent helping to control a drone to finish an aerial navigation task.

Navigation instruction:
{instruction_text}

All landmarks in order:
{landmarks_text}

Traversed landmarks:
{traversed_text}

Current next landmark:
{next_landmark}

You are given five images from different viewpoints: left, slightly left, front, slightly right, right.
Your current altitude is {altitude:.2f} meters.
Your recent actions are: {recent_action_text}

Focus only on reaching or getting closer to the current next landmark.
Do not execute the instruction as a full script in one step.
Choose exactly one next action from:
{action_space}

Return JSON only:
{{"action":"ACTION_NAME","reason":"short reason"}}
""".strip()
    return prompt


def query_qwen_action(
        llm,
        instruction_text,
        landmarks,
        next_landmark_idx,
        current_pose,
        viewpoint_img_path,
        action_history,
):
    traversed_landmarks = landmarks[:next_landmark_idx]
    next_landmark = landmarks[min(next_landmark_idx, len(landmarks) - 1)]
    prompt = build_qwen_action_prompt(
        instruction_text,
        landmarks,
        traversed_landmarks,
        next_landmark,
        current_pose,
        action_history[-5:],
    )
    raw_response = llm.query_viewpoint_api(prompt, viewpoint_img_path, show_response=False)
    action_name, reason, cleaned_response = parse_action_response(raw_response)
    return action_name, reason, cleaned_response


def CityNavAgentMCTS(scene_id, split, data_dir="./data", max_step_size=200, record=False):
    data_root = os.path.join(data_dir, f"gt_by_env/{scene_id}/{split}_landmk.json")

    os.makedirs("obs_imgs", exist_ok=True)

    predict_routes = []
    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']

    nav_evaluator = CityNavEvaluator()

    llm = OpenAI_LLM_v2(
        max_tokens=10000,
        model_name="qwen3-max-2026-01-23",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        client_type="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        cache_name="navigation_mcts",
        finish_reasons=["stop", "length"],
    )

    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]
        episode_id = navi_task['episode_id']
        print(f"================================ Start episode {episode_id} ==================================")

        landmarks = navi_task["instruction"]["landmarks"]
        if len(landmarks) == 0:
            continue

        next_landmark_idx = 0
        landmark_step_count = 0
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']

        step_size = 0
        action_history = []

        curr_pose = convert_airsim_pose(
            navi_task["start_position"] + navi_task["start_rotation"][1:] + [navi_task["start_rotation"][0]]
        )
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position'] + [0, 0, 0, 1])

        tool.setPoses([[curr_pose]])

        data_dict = {
            "episode_id": episode_id,
            "instruction": instruction,
            "gt_traj": [pose[:3] for pose in reference_path],
            "pred_traj": [],
            "pred_traj_explore": [list(curr_pose.position) + list(airsim.to_eularian_angles(curr_pose.orientation))]
        }

        while step_size < max_step_size:
            try:
                pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=scene_id)
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
                print(f"Task idx: {i}. Step size: {step_size}. Success: False. Failed to get images. Exception: {e}")
                break

            print("Qwen action loop, keep exploring ...")
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
                landmarks=landmarks,
                next_landmark_idx=next_landmark_idx,
                current_pose=curr_pose,
                viewpoint_img_path=viewpoint_img_path,
                action_history=action_history,
            )
            print(f"Qwen decision: {action_name}.")
            print(f"Qwen reason: {reason}")

            current_landmark = landmarks[min(next_landmark_idx, len(landmarks) - 1)]
            landmark_step_count += 1
            if should_advance_landmark(reason, current_landmark, landmark_step_count):
                if next_landmark_idx < len(landmarks) - 1:
                    next_landmark_idx += 1
                    landmark_step_count = 0

            if action_name == STOP_ACTION:
                if len(data_dict["pred_traj"]) == 0:
                    data_dict["pred_traj"].append(list(curr_pose.position))
                break

            new_pose = getPoseAfterMakeActions(curr_pose, [DefaultAirsimActionCodes[action_name]])
            curr_pose = new_pose
            tool.setPoses([[curr_pose]])
            step_size += 1
            action_history.append(action_name)

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
            for pr in predict_routes:
                pr.update({'final_pred_traj': pr['pred_traj_explore']})
            with open(f'output/output_data_mcts_{scene_id}.json', 'w') as f:
                json.dump(predict_routes, f, indent=4)

    nav_evaluator.log_metrics()


def replay_path(trajectory_files, scene_id, img_type='rgb'):
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    with open(trajectory_files, 'r') as f:
        meta_data = json.load(f)

    for i, traj_info in enumerate(meta_data):
        episode_id = traj_info['episode_id']
        try:
            pred_traj = traj_info['final_pred_traj']
        except Exception as e:
            print(e)
            continue
        if not traj_info['success']:
            continue
        if len(pred_traj) > 2000:
            continue

        save_dir_rgb = os.path.join(f"./output/video/{scene_id}", episode_id, 'rgb')
        os.makedirs(save_dir_rgb, exist_ok=True)
        print(f"image saved in :{save_dir_rgb}")

        save_dir_dep = os.path.join(f"./output/video/{scene_id}", episode_id, 'dep')
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
                pano_obs, pano_pose = get_front_observations(curr_pose, tool, scene_id=scene_id)
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


def make_demo_video(data_root, env_id, episode_id):
    data_dir = f"{data_root}/{env_id}/{episode_id}/rgb"
    save_dir = f"{data_root}/{env_id}/{episode_id}"
    traj_data_path = os.path.join('output', f'output_data_mcts_{env_id}.json')

    tgt_traj = None
    with open(traj_data_path, 'r') as f:
        output_trajs = json.load(f)
    for out_traj in output_trajs:
        if out_traj['episode_id'] == episode_id:
            tgt_traj = out_traj
            break

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


if __name__ == '__main__':
    env_id = 3
    split = "val_seen"
    save_demo = True

    CityNavAgentMCTS(env_id, split, max_step_size=60, record=save_demo)
    if save_demo:
        replay_path(f"./output/output_data_mcts_{env_id}.json", env_id, img_type='rgb')
        make_demo_video('./output/video', env_id=env_id, episode_id='3IRIK4HM3JIZ640FRHTYZU0EJ9Y6CH')
