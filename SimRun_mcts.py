import json
import os
import re
import sys
import unicodedata
import time
from dataclasses import dataclass, field

import airsim
import cv2
import numpy as np

from tqdm import tqdm

from src.llm.query_llm import OpenAI_LLM_v2
from src.common.param import args
from airsim_plugin.airsim_settings import (
    DefaultAirsimActionCodes,
    ObservationDirections,
)
from utils.env_utils import getPoseAfterMakeActions, get_pano_observations, get_front_observations
from utils.utils import append_text_to_image
from evaluator.nav_evaluator import CityNavEvaluator
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool

TOP_K_CANDIDATES = 3
SEARCH_DEPTH = 3
NUM_SIMULATIONS = 12
PUCT_C = 1.2
DEFAULT_RGB_IMAGE_SIZE = 512
DEFAULT_DEPTH_IMAGE_SIZE = 512


def configure_runtime_image_sizes(
    rgb_width=DEFAULT_RGB_IMAGE_SIZE,
    rgb_height=DEFAULT_RGB_IMAGE_SIZE,
    depth_width=DEFAULT_DEPTH_IMAGE_SIZE,
    depth_height=DEFAULT_DEPTH_IMAGE_SIZE,
):
    args.Image_Width_RGB = rgb_width
    args.Image_Height_RGB = rgb_height
    args.Image_Width_DEPTH = depth_width
    args.Image_Height_DEPTH = depth_height


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
            raise KeyError(
                "Episode is missing scene_id while running in multi-scene mode"
            )

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
    formatted_airsim_pose = airsim.Pose(position_val=airsim.Vector3r(
        pose[0], pose[1], pose[2]),
                                        orientation_val=airsim.Quaternionr(
                                            x_val=pose[3],
                                            y_val=pose[4],
                                            z_val=pose[5],
                                            w_val=pose[6],
                                        ))
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
            normalized = unicodedata.normalize(
                "NFKC", str(action_text)).strip().upper()
            normalized = re.sub(r"[\s\-]+", "_", normalized)
            if normalized in DefaultAirsimActionCodes:
                action_name = normalized
        reason = str(payload.get("reason", "")).strip()
    except Exception:
        pass

    if not action_name:
        action_names = sorted(DefaultAirsimActionCodes.keys(),
                              key=len,
                              reverse=True)
        for candidate in action_names:
            pattern = r"\b" + re.escape(candidate).replace("_",
                                                           r"[\s_\-]*") + r"\b"
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                action_name = candidate
                break

    if not action_name:
        action_name = "STOP"
        reason = f"[PARSE_FAILED] fallback to STOP. cleaned_response={cleaned_text}"
    elif not reason:
        reason = cleaned_text

    return action_name, reason, cleaned_text


def parse_candidate_response(raw_text):
    cleaned_text = raw_text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    candidates = []
    try:
        payload = json.loads(cleaned_text)
        raw_candidates = payload.get("candidates", [])
        for item in raw_candidates:
            action_text = item.get("action")
            if action_text is None:
                continue
            normalized = unicodedata.normalize(
                "NFKC", str(action_text)).strip().upper()
            normalized = re.sub(r"[\s\-]+", "_", normalized)
            if normalized not in DefaultAirsimActionCodes:
                continue
            score = item.get("score", 50)
            try:
                score = float(score)
            except Exception:
                score = 50.0
            score = max(0.0, min(100.0, score))
            reason = str(item.get("reason", "")).strip()
            candidates.append({
                "action": normalized,
                "score": score,
                "reason": reason
            })
    except Exception:
        pass

    unique = []
    seen = set()
    for item in candidates:
        if item["action"] in seen:
            continue
        seen.add(item["action"])
        unique.append(item)

    return unique, cleaned_text


def parse_prm_response(raw_text):
    cleaned_text = raw_text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    default_eval = {
        "score": 50.0,
        "progress_score": 50.0,
        "alignment_score": 50.0,
        "risk_score": 50.0,
        "reason": f"[PRM_PARSE_FAILED] {cleaned_text}",
    }

    try:
        payload = json.loads(cleaned_text)
    except Exception:
        return default_eval, cleaned_text

    result = {}
    for key in ["score", "progress_score", "alignment_score", "risk_score"]:
        value = payload.get(key, default_eval[key])
        try:
            value = float(value)
        except Exception:
            value = default_eval[key]
        result[key] = max(0.0, min(100.0, value))
    result["reason"] = str(payload.get("reason",
                                       "")).strip() or default_eval["reason"]
    return result, cleaned_text


def build_qwen_candidate_prompt(instruction_text):
    action_space = ", ".join(DefaultAirsimActionCodes.keys())
    prompt = f"""
You are a drone that is currently executing an aerial navigation task.
You will make one navigation decision at a time and continue iterating until the task is finished.

Navigation instruction:
{instruction_text}

Your visual observations from different viewpoints are provided above: left, slightly left, front, slightly right, right.
Based on the navigation instruction and the current multi-view observations, propose the top {TOP_K_CANDIDATES} next actions.
Rank them from most promising to least promising.
Do not treat the instruction as a script to complete in one step.
Only include actions from this action space:
{action_space}

Return JSON only:
{{
  "candidates": [
    {{"action":"ACTION_NAME","score":0-100,"reason":"short reason"}},
    {{"action":"ACTION_NAME","score":0-100,"reason":"short reason"}},
    {{"action":"ACTION_NAME","score":0-100,"reason":"short reason"}}
  ]
}}
""".strip()
    return prompt


def build_qwen_prm_prompt(instruction_text, action_prefix):
    action_sequence_text = " -> ".join(
        action_prefix) if action_prefix else "NONE"
    prompt = f"""
You are evaluating a candidate short-horizon navigation rollout for a drone.

Navigation instruction:
{instruction_text}

Candidate action sequence:
{action_sequence_text}

The current 5-view observations at the leaf state are provided above: left, slightly left, front, slightly right, right.
Judge how promising this leaf state is for eventually reaching the target location.
Higher score means more promising.
High risk means the rollout looks unstable, oscillatory, or likely to move away from the target.

Return JSON only:
{{
  "score": 0-100,
  "progress_score": 0-100,
  "alignment_score": 0-100,
  "risk_score": 0-100,
  "reason": "short reason"
}}
""".strip()
    return prompt


def query_qwen_action_candidates(llm, instruction_text, viewpoint_img_path):
    prompt = build_qwen_candidate_prompt(instruction_text)
    raw_response = llm.query_viewpoint_api(prompt,
                                           viewpoint_img_path,
                                           show_response=False)
    candidates, cleaned_response = parse_candidate_response(raw_response)

    if not candidates:
        retry_raw_response = llm.query_viewpoint_api(prompt,
                                                     viewpoint_img_path,
                                                     show_response=False)
        candidates, cleaned_response = parse_candidate_response(
            retry_raw_response)
        if candidates:
            print("[Candidate Retry] Successfully parsed candidates on retry.")
        else:
            print(
                "[Candidate Retry] Failed twice. Falling back to single-action baseline."
            )
            action_name, reason, fallback_raw = query_qwen_action(
                llm, instruction_text, viewpoint_img_path)
            candidates = [{
                "action": action_name,
                "score": 50.0,
                "reason": reason
            }]
            cleaned_response = (
                f"[CANDIDATE_FALLBACK]\n"
                f"first_try={cleaned_response}\n"
                f"second_try={retry_raw_response}\n"
                f"fallback_raw={fallback_raw}"
            )

    return candidates[:TOP_K_CANDIDATES], cleaned_response


def query_qwen_prm_score(llm, instruction_text, viewpoint_img_path,
                         action_prefix):
    prompt = build_qwen_prm_prompt(instruction_text, action_prefix)
    raw_response = llm.query_viewpoint_api(prompt,
                                           viewpoint_img_path,
                                           show_response=False)
    prm_eval, cleaned_response = parse_prm_response(raw_response)
    return prm_eval, cleaned_response


def capture_five_view_images(curr_pose, tool, scene_id):
    pano_obs, pano_pose = get_pano_observations(curr_pose,
                                                tool,
                                                scene_id=scene_id)
    pano_obs_imgs = [
        pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0],
        pano_obs[2][0]
    ]
    pano_obs_imgs_path = [
        "obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_"))
        for view_drc in ObservationDirections
    ]
    for j in range(len(pano_obs_imgs_path)):
        cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])

    return {
        "left": pano_obs_imgs_path[0],
        "slightly_left": pano_obs_imgs_path[1],
        "front": pano_obs_imgs_path[2],
        "slightly_right": pano_obs_imgs_path[3],
        "right": pano_obs_imgs_path[4],
    }


@dataclass
class SearchNode:
    pose: airsim.Pose
    step_idx: int
    action_prefix: list
    prior_score: float
    parent: "SearchNode" = None
    children: dict = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    leaf_eval: dict = None
    candidate_actions: list = field(default_factory=list)
    candidate_priors: dict = field(default_factory=dict)
    candidate_raw: str = ""

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child_by_puct(node):
    best_action = None
    best_child = None
    best_score = None
    parent_visits = max(1, node.visit_count)

    for action, child in node.children.items():
        q_score = child.mean_value
        u_score = PUCT_C * child.prior_score * np.sqrt(parent_visits) / (
            1 + child.visit_count)
        total_score = q_score + u_score
        if best_score is None or total_score > best_score:
            best_score = total_score
            best_action = action
            best_child = child

    return best_action, best_child


def expand_child(node, action, prior_score):
    next_pose = getPoseAfterMakeActions(node.pose,
                                        [DefaultAirsimActionCodes[action]])
    child = SearchNode(
        pose=next_pose,
        step_idx=node.step_idx + 1,
        action_prefix=node.action_prefix + [action],
        prior_score=prior_score,
        parent=node,
    )
    node.children[action] = child
    return child


def evaluate_leaf_node(llm, tool, scene_id, root_pose, instruction_text, node):
    tool.setPoses([[node.pose]])
    viewpoint_img_path = capture_five_view_images(node.pose, tool, scene_id)
    prm_eval, prm_raw = query_qwen_prm_score(
        llm=llm,
        instruction_text=instruction_text,
        viewpoint_img_path=viewpoint_img_path,
        action_prefix=node.action_prefix,
    )
    node.leaf_eval = {"prm": prm_eval, "raw_response": prm_raw}
    tool.setPoses([[root_pose]])
    return prm_eval["score"] / 100.0


def ensure_node_candidates(llm, tool, scene_id, root_pose, instruction_text,
                           node):
    if node.candidate_actions:
        return

    if node.parent is None:
        raise ValueError("Root node candidates should be initialized upfront.")

    tool.setPoses([[node.pose]])
    viewpoint_img_path = capture_five_view_images(node.pose, tool, scene_id)
    node_candidates, candidate_raw = query_qwen_action_candidates(
        llm=llm,
        instruction_text=instruction_text,
        viewpoint_img_path=viewpoint_img_path,
    )
    tool.setPoses([[root_pose]])

    node.candidate_actions = [item["action"] for item in node_candidates]
    node.candidate_priors = {
        item["action"]: max(0.01, item["score"] / 100.0)
        for item in node_candidates
    }
    node.candidate_raw = candidate_raw


def run_mcts_search(llm, tool, scene_id, instruction_text, root_pose,
                    root_candidates):
    root = SearchNode(
        pose=root_pose,
        step_idx=0,
        action_prefix=[],
        prior_score=1.0,
        parent=None,
        candidate_actions=[item["action"] for item in root_candidates],
        candidate_priors={
            item["action"]: max(0.01, item["score"] / 100.0)
            for item in root_candidates
        },
        candidate_raw="",
    )

    simulation_logs = []
    for sim_idx in range(NUM_SIMULATIONS):
        node = root
        path = [root]

        while len(node.action_prefix) < SEARCH_DEPTH:
            ensure_node_candidates(
                llm=llm,
                tool=tool,
                scene_id=scene_id,
                root_pose=root_pose,
                instruction_text=instruction_text,
                node=node,
            )
            unexpanded = [
                action for action in node.candidate_actions
                if action not in node.children
            ]
            if unexpanded:
                action = unexpanded[0]
                node = expand_child(node, action, node.candidate_priors[action])
                path.append(node)
                break

            _, node = select_child_by_puct(node)
            if node is None:
                break
            path.append(node)

        leaf_value = evaluate_leaf_node(
            llm=llm,
            tool=tool,
            scene_id=scene_id,
            root_pose=root_pose,
            instruction_text=instruction_text,
            node=node,
        )

        for path_node in path:
            path_node.visit_count += 1
            path_node.value_sum += leaf_value

        prm = node.leaf_eval["prm"]
        simulation_logs.append({
            "simulation": sim_idx + 1,
            "action_prefix": node.action_prefix,
            "score": prm["score"],
            "progress_score": prm["progress_score"],
            "alignment_score": prm["alignment_score"],
            "risk_score": prm["risk_score"],
            "reason": prm["reason"],
        })

    if not root.children:
        fallback_action = root_candidates[0]["action"]
        return fallback_action, simulation_logs, root

    best_action = max(
        root.children.items(),
        key=lambda item: (item[1].visit_count, item[1].mean_value),
    )[0]
    return best_action, simulation_logs, root


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


def CityNavAgentMCTS(split,
                     data_dir,
                     max_step_size,
                     record,
                     model_name,
                     max_episodes=0,
                     planner_mode="mcts"):
    os.makedirs("obs_imgs", exist_ok=True)
    configure_runtime_image_sizes()
    planner_mode = planner_mode.lower().strip()
    if planner_mode not in {"baseline", "mcts"}:
        raise ValueError(
            f"Unsupported planner_mode={planner_mode}. Expected 'baseline' or 'mcts'."
        )
    output_data_path = os.path.join("output",
                                    f"output_data_{planner_mode}.json")
    print(f"Planner mode: {planner_mode}")

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
        curr_pose = convert_airsim_pose(navi_task["start_position"] +
                                        navi_task["start_rotation"][1:] +
                                        [navi_task["start_rotation"][0]])

        if tool is None:
            tool = AirVLNSimulatorClientTool(
                machines_info=build_single_scene_machines_info(
                    episode_scene_id))

        if active_scene_id != episode_scene_id:
            print(
                f"Switch simulator scene: {active_scene_id} -> {episode_scene_id} for episode {episode_id}"
            )
            tool.machines_info = build_single_scene_machines_info(
                episode_scene_id)
            tool.run_call()
            active_scene_id = episode_scene_id

        print(f"================================ Start episode {episode_id} "
              f"(scene {episode_scene_id}) ==================================")
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']

        step_size = 0
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position'] +
                                          [0, 0, 0, 1])

        tool.setPoses([[curr_pose]])

        data_dict = {
            "episode_id":
            episode_id,
            "scene_id":
            episode_scene_id,
            "instruction":
            instruction,
            "gt_traj": [pose[:3] for pose in reference_path],
            "pred_traj": [],
            "pred_traj_explore": [
                list(curr_pose.position) +
                list(airsim.to_eularian_angles(curr_pose.orientation))
            ]
        }

        while step_size < max_step_size:
            action_idx = step_size + 1
            try:
                viewpoint_img_path = capture_five_view_images(
                    curr_pose, tool, episode_scene_id)
            except Exception as e:
                data_dict['pred_traj'].append(list(curr_pose.position))
                print(
                    f"Task idx: {i}. Action idx: {action_idx}. Step size: {step_size}. "
                    f"Success: False. Failed to get images. Exception: {e}")
                break

            root_pose = curr_pose
            if planner_mode == "baseline":
                action_name, selected_reason, raw_response = query_qwen_action(
                    llm=llm,
                    instruction_text=instruction,
                    viewpoint_img_path=viewpoint_img_path,
                )
                print(
                    f"[Action {action_idx}] baseline decision: {action_name}.")
                print(
                    f"[Action {action_idx}] baseline reason: {selected_reason}"
                )
            else:
                root_candidates, candidate_raw = query_qwen_action_candidates(
                    llm=llm,
                    instruction_text=instruction,
                    viewpoint_img_path=viewpoint_img_path,
                )
                candidate_desc = ", ".join([
                    f"{item['action']}({item['score']:.1f})"
                    for item in root_candidates
                ])
                print(
                    f"[Action {action_idx}] baseline candidates: {candidate_desc}"
                )
                # print(
                #     f"[Action {action_idx}] baseline raw_response: {candidate_raw}"
                # )

                action_name, simulation_logs, root_node = run_mcts_search(
                    llm=llm,
                    tool=tool,
                    scene_id=episode_scene_id,
                    instruction_text=instruction,
                    root_pose=root_pose,
                    root_candidates=root_candidates,
                )
                for sim_log in simulation_logs:
                    path_text = " -> ".join(
                        sim_log["action_prefix"]
                    ) if sim_log["action_prefix"] else "NONE"
                    print(
                        f"[Action {action_idx}] sim {sim_log['simulation']}: "
                        f"path={path_text} "
                        f"score={sim_log['score']:.1f} "
                        f"progress={sim_log['progress_score']:.1f} "
                        f"alignment={sim_log['alignment_score']:.1f} "
                        f"risk={sim_log['risk_score']:.1f}")
                    print(
                        f"[Action {action_idx}] sim {sim_log['simulation']} reason: {sim_log['reason']}"
                    )

                selected_child = root_node.children.get(action_name)
                selected_reason = ""
                if selected_child is not None and selected_child.leaf_eval is not None:
                    selected_reason = selected_child.leaf_eval["prm"]["reason"]
                else:
                    selected_reason = root_candidates[0]["reason"]

                print(
                    f"[Action {action_idx}] selected root action: {action_name}."
                )
                print(
                    f"[Action {action_idx}] selected reason: {selected_reason}"
                )

            if action_name == "STOP":
                if len(data_dict["pred_traj"]) == 0:
                    data_dict["pred_traj"].append(list(curr_pose.position))
                break

            tool.setPoses([[root_pose]])
            new_pose = getPoseAfterMakeActions(
                curr_pose, [DefaultAirsimActionCodes[action_name]])
            curr_pose = new_pose
            tool.setPoses([[curr_pose]])
            step_size += 1

            curr_pos = [
                curr_pose.position.x_val, curr_pose.position.y_val,
                curr_pose.position.z_val
            ]
            curr_ori = list(airsim.to_eularian_angles(curr_pose.orientation))
            data_dict['pred_traj'].append(curr_pos)
            data_dict['pred_traj_explore'].append(curr_pos + curr_ori)

        stop_pos = np.array(list(curr_pose.position))
        target_pos = np.array(list(target_pose.position))
        ne = np.linalg.norm(np.array(target_pos) - np.array(stop_pos))

        if ne < 20:
            data_dict.update({"success": True})
            print(
                f"############## Episode {episode_id}: success, NE: {ne}. Step size: {step_size}"
            )
        else:
            data_dict.update({"success": False})
            print(f"############## Episode {episode_id}: failed. NE: {ne}")

        if len(data_dict["pred_traj"]) == 0:
            data_dict["pred_traj"].append(list(curr_pose.position))

        nav_evaluator.update(data_dict)
        nav_evaluator.log_metrics()
        predict_routes.append(data_dict)

        if record:
            with open(output_data_path, 'w') as f:
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
            tool = AirVLNSimulatorClientTool(
                machines_info=build_single_scene_machines_info(traj_scene_id))

        try:
            pred_traj = traj_info['pred_traj_explore']
        except Exception as e:
            print(e)
            continue

        if active_scene_id != traj_scene_id:
            print(
                f"Switch replay scene: {active_scene_id} -> {traj_scene_id} for episode {episode_id}"
            )
            tool.machines_info = build_single_scene_machines_info(
                traj_scene_id)
            tool.run_call()
            active_scene_id = traj_scene_id

        if len(pred_traj) > 2000:
            continue

        save_dir_rgb = os.path.join(f"./output/video/{traj_scene_id}",
                                    episode_id, 'rgb')
        os.makedirs(save_dir_rgb, exist_ok=True)
        print(f"image saved in :{save_dir_rgb}")

        save_dir_dep = os.path.join(f"./output/video/{traj_scene_id}",
                                    episode_id, 'dep')
        os.makedirs(save_dir_dep, exist_ok=True)
        print(f"depth saved in :{save_dir_dep}")

        for j in tqdm(range(len(pred_traj))):
            pose = pred_traj[j]
            pos = pose[:3]
            p, r, y = pose[3:]
            ori = airsim.to_quaternion(p, r, y)

            curr_pose = convert_airsim_pose(
                pos + [ori.x_val, ori.y_val, ori.z_val, ori.w_val])
            tool.setPoses([[curr_pose]])

            try:
                pano_obs, pano_pose = get_front_observations(
                    curr_pose, tool, scene_id=traj_scene_id)
                pano_obs_imgs = pano_obs[0][0]
                pano_obs_deps = pano_obs[0][1] * 300

                if img_type == 'rgb':
                    pano_obs_imgs_path = os.path.join(
                        save_dir_rgb, f"rgb_obs_front_{j}.png")
                    cv2.imwrite(pano_obs_imgs_path, pano_obs_imgs)
                elif img_type == 'dep':
                    pano_obs_imgs_path = os.path.join(
                        save_dir_dep, f"dep_obs_front_{j}.npy")
                    np.save(pano_obs_imgs_path, pano_obs_deps)
                elif img_type == 'all':
                    pano_obs_imgs_path = os.path.join(
                        save_dir_rgb, f"rgb_obs_front_{j}.png")
                    cv2.imwrite(pano_obs_imgs_path, pano_obs_imgs)

                    pano_obs_imgs_path = os.path.join(
                        save_dir_dep, f"dep_obs_front_{j}.npy")
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
        raise ValueError(
            f"Cannot find episode {episode_id} from {traj_data_path}")

    scene_id = int(tgt_traj['scene_id'])
    data_dir = f"{data_root}/{scene_id}/{episode_id}/rgb"
    save_dir = f"{data_root}/{scene_id}/{episode_id}"
    instruction = tgt_traj['instruction']
    img_files = os.listdir(data_dir)
    sorted_img_files = sorted(
        img_files, key=lambda name: int(name.split('_')[-1].split('.')[0]))

    frames = []
    for img_f in sorted_img_files:
        img = cv2.imread(os.path.join(data_dir, img_f))
        frame = append_text_to_image(img, instruction)
        frames.append(frame)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(os.path.join(save_dir, 'demo.avi'), fourcc, 10,
                          (w, h))

    for frame in frames:
        out.write(frame)

    out.release()
    print("Video processing complete.")


def setup_auto_log(split, model_name, planner_mode="mcts"):

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
        f"{planner_mode}_{split}_{safe_model_name}_{timestamp}.log",
    )
    setup_auto_log.original_stdout = getattr(setup_auto_log, "original_stdout",
                                             sys.__stdout__)
    setup_auto_log.original_stderr = getattr(setup_auto_log, "original_stderr",
                                             sys.__stderr__)
    setup_auto_log.log_path = log_path
    setup_auto_log.log_file = open(log_path, "w", buffering=1)
    sys.stdout = _TeeStream(setup_auto_log.original_stdout,
                            setup_auto_log.log_file)
    sys.stderr = _TeeStream(setup_auto_log.original_stderr,
                            setup_auto_log.log_file)
    print(f"Auto log file: {log_path}")
    return log_path


if __name__ == '__main__':
    split = "val_seen"
    save_demo = True
    save_failed_demo = False
    data_dir = os.path.join("..", "DATA", "data", "aerialvln-slim")
    planner_mode = "mcts"
    # model_name = "qwen3-max"
    # model_name = "qwen3-vl-plus"
    # model_name = "qwen3.5-omni-plus"
    model_name = "qwen3.6-plus"
    max_episodes = 10

    setup_auto_log(split, model_name, planner_mode=planner_mode)
    CityNavAgentMCTS(
        split,
        data_dir=data_dir,
        max_step_size=200,
        record=save_demo,
        model_name=model_name,
        max_episodes=max_episodes,
        planner_mode=planner_mode,
    )
    if save_demo:
        output_data_path = f"./output/output_data_{planner_mode}.json"
   
   
        replay_path(output_data_path,
                    img_type='rgb',
                    save_failed_demo=save_failed_demo)

        with open(output_data_path, 'r') as f:
            output_trajs = json.load(f)

        for out_traj in output_trajs:
            if save_failed_demo or out_traj.get('success'):
                make_demo_video('./output/video',
                                episode_id=out_traj['episode_id'])
