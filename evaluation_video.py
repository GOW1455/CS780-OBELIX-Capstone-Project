import argparse
import csv
import importlib.util
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from obelix import OBELIX


ActionFn = Callable[[np.ndarray, np.random.Generator], str]


@dataclass
class EvalResult:
    agent_name: str
    mean_score: float
    std_score: float
    runs: int
    max_steps: int
    scaling_factor: int
    arena_size: int
    wall_obstacles: bool
    difficulty: int
    box_speed: int


def load_agent_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_agent_state(agent_module: ModuleType) -> None:
    if hasattr(agent_module, "reset") and callable(getattr(agent_module, "reset")):
        agent_module.reset()

    for name in ("step_counter", "recovery_steps"):
        if hasattr(agent_module, name):
            setattr(agent_module, name, 0)


def render_info_panel(
    *,
    panel_height: int,
    episode_idx: int,
    runs: int,
    difficulty: int,
    wall_obstacles: bool,
    step_idx: int,
    action: str,
    step_reward: float,
    total_reward: float,
    seed: int,
    status: str,
) -> np.ndarray:
    panel = np.zeros((panel_height, 980, 3), dtype=np.uint8)
    lines = [
        "OBELIX Episode Info",
        f"Status: {status}",
        f"Difficulty: {difficulty}",
        f"Walls: {'ON' if wall_obstacles else 'OFF'}",
        f"Episode: {episode_idx}/{runs}",
        f"Step: {step_idx}",
        f"Seed: {seed}",
        f"Action: {action}",
        f"Step Reward: {step_reward:.2f}",
        f"Total Reward: {total_reward:.2f}",
    ]

    y = 44
    for idx, line in enumerate(lines):
        color = (0, 255, 200) if idx == 0 else (255, 255, 255)
        scale = 0.7 if idx == 0 else 0.6
        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 34

    return panel


def wait_for_escape_start(
    window_name: str,
    frame: np.ndarray,
    info_window_name: str,
    info_panel_builder: Callable[[int, bool], np.ndarray],
    initial_difficulty: int,
    initial_wall_obstacles: bool,
    allow_setting_keys: bool,
) -> Tuple[bool, int, bool]:
    selected_difficulty = initial_difficulty
    selected_wall_obstacles = initial_wall_obstacles
    while True:
        cv2.imshow(window_name, frame)
        cv2.imshow(
            info_window_name,
            info_panel_builder(selected_difficulty, selected_wall_obstacles),
        )
        key = cv2.waitKeyEx(20)
        key_low = key & 0xFF
        if allow_setting_keys and key_low in (ord("0"), ord("2"), ord("3")):
            selected_difficulty = int(chr(key_low))
            continue
        if allow_setting_keys and key_low in (ord("w"), ord("W")):
            selected_wall_obstacles = not selected_wall_obstacles
            continue
        if key == 2555904:  # Right Arrow
            return True, selected_difficulty, selected_wall_obstacles
        if key_low in (ord("q"), ord("Q")):
            return False, selected_difficulty, selected_wall_obstacles


def evaluate_agent(
    agent_policy: ActionFn,
    agent_module: ModuleType,
    *,
    agent_name: str,
    runs: int,
    base_seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    wall_obstacles: bool,
    difficulty: int,
    box_speed: int,
    episode_plan: Optional[List[Tuple[int, bool]]] = None,
) -> EvalResult:
    scores: List[float] = []

    window_name = "OBELIX Evaluation Video"
    info_window_name = "Episode Info"

    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=base_seed,
    )
    panel_height = int(env.frame_size[0])

    try:
        user_quit = False
        idle_frame = env.frame.copy()
        last_completed_episode = 0
        last_step_idx = 0
        last_action = "NONE"
        last_step_reward = 0.0
        last_total_reward = 0.0
        last_seed = base_seed
        last_frame = idle_frame.copy()
        selected_difficulty = difficulty
        selected_wall_obstacles = wall_obstacles
        used_difficulties: List[int] = []
        used_walls: List[bool] = []

        total_runs = len(episode_plan) if episode_plan is not None else runs

        for i in range(runs):
            episode_idx = i + 1
            seed = base_seed + i*42

            if episode_plan is not None:
                selected_difficulty, selected_wall_obstacles = episode_plan[i]

            if last_completed_episode == 0:
                preview_base = idle_frame
                preview_action = "WAIT"
                preview_step = 0
                preview_step_reward = 0.0
                preview_total_reward = 0.0
                preview_seed = seed
                preview_status = "Waiting for Right Arrow (episode not started yet)"
            else:
                preview_base = last_frame
                preview_action = last_action
                preview_step = last_step_idx
                preview_step_reward = last_step_reward
                preview_total_reward = last_total_reward
                preview_seed = last_seed
                preview_status = (
                    f"Episode {last_completed_episode} finished. Press Right Arrow for next"
                )

            preview = preview_base.copy()
            should_continue, selected_difficulty, selected_wall_obstacles = wait_for_escape_start(
                window_name,
                preview,
                info_window_name,
                lambda d, w, episode_idx=episode_idx, preview_step=preview_step,
                preview_action=preview_action, preview_step_reward=preview_step_reward,
                preview_total_reward=preview_total_reward, preview_seed=preview_seed,
                preview_status=preview_status, runs=total_runs: render_info_panel(
                    panel_height=panel_height,
                    episode_idx=last_completed_episode
                    if last_completed_episode > 0
                    else episode_idx,
                    runs=runs,
                    difficulty=d,
                    wall_obstacles=w,
                    step_idx=preview_step,
                    action=preview_action,
                    step_reward=preview_step_reward,
                    total_reward=preview_total_reward,
                    seed=preview_seed,
                    status=(
                        f"{preview_status} | scheduled settings"
                        if episode_plan is not None
                        else f"{preview_status} | choose difficulty: 0/2/3, toggle walls: W"
                    ),
                ),
                selected_difficulty,
                selected_wall_obstacles,
                episode_plan is None,
            )
            if not should_continue:
                print("Stopped by user before starting the next episode.")
                break

            reset_agent_state(agent_module)
            env.difficulty = selected_difficulty
            env.wall_obstacles = selected_wall_obstacles
            env._build_obstacles()
            env.box_blink_enabled = selected_difficulty >= 2
            env.box_move_enabled = selected_difficulty >= 3
            obs = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            used_difficulties.append(selected_difficulty)
            used_walls.append(selected_wall_obstacles)

            total = 0.0
            step_idx = 0
            done = False
            step_reward = 0.0
            action = "NONE"
            frame = env.frame.copy()

            while not done:
                action = agent_policy(obs, rng)
                obs, reward, done = env.step(action, render=False)
                step_idx += 1
                step_reward = float(reward)
                total += step_reward

                frame = env.frame.copy()
                cv2.imshow(window_name, frame)
                info_panel = render_info_panel(
                    panel_height=panel_height,
                    episode_idx=episode_idx,
                    runs=total_runs,
                    difficulty=selected_difficulty,
                    wall_obstacles=selected_wall_obstacles,
                    step_idx=step_idx,
                    action=action,
                    step_reward=step_reward,
                    total_reward=total,
                    seed=seed,
                    status="Running",
                )
                cv2.imshow(info_window_name, info_panel)

                key = cv2.waitKeyEx(1)
                key_low = key & 0xFF
                if key == 2555904:  # Right Arrow: skip to next episode
                    print(
                        f"Episode {episode_idx} skipped by user at step {step_idx}; total_reward={total:.3f}"
                    )
                    break
                if key_low in (ord("q"), ord("Q")):
                    print("Stopped by user during episode playback.")
                    user_quit = True
                    done = True
                    break

            scores.append(total)
            last_completed_episode = episode_idx
            last_step_idx = step_idx
            last_action = action
            last_step_reward = step_reward
            last_total_reward = total
            last_seed = seed
            last_frame = frame.copy()
            print(
                f"episode={episode_idx}/{total_runs} seed={seed} total_reward={total:.3f} steps={step_idx}"
            )
            if user_quit:
                break
    finally:
        cv2.destroyAllWindows()

    if not scores:
        scores = [0.0]

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if used_difficulties:
        summary_difficulty = (
            used_difficulties[0]
            if all(d == used_difficulties[0] for d in used_difficulties)
            else -1
        )
    else:
        summary_difficulty = difficulty

    if used_walls:
        summary_walls = (
            used_walls[0]
            if all(w == used_walls[0] for w in used_walls)
            else wall_obstacles
        )
        if not all(w == used_walls[0] for w in used_walls):
            print(
                "Mixed wall settings used across episodes; leaderboard stores the initial --wall_obstacles value."
            )
    else:
        summary_walls = wall_obstacles

    return EvalResult(
        agent_name=agent_name,
        mean_score=mean,
        std_score=std,
        runs=runs,
        max_steps=max_steps,
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        wall_obstacles=summary_walls,
        difficulty=summary_difficulty,
        box_speed=box_speed,
    )


def append_leaderboard(path: str, result: EvalResult) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "agent_name",
                "mean_score",
                "std_score",
                "runs",
                "max_steps",
                "scaling_factor",
                "arena_size",
                "wall_obstacles",
                "difficulty",
                "box_speed",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "agent_name": result.agent_name,
                "mean_score": f"{result.mean_score:.6f}",
                "std_score": f"{result.std_score:.6f}",
                "runs": result.runs,
                "max_steps": result.max_steps,
                "scaling_factor": result.scaling_factor,
                "arena_size": result.arena_size,
                "wall_obstacles": int(result.wall_obstacles),
                "difficulty": result.difficulty,
                "box_speed": result.box_speed,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", type=str, required=True)
    parser.add_argument("--agent_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--episodes_per_setting",
        type=int,
        default=2,
        help="episodes for each (difficulty, wall) setting",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="difficulty level: 0=static, 2=blinking box, 3=moving+blinking",
    )
    parser.add_argument(
        "--box_speed",
        type=int,
        default=2,
        help="speed of moving box (pixels/step) for difficulty>=3",
    )
    parser.add_argument(
        "--leaderboard_csv", type=str, default="leaderboard.csv"
    )

    args = parser.parse_args()

    agent_mod = load_agent_module(args.agent_file)
    if not hasattr(agent_mod, "policy"):
        raise AttributeError("Submission must define: policy(obs, rng) -> action_str")

    policy_fn = getattr(agent_mod, "policy")
    agent_name = (
        args.agent_name or os.path.splitext(os.path.basename(args.agent_file))[0]
    )

    difficulty_levels = [0, 2, 3]
    wall_options = [False, True]
    episode_plan: List[Tuple[int, bool]] = []
    for difficulty_level in difficulty_levels:
        for walls in wall_options:
            for _ in range(args.episodes_per_setting):
                episode_plan.append((difficulty_level, walls))

    print(
        "Running fixed schedule: "
        f"{args.episodes_per_setting} episodes x {len(difficulty_levels)} difficulties x {len(wall_options)} wall settings "
        f"= {len(episode_plan)} total episodes"
    )

    result = evaluate_agent(
        policy_fn,
        agent_mod,
        agent_name=agent_name,
        runs=len(episode_plan),
        base_seed=args.seed,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        episode_plan=episode_plan,
    )

    print(
        f"agent={result.agent_name} mean={result.mean_score:.3f} std={result.std_score:.3f} "
        f"runs={result.runs} steps={result.max_steps} arena={result.arena_size} "
        f"wall_obstacles={result.wall_obstacles} difficulty={result.difficulty} box_speed={result.box_speed}"
    )

    append_leaderboard(args.leaderboard_csv, result)
    print(f"Appended to {args.leaderboard_csv}")


if __name__ == "__main__":
    main()
