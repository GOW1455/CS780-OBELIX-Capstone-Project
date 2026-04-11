import argparse
import csv
import importlib.util
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType
from typing import Callable, List

import numpy as np

from obelix import OBELIX


ActionFn = Callable[[np.ndarray, np.random.Generator], str]


@dataclass
class LevelMetrics:
    difficulty: int
    seeds: int
    success_rate: float
    average_reward: float
    average_steps_without_wall: float


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


def run_level_evaluation(
    policy_fn: ActionFn,
    agent_module: ModuleType,
    *,
    difficulty: int,
    seed_start: int,
    num_seeds: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    box_speed: int,
    wall_obstacles: bool,
    render: bool,
) -> LevelMetrics:
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    successes = 0

    for offset in range(num_seeds):
        seed = seed_start + offset
        reset_agent_state(agent_module)
        env = OBELIX(
            scaling_factor=scaling_factor,
            arena_size=arena_size,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=box_speed,
            seed=seed,
        )

        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = policy_fn(obs, rng)
            obs, reward, done = env.step(action, render=render)
            total_reward += float(reward)
            steps += 1

        # Success definition in OBELIX: attached box reaches boundary.
        succeeded = bool(
            env.enable_push
            and env._box_touches_boundary(env.box_center_x, env.box_center_y)
        )
        if succeeded:
            successes += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    return LevelMetrics(
        difficulty=difficulty,
        seeds=num_seeds,
        success_rate=float(successes / num_seeds),
        average_reward=float(np.mean(episode_rewards)),
        average_steps_without_wall=float(np.mean(episode_steps)),
    )


def parse_difficulties(raw: str) -> List[int]:
    levels: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        levels.append(int(token))
    if not levels:
        raise ValueError("At least one difficulty level is required.")
    return levels


def save_results(
    *,
    csv_path: str,
    agent_name: str,
    seed_start: int,
    max_steps: int,
    scaling_factor: int,
    arena_size: int,
    box_speed: int,
    wall_obstacles: bool,
    per_level: List[LevelMetrics],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    all_rewards = [m.average_reward for m in per_level]
    all_steps = [m.average_steps_without_wall for m in per_level]
    all_success = [m.success_rate for m in per_level]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "agent_name",
                "difficulty_level",
                "wall_obstacles",
                "seeds",
                "seed_start",
                "success_rate",
                "average_reward",
                "average_steps_without_wall",
                "total_average_steps_without_wall",
                "max_steps",
                "scaling_factor",
                "arena_size",
                "box_speed",
            ],
        )
        if not file_exists:
            writer.writeheader()

        for metrics in per_level:
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "agent_name": agent_name,
                    "difficulty_level": metrics.difficulty,
                    "wall_obstacles": int(wall_obstacles),
                    "seeds": metrics.seeds,
                    "seed_start": seed_start,
                    "success_rate": f"{metrics.success_rate:.6f}",
                    "average_reward": f"{metrics.average_reward:.6f}",
                    "average_steps_without_wall": f"{metrics.average_steps_without_wall:.6f}",
                    "total_average_steps_without_wall": "",
                    "max_steps": max_steps,
                    "scaling_factor": scaling_factor,
                    "arena_size": arena_size,
                    "box_speed": box_speed,
                }
            )

        writer.writerow(
            {
                "timestamp_utc": timestamp,
                "agent_name": agent_name,
                "difficulty_level": "ALL",
                "wall_obstacles": int(wall_obstacles),
                "seeds": int(sum(m.seeds for m in per_level)),
                "seed_start": seed_start,
                "success_rate": f"{float(np.mean(all_success)):.6f}",
                "average_reward": f"{float(np.mean(all_rewards)):.6f}",
                "average_steps_without_wall": f"{float(np.mean(all_steps)):.6f}",
                "total_average_steps_without_wall": f"{float(np.mean(all_steps)):.6f}",
                "max_steps": max_steps,
                "scaling_factor": scaling_factor,
                "arena_size": arena_size,
                "box_speed": box_speed,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OBELIX agent on each difficulty level for multiple seeds "
            "with wall_obstacles disabled."
        )
    )
    parser.add_argument("--agent_file", type=str, default="agent.py")
    parser.add_argument("--agent_name", type=str, default=None)
    parser.add_argument("--difficulty_levels", type=str, default="0,2,3")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output_csv", type=str, default="leaderboard.csv")

    args = parser.parse_args()

    levels = parse_difficulties(args.difficulty_levels)

    agent_mod = load_agent_module(args.agent_file)
    if not hasattr(agent_mod, "policy"):
        raise AttributeError("Submission must define: policy(obs, rng) -> action_str")

    policy_fn = getattr(agent_mod, "policy")
    agent_name = (
        args.agent_name or os.path.splitext(os.path.basename(args.agent_file))[0]
    )

    per_level: List[LevelMetrics] = []
    for difficulty in levels:
        metrics = run_level_evaluation(
            policy_fn,
            agent_mod,
            difficulty=difficulty,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            box_speed=args.box_speed,
            wall_obstacles=args.wall_obstacles,
            render=args.render,
        )
        per_level.append(metrics)

        print(
            f"difficulty={metrics.difficulty} seeds={metrics.seeds} "
            f"success_rate={metrics.success_rate:.3f} "
            f"avg_reward={metrics.average_reward:.3f} "
            f"avg_steps={metrics.average_steps_without_wall:.3f} "
            f"walls={int(args.wall_obstacles)}"
        )

    total_avg_steps = float(
        np.mean([m.average_steps_without_wall for m in per_level])
    )
    print(f"total_average_steps_without_wall={total_avg_steps:.3f}")

    save_results(
        csv_path=args.output_csv,
        agent_name=agent_name,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        box_speed=args.box_speed,
        wall_obstacles=args.wall_obstacles,
        per_level=per_level,
    )

    print(f"Appended results to {args.output_csv}")


if __name__ == "__main__":
    main()
