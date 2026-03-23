"""
Test agent for Food Delivery Dispatch Optimization environment.

Runs a real LLM agent against the environment and logs full trajectories
to a .jsonl file for inspection and debugging.

Usage:
    # With OpenReward server running:
    python test_agent.py

    # Local mode (no server needed):
    python test_agent.py --local
"""

import argparse
import asyncio
import json
import os
import sys
import time

from openai import AsyncOpenAI

# For local mode
sys.path.insert(0, os.path.dirname(__file__))
from fooddelivery import FoodDelivery
from simulation import DispatchActions, GetInfoParams


async def run_local(task_id: str, max_steps: int = 50):
    """Run the agent locally without an OpenReward server."""
    oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL_NAME = "gpt-5.4"

    log_file = f"trajectory_{task_id}.jsonl"

    # Clear old log
    if os.path.exists(log_file):
        os.remove(log_file)

    env = FoodDelivery(task_spec={"id": task_id})
    prompt_blocks = await env.get_prompt()
    prompt_text = prompt_blocks[0].text

    # Define tools for the OpenAI API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_info",
                "description": "Get static city layout: zones, restaurants, courier positions. Does not advance time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "step",
                "description": (
                    "Advance the simulation by one minute with optional dispatch actions. "
                    "assignments: list of {order_ids: [int], courier_id: int}. "
                    "surge_multipliers: dict of zone_id (string) to float (1.0-3.0). "
                    "repositions: list of {courier_id: int, zone_id: int}. "
                    "All fields are optional. Call with empty object {} to advance time with no actions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "assignments": {
                            "type": "array",
                            "description": "List of order-to-courier assignments. Each has order_ids (list of ints) and courier_id (int).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "order_ids": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                    "courier_id": {"type": "integer"},
                                },
                                "required": ["order_ids", "courier_id"],
                            },
                        },
                        "surge_multipliers": {
                            "type": "object",
                            "description": "Zone surge pricing. Keys are zone IDs as strings, values are multipliers (1.0-3.0).",
                            "additionalProperties": {"type": "number"},
                        },
                        "repositions": {
                            "type": "array",
                            "description": "List of courier repositioning commands. Each has courier_id and zone_id.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "courier_id": {"type": "integer"},
                                    "zone_id": {"type": "integer"},
                                },
                                "required": ["courier_id", "zone_id"],
                            },
                        },
                    },
                },
            },
        },
    ]

    input_list = [{"role": "user", "content": prompt_text}]
    finished = False
    step_count = 0
    total_reward = 0.0

    # Log prompt
    with open(log_file, "a") as f:
        f.write(json.dumps({
            "type": "prompt",
            "task_id": task_id,
            "timestamp": time.time(),
            "content_preview": prompt_text[:500],
        }) + "\n")

    print(f"Starting agent on task: {task_id}")
    print(f"Logging trajectory to: {log_file}")
    print("-" * 60)

    while not finished and step_count < max_steps:
        try:
            response = await oai_client.chat.completions.create(
                model=MODEL_NAME,
                tools=tools,
                messages=input_list,
            )
        except Exception as e:
            print(f"LLM API error: {e}")
            break

        msg = response.choices[0].message
        input_list.append(msg)

        # Log LLM response
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "type": "llm_response",
                "step": step_count,
                "timestamp": time.time(),
                "has_tool_calls": bool(msg.tool_calls),
                "content_preview": (msg.content or "")[:200],
                "num_tool_calls": len(msg.tool_calls) if msg.tool_calls else 0,
            }) + "\n")

        if not msg.tool_calls:
            # Agent sent a text message, add to context and continue
            print(f"  Agent message: {(msg.content or '')[:100]}")
            continue

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Call the tool
            if tool_name == "get_info":
                result = await env.get_info(GetInfoParams())
            elif tool_name == "step":
                result = await env.step(DispatchActions(**args))
            else:
                result = None

            if result is None:
                tool_output = "Unknown tool"
                reward = 0.0
                finished_flag = False
            else:
                tool_output = result.blocks[0].text if result.blocks else ""
                reward = result.reward
                finished_flag = result.finished

            total_reward = reward if finished_flag else total_reward

            # Add tool response to conversation
            input_list.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_output,
            })

            step_count += 1

            # Log tool call
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "type": "tool_call",
                    "step": step_count,
                    "tool": tool_name,
                    "arguments": args,
                    "reward": reward,
                    "finished": finished_flag,
                    "timestamp": time.time(),
                    "output_preview": tool_output[:400],
                }) + "\n")

            # Print summary
            if tool_name == "step":
                n_assignments = len(args.get("assignments", []))
                n_repos = len(args.get("repositions", []))
                surge = args.get("surge_multipliers", {})
                print(
                    f"  Step {step_count}: "
                    f"assign={n_assignments} repos={n_repos} "
                    f"surge={len(surge)} zones | "
                    f"reward={reward:+.3f}"
                )
            else:
                print(f"  Step {step_count}: {tool_name}()")

            if finished_flag:
                finished = True
                print(f"\n{'='*60}")
                print(f"FINISHED! Final reward: {reward:.4f}")
                print(f"{'='*60}")
                break

    # Final summary
    print(f"\nSummary:")
    print(f"  Total tool calls: {step_count}")
    print(f"  Finished: {finished}")
    print(f"  Final reward: {total_reward:.4f}")
    print(f"  Trajectory: {log_file}")

    # Log summary
    with open(log_file, "a") as f:
        f.write(json.dumps({
            "type": "summary",
            "task_id": task_id,
            "total_tool_calls": step_count,
            "finished": finished,
            "final_reward": total_reward,
            "timestamp": time.time(),
        }) + "\n")

    return finished, total_reward


async def run_remote():
    """Run agent against an OpenReward server."""
    from openreward import OpenReward

    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/fooddelivery"
    SPLIT = "train"

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    for task in tasks[:1]:
        task_id = task.task_spec["id"]
        log_file = f"trajectory_{task_id}.jsonl"

        if os.path.exists(log_file):
            os.remove(log_file)

        rollout = or_client.rollout.create(
            run_name="fooddelivery_test",
            rollout_name=f"test_{task_id}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
        )

        async with environment.session(task=task) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False
            step_count = 0

            rollout.log_openai_response(message=input_list[0], is_finished=finished)

            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "type": "prompt",
                    "task_id": task_id,
                    "timestamp": time.time(),
                    "content_preview": prompt[0].text[:500],
                }) + "\n")

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name, json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        rollout.log_openai_response(
                            input_list[-1], reward=reward, is_finished=finished
                        )

                        step_count += 1

                        with open(log_file, "a") as f:
                            f.write(json.dumps({
                                "type": "tool_call",
                                "step": step_count,
                                "tool": item.name,
                                "arguments": json.loads(str(item.arguments)),
                                "reward": reward,
                                "finished": finished,
                                "timestamp": time.time(),
                                "output_preview": tool_result.blocks[0].text[:400],
                            }) + "\n")

                        print(f"Step {step_count} | Tool: {item.name} | Reward: {reward:.3f}")

                        if finished:
                            print(f"\nFINISHED! Final reward: {reward:.4f}")
                            break

            print(f"\nTotal tool calls: {step_count}")
            print(f"Trajectory logged to: {log_file}")


async def main():
    parser = argparse.ArgumentParser(description="Test Food Delivery agent")
    parser.add_argument("--local", action="store_true",
                        help="Run locally without OpenReward server")
    parser.add_argument("--task", default="weekday_calm_seed42",
                        help="Task ID to run (default: weekday_calm_seed42)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max tool calls before stopping (default: 300)")
    args = parser.parse_args()

    if args.local:
        await run_local(args.task, args.max_steps)
    else:
        await run_remote()


if __name__ == "__main__":
    asyncio.run(main())
