"""
robot/maniskill/gemini_highlevel.py

Gemini-based high-level policy for in-context reasoning over steering commands.

Implements Section V-B ("In-context Reasoning for Choosing Steering Abstractions")
and Appendix B of the Steerable Policies paper.

The GeminiHighLevelPolicy:
  - Maintains a within-episode history of (image, command) pairs.
  - Re-queries Gemini every `steps_per_query` environment steps.
  - Resolves <object description> placeholders in commands to [x, y] pixel
    coordinates (normalised 0-255 as per Appendix A).
  - Supports three modes (matching paper experiments):
      "full"          – full reasoning + all steering abstractions  (Fig. 22)
      "no_reasoning"  – single-line output, all abstractions         (Fig. 23)
      "saycan"        – subtask-level commands only, with reasoning  (Fig. 24)

Usage:
    from robot.maniskill.gemini_highlevel import GeminiHighLevelPolicy

    policy = GeminiHighLevelPolicy(
        task_description="put the carrot on the plate",
        mode="full",
        model_name="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    policy.reset()          # call at start of each episode
    cmd = policy.step(img)  # call every env step; returns steering command
"""

import io
import json
import re
import time
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Prompts — verbatim from Appendix B (Figs. 22–25) of the paper
# ---------------------------------------------------------------------------

_HEADER_FULL = """\
You are guiding a robot to complete a task by generating a subtask instruction \
for the robot to follow given an image of the robot's current observation, and \
a history of observations and past commands.

The robot is trying to accomplish the overall task: '{task}'. You should keep \
the rest of the environment unchanged. Note that all objects required for the \
task are present (no need to search for objects). There are several kinds of \
instructions the robot understands, each at a different level of abstraction, \
detailed below. Sometimes, a high-level instruction is sufficient for the robot \
to make progress towards solving the task, but other times, a more specific \
command is required.

Below are examples of the kinds of instructions you can generate for the robot.

Some examples of High-level instructions are:
- 'grasp the mushroom in the pot'
- 'wipe the cloth across the table'
- 'move the lid above the pot'
- 'lift the carrot up'
- 'move towards the stuffed toy'
- 'drop the eggplant in the bowl'

The robot is generally reliable at following high-level commands, especially \
when interacting with non-novel objects. They can also be used to better \
understand the scene and robot state, such as by saying 'lift the object' to \
ascertain if the robot has actually firmly grasped it. However, these commands \
may fail when interacting with novel objects, as the robot may not recognize \
them by name. The robot also may also confuse similar objects with each other \
(e.g., it may mix up a bowl and pot, or multiple instances of the same kind \
of object).

Some examples of Atomic Motion instructions are:
- 'move left'
- 'move right and down'
- 'move down and close'
- 'open the gripper'
- 'move the cloth left'
- 'pull the drawer handle backward'

Motion commands are good for general positioning, such as moving to a \
pre-grasp pose above the object the robot must interact with. They can also \
be used for more minor corrections, e.g., 'move up and right above the bowl' \
if the gripper is too low. These commands are also appropriate when the task \
involves spatial reasoning (e.g., tasks that reference left or right) or moving \
in unconventional ways (e.g., lifting the arm to grasp something on a high \
shelf). However, these commands are often underspecified, so the robot may reach \
for the wrong object if simply told 'move to the right and grasp.'

Some examples of Position-based instructions are:
- 'pick up the object at <mushroom>'
- 'go to the pot at <pot> to grab the mushroom'

Note, you simply need to specify the object name within angle brackets, e.g., \
<mushroom>, and a object detector system will fill in the pixel coordinates for \
you. If there are multiple instances of the object, provide a unique, specific \
identifier within the brackets to indicate which instance you are referring to, \
e.g., <mushroom on the left>.

Position-based commands are especially helpful for specifying novel objects that \
the robot fails to recognize by name. They can also be used to clearly specify \
locations, such as where to place or move objects. However, these commands can \
be unreliable, as the robot may get confused (e.g., reaching for the wrong \
object in scenes with clutter or with multiple instances of the same object). \
The object detector may also fail, especially when objects are occluded.

Next, some examples of Combination instructions are:
- 'move down to the object at <mushroom> and grasp it'
- 'grab the mushroom at <mushroom on the left> and move it to <plate>'
- 'move the stuffed toy toward the box at <box>'

Combination commands are useful for supplementing the weaknesses of other \
command types, such as saying 'move right to grasp the object at <object \
description>' rather than just 'move right and grasp' (if there are multiple \
objects to the right of the robot). However, being too specific may result in \
out-of-distribution commands, which the robot policy fails to follow. \
Additionally, less specific commands can be useful for positioning the robot \
in a location where it's more likely to succeed.

Finally, there is the RESET TO HOME instruction, which causes the robot to \
return to its home position. You use this if the robot is stuck or interacting \
with the wrong object.

You MUST follow these steps when generating your response:
- Step 1: Briefly describe the scene in the current observation, and any \
important differences from past observations (eg. objects moved, picked up, \
dropped etc).
- Step 2: Briefly describe what must happen next to make progress towards the \
overall task.
- Step 3: Generate a candidate instruction for EACH level of abstraction.
- Step 4: Briefly reason about which instruction is most appropriate given the \
current observation and task progress.
- Step 5: Determine the final steerable command for the robot to execute. Only \
the LAST line of your output will be interpreted as the subtask command. In the \
LAST line, answer ONLY with the subtask instruction that the robot should \
execute, and NOTHING ELSE.

Note: The robot will ask you for a new instruction after executing for a few \
steps. If you've issued the same type of command more than once without \
progress, you MUST issue a command at a different level of abstraction. If you \
previously provided a command with positions <> (without progress), you MAY NOT \
use positions in the next command. For example: - If you previously tried to \
pick up the mushroom by specifying the mushroom's position, but the robot did \
not reach the mushroom, next time you can explicitly tell the robot to move in \
a certain direction (eg. move to the right and down). - If the robot reached \
for the wrong object when you told it to pick up the mushroom at <mushroom on \
the left>. This time, you should give a command at a different level of \
abstraction: move left to the mushroom and grasp it.\
"""

_HEADER_NO_REASONING = """\
You are guiding a robot to complete a task by generating a subtask instruction \
for the robot to follow given an image of the robot's current observation, and \
a history of observations and past commands.

The robot is trying to accomplish the overall task: '{task}'. You should keep \
the rest of the environment unchanged. Note that all objects required for the \
task are present (no need to search for objects). There are several kinds of \
instructions the robot understands, each at a different level of abstraction, \
detailed below. Sometimes, a high-level instruction is sufficient for the robot \
to make progress towards solving the task, but other times, a more specific \
command is required.

Below are examples of the kinds of instructions you can generate for the robot.

Some examples of High-level instructions are:
- 'grasp the mushroom in the pot'
- 'wipe the cloth across the table'
- 'move the lid above the pot'
- 'lift the carrot up' # IMPORTANT: Lifting is generallly good post-grasping
- 'move towards the stuffed toy'
- 'drop the eggplant in the bowl'

The robot is generally reliable at following high-level commands, especially \
when interacting with non-novel objects. They can also be used to better \
understand the scene and robot state, such as by saying 'lift the object' to \
ascertain if the robot has actually firmly grasped it. However, these commands \
may fail when interacting with novel objects, as the robot may not recognize \
them by name. The robot also may also confuse similar objects with each other \
(e.g., it may mix up a bowl and pot, or multiple instances of the same kind \
of object).

Some examples of Atomic Motion instructions are:
- 'move left'
- 'move right and down'
- 'move down and close'
- 'open the gripper'
- 'move the cloth left'
- 'pull the drawer handle backward'

Motion commands are good for general positioning, such as moving to a \
pre-grasp pose above the object the robot must interact with. They can also \
be used for more minor corrections, e.g., 'move up and right above the bowl' \
if the gripper is too low. These commands are also appropriate when the task \
involves spatial reasoning (e.g., tasks that reference left or right) or moving \
in unconventional ways (e.g., lifting the arm to grasp something on a high \
shelf). However, these commands are often underspecified, so the robot may reach \
for the wrong object if simply told 'move to the right and grasp.'

Some examples of Position-based instructions are:
- 'pick up the object at <mushroom>'
- 'go to <pot>'
- 'grasp at <blue cube>'
- 'move above <plate>'
- 'open gripper at <bowl>'
- 'move towards <towel>'

Note, you simply need to specify the object name within angle brackets, e.g., \
<mushroom>, and a object detector system will fill in the pixel coordinates for \
you. If there are multiple instances of the object, provide a unique, specific \
identifier within the brackets to indicate which instance you are referring to, \
e.g., <mushroom on the left>.

Position-based commands are especially helpful for specifying novel objects that \
the robot fails to recognize by name. They can also be used to clearly specify \
locations, such as where to place or move objects. However, these commands can \
be unreliable, as the robot may get confused (e.g., reaching for the wrong \
object in scenes with clutter or with multiple instances of the same object). \
The object detector may also fail, especially when objects are occluded.

Next, some examples of Combination instructions are:
- 'move down to the object at <mushroom> and grasp it'
- 'grab the mushroom at <mushroom on the left> and move it to <plate>'
- 'move the stuffed toy toward the box at <box>'

Combination commands are useful for supplementing the weaknesses of other \
command types, such as saying 'move right to grasp the object at <object \
description>' rather than just 'move right and grasp' (if there are multiple \
objects to the right of the robot). However, being too specific may result in \
out-of-distribution commands, which the robot policy fails to follow. \
Additionally, less specific commands can be useful for positioning the robot \
in a location where it's more likely to succeed.

Finally, there is the RESET TO HOME instruction, which causes the robot to \
return to its home position. You use this if the robot is stuck or interacting \
with the wrong object.

You MUST format your response as follows: ONLY OUTPUT A SINGLE COMMAND AND \
NOTHING ELSE. YOUR RESPONSE SHOULD ONLY HAVE ONE LINE, WHICH IS THE COMMAND \
THE ROBOT MUST EXECUTE.\
"""

_HEADER_SAYCAN = """\
You are guiding a robot to complete a task by generating a subtask instruction \
for the robot to follow given an image of the robot's current observation, and \
a history of observations and past commands. The robot is trying to accomplish \
the overall task: '{task}'. You should keep the rest of the environment \
unchanged. Note that all objects required for the task are present (no need to \
search for objects). Below are some examples of the kinds of instructions you \
should generate:
- 'grasp the mushroom in the pot'
- 'wipe the cloth across the table'
- 'move the lid above the pot'

You MUST follow these steps when generating your response:
- Step 1: Briefly (1-2 sentences) describe the scene in the current \
observation, and any important differences from past observations (eg. objects \
moved, picked up, dropped etc).
- Step 2: Briefly (1 sentence) describe what must happen next to make progress \
towards the overall task.
- Step 3: Determine the final steerable command (last line) for the robot to \
execute. Only the LAST line of your output will be interpreted as the subtask \
command. In the LAST line, answer ONLY with the subtask instruction that the \
robot should execute, and NOTHING ELSE.\
"""

# Query footers appended after the current-observation image
_QUERY_FULL = (
    "Based on this observation, generate the next command. Remember the "
    "instructions: describe the scene, determine what must happen next, generate "
    "steerable commands at EACH level of abstraction, determine the best command, "
    "output the best command. Again, the LAST line of your output will be "
    "interpreted as the final steerable, subtask command. In the LAST line, "
    "answer ONLY with the subtask instruction that the robot should execute, "
    "and NOTHING ELSE."
)

_QUERY_NO_REASONING = (
    "Based on this observation, generate the next command. Again, ONLY OUTPUT "
    "A SINGLE LINE, and NOTHING ELSE."
)

_QUERY_SAYCAN = (
    "Based on this observation, generate the next command. Remember the "
    "instructions: describe the scene, determine what must happen next, and "
    "generate a command. Again, the LAST line of your output will be interpreted "
    "as the final subtask command. In the LAST line, answer ONLY with the subtask "
    "instruction that the robot should execute, and NOTHING ELSE."
)

# Prompt for resolving <object description> tags to pixel coordinates (Fig. 25)
_KEYPOINT_PROMPT = (
    "What is the pixel coordinate location of the {object_description} in the image?\n"
    "The image has width {width} and height {height}.\n"
    "Respond with a JSON object with the keys 'x' and 'y'."
)

# Thinking-token budget per mode (paper: 1024 for full/saycan, 0 for no_reasoning)
_THINKING_BUDGET: dict[str, int] = {
    "full": 1024,
    "saycan": 1024,
    "no_reasoning": 0,
}


# ---------------------------------------------------------------------------
# Helper: import google-genai
# ---------------------------------------------------------------------------

def _import_genai():
    """Import the google-genai package, raising a helpful error if absent."""
    try:
        from google import genai
        from google.genai import types
        return genai, types
    except ImportError as exc:
        raise ImportError(
            "The google-genai package is required for Gemini integration. "
            "Install it with:  pip install google-genai"
        ) from exc


# ---------------------------------------------------------------------------
# GeminiHighLevelPolicy
# ---------------------------------------------------------------------------

class GeminiHighLevelPolicy:
    """
    High-level VLM policy using Gemini for in-context reasoning over steering
    commands (Section V-B of the Steerable Policies paper).

    The policy is stateful within an episode: it accumulates a history of
    (image, command) pairs and provides that history as in-context examples
    whenever it re-queries the model.  Call `reset()` between episodes.

    Args:
        task_description: Natural-language description of the task.
        mode: One of "full" (Fig. 22), "no_reasoning" (Fig. 23),
              or "saycan" (Fig. 24).
        model_name: Gemini model to use (e.g. "gemini-2.5-flash").
        api_key: Gemini API key. If None, uses the GEMINI_API_KEY env var.
        steps_per_query: Re-query Gemini after this many env steps (paper: 20).
        max_history: Maximum number of past (image, command) pairs to keep.
        img_width: Width of raw images supplied to `step()`, used for
                   coordinate normalisation (ManiSkill: 640).
        img_height: Height of raw images supplied to `step()` (ManiSkill: 480).
    """

    MODES = ("full", "no_reasoning", "saycan")

    def __init__(
        self,
        task_description: str,
        mode: str = "full",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        steps_per_query: int = 20,
        max_history: int = 10,
        img_width: int = 640,
        img_height: int = 480,
    ) -> None:
        assert mode in self.MODES, f"mode must be one of {self.MODES}, got '{mode}'"

        self.task = task_description
        self.mode = mode
        self.steps_per_query = steps_per_query
        self.max_history = max_history
        self.img_width = img_width
        self.img_height = img_height

        # Lazy-import so the rest of the repo works without google-genai installed
        genai, self._types = _import_genai()
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

        # Select prompt templates for the chosen mode
        _headers = {
            "full": _HEADER_FULL,
            "no_reasoning": _HEADER_NO_REASONING,
            "saycan": _HEADER_SAYCAN,
        }
        _queries = {
            "full": _QUERY_FULL,
            "no_reasoning": _QUERY_NO_REASONING,
            "saycan": _QUERY_SAYCAN,
        }
        self._header = _headers[mode].format(task=task_description)
        self._query_footer = _queries[mode]

        # Episode state
        self._history: list[tuple[np.ndarray, str]] = []
        self._current_command: Optional[str] = None
        self._query_image: Optional[np.ndarray] = None
        self._steps_since_query: int = 0
        self._episode_step: int = 0
        # Trace: one entry per Gemini query, for saving to .txt after each attempt
        self._trace: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode state.  Call at the start of every new episode."""
        self._history = []
        self._current_command = None
        self._query_image = None
        self._steps_since_query = 0
        self._episode_step = 0
        self._trace = []

    def step(self, raw_image: np.ndarray) -> str:
        """
        Return the current steering command, re-querying Gemini when needed.

        Args:
            raw_image: Full-resolution uint8 RGB image (H, W, 3) from the env.
                       For ManiSkill this is the raw 640×480 camera image.

        Returns:
            Steering command string to pass to the low-level VLA.
        """
        need_requery = (
            self._current_command is None
            or self._steps_since_query >= self.steps_per_query
        )

        if need_requery:
            # Archive the previous (query-image, command) pair into history
            if self._current_command is not None and self._query_image is not None:
                self._history.append((self._query_image.copy(), self._current_command))
                if len(self._history) > self.max_history:
                    self._history = self._history[-self.max_history :]

            # Query Gemini for the next steering command (returns last line + full text)
            raw_cmd, full_response = self._query_gemini(raw_image)
            print(f"  [Gemini raw] {raw_cmd!r}")

            # Resolve <object> placeholders to [x, y] coordinates (not for saycan)
            if self.mode != "saycan":
                command = self._resolve_keypoints(raw_cmd, raw_image)
            else:
                command = raw_cmd

            # RESET TO HOME → fall back to the task description
            if "RESET TO HOME" in command.upper():
                print("  [Gemini] RESET TO HOME → falling back to task description.")
                command = self.task

            # Record trace entry for this query
            self._trace.append({
                "episode_step": self._episode_step,
                "full_response": full_response,
                "extracted_command": raw_cmd,
                "resolved_command": command,
            })

            self._current_command = command
            self._query_image = raw_image.copy()
            self._steps_since_query = 0
            print(f"  [Gemini cmd] {command!r}")

        self._episode_step += 1
        self._steps_since_query += 1
        return self._current_command

    def get_trace(self) -> list:
        """Return the list of per-query trace entries for this episode."""
        return list(self._trace)

    def format_trace(
        self, task_name: str, seed: int, attempt: int, success: bool, total_steps: int
    ) -> str:
        """Format the episode trace as a human-readable string for saving to .txt."""
        lines = [
            "=== Gemini Episode Trace ===",
            f"Task:        {task_name}",
            f"Instruction: {self.task}",
            f"Mode:        {self.mode}",
            f"Seed:        {seed}",
            f"Attempt:     {attempt}",
            "",
        ]
        for i, entry in enumerate(self._trace):
            lines.append(f"--- Query {i + 1}  (episode step {entry['episode_step']}) ---")
            lines.append("Full Gemini response:")
            lines.append(entry["full_response"])
            lines.append(f"Extracted command: \"{entry['extracted_command']}\"")
            if entry["resolved_command"] != entry["extracted_command"]:
                lines.append(f"Resolved command:  \"{entry['resolved_command']}\"")
            lines.append("")
        result = "SUCCESS" if success else "FAILED"
        lines.append(f"=== Result: {result} ({total_steps} steps) ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _np_to_part(self, image_np: np.ndarray):
        """Convert a uint8 numpy RGB image to a Gemini API Part (JPEG bytes)."""
        buf = io.BytesIO()
        Image.fromarray(image_np.astype(np.uint8)).save(buf, format="JPEG", quality=90)
        return self._types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    def _build_parts(self, current_image: np.ndarray) -> list:
        """
        Build the ordered list of content Parts for one Gemini generate_content
        call.  Structure mirrors Fig. 22 / 23 / 24:

            [header text]
            "Here are the past observations and commands:"
              "Observation:" <image> "Command: <cmd>"   ← repeated per history entry
            "Current Observation:" <image>
            [query footer]
        """
        parts = [self._types.Part.from_text(text=self._header)]

        if self._history:
            parts.append(
                self._types.Part.from_text(text="\nHere are the past observations and commands:")
            )
            for hist_img, hist_cmd in self._history:
                parts.append(self._types.Part.from_text(text="\nObservation:"))
                parts.append(self._np_to_part(hist_img))
                parts.append(self._types.Part.from_text(text=f"Command: {hist_cmd}"))

        parts.append(self._types.Part.from_text(text="\nCurrent Observation:"))
        parts.append(self._np_to_part(current_image))
        parts.append(self._types.Part.from_text(text=self._query_footer))

        return parts

    def _make_config(self, thinking_budget: int):
        """Build a GenerateContentConfig, optionally with a thinking budget."""
        try:
            return self._types.GenerateContentConfig(
                thinking_config=self._types.ThinkingConfig(thinking_budget=thinking_budget)
            )
        except (AttributeError, TypeError):
            # ThinkingConfig not available in this SDK version — proceed without it
            return None

    def _query_gemini(self, current_image: np.ndarray, max_retries: int = 3) -> tuple[str, str]:
        """
        Call Gemini and return (last_line, full_response_text).
        The last non-empty line is the steering command; the full text is saved to the trace.
        """
        parts = self._build_parts(current_image)
        budget = _THINKING_BUDGET[self.mode]
        config = self._make_config(budget)

        for attempt in range(max_retries):
            try:
                kwargs = dict(model=self._model_name, contents=parts)
                if config is not None:
                    kwargs["config"] = config
                response = self._client.models.generate_content(**kwargs)
                text = response.text.strip()
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                last_line = lines[-1] if lines else self.task
                return last_line, text
            except Exception as exc:
                print(f"  [Gemini] API error (attempt {attempt + 1}/{max_retries}): {exc}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        # All retries exhausted → fall back to task description
        print("  [Gemini] All retries failed; falling back to task description.")
        return self.task, ""

    def _resolve_keypoints(self, command: str, image_np: np.ndarray) -> str:
        """
        Replace every <object description> tag in `command` with [x, y] pixel
        coordinates (normalised to [0, 255] as per Appendix A of the paper).
        Uses a separate Gemini call per tag (Fig. 25).
        """
        tags = re.findall(r"<([^<>]+)>", command)
        if not tags:
            return command

        resolved = command
        for tag in tags:
            coords = self._get_keypoint(tag, image_np)
            if coords is not None:
                x_norm, y_norm = coords
                resolved = resolved.replace(f"<{tag}>", f"[{x_norm}, {y_norm}]", 1)
            else:
                print(f"  [Gemini] Could not resolve keypoint for '<{tag}>'; leaving as-is.")

        return resolved

    def _get_keypoint(
        self,
        object_description: str,
        image_np: np.ndarray,
        max_retries: int = 2,
    ) -> Optional[tuple[int, int]]:
        """
        Ask Gemini for the pixel location of `object_description` in `image_np`.

        Returns (x_norm, y_norm) normalised to [0, 255], or None on failure.
        Implements the prompt from Fig. 25 of the paper.
        """
        prompt = _KEYPOINT_PROMPT.format(
            object_description=object_description,
            width=self.img_width,
            height=self.img_height,
        )
        parts = [
            self._np_to_part(image_np),
            self._types.Part.from_text(text=prompt),
        ]

        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=parts,
                )
                text = response.text.strip()
                # Strip markdown code fences if present
                text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
                data = json.loads(text)
                x_raw, y_raw = float(data["x"]), float(data["y"])
                # Normalise raw pixel coords → [0, 255]
                x_norm = int(round(x_raw * 255.0 / self.img_width))
                y_norm = int(round(y_raw * 255.0 / self.img_height))
                x_norm = max(0, min(255, x_norm))
                y_norm = max(0, min(255, y_norm))
                return x_norm, y_norm
            except Exception as exc:
                print(
                    f"  [Gemini] Keypoint resolution error for '{object_description}' "
                    f"(attempt {attempt + 1}/{max_retries}): {exc}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)

        return None


# ---------------------------------------------------------------------------
# KeypointResolver — lightweight stand-alone object-detection helper
# ---------------------------------------------------------------------------

class KeypointResolver:
    """
    Resolves <object description> tags in a command string to [x, y] pixel
    coordinates (normalised 0-255), using the same Fig. 25 prompt as
    GeminiHighLevelPolicy.

    Used when the user manually types a command with <tag> placeholders during
    --interactive or --prompt_every_n_steps evaluation (without full Gemini
    high-level policy).

    Example:
        resolver = KeypointResolver(api_key=os.environ["GEMINI_API_KEY"])
        cmd = resolver.resolve("pick up the object at <carrot>", image_np)
        # → "pick up the object at [112, 198]"
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        img_width: int = 640,
        img_height: int = 480,
    ) -> None:
        genai, self._types = _import_genai()
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self.img_width = img_width
        self.img_height = img_height

    def resolve(self, command: str, image_np: np.ndarray) -> str:
        """Replace every <object description> tag in `command` with [x, y] coordinates."""
        tags = re.findall(r"<([^<>]+)>", command)
        if not tags:
            return command
        resolved = command
        for tag in tags:
            coords = self._get_keypoint(tag, image_np)
            if coords is not None:
                x_norm, y_norm = coords
                resolved = resolved.replace(f"<{tag}>", f"[{x_norm}, {y_norm}]", 1)
                print(f"  [Keypoint] <{tag}> → [{x_norm}, {y_norm}]")
            else:
                print(f"  [Keypoint] Could not resolve '<{tag}>'; leaving as-is.")
        return resolved

    def _np_to_part(self, image_np: np.ndarray):
        buf = io.BytesIO()
        Image.fromarray(image_np.astype(np.uint8)).save(buf, format="JPEG", quality=90)
        return self._types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    def _get_keypoint(
        self,
        object_description: str,
        image_np: np.ndarray,
        max_retries: int = 2,
    ) -> Optional[tuple[int, int]]:
        prompt = _KEYPOINT_PROMPT.format(
            object_description=object_description,
            width=self.img_width,
            height=self.img_height,
        )
        parts = [
            self._np_to_part(image_np),
            self._types.Part.from_text(text=prompt),
        ]
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=parts,
                )
                text = response.text.strip()
                text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
                data = json.loads(text)
                x_raw, y_raw = float(data["x"]), float(data["y"])
                x_norm = int(round(x_raw * 255.0 / self.img_width))
                y_norm = int(round(y_raw * 255.0 / self.img_height))
                x_norm = max(0, min(255, x_norm))
                y_norm = max(0, min(255, y_norm))
                return x_norm, y_norm
            except Exception as exc:
                print(
                    f"  [Keypoint] Error for '{object_description}' "
                    f"(attempt {attempt + 1}/{max_retries}): {exc}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
        return None
