# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for recording pi0.5-DROID deployment observations."""

import json

import numpy as np

from rlinf.models.embodiment.openpi.policies import observation_recording_policy


class _FakePolicy:
    def __init__(self):
        self.observations = []

    def infer(self, obs):
        self.observations.append(obs)
        return {"actions": np.zeros((15, 8), dtype=np.float32)}

    def reset(self):
        pass


class _FakeWriter:
    instances = []

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.frames = []
        self.closed = False
        self.instances.append(self)

    def write(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        self.closed = True


def test_recording_policy_writes_timestamped_observation_and_video_frames(
    tmp_path, monkeypatch
):
    _FakeWriter.instances = []
    monkeypatch.setattr(observation_recording_policy, "_H264Writer", _FakeWriter)
    wrapped = _FakePolicy()
    policy = observation_recording_policy.DroidObservationRecordingPolicy(
        wrapped,
        output_root=tmp_path,
        exterior_image_key="observation/exterior_image_2_left",
        fps=15.0,
    )
    obs = {
        "observation/exterior_image_2_left": np.full((180, 320, 3), 25, dtype=np.uint8),
        "observation/wrist_image_left": np.full((180, 320, 3), 50, dtype=np.uint8),
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "prompt": "wipe the whiteboard",
    }

    result = policy.infer(obs)
    policy.close()

    assert result["actions"].shape == (15, 8)
    assert wrapped.observations == [obs]
    record = json.loads(
        (policy.output_dir / "observations.jsonl").read_text(encoding="utf-8")
    )
    assert record["index"] == 0
    assert record["timestamp_utc"].endswith("+00:00")
    assert record["observation"]["observation/joint_position"] == list(range(7))
    assert record["observation"]["prompt"] == "wipe the whiteboard"
    image_record = record["observation"]["observation/exterior_image_2_left"]
    assert image_record == {
        "video": "exterior.mp4",
        "frame_index": 0,
        "shape": [180, 320, 3],
        "dtype": "uint8",
    }
    assert len(_FakeWriter.instances) == 3
    assert _FakeWriter.instances[0].frames[0].shape == (180, 320, 3)
    assert _FakeWriter.instances[1].frames[0].shape == (180, 320, 3)
    assert _FakeWriter.instances[2].frames[0].shape == (224, 672, 3)
    assert np.all(_FakeWriter.instances[2].frames[0][:, 448:] == 0)
    assert all(writer.closed for writer in _FakeWriter.instances)


def test_recording_policy_rejects_missing_selected_camera(tmp_path, monkeypatch):
    _FakeWriter.instances = []
    monkeypatch.setattr(observation_recording_policy, "_H264Writer", _FakeWriter)
    policy = observation_recording_policy.DroidObservationRecordingPolicy(
        _FakePolicy(),
        output_root=tmp_path,
        exterior_image_key="observation/exterior_image_1_left",
    )

    try:
        with np.testing.assert_raises_regex(KeyError, "missing image keys"):
            policy.infer(
                {"observation/wrist_image_left": np.zeros((16, 16, 3), dtype=np.uint8)}
            )
    finally:
        policy.close()
