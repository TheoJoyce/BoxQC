"""Example trainer: hand the dataset to any external program and collect its model.

Use this when your training code is not Python you want to import - another framework,
a container, a script on a cluster. BoxQC prepares the dataset, runs your command, and
takes over again once the command has written a model file: evaluation, metadata and
deployment are the same as for the built-in trainer.

To use it, make the module importable and configure it in boxqc.yaml::

    plugins: [command_trainer]
    train:
      trainer: command

and set the command in the environment before starting the studio (it is deliberately
not accepted from the web UI, which would let anyone who can reach the studio run
programs)::

    export BOXQC_TRAIN_COMMAND='python my_train.py --data {data} --epochs {epochs} --imgsz {imgsz} --out {output}'

Placeholders: ``{data}`` (the YOLO data.yaml), ``{root}`` (the dataset directory, which
also holds COCO annotations), ``{output}`` (where the command must write an ONNX model
in a format the device can run, see docs/models.md), ``{epochs}``, ``{imgsz}``,
``{batch}``, ``{device}``, ``{base_model}`` and ``{name}``.

Optional progress reporting: every output line of the form ``epoch=3 mAP50=0.71 ...``
(space-separated ``key=value`` pairs including ``epoch``) is charted on the Training page.
"""

from __future__ import annotations

import os
import shlex
import subprocess

from boxqc.studio.export import PreparedDataset, write_coco
from boxqc.studio.training import (
    TrainContext,
    Trainer,
    TrainerOutput,
    TrainingCancelled,
    TrainRequest,
    register_trainer,
)

COMMAND_ENV = "BOXQC_TRAIN_COMMAND"


def parse_progress(line: str) -> dict | None:
    """``"epoch=3 mAP50=0.71"`` -> ``{"epoch": 3, "mAP50": 0.71}``; None for any other line."""
    metrics: dict = {}
    for part in line.split():
        key, sep, value = part.partition("=")
        if not sep:
            return None
        try:
            metrics[key] = int(value) if key == "epoch" else float(value)
        except ValueError:
            return None
    return metrics if "epoch" in metrics else None


@register_trainer
class CommandTrainer(Trainer):
    name = "command"
    label = "External command"
    base_models = (("none", "decided by the command"),)

    @classmethod
    def availability(cls) -> str | None:
        return None if os.environ.get(COMMAND_ENV) else f"Set ${COMMAND_ENV} to the training command."

    def train(self, request: TrainRequest, dataset: PreparedDataset, ctx: TrainContext) -> TrainerOutput:
        write_coco(dataset)     # so the command may use either annotation format
        run_dir = self.cfg.train.workdir / "runs" / request.name
        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / "model.onnx"
        command = os.environ[COMMAND_ENV].format(
            data=shlex.quote(str(dataset.yaml_path)), root=shlex.quote(str(dataset.root)),
            output=shlex.quote(str(output)), epochs=request.epochs, imgsz=request.imgsz,
            batch=request.batch, device=shlex.quote(request.device),
            base_model=shlex.quote(request.base_model), name=shlex.quote(request.name))
        ctx.log(f"$ {command}")

        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                line = line.rstrip()
                ctx.log(line)
                if (metrics := parse_progress(line)) is not None:
                    ctx.epoch(metrics, total_epochs=request.epochs)
                if ctx.stop_requested == "cancel":      # "early" cannot be passed on: let it finish
                    raise TrainingCancelled()
            status = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
        if status != 0:
            raise RuntimeError(f"The training command exited with status {status}")
        if not output.is_file():
            raise RuntimeError(f"The training command did not write {output}")
        return TrainerOutput(model_path=output, format=str(request.options.get("format", "yolo-onnx")),
                             info={"run_dir": str(run_dir), "command": command})
