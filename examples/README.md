# Examples

| File | What it shows |
|---|---|
| [inspect_client.py](inspect_client.py) | Driving a device from another program: find it on the network, trigger an inspection, read the verdict, fetch the annotated photo. Standard library only. |
| [command_trainer.py](command_trainer.py) | A trainer plug-in that runs any external training program on the prepared dataset. |
| [boxes_scores_labels_format.py](boxes_scores_labels_format.py) | A model-format plug-in for detectors with separate `boxes` / `scores` / `labels` outputs. |

The two plug-ins are complete and tested (`tests/test_examples.py`), and are meant to be
copied and adapted. To try one as it is, put the examples folder on the module path and
name the module in `boxqc.yaml`:

```yaml
plugins: [command_trainer]
```

```bash
PYTHONPATH=examples boxqc plugins     # the new trainer / format is listed
```

[docs/trainers.md](../docs/trainers.md) and [docs/models.md](../docs/models.md) explain
the two interfaces; packaging a plug-in for `pip install` is covered there too.
