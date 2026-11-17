# Documentation

Start with the [project README](../README.md) for what BoxQC is and how to run it.

**Using it**

| | |
|---|---|
| [workflow.md](workflow.md) | From an empty station to a working inspection: ROI, capture, label, train, inspect. |
| [pi-setup.md](pi-setup.md) | The device: hardware, installation on a Raspberry Pi, USB network link, code sync, troubleshooting. |
| [configuration.md](configuration.md) | How `boxqc.yaml`, the local overlay, environment variables and runtime settings fit together. |
| [cli.md](cli.md) | Every `boxqc` command and option. |
| [upgrading.md](upgrading.md) | Moving an installation from BoxQC 0.1 to 0.2. |

**Integrating with it**

| | |
|---|---|
| [device-api.md](device-api.md) | The device's HTTP API: inspections, camera, settings, models. |
| [models.md](models.md) | The model file and metadata contract; running a model trained elsewhere; adding a model format. |
| [trainers.md](trainers.md) | How training is organised; adding a training backend. |
| [data-formats.md](data-formats.md) | Every file BoxQC stores: annotations, datasets, settings, inspection records. |
| [../examples/](../examples/README.md) | A device API client and two plug-ins to copy. |

**Changing it**

| | |
|---|---|
| [architecture.md](architecture.md) | The two programs, how they talk, and what each module is responsible for. |
| [development.md](development.md) | Setup, tests, the rules that keep the design intact, and how to make common changes. |
