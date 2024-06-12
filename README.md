## Structure of data folder

```
data
├── for_generating     # Files for generation dataset
│   ├── coins          # Images of coins
│   └── textures       # Images of textures
├── test               # Test data from task
│   ├── images
│   └── counts.json
└── dataset            # Generated dataset to YOLO model
    ├── train
    │   ├── images
    │   └── labels
    ├── valid
    │   ├── images
    │   └── labels
    └── data.yaml
```

## Run unstruction

#### Create docker image

1. `docker build -t <TAG_NAME> .`

#### Run docker in interactive mode

2. `docker run --rm -it --entrypoint bash -v ./weights:/app/weights -v ./data:/app/data <TAG_NAME>`

#### Install dependicies

By default the torch version is set for the cpu, if you need training on the gpu you will need to change this

3. `make setup_ws`

#### Generate dataset to YOLO

4. `make generate_dataset`

#### Attach clearml

5. `poetry run clearml-init`

#### Run train. ```--model_path``` default is yolov8s.pt, you can replace it your weights

6. `make train ARGS='--model_path=<some path>'`

#### Run test, ```--download=True``` if you want to download test dataset, ```--weights_path``` - path to .pt file
7. `make test ARGS='--weights_path=<some path>'`

# Results:
Метрика 1. Средняя относительная ошибка количества монет: 13.7%

Метрика 2. Средняя относительная ошибка количества денег: 40.8%