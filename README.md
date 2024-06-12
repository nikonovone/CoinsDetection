1. docker build -t \<TAG_NAME> .
1. docker run --rm -it --entrypoint bash -v ./weights:/app/weights -v ./data:/app/data \<TAG_NAME>
1. make setup_ws
1. make generate_dataset
1. make train ARGS='--model_path=<some path>'
1. make test ARGS='--weights_path=<some path>'
