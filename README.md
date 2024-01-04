# Voice Search Server

This project allows you to set up a voice search server, add items to the database along with their speech descriptions, and query the items using voice queries.

## Installation

Follow these steps to set up the voice search server:

1. Create a conda environment:
    ```bash
    conda create -y --name voice-search-server python=3.7
    conda activate voice-search-server
    ```

2. Install required packages:
    ```bash
    conda install -y -c pykaldi pykaldi
    conda install -y -c conda-forge scikit-learn onnx onnxruntime grpcio
    pip3 install click
    ```

## Running the Server

Use the following command to start the voice search server:

```bash
api_port=8080
num_workers=10
model_dir=model
data_dir=data
server.py $api_port $num_workers $model_dir $data_dir
```

## Adding Items to Database

To add a new item to the database along with its speech description, use the following command:

```bash
python3 enroll.py localhost:8080 description.wav label
```

## Querying

To run a query with a new recording, use the following command:

```bash
python3 classify.py localhost:8080 query.wav
```
