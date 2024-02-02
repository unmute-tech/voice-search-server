import click

import grpc
from messages.voice_search_server_pb2 import RetrainRequest
from messages.voice_search_server_pb2_grpc import SpeechStub

@click.command()
@click.argument('address')
def run(address):
    with grpc.insecure_channel(address) as channel:
        stub = SpeechStub(channel)
        response = stub.Retrain(RetrainRequest())

        print(response.message)

if __name__ == '__main__':
    run()
