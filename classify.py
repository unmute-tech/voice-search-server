import click
import wave

import grpc
from messages.voice_search_server_pb2 import RecognitionConfig, StreamingRecognizeRequest
from messages.voice_search_server_pb2_grpc import SpeechStub
 

def chunks(wav):
    while True:
        chunk = wav.readframes(16384)
        if len(chunk) == 0:
            break
        yield chunk

def requests(path):
    wav = wave.open(path, "rb")
    sample_rate = wav.getframerate()
    config = RecognitionConfig()

    yield StreamingRecognizeRequest(streaming_config=config)
    for chunk in chunks(wav):
        yield StreamingRecognizeRequest(audio_content=chunk) 


@click.command()
@click.argument('address')
@click.argument('wav')
def run(address, wav):
    with grpc.insecure_channel(address) as channel:
        stub = SpeechStub(channel)
        for response in stub.StreamingRecognize(requests(wav)):
            print(response)

if __name__ == '__main__':
    run()
