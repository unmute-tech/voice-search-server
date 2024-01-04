import click
import librosa

import grpc
from messages.voice_search_server_pb2 import RecognitionConfig, StreamingRecognizeRequest
from messages.voice_search_server_pb2_grpc import SpeechStub


def requests(path):
    audio, _ = librosa.load(path, mono=True, sr=16000)
    pcm = (audio * 32767).astype('<u2').tobytes()
    yield StreamingRecognizeRequest(audio_content=pcm)


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
