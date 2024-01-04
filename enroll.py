import click
import librosa

import grpc
from messages.voice_search_server_pb2 import RecognitionConfig, EnrollRequest
from messages.voice_search_server_pb2_grpc import SpeechStub
 

def pcm(path):
    audio, _ = librosa.load(path, mono=True, sr=16000)
    return (audio * 32767).astype('<u2').tobytes()

@click.command()
@click.argument('address')
@click.argument('wav')
@click.argument('label')
def run(address, wav, label):
    with grpc.insecure_channel(address) as channel:
        stub = SpeechStub(channel)
        response = stub.Enroll(EnrollRequest(
            audio_content=pcm(wav),
            label=label
        ))

        print('request_id:', response.request_id)
        print('transcript:', response.transcript)
        print('label:', response.label)

if __name__ == '__main__':
    run()
