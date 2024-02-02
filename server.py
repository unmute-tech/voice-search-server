import click
from concurrent import futures
import datetime
import grpc
import json
import multiprocessing
import time
import uuid
import wave

from lib import create_asr, create_classifier

from messages.voice_search_server_pb2 import SpeechRecognitionAlternative, StreamingRecognitionResult, StreamingRecognizeResponse, EnrollResponse, RetrainResponse
import messages.voice_search_server_pb2_grpc as voice_search_server_pb2_grpc

_ONE_DAY = datetime.timedelta(days=1)


class SpeechServicer(voice_search_server_pb2_grpc.SpeechServicer):

    def __init__(self, asr, classifier, data_dir):
        self.asr = asr
        self.classifier = classifier
        self.data_dir = data_dir

    def StreamingRecognize(self, requests, context):
        request_id = str(uuid.uuid4().int)
        pcm = b""
        for request in requests:
            pcm += request.audio_content

        transcript = self.asr.transcribe(pcm)
        predictions = self.classifier.classify(transcript)

        yield StreamingRecognizeResponse(results=[
            StreamingRecognitionResult(alternatives=[
                SpeechRecognitionAlternative(transcript=label, confidence=confidence) for confidence, label in predictions
            ])
        ])

        self.saveWav(request_id, pcm)
        self.saveTranscript(request_id, transcript)
        self.savePredictions(request_id, predictions)

    def Enroll(self, request, context):
        pcm = request.audio_content
        label = request.label
        request_id = str(uuid.uuid4().int)

        transcript = self.asr.transcribe(pcm)

        self.saveWav(request_id, pcm)
        self.saveTranscript(request_id, transcript)
        self.saveLabel(request_id, label)

        return EnrollResponse(
            request_id=request_id,
            transcript=transcript,
            label=label
        )

    def Retrain(self, request, context):
        self.classifier.train(self.data_dir)

        return RetrainResponse(
            message='Model trained successfully'
        )

    def saveWav(self, request_id, pcm):
        wav = wave.open(f'{self.data_dir}/{request_id}.wav', 'w')
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
        wav.close()

    def saveTranscript(self, request_id, transcript):
        with open(f'{self.data_dir}/{request_id}.txt', 'w') as f:
            print(transcript, file=f)

    def saveLabel(self, request_id, label):
        with open(f'{self.data_dir}/{request_id}.label', 'w') as f:
            print(label, file=f)

    def savePredictions(self, request_id, predictions):
        with open(f'{self.data_dir}/{request_id}.json', 'w') as f:
            json.dump(predictions, f)

def _run_server(api_port, model_dir, data_dir):
    asr = create_asr(model_dir)
    classifier = create_classifier(model_dir)
    servicer = SpeechServicer(asr, classifier, data_dir)

    options = (("grpc.so_reuseport", 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    voice_search_server_pb2_grpc.add_SpeechServicer_to_server(servicer, server)

    server.add_insecure_port(f'[::]:{api_port}')
    server.start()
    _wait_forever(server)

def _wait_forever(server):
    try:
        while True:
            time.sleep(_ONE_DAY.total_seconds())
    except KeyboardInterrupt:
        server.stop(None)

@click.command()
@click.argument('api_port')
@click.argument('num_workers')
@click.argument('model_dir')
@click.argument('data_dir')
def run_server(api_port, num_workers, model_dir, data_dir):
    workers = []
    for _ in range(int(num_workers)):
        worker = multiprocessing.Process(
            target=_run_server, args=(api_port, model_dir, data_dir,)
        )
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()

if __name__ == '__main__':
    run_server()
