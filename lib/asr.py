import joblib
import numpy as np
import onnxruntime as rt

from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.matrix import Matrix


def create_asr(model_path):
    xlsr_model = load_xlsr_model(model_path)
    asr_model = load_asr_model(model_path)

    return ASR(xlsr_model, asr_model)

def load_xlsr_model(path):
    sess_opt = rt.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    xlsr_model = rt.InferenceSession(f'{path}/xls_r_300m_18.onnx', sess_opt)

    return xlsr_model

def load_asr_model(path):
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = 15
    decoder_opts.lattice_beam = 4
    decoder_opts.max_active = 7000
    decoder_opts.min_active = 200
    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = 1.0
    decodable_opts.frame_subsampling_factor = 1

    return NnetLatticeFasterRecognizer.from_files(
        f'{path}/final.mdl', f'{path}/HCLG.fst', f'{path}/words.txt',
        decoder_opts=decoder_opts, decodable_opts=decodable_opts)

class ASR:

    def __init__(self, xlsr_model, asr_model):
        self.xlsr_model = xlsr_model
        self.asr_model = asr_model

    def transcribe(self, wav):
        if len(wav) == 0:
            return ''

        wav = np.frombuffer(wav, dtype='int16') / 2**15
        feats = self.extract_features(wav)
        return self.decode(feats)

    def extract_features(self, wav):
        chunks = []
        chunk_length = 30
        for start in range(0, wav.shape[0], chunk_length * 16000):
            chunk = wav[start:start + chunk_length * 16000].astype('float32')
            chunks.append(self.xlsr_model.run(['output'], {'input': chunk.reshape((1, -1))})[0])

        return np.concatenate(chunks, axis=1)[0]

    def decode(self, feats):
        output = self.asr_model.decode(Matrix(feats))['text']
        output = output.replace('oU','o u').replace('aI', 'a i').replace('eI', 'e i')
        return output
