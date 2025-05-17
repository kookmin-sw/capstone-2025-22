import subprocess
from io import BytesIO
import librosa

def decode_audio_to_wav(audio_buffer: BytesIO) -> BytesIO:
    """ffmpeg를 사용해 어떤 포맷이든 WAV로 변환"""
    process = subprocess.Popen(
        ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL  # 오류 무시 or 로깅
    )
    wav_bytes, _ = process.communicate(input=audio_buffer.getvalue())
    return BytesIO(wav_bytes)