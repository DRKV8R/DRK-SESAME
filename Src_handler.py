import os, base64, tempfile, logging, runpod, torch, torchaudio, sys
sys.path.append('/app/csm_repo')

from generator import load_csm_1b, Segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL = None

def init_model(repo):
    global MODEL
    if MODEL: return MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CSM 1B from {repo} on {device}")
    MODEL = load_csm_1b(device=device, repo=repo)
    return MODEL

def prepare_context(audio_b64, text, speaker):
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
        fp.write(audio_bytes); path = fp.name
    wav,sr = torchaudio.load(path); os.unlink(path)
    if sr!=24000: wav=torchaudio.functional.resample(wav, sr, 24000)
    return Segment(text=text, speaker=speaker, audio=wav.squeeze(0))

def handler(event):
    inp = event.get('input',{})
    text = inp.get('text')
    if not text: return {"error":"text required"}
    
    # Environment variable support with input overrides
    repo    = inp.get('model_repo')            or os.getenv('MODEL_REPO','BiggestLab/csm-1b')
    speakID = inp.get('speaker_id',            int(os.getenv('DEFAULT_SPEAKER_ID',0)))
    maxms   = inp.get('max_length_ms',         int(os.getenv('MAX_LENGTH_MS',10000)))
    
    ctx=[]
    if inp.get('reference_audio') and inp.get('reference_text'):
        try:
            ctx.append(prepare_context(inp['reference_audio'], inp['reference_text'], speakID))
        except Exception as e:
            logger.warning(f"context prep failed: {e}")
    
    try:
        model = init_model(repo)
        audio = model.generate(text=text, speaker=speakID, context=ctx, max_audio_length_ms=maxms)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
            torchaudio.save(fp.name, audio.unsqueeze(0).cpu(), 24000)
            b64 = base64.b64encode(open(fp.name,'rb').read()).decode(); os.unlink(fp.name)
        return {"audio_base64":b64,"sample_rate":24000,"format":"wav","text":text,"speaker_id":speakID}
    except Exception as e:
        logger.error(e); return {"error":str(e)}

if __name__=="__main__":
    runpod.serverless.start({"handler":handler})
