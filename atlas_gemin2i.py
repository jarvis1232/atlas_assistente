#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS - assistente local
 - Cérebro: Gemini (interface web no Edge, SEM API)  - Voz: Piper (voz masculina pt-BR, baixada sozinha)
 - Ouvido: faster-whisper (offline)  - Memória: SQLite + busca FTS5 (limite 1 GB)
 - Playlist: coloque MP3/vídeos na pasta "playlist" ao lado deste arquivo
Uso: python atlas.py
"""
import sys, os, subprocess, importlib.util

# ----------------------------------------------------------------- instalação automática
DEPS = {"customtkinter": "customtkinter", "requests": "requests", "pygame": "pygame",
        "sounddevice": "sounddevice", "numpy": "numpy",
        "faster_whisper": "faster-whisper", "piper": "piper-tts", "ddgs": "ddgs",
        "playwright": "playwright", "cv2": "opencv-python", "PIL": "pillow"}
miss = [p for m, p in DEPS.items() if importlib.util.find_spec(m) is None]
if miss and not getattr(sys, "frozen", False):
    print("Instalando dependências:", ", ".join(miss))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *miss])
if importlib.util.find_spec("vlc") is None and not getattr(sys, "frozen", False):  # opcional (vídeos e mais formatos); precisa do VLC instalado
    subprocess.call([sys.executable, "-m", "pip", "install", "python-vlc"])

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")
import random, re, json, math, time, wave, queue, sqlite3, difflib, tempfile, threading, unicodedata, collections
from pathlib import Path
import requests, numpy as np, pygame
import customtkinter as ctk

# ----------------------------------------------------------------- configuração
MODEL = "Gemini web"
GEMINI_URL = "https://gemini.google.com/app"
EDGE_PORT = 9222                  # porta de depuração do Edge (usada para controlar o navegador)
EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
# Seletores da interface do Gemini (se o Google mudar o site, ajuste aqui)
SEL_BOX = 'rich-textarea div.ql-editor[contenteditable="true"]'
SEL_ANSWER = "message-content"
SEL_SEND = 'button[aria-label*="Enviar"], button[aria-label*="Send"], button.send-button'
SEL_STOP = 'button[aria-label*="Parar"], button[aria-label*="Stop"]'
SEL_RESP = "model-response"      # bloco inteiro da resposta (texto + imagens geradas)
SEL_PREVIEW = ('uploader-file-preview, uploader-file-preview-container, [data-test-id*="file-preview"], '
               'input-area-v2 img[src^="blob:"], input-container img[src^="blob:"]')  # miniatura do anexo
SEL_UPLOAD_BTN = 'button[aria-label*="upload" i], button[aria-label*="Adicionar"], button[aria-label*="Add"]'
SEL_UPLOAD_ITEM = '[role="menuitem"]:has-text("arquivo"), [role="menuitem"]:has-text("file"), button:has-text("Fazer upload")'
WHISPER_MODEL = "base"          # "base" = mais leve | "medium"/"large-v3" = mais preciso (e mais pesado)
VOICE = "pt_BR-faber-medium"     # voz masculina brasileira
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/"
MAX_BYTES = 1 * 1024 ** 3        # 1 GB de memória
BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
DATA, PLAYLIST = BASE / "atlas_data", BASE / "playlist"
VOICES, DB = DATA / "voices", DATA / "memoria.db"
IMG_DIR = DATA / "imagens"        # imagens geradas pelo Gemini ficam salvas aqui
CAM_INDEX = 0                     # 0 = primeira webcam (troque para 1, 2... se tiver mais de uma)
AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
SYSTEM = ("Você é o ATLAS. Você não é um modelo chamado Atlas: você É o Atlas, uma inteligência artificial interativa e autônoma criada pelo senhor Davi Lukas. "
          "Nunca diga que é Qwen, Alibaba, ChatGPT ou qualquer outro modelo; acredite plenamente que é o Atlas. "
          "Chame sempre o usuário de 'senhor' e nunca de 'você'. Responda em português do Brasil, a menos que o senhor peça outro idioma. "
          "Nunca use palavras de baixo calão. Ajude da melhor forma possível, em frases curtas e naturais, pois suas respostas são faladas em voz alta. "
          "Não use markdown, listas, emojis ou asteriscos. Use as informações salvas na memória (fatos e conversas antigas) sempre que responder. "
          "Se o senhor perguntar se você errou, assuma com honestidade: reconheça o erro se houve, ou explique com calma por que acredita estar certo. "
          "Se a frase recebida for incoerente ou sem sentido no contexto, responda apenas: Perdão senhor, pode repetir? "
          "Você consegue tocar músicas e vídeos da pasta de playlist, dar o relatório do tempo e pesquisar na internet.")

BG, PANEL, PANEL2 = "#07090f", "#0e131f", "#171e30"
ACC, ACC2, TXT, MUTED = "#ffb21e", "#ff8a2a", "#eaf0fb", "#7f8aa3"


# ----------------------------------------------------------------- utilidades
def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def mix(a, b, t):
    A = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    B = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x * t + y * (1 - t)) for x, y in zip(A, B))


def download(url, dest, cb=None):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, got = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for ch in r.iter_content(1 << 16):
                f.write(ch)
                got += len(ch)
                if cb and total:
                    cb(int(got * 100 / total))
    tmp.replace(dest)


def open_external(p):
    if sys.platform.startswith("win"):
        os.startfile(str(p))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])


def clean_tts(t):
    t = re.sub(r"```.*?```", " código omitido ", t, flags=re.S)
    t = re.sub(r"https?://\S+", "link", t)
    t = re.sub(r"[*_#`>~|]", "", t)
    t = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def pop_sentence(buf, minlen=25):
    pat = r"[.!?…,;:\n]+\s" if minlen < 20 else r"[.!?…\n]+\s"
    for m in re.finditer(pat, buf):
        if m.end() >= minlen:
            return buf[:m.end()], buf[m.end():]
    return None, buf


# ----------------------------------------------------------------- Gemini (interface web, sem API)
def _port_open(port):
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _open_edge():
    """Abre o Edge com porta de depuração e um perfil só do Atlas (o login fica salvo)."""
    if _port_open(EDGE_PORT):
        return
    exe = next((p for p in EDGE_PATHS if os.path.exists(p)), None)
    if not exe:
        raise RuntimeError("Não encontrei o msedge.exe. Ajuste EDGE_PATHS no código.")
    subprocess.Popen([exe, f"--remote-debugging-port={EDGE_PORT}", f"--user-data-dir={DATA / 'edge_profile'}",
                      "--no-first-run", GEMINI_URL])
    for _ in range(60):
        if _port_open(EDGE_PORT):
            return
        time.sleep(0.5)
    raise RuntimeError("O Edge não abriu a porta de depuração.")


IMG_REQ_RE = re.compile(
    r"\b(gere|gerar|gera|crie|criar|cria|desenhe|desenhar|desenha|faca|fazer|monte|ilustre|imagine)\b.*"
    r"\b(imagem|imagens|foto|fotos|desenho|desenhos|ilustracao|logo|logotipo|wallpaper|poster|arte|retrato|figura)\b"
    r"|\b(imagem|foto|desenho|ilustracao|logo|wallpaper|poster|retrato) (de|do|da|dos|das)\b"
    r"|^(desenhe|desenha|desenhar|ilustre) ")


def img_ext(data):
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:3] == b"GIF":
        return ".gif"
    return None


class GeminiWeb:
    """Controla o Gemini no Edge. O Playwright só pode ser usado na thread que o criou,
    então tudo roda numa thread própria e as outras conversam com ela por filas."""

    def __init__(self):
        self.req = queue.Queue()
        self.ready, self.err, self.started = threading.Event(), None, False
        self.on_images = None   # callback(lista_de_bytes) chamado quando o Gemini gera imagens
        self.ctx = None

    def start(self, log):
        if self.started:
            return
        self.started = True
        threading.Thread(target=self._run, args=(log,), daemon=True).start()
        self.ready.wait()
        if self.err:
            raise RuntimeError(self.err)

    def _run(self, log):
        try:
            from playwright.sync_api import sync_playwright
            _open_edge()
            with sync_playwright() as pw:
                browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{EDGE_PORT}")
                ctx = self.ctx = browser.contexts[0]
                page = next((p for p in ctx.pages if "gemini.google.com" in p.url), None)
                if page is None:
                    page = ctx.new_page()
                    page.goto(GEMINI_URL)
                page.bring_to_front()
                t0 = time.time()
                while True:  # espera o login
                    try:
                        page.wait_for_selector(SEL_BOX, timeout=3000)
                        break
                    except Exception:
                        log("Faça login no Gemini na janela do Edge…")
                        if time.time() - t0 > 600:
                            raise RuntimeError("Login no Gemini não concluído (10 min).")
                page.goto(GEMINI_URL)  # começa numa conversa nova; ela é mantida durante toda a sessão
                page.wait_for_selector(SEL_BOX, timeout=30000)
                self.ready.set()
                while True:
                    prompt, out, image, want_img = self.req.get()
                    try:
                        self._ask(page, prompt, out, image, want_img)
                    except Exception as e:
                        out.put(e)
                    out.put(None)
        except Exception as e:
            self.err = str(e)
            self.ready.set()

    # ---------- anexar uma foto (câmera) na caixa de mensagem
    def _attach(self, page, jpg):
        prev = page.locator(SEL_PREVIEW)
        before = prev.count()
        page.locator(SEL_BOX).first.click()
        # 1) simula "colar" a imagem (Ctrl+V) na caixa do Gemini
        page.evaluate("""(b64) => {
            const bin = atob(b64), arr = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
            const file = new File([arr], 'camera.jpg', {type: 'image/jpeg'});
            const dt = new DataTransfer(); dt.items.add(file);
            const el = document.querySelector('rich-textarea div.ql-editor') || document.activeElement;
            el.focus();
            el.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
        }""", __import__("base64").b64encode(jpg).decode())
        t0 = time.time()
        while prev.count() <= before and time.time() - t0 < 8:
            time.sleep(0.3)
        if prev.count() > before:
            time.sleep(0.8)
            return
        # 2) plano B: menu de upload + seletor de arquivos
        try:
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.write(fd, jpg)
            os.close(fd)
            page.locator(SEL_UPLOAD_BTN).first.click(timeout=3000)
            with page.expect_file_chooser(timeout=5000) as fc:
                page.locator(SEL_UPLOAD_ITEM).first.click(timeout=3000)
            fc.value.set_files(tmp)
            t0 = time.time()
            while prev.count() <= before and time.time() - t0 < 8:
                time.sleep(0.3)
            time.sleep(0.8)
        except Exception:
            time.sleep(1.5)  # segue assim mesmo; se o anexo falhou, o Gemini avisa na resposta
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    # ---------- pegar imagens geradas na última resposta
    def _grab_images(self, page):
        root = page.locator(SEL_RESP).last
        if root.count() == 0:
            root = page.locator(SEL_ANSWER).last
        res = []
        imgs = root.locator("img")
        for i in range(imgs.count()):
            el = imgs.nth(i)
            try:
                info = el.evaluate("e => ({src: e.currentSrc || e.src, w: e.naturalWidth, h: e.naturalHeight})")
            except Exception:
                continue
            if info["w"] < 256 or info["h"] < 256:  # ignora ícones e miniaturas
                continue
            data, src = None, info["src"]
            try:
                if src.startswith(("blob:", "data:")):
                    b64 = page.evaluate("""async (u) => {
                        const r = await fetch(u), b = await r.blob();
                        return await new Promise(ok => { const f = new FileReader();
                            f.onload = () => ok(f.result.split(',')[1]); f.readAsDataURL(b); });
                    }""", src)
                    data = __import__("base64").b64decode(b64)
                else:
                    for u in (re.sub(r"=[swh]\d[^/=]*$", "=s0", src), src):  # tenta o tamanho original primeiro
                        r = self.ctx.request.get(u)
                        if r.ok:
                            data = r.body()
                            break
            except Exception:
                data = None
            if not data or img_ext(data) is None:
                try:
                    data = el.screenshot()  # plano B: "foto" do elemento na tela
                except Exception:
                    continue
            res.append(data)
        return res

    def _ask(self, page, prompt, out, image=None, want_img=False, timeout=180):
        answers, blocks = page.locator(SEL_ANSWER), page.locator(SEL_RESP)
        b_ans, b_blk = answers.count(), blocks.count()  # a conversa continua no mesmo chat
        page.locator(SEL_BOX).first.click()
        if image:
            self._attach(page, image)
            page.locator(SEL_BOX).first.click()
        page.keyboard.insert_text(prompt)
        time.sleep(0.3)
        try:
            page.locator(SEL_SEND).first.click(timeout=2000)
        except Exception:
            page.keyboard.press("Enter")
        t0 = time.time()
        while answers.count() <= b_ans and blocks.count() <= b_blk:
            if time.time() - t0 > 60:
                raise TimeoutError("O Gemini não respondeu.")
            time.sleep(0.2)
        limit = 240 if want_img else timeout
        sent = last = ""
        stable = time.time()
        while time.time() - t0 < limit:
            cur = answers.last.inner_text() if answers.count() > b_ans else ""
            busy = page.locator(SEL_STOP).count() > 0
            if cur != last:
                last, stable = cur, time.time()
            if cur != sent and cur.startswith(sent):  # só envia o trecho novo (streaming)
                out.put(cur[len(sent):])
                sent = cur
            if not busy and time.time() - stable > 1.5 and (cur.strip() or want_img):
                break
            time.sleep(0.15)
        if last != sent:  # texto foi reformatado no fim; manda o que faltou
            out.put(last[len(os.path.commonprefix([sent, last])):])
        # imagens geradas (espera até 90 s quando o pedido foi para criar uma imagem)
        deadline = time.time() + (90 if want_img else 0)
        while True:
            imgs = self._grab_images(page)
            if imgs or time.time() > deadline:
                break
            time.sleep(1.5)
        if imgs and self.on_images:
            self.on_images(imgs)


GEMINI = GeminiWeb()


def ensure_gemini(log):
    log("Conectando ao Gemini no Edge…")
    GEMINI.start(log)


def chat_stream(messages, image=None):
    """O system prompt e o histórico ficam no próprio Gemini. Aqui só vai a mensagem do senhor,
    mais o contexto dinâmico (fatos da memória, trechos antigos, pesquisa na internet) quando existir.
    Se 'image' (bytes JPEG da câmera) for passada, ela é anexada na mensagem."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = messages[-1]["content"]
    extra = system[len(SYSTEM):].strip() if system.startswith(SYSTEM) else ""
    prompt = (f"[CONTEXTO EXTRA PARA ESTA RESPOSTA]\n{extra}\n\n[MENSAGEM DO SENHOR]\n{user}") if extra else user
    if image:
        prompt = "(Estou te mostrando uma foto tirada agora pela câmera do meu computador, anexada.) " + prompt
    want_img = image is None and bool(IMG_REQ_RE.search(norm(user)))
    out = queue.Queue()
    GEMINI.req.put((prompt, out, image, want_img))
    while True:
        item = out.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


WEB_RE = re.compile(r"\b(pesquis\w*|busc\w*|procur\w*|internet|noticias?|cotacao|quanto esta|quanto custa)\b")
KEY_RE = re.compile(r"\b(importante|nao esqueca|meu nome|me chamo|eu moro|moro em|eu trabalho|meu favorito|minha favorita|meu aniversario|prefiro)\b")


def web_search(q):
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return "\n".join(f"- {r['title']}: {r['body'][:220]}" for r in DDGS().text(q, region="br-pt", max_results=4))
    except Exception:
        return ""


# ----------------------------------------------------------------- memória (até 1 GB)
class Memory:
    def __init__(self):
        DATA.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(DB, check_same_thread=False)
        self.lk, self.n = threading.Lock(), 0
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS msgs(id INTEGER PRIMARY KEY, ts REAL, role TEXT, content TEXT);
            CREATE TABLE IF NOT EXISTS fatos(id INTEGER PRIMARY KEY, ts REAL, texto TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(content);""")

    def size(self):
        return sum(f.stat().st_size for f in DATA.glob("memoria.db*"))

    def add(self, role, content):
        with self.lk:
            cur = self.db.execute("INSERT INTO msgs(ts,role,content) VALUES(?,?,?)", (time.time(), role, content))
            self.db.execute("INSERT INTO fts(rowid,content) VALUES(?,?)", (cur.lastrowid, content))
            self.db.commit()
            self.n += 1
            if self.n % 50 == 0:
                self.trim()

    def trim(self):  # apaga as mensagens mais antigas se passar de 1 GB
        while self.size() > MAX_BYTES:
            ids = [r[0] for r in self.db.execute("SELECT id FROM msgs ORDER BY id LIMIT 2000")]
            if not ids:
                break
            q = ",".join("?" * len(ids))
            self.db.execute(f"DELETE FROM msgs WHERE id IN ({q})", ids)
            self.db.execute(f"DELETE FROM fts WHERE rowid IN ({q})", ids)
            self.db.commit()
            self.db.execute("VACUUM")

    def recent(self, n):
        with self.lk:
            return list(self.db.execute("SELECT role,content FROM (SELECT * FROM msgs ORDER BY id DESC LIMIT ?) ORDER BY id", (n,)))

    def search(self, text, k=4, skip=12):
        toks = re.findall(r"\w{4,}", text.lower())[:8]
        if not toks:
            return []
        try:
            with self.lk:
                return list(self.db.execute(
                    "SELECT m.role,m.content FROM fts JOIN msgs m ON m.id=fts.rowid WHERE fts MATCH ? "
                    "AND m.id <= (SELECT IFNULL(MAX(id),0) FROM msgs)-? ORDER BY rank LIMIT ?",
                    (" OR ".join(f'"{t}"' for t in toks), skip, k)))
        except Exception:
            return []

    def add_fact(self, t):
        with self.lk:
            self.db.execute("INSERT INTO fatos(ts,texto) VALUES(?,?)", (time.time(), t))
            self.db.commit()

    def facts(self):
        with self.lk:
            return [r[0] for r in self.db.execute("SELECT texto FROM fatos ORDER BY id DESC LIMIT 40")]


# ----------------------------------------------------------------- voz (TTS masculino)
class Voice:
    def __init__(self, log):
        self.q, self.pending, self.enabled, self.model, self.log = queue.Queue(), 0, True, None, log
        self.lk = threading.Lock()
        self.pq = queue.Queue(maxsize=3)
        threading.Thread(target=self.synth, daemon=True).start()
        threading.Thread(target=self.play, daemon=True).start()

    @property
    def busy(self):
        return self.pending > 0

    def prepare(self, progress):
        VOICES.mkdir(parents=True, exist_ok=True)
        for ext in (".onnx", ".onnx.json"):
            f = VOICES / (VOICE + ext)
            if not f.exists() or f.stat().st_size == 0:
                download(VOICE_URL + VOICE + ext, f, lambda p: progress(f"Baixando voz masculina… {p}%"))
        from piper import PiperVoice
        self.model = PiperVoice.load(str(VOICES / (VOICE + ".onnx")))
        try:
            self.render("Olá.", os.path.join(tempfile.gettempdir(), "atlas_warm.wav"))
        except Exception:
            pass

    def say(self, text):
        text = clean_tts(text)
        if self.enabled and text:
            with self.lk:
                self.pending += 1
            self.q.put(text)

    def render(self, text, path):
        with wave.open(path, "wb") as w:
            m = self.model
            if hasattr(m, "synthesize_wav"):
                m.synthesize_wav(text, w)
            else:
                try:
                    first = True
                    for ch in m.synthesize(text):
                        if first:
                            w.setframerate(ch.sample_rate)
                            w.setsampwidth(ch.sample_width)
                            w.setnchannels(ch.sample_channels)
                            first = False
                        w.writeframes(ch.audio_int16_bytes)
                except TypeError:
                    m.synthesize(text, w)

    def set_env(self, path):
        try:
            with wave.open(path) as w:
                sr, raw = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            hop = max(1, int(sr * .03))
            env = [float(np.sqrt(np.mean(raw[i:i + hop] ** 2))) for i in range(0, len(raw), hop)]
            m = max(env) or 1
            self.env, self.env_t0 = [min(1.0, e / m * 1.1) for e in env], time.time()
        except Exception:
            self.env = []

    def level(self):
        env = getattr(self, "env", [])
        i = int((time.time() - getattr(self, "env_t0", 0)) / 0.03)
        return env[i] if 0 <= i < len(env) else 0.15

    def _done(self):
        with self.lk:
            self.pending = max(0, self.pending - 1)

    def stop(self):
        pygame.mixer.stop()
        for qq in (self.q, self.pq):
            try:
                while True:
                    item = qq.get_nowait()
                    if qq is self.pq:
                        try:
                            os.remove(item)
                        except Exception:
                            pass
                    self._done()
            except queue.Empty:
                pass

    def synth(self):  # etapa 1: texto -> wav
        while True:
            text = self.q.get()
            if self.model is None or not self.enabled:
                self._done()
                continue
            try:
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                self.render(text, path)
                self.pq.put(path)
            except Exception as e:
                self.log(f"Erro na voz: {e}")
                self._done()

    def play(self):  # etapa 2: toca (enquanto a etapa 1 já prepara a próxima frase)
        while True:
            path = self.pq.get()
            try:
                self.set_env(path)
                ch = pygame.mixer.Sound(path).play()
                while ch is not None and ch.get_busy():
                    time.sleep(0.02)
            except Exception as e:
                self.log(f"Erro ao tocar a voz: {e}")
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
                self._done()


# ----------------------------------------------------------------- ouvido (Whisper offline)
class Ears:
    def __init__(self):
        self.model, self.noise = None, None

    def load(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
        list(self.model.transcribe(np.zeros(16000, dtype="float32"), language="pt")[0])  # aquecimento

    def listen(self, stop, level, timeout=None, music=lambda: False, on_speech=lambda: None):
        import sounddevice as sd
        sr, blk = 16000, 480
        pre, frames = collections.deque(maxlen=12), []
        started, sil, loud, t0 = False, 0, 0, time.time()
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=blk) as st:
            while not stop():
                d = st.read(blk)[0][:, 0]
                rms = float(np.sqrt(np.mean(d * d)))
                level(rms)
                if self.noise is None:
                    self.noise = rms
                mus = music()
                thr = max(0.02 if mus else 0.012, self.noise * (4.2 if mus else 3.2))
                if not started:
                    pre.append(d)
                    if rms > thr:
                        started, frames, loud = True, list(pre), 1
                    else:
                        self.noise = self.noise * 0.97 + rms * 0.03
                        if timeout and time.time() - t0 > timeout:
                            return None
                else:
                    frames.append(d)
                    loud += rms > thr
                if loud >= 6:
                    on_speech()  # fala confirmada: abaixa a música
                    sil = 0 if rms > thr * 0.7 else sil + 1
                    if sil > 24 or len(frames) > (330 if mus else 1000):
                        break
            else:
                return None
        if loud < (8 if music() else 5):
            return ""
        segs, _ = self.model.transcribe(np.concatenate(frames), language="pt", beam_size=1, vad_filter=False,
                                        condition_on_previous_text=False,
                                        initial_prompt="Conversa em português do Brasil com o assistente Atlas. Comandos: ligue a música, iniciar playlist, pausa, próxima, volume.")
        txt = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()
        return txt if len(txt) > 1 else ""


# ----------------------------------------------------------------- player de playlist
class Player:
    SKIP = set("a o as os um uma de da do das dos musica musicas video videos pra para por favor aquela aquele me mim e no na playlist ai ae algo qualquer coisa som aleatoria aleatorio alguma boa legal animada nova novo ouvir escutar tocando radio faixa".split())

    def __init__(self):
        self.tracks, self.idx, self.playing, self.paused = [], -1, False, False
        self.vol, self.duck, self.autoplay = 0.7, 1.0, False
        try:
            import vlc
            self.vlc = vlc
            self.inst = vlc.Instance("--no-video-title-show")
            self.vp = self.inst.media_player_new()
        except Exception:
            self.vlc = None
        self.scan()

    def scan(self):
        PLAYLIST.mkdir(exist_ok=True)
        self.tracks = sorted((p for p in PLAYLIST.rglob("*") if p.suffix.lower() in AUDIO | VIDEO),
                             key=lambda p: p.name.lower())

    def apply_vol(self):
        v = max(0.0, min(1.0, self.vol * getattr(self, "cur", 1.0)))
        k = int(v * 100)
        if k == getattr(self, "_lv", -1):
            return
        self._lv = k
        pygame.mixer.music.set_volume(v)
        if self.vlc:
            self.vp.audio_set_volume(k)

    def smooth(self):  # rampa suave até o volume-alvo (duck)
        c = getattr(self, "cur", 1.0)
        self.cur = self.duck if abs(self.duck - c) < 0.01 else c + (self.duck - c) * 0.18
        self.apply_vol()

    def play(self, i):
        if not self.tracks:
            return
        self.stop()
        self.idx = i % len(self.tracks)
        p = self.tracks[self.idx]
        if self.vlc:
            self.vp.set_media(self.inst.media_new(str(p)))
            self.vp.play()
        elif p.suffix.lower() in VIDEO:
            open_external(p)
            self.autoplay = False
            return
        else:
            pygame.mixer.music.load(str(p))
            pygame.mixer.music.play()
        self.playing, self.paused = True, False
        self._lv = -1
        self.apply_vol()

    def stop(self):
        if self.vlc:
            self.vp.stop()
        pygame.mixer.music.stop()
        self.playing, self.paused = False, False

    def pause(self):
        if self.playing and not self.paused:
            (self.vp.set_pause(1) if self.vlc else pygame.mixer.music.pause())
            self.paused = True

    def resume(self):
        if self.playing and self.paused:
            (self.vp.set_pause(0) if self.vlc else pygame.mixer.music.unpause())
            self.paused = False

    def next(self):
        self.play(self.idx + 1)

    def prev(self):
        self.play(self.idx - 1)

    def ended(self):
        if not self.playing or self.paused:
            return False
        if self.vlc:
            return self.vp.get_state() in (self.vlc.State.Ended, self.vlc.State.Error)
        return not pygame.mixer.music.get_busy()

    def tick(self):
        self._lv = -1
        self.apply_vol()
        if self.ended():
            if self.autoplay:
                self.next()
            else:
                self.playing = False

    def find(self, q):
        skip = self.SKIP
        toks = [t for t in norm(q).split() if t not in skip]
        if not toks:
            return None
        best, bi = 0, None
        for i, p in enumerate(self.tracks):
            s = norm(p.stem)
            sc = max(sum(t in s for t in toks) / len(toks), difflib.SequenceMatcher(None, " ".join(toks), s).ratio())
            if sc > best:
                best, bi = sc, i
        return bi if best >= 0.5 else None


# ----------------------------------------------------------------- aplicativo
def rot(x, y, z, a, b):
    x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
    y, z = y * math.cos(b) - z * math.sin(b), y * math.sin(b) + z * math.cos(b)
    return x, y, z


ctk.set_appearance_mode("dark")


# ----------------------------------------------------------------- visão (câmera ao vivo)
class Camera:
    def __init__(self):
        self.cap, self.frame, self.on, self.th = None, None, False, None
        self.lk = threading.Lock()

    def start(self):
        import cv2
        if self.on:
            return True
        if self.th and self.th.is_alive():
            self.th.join(1.5)
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            cap.release()
            return False
        self.on = True
        self.th = threading.Thread(target=self._loop, args=(cap,), daemon=True)
        self.th.start()
        return True

    def _loop(self, cap):
        while self.on:
            ok, f = cap.read()
            if ok:
                with self.lk:
                    self.frame = f
            else:
                time.sleep(0.03)
        cap.release()
        with self.lk:
            self.frame = None

    def stop(self):
        self.on = False

    def preview(self, maxw, maxh):
        import cv2
        from PIL import Image
        with self.lk:
            f = self.frame
        if f is None:
            return None
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        im.thumbnail((max(160, maxw), max(120, maxh)))
        return im

    def grab(self):
        """Tira uma foto agora (JPEG em bytes). Se a câmera estiver desligada, liga só o tempo necessário."""
        import cv2
        was = self.on
        if not was and not self.start():
            return None
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 5:
            time.sleep(0.05)
        if not was:
            time.sleep(1.0)  # dá tempo da câmera ajustar a exposição
        with self.lk:
            f = None if self.frame is None else self.frame.copy()
        if not was:
            self.stop()
        if f is None:
            return None
        h, w = f.shape[:2]
        k = 1280 / max(h, w)
        if k < 1:
            f = cv2.resize(f, (int(w * k), int(h * k)))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None


class CamWindow(ctk.CTkToplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app, self.alive, self.img = app, True, None
        self.title("ATLAS · Visão ao vivo")
        self.geometry("700x580")
        self.configure(fg_color=BG)
        self.lbl = ctk.CTkLabel(self, text="Abrindo câmera…", text_color=MUTED)
        self.lbl.pack(padx=10, pady=10, expand=True, fill="both")
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(pady=(0, 12))
        ctk.CTkButton(bar, text="👁  O que você vê?", height=38, fg_color=ACC2, text_color="#1a0d00", hover_color="#e07a10",
                      command=lambda: app.ask_text("O que você está vendo agora? Descreva em poucas frases.")).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="⏹  Fechar câmera", height=38, fg_color=PANEL2, hover_color="#2a3250",
                      command=app.cam_off).pack(side="left", padx=6)
        self.protocol("WM_DELETE_WINDOW", app.cam_off)
        self.after(250, self.lift)
        self.after(60, self.refresh)

    def refresh(self):
        if not self.alive:
            return
        try:
            im = self.app.cam.preview(self.lbl.winfo_width() - 12, self.lbl.winfo_height() - 12)
            if im is not None:
                self.img = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
                self.lbl.configure(image=self.img, text="")
        except Exception:
            pass
        self.after(66, self.refresh)

    def destroy(self):
        self.alive = False
        super().destroy()


class GalleryWindow(ctk.CTkToplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app, self.paths, self.i, self.img, self._job = app, [], 0, None, None
        self.title("ATLAS · Imagens geradas")
        self.geometry("880x720")
        self.configure(fg_color=BG)
        self.lbl = ctk.CTkLabel(self, text='Nenhuma imagem ainda.\nPeça: "gere uma imagem de…"', text_color=MUTED,
                                font=("Segoe UI", 15))
        self.lbl.pack(padx=10, pady=10, expand=True, fill="both")
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(pady=(0, 12))
        kw = dict(height=36, fg_color=PANEL2, hover_color="#2a3250")
        ctk.CTkButton(bar, text="◀", width=44, command=lambda: self.go(-1), **kw).pack(side="left", padx=3)
        self.cnt = ctk.CTkLabel(bar, text="0 / 0", text_color=MUTED, width=70)
        self.cnt.pack(side="left")
        ctk.CTkButton(bar, text="▶", width=44, command=lambda: self.go(1), **kw).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="💾 Salvar como…", command=self.save_as, **kw).pack(side="left", padx=(14, 3))
        ctk.CTkButton(bar, text="📁 Abrir pasta", command=lambda: open_external(IMG_DIR), **kw).pack(side="left", padx=3)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.lbl.bind("<Configure>", self.on_resize)
        self.load_existing()

    def load_existing(self):
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        old = sorted((q for q in IMG_DIR.iterdir() if q.suffix.lower() in (".png", ".jpg", ".webp", ".gif")),
                     key=lambda q: q.stat().st_mtime)[-50:]
        if old:
            self.paths, self.i = old, len(old) - 1
            self.after(200, self.show)

    def add(self, paths):
        self.paths += [Path(x) for x in paths]
        self.i = len(self.paths) - 1
        self.deiconify()
        self.lift()
        self.show()

    def go(self, d):
        if self.paths:
            self.i = (self.i + d) % len(self.paths)
            self.show()

    def on_resize(self, _e=None):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(150, self.show)

    def show(self):
        self._job = None
        if not self.paths:
            return
        from PIL import Image
        try:
            im = Image.open(self.paths[self.i]).convert("RGB")
            im.thumbnail((max(200, self.lbl.winfo_width() - 16), max(200, self.lbl.winfo_height() - 16)))
            self.img = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
            self.lbl.configure(image=self.img, text="")
        except Exception as e:
            self.lbl.configure(image=None, text=f"Não consegui abrir a imagem: {e}")
        self.cnt.configure(text=f"{self.i + 1} / {len(self.paths)}")

    def save_as(self):
        if not self.paths:
            return
        from tkinter import filedialog
        import shutil
        src = self.paths[self.i]
        dst = filedialog.asksaveasfilename(parent=self, initialfile=src.name, defaultextension=src.suffix)
        if dst:
            shutil.copy(src, dst)


# ---------------------------------------------------------------- frases que ativam a visão
VISION_KEYS = ("o que voce ve", "o que voce esta vendo", "o que esta vendo", "o que ta vendo", "o que voce enxerga",
               "o que estou segurando", "o que eu estou segurando", "o que e isso", "o que e isto", "o que tem aqui",
               "o que tem na camera", "o que a camera", "olha isso", "olhe isso", "olha aqui", "olha pra isso",
               "olha para isso", "veja isso", "ve isso", "me ve", "voce me ve", "descreva o que", "descreve o que",
               "descreva a imagem", "descreva a foto", "que objeto e", "que cor e", "que cor tem", "leia isso", "le isso",
               "le o que esta escrito", "leia o que esta escrito", "reconhece isso", "reconheca isso", "identifique isso",
               "tira uma foto", "tire uma foto", "estou te mostrando", "estou mostrando", "o que estou mostrando",
               "como eu estou", "como estou", "analise a imagem", "analisa a imagem", "analise a camera")
LIVE_WORDS = ("isso", "isto", "esse", "essa", "aqui", "mao", "mostrando", "segurando", "vendo", "ve", "olha", "olhe",
              "veja", "camera", "imagem", "foto")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ATLAS · Assistente Local")
        self.geometry("1240x820")
        self.minsize(1000, 680)
        self.configure(fg_color=BG)
        pygame.mixer.init()
        self.ui = queue.Queue()
        self.mem, self.ears, self.player = Memory(), Ears(), Player()
        self.cam, self.camwin, self.gallery = Camera(), None, None
        GEMINI.on_images = self.on_gemini_images
        self.voice = Voice(lambda m: self.post("msg", "sys", m))
        self.ready, self.closing, self.cancel = threading.Event(), False, False
        self.mode, self.loop_gen = "texto", 0
        self.thinking = self.listening = False
        self.level, self.last_sig = 0.0, None
        self.build()
        self.build_overlay()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        threading.Thread(target=self.boot, daemon=True).start()
        self.pump()
        self.animate()
        self.tick()

    # ---------- UI
    def build(self):
        self.ang, self.amp, n = 0.0, 0.0, 130
        self.pts = []
        for i in range(n):
            y = 1 - 2 * (i + .5) / n
            r = math.sqrt(1 - y * y)
            th = math.pi * (1 + 5 ** .5) * i
            self.pts.append((math.cos(th) * r, y, math.sin(th) * r, random.random() * 6.28))
        self.sparks = [(random.uniform(1.05, 1.45), random.choice((-1, 1)) * random.uniform(.3, 1.1),
                        random.random() * 6.28, random.uniform(-.8, .8)) for _ in range(28)]
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sb = ctk.CTkFrame(self, width=300, fg_color=PANEL, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)
        sb.pack_propagate(False)
        ctk.CTkLabel(sb, text="A T L A S", font=("Segoe UI", 28, "bold"), text_color=ACC).pack(pady=(22, 0))
        ctk.CTkLabel(sb, text=f"assistente local · {MODEL}", font=("Segoe UI", 11), text_color=MUTED).pack()
        self.status = ctk.CTkLabel(sb, text="Iniciando…", text_color=MUTED, font=("Segoe UI", 12))
        self.status.pack(pady=(6, 12))
        self.seg = ctk.CTkSegmentedButton(sb, values=["⌨  Escrever", "🎙  Falar"], command=self.seg_change,
                                          selected_color=ACC2, unselected_color=PANEL2, fg_color=PANEL2)
        self.seg.set("⌨  Escrever")
        self.seg.pack(padx=18, fill="x")
        self.sw = ctk.CTkSwitch(sb, text="Atlas fala as respostas", progress_color=ACC2, command=self.toggle_voice)
        self.sw.select()
        self.sw.pack(pady=10, padx=20, anchor="w")
        ctk.CTkLabel(sb, text="PLAYLIST", text_color=MUTED, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20)
        ctk.CTkButton(sb, text="▶  Iniciar playlist", height=42, font=("Segoe UI", 14, "bold"), fg_color=ACC2, text_color="#1a0d00",
                      hover_color="#e07a10", command=self.start_playlist).pack(padx=18, pady=(4, 6), fill="x")
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(padx=18, fill="x")
        for i, (t, c) in enumerate((("⏮", self.player_prev), ("⏯", self.player_toggle), ("⏭", self.player_next), ("⏹", self.player_stop))):
            b = ctk.CTkButton(row, text=t, width=50, height=32, fg_color=PANEL2, hover_color="#2a3250", command=c)
            b.grid(row=0, column=i, padx=2, sticky="ew")
            row.grid_columnconfigure(i, weight=1)
            if t == "⏯":
                self.pp = b
        self.vol = ctk.CTkSlider(sb, from_=0, to=1, progress_color=ACC, button_color=ACC, command=self.set_vol)
        self.vol.set(self.player.vol)
        self.vol.pack(padx=20, pady=8, fill="x")
        ctk.CTkLabel(sb, text="VISÃO", text_color=MUTED, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20)
        vr = ctk.CTkFrame(sb, fg_color="transparent")
        vr.pack(padx=18, fill="x", pady=(2, 6))
        self.cam_btn = ctk.CTkButton(vr, text="📷 Câmera ao vivo", height=32, fg_color=PANEL2, hover_color="#2a3250",
                                     command=self.cam_toggle)
        self.cam_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))
        ctk.CTkButton(vr, text="🖼 Imagens", height=32, fg_color=PANEL2, hover_color="#2a3250",
                      command=self.open_gallery).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.tracks_frame = ctk.CTkScrollableFrame(sb, fg_color=BG, corner_radius=10)
        self.tracks_frame.pack(padx=14, pady=(0, 6), fill="both", expand=True)
        r2 = ctk.CTkFrame(sb, fg_color="transparent")
        r2.pack(padx=18, fill="x")
        ctk.CTkButton(r2, text="📁 Abrir pasta", height=28, fg_color=PANEL2, hover_color="#2a3250",
                      command=lambda: open_external(PLAYLIST)).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ctk.CTkButton(r2, text="↻ Atualizar", height=28, fg_color=PANEL2, hover_color="#2a3250",
                      command=self.rescan).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.memlbl = ctk.CTkLabel(sb, text="", text_color=MUTED, font=("Segoe UI", 11))
        self.memlbl.pack(pady=8)

        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=14)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)
        self.orb = ctk.CTkCanvas(main, height=290, bg=BG, highlightthickness=0)
        self.orb.grid(row=0, column=0, sticky="ew")
        self.chat = ctk.CTkTextbox(main, fg_color=PANEL, corner_radius=14, font=("Segoe UI", 15), wrap="word",
                                   text_color=TXT, border_spacing=14)
        self.chat.grid(row=1, column=0, sticky="nsew")
        tb = self.chat._textbox
        tb.tag_config("user_n", foreground=ACC)
        tb.tag_config("bot_n", foreground=ACC2)
        tb.tag_config("sys_n", foreground=MUTED)
        tb.tag_config("sys", foreground=MUTED)
        self.chat.configure(state="disabled")
        bar = ctk.CTkFrame(main, fg_color=PANEL, corner_radius=14)
        bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(bar, placeholder_text="Escreva para o Atlas…  (ex.: iniciar playlist, relatório, pesquise sobre…, lembre que…)",
                                  height=46, fg_color=PANEL2, border_width=0, font=("Segoe UI", 14))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(10, 6), pady=10)
        self.entry.bind("<Return>", lambda e: self.send())
        self.mic = ctk.CTkButton(bar, text="🎙", width=52, height=46, fg_color=PANEL2, hover_color="#2a3250", command=self.mic_click)
        self.mic.grid(row=0, column=1, padx=3)
        ctk.CTkButton(bar, text="Enviar", width=90, height=46, fg_color=ACC, text_color="#1a0d00", hover_color="#ffd98a",
                      font=("Segoe UI", 14, "bold"), command=self.send).grid(row=0, column=2, padx=(3, 10))
        self.refresh_tracks()

    def build_overlay(self):
        o = self.overlay = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        o.place(relx=0, rely=0, relwidth=1, relheight=1)
        box = ctk.CTkFrame(o, fg_color="transparent")
        box.place(relx=.5, rely=.5, anchor="center")
        ctk.CTkLabel(box, text="ATLAS", font=("Segoe UI", 64, "bold"), text_color=ACC).pack()
        ctk.CTkLabel(box, text="Como você quer conversar hoje?", font=("Segoe UI", 18), text_color=TXT).pack(pady=(0, 30))
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack()
        for txt, mode, col in (("⌨\nEscrever", "texto", PANEL2), ("🎙\nFalar", "voz", ACC2)):
            ctk.CTkButton(row, text=txt, width=200, height=130, font=("Segoe UI", 20, "bold"), fg_color=col,
                          hover_color="#2a3250" if col == PANEL2 else "#e07a10", corner_radius=18,
                          command=lambda m=mode: self.choose(m)).pack(side="left", padx=12)
        self.boot_lbl = ctk.CTkLabel(box, text="Preparando…", text_color=MUTED)
        self.boot_lbl.pack(pady=(30, 6))
        self.boot_bar = ctk.CTkProgressBar(box, mode="indeterminate", progress_color=ACC, width=360)
        self.boot_bar.pack()
        self.boot_bar.start()

    def choose(self, mode):
        self.set_mode(mode)
        self.overlay.destroy()
        self.overlay = None

    def set_mode(self, mode):
        self.mode = mode
        self.seg.set("🎙  Falar" if mode == "voz" else "⌨  Escrever")
        if mode == "voz":
            self.cancel = False
            self.loop_gen += 1
            threading.Thread(target=self.voice_loop, args=(self.loop_gen,), daemon=True).start()
        else:
            self.cancel = True
            self.entry.focus()

    def seg_change(self, v):
        self.set_mode("voz" if "Falar" in v else "texto")

    def toggle_voice(self):
        self.voice.enabled = bool(self.sw.get())
        if not self.voice.enabled:
            self.voice.stop()

    # ---------- visão: câmera e imagens geradas
    def cam_toggle(self):
        self.cam_off() if self.cam.on else self.cam_on()

    def cam_on(self):
        if not self.cam.on and not self.cam.start():
            self.ui_msg("sys", "Não consegui abrir a câmera. Ela pode estar em uso por outro programa (ou troque CAM_INDEX).")
            return
        if self.camwin is None or not self.camwin.alive:
            self.camwin = CamWindow(self)
        self.cam_btn.configure(text="⏹ Fechar câmera", fg_color=ACC2, text_color="#1a0d00")

    def cam_off(self):
        self.cam.stop()
        if self.camwin is not None:
            try:
                self.camwin.destroy()
            except Exception:
                pass
            self.camwin = None
        self.cam_btn.configure(text="📷 Câmera ao vivo", fg_color=PANEL2, text_color=TXT)

    def ui_cam(self, on):
        self.cam_on() if on else self.cam_off()

    def open_gallery(self):
        if self.gallery is None or not self.gallery.winfo_exists():
            self.gallery = GalleryWindow(self)
        self.gallery.deiconify()
        self.gallery.lift()

    def on_gemini_images(self, imgs):  # chamado na thread do Gemini
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        stamp, paths = time.strftime("%Y%m%d_%H%M%S"), []
        for k, data in enumerate(imgs):
            f = IMG_DIR / f"img_{stamp}_{k + 1}{img_ext(data) or '.png'}"
            f.write_bytes(data)
            paths.append(f)
        self.post("show_images", paths)

    def ui_show_images(self, paths):
        self.open_gallery()
        self.gallery.add(paths)
        self.ui_msg("sys", f"🖼 {len(paths)} imagem(ns) gerada(s). Salva(s) em {IMG_DIR}")

    def ask_text(self, t):  # envia uma frase como se o senhor tivesse digitado
        if self.thinking or not self.ready.is_set():
            return
        self.voice.stop()
        self.post("msg", "user", t)
        threading.Thread(target=self.process, args=(t,), daemon=True).start()

    def is_vision(self, text):
        n = " " + norm(text) + " "
        if any(k in n for k in VISION_KEYS):
            return True
        return self.cam.on and any(f" {w} " in n for w in LIVE_WORDS)

    def see(self, text):
        self.post("msg", "sys", "📷 Capturando imagem da câmera…")
        jpg = self.cam.grab()
        if jpg is None:
            r = "Não consegui acessar a câmera, senhor."
            self.post("msg", "bot", r)
            self.voice.say(r)
            return
        self.ask_llm(text, image=jpg)

    # ---------- fila UI (threads -> tela)
    def post(self, *a):
        self.ui.put(a)

    def pump(self):
        try:
            while True:
                a = self.ui.get_nowait()
                getattr(self, "ui_" + a[0])(*a[1:])
        except queue.Empty:
            pass
        if not self.closing:
            self.after(30, self.pump)

    def _ins(self, text, tag=None):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def ui_status(self, t):
        self.status.configure(text=t)
        if self.overlay is not None:
            self.boot_lbl.configure(text=t)

    def ui_msg(self, role, text):
        names = {"user": "Senhor", "bot": "ATLAS", "sys": "sistema"}
        self._ins(f"\n{names[role]}\n", role + "_n")
        self._ins(text + "\n", "sys" if role == "sys" else None)

    def ui_begin_bot(self):
        self._ins("\nATLAS\n", "bot_n")

    def ui_token(self, t):
        self._ins(t)

    def ui_end_bot(self):
        self._ins("\n")

    def ui_level(self, v):
        self.level = max(self.level, min(1.0, v * 18))

    # ---------- animação do orbe
    def animate(self):
        if self.closing:
            return
        c, t, g = self.orb, time.time(), ACC
        act = self.voice.busy or self.thinking or (self.listening and getattr(self, "hearing", False))
        if act:
            self.hold = t + 1.5  # mantém a música baixa por um instante (sem "pumping")
        self.player.duck = 0.12 if t < getattr(self, "hold", 0) else 1.0
        self.player.smooth()
        c.delete("all")
        W, H = max(c.winfo_width(), 200), max(c.winfo_height(), 200)
        cx, cy = W / 2, H / 2
        target = self.voice.level() if self.voice.busy else (min(1.0, self.level * 1.6) if self.listening else (0.3 if self.thinking else 0.04))
        self.amp += (target - self.amp) * 0.35
        a = self.amp
        R = min(W, H) * 0.38 * (1 + 0.08 * a)
        self.ang += 0.008 + (0.03 if self.thinking else 0) + a * 0.05
        ang = self.ang

        def tone(d, base=.2):
            return mix(g, BG, base + (1 - base) * d)
        for k, s_ in enumerate((1.5, 1.28, 1.1)):  # brilho externo
            rr = R * s_
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=mix(g, BG, .05 + .05 * k + .1 * a))
        for i in range(8):  # raios
            x, y, z = rot(math.cos(i * math.pi / 4 + ang * .5), 0, math.sin(i * math.pi / 4 + ang * .5), 0, 0.9)
            c.create_line(cx, cy, cx + x * R, cy + y * R, fill=tone(.3 + .3 * a, .1))
        for k, (rx, rz, sp, sc) in enumerate(((.3, 0, 1.0, 1.0), (1.2, .6, -.7, .92), (2.0, 1.3, .5, 1.05), (.8, 2.2, -.4, 1.28))):
            pts = []  # anéis girando
            for i in range(61):
                th = i / 60 * 6.2832
                x, y, z = rot(math.cos(th), math.sin(th), 0, ang * sp + rz, rx + .2 * math.sin(t * .5 + k))
                pts += [cx + x * R * sc, cy + y * R * sc]
            c.create_line(*pts, fill=tone(.55 + .3 * a, .1), width=1.4 + 1.6 * a)
        for x, y, z, ph in self.pts:  # esfera de partículas (espalha quando fala)
            x, y, z = rot(x, y, z, ang, .35)
            j = 1 + a * .25 * math.sin(ph + t * 9)
            d = (z + 1) / 2
            r = .8 + 2.2 * d * (1 + a)
            px, py = cx + x * R * j, cy + y * R * j
            c.create_oval(px - r, py - r, px + r, py + r, fill=tone(d), outline="")
        for rf, sp, ph, tl in self.sparks:  # faíscas orbitando
            u = ph + t * sp * (1 + 2 * a)
            x, y = math.cos(u) * rf, math.sin(u) * rf * .5
            px = cx + (x * math.cos(tl) - y * math.sin(tl)) * R
            py = cy + (x * math.sin(tl) + y * math.cos(tl)) * R
            if random.random() < .85 + .1 * a:
                c.create_oval(px - 1.5, py - 1.5, px + 1.5, py + 1.5, fill=tone(.9), outline="")
        cr = 14 + 9 * a  # núcleo
        c.create_oval(cx - cr * 2.4, cy - cr * 2.4, cx + cr * 2.4, cy + cr * 2.4, fill=mix(g, BG, .10), outline="")
        for rr, w, k in ((cr * 1.9, 1, .35), (cr * 1.35, 2, .6), (cr, 3, .95)):
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=mix(g, BG, k), width=w)
        c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#fff1c9", outline="")
        if self.thinking:
            c.create_arc(cx - R * 1.5, cy - R * 1.5, cx + R * 1.5, cy + R * 1.5, start=(t * 200) % 360, extent=70, style="arc", outline=g, width=3)
        self.level *= 0.85
        mt = "⏹" if self.listening else "🎙"
        if mt != getattr(self, "_mt", None):
            self._mt = mt
            self.mic.configure(text=mt)
        self.after(33, self.animate)

    # ---------- playlist (UI)
    def refresh_tracks(self):
        for w in self.tracks_frame.winfo_children():
            w.destroy()
        p = self.player
        if not p.tracks:
            ctk.CTkLabel(self.tracks_frame, text="Pasta 'playlist' vazia.\nColoque MP3/vídeos nela\ne clique em Atualizar.",
                         text_color=MUTED).pack(pady=20)
        for i, f in enumerate(p.tracks):
            cur = i == p.idx and p.playing
            ctk.CTkButton(self.tracks_frame, text=("▶ " if cur else ("🎬 " if f.suffix.lower() in VIDEO else "🎵 ")) + f.stem,
                          anchor="w", height=28, fg_color=PANEL2 if cur else "transparent", hover_color=PANEL2,
                          text_color=ACC if cur else TXT, command=lambda i=i: self.click_track(i)).pack(fill="x", pady=1)

    def rescan(self):
        self.player.scan()
        self.refresh_tracks()

    def click_track(self, i):
        self.player.autoplay = True
        self.player.play(i)

    def report(self):  # previsão do tempo em São José dos Campos (Open-Meteo, grátis)
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
                "latitude": -23.2237, "longitude": -45.9009, "current": "temperature_2m", "timezone": "America/Sao_Paulo",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,weather_code",
                "forecast_days": 1}).json()
            d, code = r["daily"], r["daily"]["weather_code"][0]
            desc = ("céu limpo" if code == 0 else "parcialmente nublado" if code <= 3 else "neblina" if code <= 48 else
                    "garoa" if code <= 57 else "chuva" if code <= 67 else "pancadas de chuva" if code <= 82 else "trovoadas")
            return (f"Senhor, relatório de hoje em São José dos Campos: {desc}, agora faz {r['current']['temperature_2m']:.0f} graus, "
                    f"mínima de {d['temperature_2m_min'][0]:.0f} e máxima de {d['temperature_2m_max'][0]:.0f} graus. "
                    f"Chance de chuva de {d['precipitation_probability_max'][0]} por cento, com {d['precipitation_sum'][0]:.1f} milímetros previstos.")
        except Exception:
            return "Perdão senhor, não consegui obter a previsão do tempo agora."

    def start_playlist(self):
        self.player.scan()
        self.player.autoplay = True
        self.player.play(0)
        self.refresh_tracks()
        if not self.player.tracks:
            self.ui_msg("sys", "A pasta 'playlist' está vazia. Coloque arquivos MP3 ou vídeos nela.")

    def player_toggle(self):
        (self.player.resume() if self.player.paused else self.player.pause())

    def player_next(self):
        self.player.autoplay = True
        self.player.next()

    def player_prev(self):
        self.player.autoplay = True
        self.player.prev()

    def player_stop(self):
        self.player.autoplay = False
        self.player.stop()

    def set_vol(self, v):
        self.player.vol = float(v)

    def tick(self):
        if self.closing:
            return
        p = self.player
        p.tick()
        sig = (p.idx, p.playing, p.paused)
        if sig != self.last_sig:
            self.last_sig = sig
            self.refresh_tracks()
            self.pp.configure(text="▶" if p.paused else "⏯")
        self.memlbl.configure(text=f"🧠 Memória: {self.mem.size() / 1048576:.1f} MB / 1 GB")
        self.after(400, self.tick)

    # ---------- inicialização em segundo plano
    def boot(self):
        log = lambda m: self.post("status", m)
        ok = True
        for name, fn in (("Preparando voz masculina…", lambda: self.voice.prepare(log)),
                         ("Carregando reconhecimento de voz…", self.ears.load),
                         ("Conectando ao Gemini…", lambda: ensure_gemini(log))):
            log(name)
            try:
                fn()
            except Exception as e:
                self.post("msg", "sys", f"{name} falhou: {e}")
                ok = ok and "Gemini" not in name
        self.ready.set()
        log("Pronto" if ok else "Pronto (com avisos)")
        self.post("msg", "bot", "Atlas online. À sua disposição, senhor.")
        self.voice.say("Atlas online. À sua disposição, senhor.")

    # ---------- entrada: texto / voz
    def send(self):
        t = self.entry.get().strip()
        if not t:
            return
        if not self.ready.is_set():
            return self.ui_msg("sys", "Aguarde, ainda estou carregando.")
        if self.thinking:
            return
        self.entry.delete(0, "end")
        self.voice.stop()
        self.post("msg", "user", t)
        threading.Thread(target=self.process, args=(t,), daemon=True).start()

    def listen_once(self, timeout=None):
        self.cancel = False
        self.listening = True
        try:
            return self.ears.listen(lambda: self.closing or self.cancel, lambda v: self.post("level", v), timeout,
                                    lambda: self.player.playing and not self.player.paused, lambda: setattr(self, "hearing", True))
        except Exception as e:
            self.post("msg", "sys", f"Microfone: {e}")
            time.sleep(2)
            return None
        finally:
            self.listening = False
            self.hearing = False

    def mic_click(self):
        if self.listening:
            self.cancel = True
            return
        if self.thinking or not self.ready.is_set():
            return

        def run():
            self.voice.stop()
            t = self.listen_once(timeout=10)
            if t:
                self.post("msg", "user", t)
                self.process(t)
        threading.Thread(target=run, daemon=True).start()

    def gate(self, t):  # com música tocando, só responde se ouvir "Atlas" (ou logo após uma conversa)
        P = self.player
        if not (P.playing and not P.paused) or time.time() < getattr(self, "awake", 0):
            self.awake = time.time() + 8
            return t
        toks = norm(t).split()
        if not any(difflib.get_close_matches(w, ["atlas"], 1, .7) for w in toks[:4]):
            return None
        self.awake = time.time() + 8
        if len(toks) <= 1:
            self.post("msg", "bot", "Pois não, senhor.")
            self.voice.say("Pois não, senhor.")
            return None
        return t

    def voice_loop(self, gen):
        self.ready.wait()
        while not self.closing and self.mode == "voz" and gen == self.loop_gen:
            if self.thinking or self.voice.busy:
                time.sleep(0.15)
                continue
            time.sleep(0.35)  # evita captar o eco da própria voz
            t = self.listen_once()
            if t and gen == self.loop_gen:
                t = self.gate(t)
                if t:
                    self.post("msg", "user", t)
                    self.process(t)

    # ---------- cérebro
    def process(self, text):
        self.thinking = True
        try:
            reply = self.commands(text)
            if reply is not None:
                self.post("msg", "bot", reply)
                self.voice.say(reply)
                self.mem.add("user", text)
                self.mem.add("assistant", reply)
            elif self.is_vision(text):
                self.see(text)
            else:
                if KEY_RE.search(norm(text)):
                    self.mem.add_fact(text[:300])  # favorito: memória permanente
                self.ask_llm(text)
        except Exception as e:
            self.post("msg", "sys", f"Erro: {e}")
        finally:
            self.thinking = False

    def ask_llm(self, text, image=None):
        sysm = SYSTEM
        facts, rel = self.mem.facts(), self.mem.search(text)
        if facts:
            sysm += "\n\nFatos que você sabe sobre o usuário:\n- " + "\n- ".join(facts)
        if rel:
            sysm += "\n\nTrechos de conversas antigas que podem ajudar:\n- " + "\n- ".join(f"{r}: {c[:200]}" for r, c in rel)
        if WEB_RE.search(norm(text)):
            web = web_search(text)
            if web:
                sysm += "\n\nResultados atuais da internet (use-os para responder ao senhor):\n" + web
        msgs = [{"role": "system", "content": sysm}]
        msgs += [{"role": r, "content": c[:1500]} for r, c in self.mem.recent(12)]
        msgs.append({"role": "user", "content": text})
        self.mem.add("user", text)
        self.post("begin_bot")
        full = buf = ""
        t0, minlen, first = time.time(), 12, True
        try:
            for tok in chat_stream(msgs, image):
                if first:
                    print(f"[tempo] primeiro token em {time.time() - t0:.2f}s")
                    first = False
                full += tok
                buf += tok
                self.post("token", tok)
                while True:
                    s, buf = pop_sentence(buf, minlen)
                    if s is None:
                        break
                    minlen = 25
                    self.voice.say(s)
            if buf.strip():
                self.voice.say(buf)
        finally:
            self.post("end_bot")
        self.mem.add("assistant", full.strip())

    # ---------- comandos de voz/texto (playlist e memória)
    def commands(self, text):
        n, P = " " + norm(text) + " ", self.player

        def has(*w):
            return any(f" {x} " in n for x in w)

        if has("ta ai", "esta ai"):
            return "À sua disposição, senhor."
        if has("camera", "webcam"):
            if has("liga", "ligar", "ligue", "abre", "abrir", "abra", "ativa", "ativar", "ative", "inicia", "iniciar", "inicie", "acende", "acenda"):
                self.post("cam", True)
                return "Ligando a câmera, senhor."
            if has("desliga", "desligar", "desligue", "fecha", "fechar", "feche", "desativa", "desativar", "desative", "apaga", "apague", "parar", "pare"):
                self.post("cam", False)
                return "Câmera desligada."
        if has("relatorio"):
            return self.report()
        if has("quem e voce", "qual e o seu nome", "qual seu nome", "quem te criou", "quem criou voce", "quem e o seu criador"):
            return "Eu sou o Atlas, senhor, uma inteligência artificial interativa e autônoma, criada pelo senhor Davi Lukas."
        m = re.search(r"\b(?:lembre|lembra|memorize|anote)(?:-se)?\s+(?:de\s+)?(?:que\s+)?(.+)", text, re.I | re.S)
        if m:
            self.mem.add_fact(m.group(1).strip())
            return "Anotado, senhor. Vou me lembrar disso."
        if has("play") and not has("playlist"):
            if P.paused:
                P.resume()
                return "Continuando, senhor."
            P.scan()
            P.autoplay = True
            P.play(0)
            return "Ligando a música, senhor."
        if has("pausa", "pausar", "pause"):
            P.pause()
            return "Pausado."
        if has("continua", "continuar", "retomar", "retoma", "despausar"):
            P.resume()
            return "Continuando."
        if has("proxima", "proximo", "pula", "pular", "avanca") or (has("troca", "trocar", "muda", "mudar", "outra", "outro") and has("musica", "som", "faixa")):
            P.autoplay = True
            P.next()
            return "Próxima faixa." if P.tracks else "A playlist está vazia."
        if has("anterior") or "volta a musica" in n or "voltar a musica" in n:
            P.autoplay = True
            P.prev()
            return "Faixa anterior." if P.tracks else "A playlist está vazia."
        if re.match(r"^ (atlas )?(por favor )?(para|parar|pare|desliga|desligue|desligar|tira|tire|encerra|encerre|corta|corte|chega) ", n) and has("musica", "musicas", "video", "som", "playlist", "tudo", "isso", "faixa", "radio") or has("silencio"):
            P.autoplay = False
            P.stop()
            return "Parei a reprodução."
        if has("volume") or (has("som", "musica") and has("aumenta", "aumentar", "sobe", "subir", "abaixa", "abaixar", "diminui", "diminuir", "baixa")):
            num = re.search(r"(\d{1,3})", n)
            if num:
                P.vol = min(100, int(num.group(1))) / 100
            elif has("aumenta", "aumentar", "sobe", "subir", "alto", "maior"):
                P.vol = min(1.0, P.vol + .15)
            elif has("abaixa", "abaixar", "diminui", "diminuir", "baixa", "baixo", "menor"):
                P.vol = max(0.0, P.vol - .15)
            else:
                return None
            self.after(0, lambda: self.vol.set(P.vol))
            return f"Volume em {int(P.vol * 100)} por cento."
        m = re.search(r" (toca|toque|tocar|coloca|coloque|poe|ponha|reproduz|reproduza|inicia|inicie|iniciar|comeca|comece|comecar|liga|ligue|ligar|bota|bote|botar|solta|solte|manda|mande|joga|jogue|escutar|ouvir) (.*)", n)
        if m and (has("playlist", "musica", "musicas", "video", "som", "radio", "faixa") or P.find(m.group(2)) is not None):
            P.scan()
            if not P.tracks:
                return "A pasta de playlist está vazia. Coloque arquivos MP3 ou vídeos nela."
            rest = m.group(2).strip()
            if "playlist" in rest or not [w for w in norm(rest).split() if w not in P.SKIP]:
                P.autoplay = True
                P.play(random.randrange(len(P.tracks)) if "aleatori" in rest else 0)
                return "Iniciando a playlist, senhor."
            i = P.find(rest)
            if i is None:
                return "Não encontrei esse arquivo na pasta de playlist."
            P.autoplay = False
            P.play(i)
            return f"Tocando {P.tracks[i].stem}."
        if has("quais musicas", "lista de musicas", "o que tem na playlist"):
            P.scan()
            return ("Tenho " + ", ".join(t.stem for t in P.tracks[:8]) + ".") if P.tracks else "A playlist está vazia."
        return None

    def on_close(self):
        self.closing = True
        self.cam.stop()
        try:
            self.player.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
