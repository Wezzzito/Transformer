"""
Web sencilla para el clasificador CCT (especie + salud).

Uso:
    pip install flask pillow torch torchvision
    python app.py --modelo resultados_cct_mixto_2/mejor_cct.pt
Luego abre http://localhost:5000
"""
import argparse
import io

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms

# Misma arquitectura que en test.py
class CCTMultitarea(nn.Module):
    def __init__(self, num_species, image_size=128, embed_dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, embed_dim, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        with torch.no_grad():
            token_count = self.tokenizer(torch.zeros(1, 3, image_size, image_size)).flatten(2).shape[2]
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads,
                dim_feedforward=embed_dim * 2, dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.species_head = nn.Linear(embed_dim, num_species)
        self.health_head = nn.Linear(embed_dim, 2)

    def forward(self, images):
        tokens = self.tokenizer(images).flatten(2).transpose(1, 2)
        tokens = self.dropout(tokens + self.position)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        weights = self.attention_pool(tokens).softmax(dim=1)
        pooled = (weights * tokens).sum(dim=1)
        return self.species_head(pooled), self.health_head(pooled)


parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="resultados_cct_mixto_2/mejor_cct.pt")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.modelo, map_location="cpu", weights_only=True)
ESPECIES = list(ckpt["species_names"])
SALUD = ["sana", "enferma"]
cfg = ckpt["config"]
modelo = CCTMultitarea(len(ESPECIES), **cfg)
modelo.load_state_dict(ckpt["state_dict"], strict=True)
modelo.to(device).eval()
tf = transforms.Compose([
    transforms.Resize((cfg["image_size"], cfg["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(ckpt["mean"], ckpt["std"]),
])

app = Flask(__name__)


@app.post("/predecir")
def predecir():
    archivo = request.files.get("imagen")
    if not archivo:
        return jsonify(error="No se envio ninguna imagen"), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(archivo.read()))).convert("RGB")
    except Exception:
        return jsonify(error="Archivo de imagen no valido"), 400

    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        le, ls = modelo(x)
        pe, ps = F.softmax(le, 1)[0], F.softmax(ls, 1)[0]

    top = torch.topk(pe, 3)
    return jsonify(
        especie=ESPECIES[top.indices[0]],
        confianza_especie=float(top.values[0]),
        top3=[{"especie": ESPECIES[i], "prob": float(p)} for p, i in zip(top.values, top.indices)],
        salud=SALUD[int(ps.argmax())],
        confianza_salud=float(ps.max()),
    )


@app.get("/")
def index():
    return PAGINA


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnostico de plantas</title>
<style>
  :root{--bg:#f4f7f2;--card:#fff;--tx:#1d2b1f;--mut:#6b7a6d;--ac:#2f7d46;--bad:#c0392b;--bd:#d9e2d6}
  @media (prefers-color-scheme:dark){:root{--bg:#121a14;--card:#1b261e;--tx:#e8f0e9;--mut:#93a396;--ac:#4fb26b;--bad:#ef6a5b;--bd:#2c3b2f}}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--tx);font-family:system-ui,sans-serif;
    min-height:100vh;display:flex;justify-content:center;padding:24px}
  main{width:100%;max-width:460px}
  h1{font-size:1.4rem;margin:0 0 16px}
  .card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:18px}
  #drop{border:2px dashed var(--bd);border-radius:12px;min-height:220px;display:flex;align-items:center;
    justify-content:center;text-align:center;color:var(--mut);cursor:pointer;overflow:hidden}
  #drop.over{border-color:var(--ac)} #drop img{width:100%;display:block}
  #res{margin-top:16px;display:none}
  .row{display:flex;justify-content:space-between;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--bd)}
  .row:last-child{border:0} .big{font-size:1.2rem;font-weight:600}
  .ok{color:var(--ac)} .bad{color:var(--bad)} small{color:var(--mut)}
  #err{color:var(--bad);margin-top:12px}
</style></head><body><main>
<h1>Diagnostico de plantas</h1>
<div class="card">
  <label id="drop"><input id="f" type="file" accept="image/*" hidden>
    <span id="hint">Haz clic o arrastra una imagen de una hoja</span></label>
  <div id="res">
    <div class="row"><span>Especie</span><span><span class="big" id="esp"></span> <small id="cesp"></small></span></div>
    <div class="row"><span>Salud</span><span><span class="big" id="sal"></span> <small id="csal"></small></span></div>
    <div class="row"><span><small>Otras opciones</small></span><small id="top"></small></div>
  </div>
  <div id="err"></div>
</div>
<script>
const f=document.getElementById('f'),drop=document.getElementById('drop'),$=id=>document.getElementById(id);
const pct=x=>(x*100).toFixed(1)+'%', nombre=s=>s.replace(/_/g,' ');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over')});
drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files[0])enviar(e.dataTransfer.files[0])});
f.addEventListener('change',()=>f.files[0]&&enviar(f.files[0]));
async function enviar(file){
  $('err').textContent='';$('res').style.display='none';
  drop.innerHTML='';const im=new Image();im.src=URL.createObjectURL(file);drop.appendChild(im);
  const fd=new FormData();fd.append('imagen',file);
  try{
    const r=await fetch('/predecir',{method:'POST',body:fd});const d=await r.json();
    if(!r.ok)throw new Error(d.error);
    $('esp').textContent=nombre(d.especie);$('cesp').textContent=pct(d.confianza_especie);
    const sana=d.salud==='sana';$('sal').textContent=sana?'Sana':'Enferma';
    $('sal').className='big '+(sana?'ok':'bad');$('csal').textContent=pct(d.confianza_salud);
    $('top').textContent=d.top3.slice(1).map(t=>nombre(t.especie)+' '+pct(t.prob)).join(' · ');
    $('res').style.display='block';
  }catch(e){$('err').textContent='Error: '+e.message}
}
</script></main></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
