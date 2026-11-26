"""Ajuste del CCT de Kaggle con replay PlantVillage + PlantDoc.
Uso: python ajustar_cct_mixto.py
Requiere torch torchvision pillow numpy pandas scikit-learn tqdm.
No ejecuta el codigo original ni entrena desde cero. Usa sus CSV guardados.
No usa imagenes de test para entrenar ni para escoger checkpoints.
El test.csv de PV se lee SOLO para comprobar solapamientos de rutas/hashes.
El optimizador es nuevo: el checkpoint original no incluye su estado.
Una epoca equivale a --pasos lotes mixtos, no a todo PlantVillage.
Los controles exactos no detectan todas las variantes aumentadas/recortes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

class CCTMultitarea(nn.Module):
    def __init__(self, num_species, image_size=128, embed_dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        if embed_dim % heads:
            raise ValueError('embed_dim debe ser múltiplo de heads')
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, embed_dim, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        )
        with torch.no_grad():
            token_count = self.tokenizer(torch.zeros(1, 3, image_size, image_size)).flatten(2).shape[2]
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=heads,
                dim_feedforward=embed_dim * 2, dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.species_head = nn.Linear(embed_dim, num_species)
        self.health_head = nn.Linear(embed_dim, 2)

    def forward(self, images):
        tokens = self.tokenizer(images).flatten(2).transpose(1, 2)  # [B,N,D]
        if tokens.shape[1] != self.position.shape[1]:
            raise ValueError('La resolución debe coincidir con la del entrenamiento')
        tokens = self.dropout(tokens + self.position)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        weights = self.attention_pool(tokens).softmax(dim=1)  # [B,N,1]
        pooled = (weights * tokens).sum(dim=1)                # [B,D]
        return self.species_head(pooled), self.health_head(pooled)

LABELS = {
    'Apple Scab Leaf': ('Apple', 1),
    'Apple leaf': ('Apple', 0),
    'Apple rust leaf': ('Apple', 1),
    'Bell_pepper leaf spot': ('Pepper,_bell', 1),
    'Bell_pepper leaf': ('Pepper,_bell', 0),
    'Blueberry leaf': ('Blueberry', 0),
    'Cherry leaf': ('Cherry_(including_sour)', 0),
    'Corn Gray leaf spot': ('Corn_(maize)', 1),
    'Corn leaf blight': ('Corn_(maize)', 1),
    'Corn rust leaf': ('Corn_(maize)', 1),
    'Peach leaf': ('Peach', 0),
    'Potato leaf early blight': ('Potato', 1),
    'Potato leaf late blight': ('Potato', 1),
    'Raspberry leaf': ('Raspberry', 0),
    'Soyabean leaf': ('Soybean', 0),
    'Squash Powdery mildew leaf': ('Squash', 1),
    'Strawberry leaf': ('Strawberry', 0),
    'Tomato Early blight leaf': ('Tomato', 1),
    'Tomato Septoria leaf spot': ('Tomato', 1),
    'Tomato leaf bacterial spot': ('Tomato', 1),
    'Tomato leaf late blight': ('Tomato', 1),
    'Tomato leaf mosaic virus': ('Tomato', 1),
    'Tomato leaf yellow virus': ('Tomato', 1),
    'Tomato leaf': ('Tomato', 0),
    'Tomato mold leaf': ('Tomato', 1),
    'Tomato two spotted spider mites leaf': ('Tomato', 1),
    'grape leaf black rot': ('Grape', 1),
    'grape leaf': ('Grape', 0),
}

HEALTH = ['sana', 'enferma']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def read_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def leer_csv(path, root, names, require_files=True):
    df = pd.read_csv(path)
    required = {'path', 'species', 'health', 'digest'}
    if not required <= set(df):
        raise ValueError(f'{path}: faltan columnas {required-set(df)}. Usa los CSV originales.')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError(f'{path}: CSV vacio o valores ausentes.')
    if set(df.species)-set(names) or not df.health.isin([0, 1]).all():
        raise ValueError(f'{path}: etiquetas incompatibles con el checkpoint.')
    df['path'] = df.path.map(lambda p: str((root/str(p)).resolve()))
    if require_files:
        missing = [p for p in df.path if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'No existe {missing[0]}. Revisa --raiz-rutas y las rutas del CSV.')
    return df.reset_index(drop=True)


def comprobar_particiones(parts):
    keys = list(parts)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            for col in ['path', 'digest']:
                overlap = set(parts[a][col]) & set(parts[b][col])
                if overlap:
                    raise ValueError(f'PV {a}/{b}: {len(overlap)} solapamientos de {col}. Revisa las particiones.')


def preparar_pd(root, pv_parts, seed):
    if not root.is_dir() or root.name.casefold() != 'train':
        raise ValueError('--plantdoc-train debe ser la carpeta train de PlantDoc.')
    lookup = {k.casefold(): v for k, v in LABELS.items()}
    rows = []
    forbidden = set().union(*(set(p.digest) for p in pv_parts.values()))
    excluded = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'): continue
        if folder.name.casefold() not in lookup:
            raise ValueError(f'Carpeta PlantDoc desconocida: {folder.name}')
        species, health = lookup[folder.name.casefold()]
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in forbidden:
                excluded.append(dict(path=str(path), motivo='duplicado exacto de PlantVillage'))
                continue
            im = read_rgb(path)
            pixels = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            rows.append(dict(path=str(path.resolve()), species=species, health=health,
                             joint=folder.name, digest=digest, pixels=pixels))
    if not rows: raise ValueError('PlantDoc/train no tiene imagenes utilizables.')
    df = pd.DataFrame(rows)
    df['target'] = df.species+'|'+df.health.astype(str)
    conflicts = df.groupby('pixels').target.nunique()
    bad = df.pixels.isin(conflicts[conflicts > 1].index)
    for p in df.loc[bad, 'path']:
        excluded.append(dict(path=p, motivo='pixeles iguales con etiquetas contradictorias'))
    df = df.loc[~bad].copy()
    duplicate = df.duplicated('pixels')
    for p in df.loc[duplicate, 'path']:
        excluded.append(dict(path=p, motivo='duplicado de pixeles dentro de PlantDoc/train'))
    df = df.loc[~duplicate].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for label, group in df.groupby('joint'):
        ids = rng.permutation(group.index)
        if len(ids) < 2:
            raise ValueError(f'{label}: menos de 2 imagenes tras filtrar; no puede dividirse train/val.')
        n = min(len(ids)-1, max(1, round(0.2*len(ids))))
        va.extend(ids[:n]); tr.extend(ids[n:])
    return (df.loc[tr].reset_index(drop=True), df.loc[va].reset_index(drop=True),
            pd.DataFrame(excluded, columns=['path', 'motivo']))


def transform(size, mean, std, train=False):
    ops = [T.Resize((size, size))]
    if train:
        ops = [T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
               T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.RandomRotation(15),
               T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
               T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.1)]
    return T.Compose(ops+[T.ToTensor(), T.Normalize(mean, std)])


class Fotos(Dataset):
    def __init__(self, frame, names, tf):
        self.frame, self.tf = frame, tf
        self.mapping = {s:i for i,s in enumerate(names)}
        if set(frame.species)-set(names): raise ValueError('Especies desconocidas en los datos.')
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        r = self.frame.iloc[idx]
        return self.tf(read_rgb(r.path)), self.mapping[r.species], int(r.health)


class LotesMixtos(Sampler):
    """Mitad PV, mitad PD. Recorre permutaciones nuevas antes de repetir indices.
    La longitud de cada epoca se limita para no repetir PD decenas de veces.
    """
    def __init__(self, n_pv, n_pd, batch, steps, seed):
        self.n_pv, self.n_pd = n_pv, n_pd
        self.half, self.steps, self.seed, self.epoch = batch//2, steps, seed, 0
    def __len__(self): return self.steps
    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        def stream(n):
            while True:
                yield from rng.permutation(n).tolist()
        pv, pd_ = stream(self.n_pv), stream(self.n_pd)
        for _ in range(self.steps):
            ids = [next(pv) for _ in range(self.half)]
            ids += [self.n_pv+next(pd_) for _ in range(self.half)]
            rng.shuffle(ids)
            yield ids


def epoca(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    ys, ps, yh, ph = [], [], [], []
    loss_sum, count = 0., 0
    for x, s, h in tqdm(loader, leave=False):
        x, s, h = x.to(device), s.to(device), h.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            ls, lh = model(x)
            loss = nn.functional.cross_entropy(ls, s)+nn.functional.cross_entropy(lh, h)
            if not torch.isfinite(loss): raise ValueError('Perdida no finita.')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        count += len(x); loss_sum += loss.item()*len(x)
        ys.extend(s.cpu().tolist()); yh.extend(h.cpu().tolist())
        ps.extend(ls.detach().argmax(1).cpu().tolist())
        ph.extend(lh.detach().argmax(1).cpu().tolist())
    return dict(loss=loss_sum/count, accuracy_especie=accuracy_score(ys, ps),
                accuracy_salud=accuracy_score(yh, ph),
                f1_especie=f1_score(ys, ps, labels=sorted(set(ys)), average='macro', zero_division=0),
                f1_salud=f1_score(yh, ph, labels=[0,1], average='macro', zero_division=0),
                accuracy_conjunta=float(np.mean((np.array(ys)==ps)&(np.array(yh)==ph))))


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo', type=Path, default=base/'resultados_cct_kaggle/mejor_cct.pt')
    p.add_argument('--splits-pv', type=Path, default=base/'resultados_cct_kaggle')
    p.add_argument('--raiz-rutas', type=Path, default=base, help='Base para rutas relativas de los CSV de PV.')
    p.add_argument('--plantdoc-train', type=Path, default=base/'PlantDoc-Dataset-master/train')
    p.add_argument('--salida', type=Path, default=base/'resultados_cct_mixto')
    p.add_argument('--epocas', type=int, default=15)
    p.add_argument('--paciencia', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--pasos', type=int, default=0, help='0: una pasada aproximada por PD por epoca; PV se muestrea.')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    if a.epocas < 1 or a.paciencia < 1 or a.batch_size < 2 or a.batch_size % 2 or a.pasos < 0 or a.lr <= 0 or a.workers < 0:
        p.error('Revisa parametros: batch par >=2; epocas/paciencia/lr positivos; pasos/workers >=0.')
    try:
        if a.salida.exists() and (not a.salida.is_dir() or any(a.salida.iterdir())):
            raise ValueError('La salida ya contiene archivos. Usa --salida con una carpeta nueva.')
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(a.modelo, map_location='cpu', weights_only=True)
        names = ckpt['species_names']; cfg = ckpt['config']
        if len(names) != len(set(names)) or list(ckpt['health_names']) != HEALTH:
            raise ValueError('Etiquetas del checkpoint incompatibles.')
        model = CCTMultitarea(len(names), **cfg)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model.to(device)
        parts = {s:leer_csv(a.splits_pv/f'{s}.csv', a.raiz_rutas, names, s!='test')
                 for s in ['train','val','test']}
        comprobar_particiones(parts)
        pd_train, pd_val, excluded = preparar_pd(a.plantdoc_train, parts, a.seed)
        train_tf = transform(cfg['image_size'], ckpt['mean'], ckpt['std'], True)
        val_tf = transform(cfg['image_size'], ckpt['mean'], ckpt['std'])
        pv_ds = Fotos(parts['train'], names, train_tf)
        pd_ds = Fotos(pd_train, names, train_tf)
        steps = a.pasos or math.ceil(len(pd_ds)/(a.batch_size//2))
        sampler = LotesMixtos(len(pv_ds), len(pd_ds), a.batch_size, steps, a.seed)
        kw = dict(num_workers=a.workers, pin_memory=device.type=='cuda')
        loader = DataLoader(ConcatDataset([pv_ds,pd_ds]), batch_sampler=sampler, **kw)
        vals = {'pv':DataLoader(Fotos(parts['val'], names, val_tf), batch_size=a.batch_size, **kw),
                'pd':DataLoader(Fotos(pd_val, names, val_tf), batch_size=a.batch_size, **kw)}
        a.salida.mkdir(parents=True)
        for name, frame in [('pv_train',parts['train']),('pv_val',parts['val']),('pd_train',pd_train),('pd_val',pd_val),('excluidas_pd',excluded)]:
            frame.to_csv(a.salida/f'{name}.csv', index=False)
        settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        settings['pasos_efectivos'] = steps
        (a.salida/'config_ajuste.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        print(f'Modelo inicial: {a.modelo.resolve()} (epoca original {ckpt.get("epoch")}) | {device}')
        print(f'PV train={len(pv_ds)}, val={len(parts["val"])}; PD train={len(pd_ds)}, val={len(pd_val)}')
        print(f'Cada epoca: {steps} lotes, {a.batch_size//2} PV + {a.batch_size//2} PD. No recorre todo PV.')
        print('F1 especie usa clases presentes en cada validacion. Test no se evalua durante el ajuste.')
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        history = []; best = -1.; stale = 0
        # Epoca cero: conserva el modelo original si el ajuste no mejora validacion.
        for epoch in range(a.epocas+1):
            tr = {}
            if epoch:
                sampler.epoch = epoch
                tr = epoca(model, loader, device, optimizer)
            scores = {d:epoca(model, dl, device) for d,dl in vals.items()}
            score = sum(v[k] for v in scores.values() for k in ['f1_especie','f1_salud'])/4
            row = dict(epoch=epoch, score=score, **{'train_'+k:v for k,v in tr.items()})
            for d, metrics in scores.items(): row.update({f'val_{d}_{k}':v for k,v in metrics.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(a.salida/'historial.csv', index=False)
            print(f'Epoca {epoch}: score={score:.4f} | PV especie/salud={scores["pv"]["accuracy_especie"]:.2%}/{scores["pv"]["accuracy_salud"]:.2%}'
                  f' | PD especie/salud={scores["pd"]["accuracy_especie"]:.2%}/{scores["pd"]["accuracy_salud"]:.2%}', flush=True)
            payload = dict(state_dict=model.state_dict(), config=cfg, species_names=names,
                           health_names=HEALTH, mean=ckpt['mean'], std=ckpt['std'], epoch=epoch,
                           val_score=score, seed=a.seed, parent_checkpoint=str(a.modelo.resolve()),
                           parent_epoch=ckpt.get('epoch'), validation=scores,
                           selection='mean(F1_species_PV,F1_health_PV,F1_species_PD,F1_health_PD)')
            torch.save(payload, a.salida/'ultimo_cct.pt')
            if score > best + 1e-6:
                best, stale = score, 0
                torch.save(payload, a.salida/'mejor_cct.pt')
            elif epoch:
                stale += 1
            if stale >= a.paciencia:
                print('Parada temprana.'); break
        print('Terminado. Mejor checkpoint:', (a.salida/'mejor_cct.pt').resolve())
        print('Compara historial.csv antes de evaluar el test. No se garantiza mejorar ambos dominios.')
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
