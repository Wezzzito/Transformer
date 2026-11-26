from pathlib import Path
import hashlib, json, os, random, warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from tqdm.auto import tqdm

###########################
#Integrantes:
#- Julio Pino Charun
#- Emanuel Salas Ramos
#- Gianella Rosas Lipa
###########################

# --- CONSTANTES Y CONFIGURACIÓN ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE = Path('/content') if Path('/content').exists() else Path.cwd()
RAW_DATA_DIR = BASE / 'New Plant Diseases Dataset(Augmented)'
OUT_DIR = BASE / 'resultados_cct_kaggle'

VERIFICAR_IMAGENES = False
CONFIG = dict(image_size=128, embed_dim=128, depth=4, heads=4, dropout=0.1)
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 7
LR = 3e-4
LAMBDA_ESPECIE = 1.0
LAMBDA_SALUD = 1.0
SALUD = ['sana', 'enferma']

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NOMBRES_ES = {
    'Apple': 'Manzano', 'Blueberry': 'Arándano', 'Cherry_(including_sour)': 'Cerezo',
    'Corn_(maize)': 'Maíz', 'Grape': 'Vid', 'Orange': 'Naranjo', 'Peach': 'Duraznero',
    'Pepper,_bell': 'Pimiento', 'Potato': 'Papa', 'Raspberry': 'Frambueso',
    'Soybean': 'Soya', 'Squash': 'Calabaza', 'Strawberry': 'Fresa', 'Tomato': 'Tomate'
}


# --- FUNCIONES Y CLASES ---
def fijar_semilla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def localizar_splits(base_dir):
    """Encuentra automáticamente las carpetas 'train' y 'valid' incluso si el ZIP se anidó dos veces."""
    base_dir = Path(base_dir)
    candidatos = [
        base_dir,
        base_dir / 'New Plant Diseases Dataset(Augmented)',
        base_dir / 'new plant diseases dataset(augmented)' / 'New Plant Diseases Dataset(Augmented)',
    ]
    for c in candidatos:
        if (c / 'train').is_dir() and (c / 'valid').is_dir():
            return c / 'train', c / 'valid'
    raise FileNotFoundError(
        f'No se encontraron las carpetas "train" y "valid" dentro de: {base_dir}\n'
        'Verifica la ruta en RAW_DATA_DIR.'
    )


def leer_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def indexar_carpeta(root, verificar=False):
    root = Path(root)
    rows = []
    carpetas = [f for f in sorted(root.iterdir()) if f.is_dir() and '___' in f.name]
    if not carpetas:
        raise ValueError(f'No hay carpetas con formato especie___condicion en {root}')

    for folder in tqdm(carpetas, desc=f'Indexando {root.name}'):
        species, condition = folder.name.split('___', 1)
        if not species or not condition:
            raise ValueError(f'Etiqueta inválida: {folder.name}')
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            if verificar:
                try:
                    leer_rgb(path)
                except Exception as exc:
                    raise ValueError(f'Imagen dañada: {path}') from exc
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(dict(
                path=str(path),
                species=species,
                health=int(condition.lower() != 'healthy'),
                joint=folder.name,
                digest=digest
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f'No se encontraron imágenes en {root}')
    if (df.groupby('digest').joint.nunique() > 1).any():
        warnings.warn(f'Se encontraron imágenes idénticas con etiquetas distintas en {root.name}.')
    before = len(df)
    df = df.drop_duplicates('digest').reset_index(drop=True)
    print(f'[{root.name}] Imágenes: {len(df)} (duplicados eliminados: {before - len(df)})')
    return df


def crear_transform(size, train=False):
    ops = ([
        T.RandomResizedCrop(size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
    ] if train else [T.Resize((size, size))])
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class PlantasDataset(Dataset):
    def __init__(self, frame, mapping, transform):
        self.frame = frame.reset_index(drop=True)
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        r = self.frame.iloc[index]
        x = self.transform(leer_rgb(r.path))
        return x, self.mapping[r.species], int(r.health)


def crear_loader(frame, mapping, train=False, size=128):
    ds = PlantasDataset(frame, mapping, crear_transform(size, train))
    workers = 2 if (os.name != 'nt' and DEVICE.type == 'cuda') else 0
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(SEED)
    )


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


def pesos_clase(labels, n):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n)
    if (counts == 0).any():
        raise ValueError('Hay clases sin ejemplos en entrenamiento')
    return torch.tensor(counts.sum() / (n * counts), dtype=torch.float32, device=DEVICE)


def ejecutar_epoca(model, loader, loss_species, loss_health, num_species, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total = 0.0, 0
    ys, yh, ps, ph = [], [], [], []

    for x, s, h in tqdm(loader, leave=False):
        x, s, h = x.to(DEVICE), s.to(DEVICE), h.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits_s, logits_h = model(x)
            loss = LAMBDA_ESPECIE * loss_species(logits_s, s) + LAMBDA_SALUD * loss_health(logits_h, h)
            if not torch.isfinite(loss):
                raise FloatingPointError('Pérdida no finita')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * len(x)
        total += len(x)
        ys.extend(s.cpu().tolist())
        yh.extend(h.cpu().tolist())
        ps.extend(logits_s.argmax(1).detach().cpu().tolist())
        ph.extend(logits_h.argmax(1).detach().cpu().tolist())

    if total == 0:
        raise ValueError('Conjunto vacío')

    metrics = dict(
        loss=total_loss / total,
        accuracy_especie=accuracy_score(ys, ps),
        accuracy_salud=accuracy_score(yh, ph),
        f1_especie=f1_score(ys, ps, labels=list(range(num_species)), average='macro', zero_division=0),
        f1_salud=f1_score(yh, ph, labels=[0, 1], average='macro', zero_division=0),
        accuracy_conjunta=float(np.mean((np.array(ys) == ps) & (np.array(yh) == ph)))
    )
    return metrics, (ys, ps, yh, ph)


def entrenar(model, train_loader, val_loader, loss_species, loss_health, species_names):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best, stale, history = -1.0, 0, []
    num_species = len(species_names)

    for epoch in range(1, EPOCHS + 1):
        tr, _ = ejecutar_epoca(model, train_loader, loss_species, loss_health, num_species, optimizer)
        va, _ = ejecutar_epoca(model, val_loader, loss_species, loss_health, num_species)
        score = (va['f1_especie'] + va['f1_salud']) / 2
        history.append({
            'epoch': epoch,
            **{'train_' + k: v for k, v in tr.items()},
            **{'val_' + k: v for k, v in va.items()}
        })
        pd.DataFrame(history).to_csv(OUT_DIR / 'historial.csv', index=False)
        print(f'Época {epoch}: pérdida={tr["loss"]:.4f}, F1 especie={va["f1_especie"]:.3f}, '
              f'F1 salud={va["f1_salud"]:.3f}')
        if score > best:
            best, stale = score, 0
            torch.save({
                'state_dict': model.state_dict(), 'config': CONFIG,
                'species_names': species_names, 'health_names': SALUD,
                'mean': MEAN, 'std': STD, 'epoch': epoch, 'val_score': score,
                'seed': SEED
            }, OUT_DIR / 'mejor_cct.pt')
        else:
            stale += 1
        scheduler.step()
        if stale >= PATIENCE:
            print('Parada temprana. Se conserva el mejor modelo de validación.')
            break
    return pd.DataFrame(history)


def cargar_modelo(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    net = CCTMultitarea(len(ckpt['species_names']), **ckpt['config']).to(DEVICE)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net, ckpt


@torch.inference_mode()
def predecir(image_path, model_path):
    net, ckpt = cargar_modelo(model_path)
    im = leer_rgb(image_path)
    transform = T.Compose([
        T.Resize((ckpt['config']['image_size'],) * 2),
        T.ToTensor(),
        T.Normalize(ckpt['mean'], ckpt['std'])
    ])
    s, h = net(transform(im).unsqueeze(0).to(DEVICE))
    s, h = s.softmax(1)[0], h.softmax(1)[0]
    idx_s, idx_h = s.argmax().item(), h.argmax().item()
    label = ckpt['species_names'][idx_s]
    return {
        'planta': NOMBRES_ES.get(label, label),
        'etiqueta_dataset': label,
        'estado_predicho': ckpt['health_names'][idx_h],
        'probabilidad_modelo_especie': round(s[idx_s].item(), 4),
        'probabilidad_modelo_estado': round(h[idx_h].item(), 4)
    }


def evaluar_externo(csv_path, root, model, species_to_id, loss_species, loss_health):
    frame = pd.read_csv(csv_path)
    if not {'path', 'species', 'health'} <= set(frame.columns):
        raise ValueError('El CSV debe tener path,species,health')
    unknown = set(frame.species) - set(species_to_id)
    if unknown:
        raise ValueError(f'Plantas no entrenadas: {unknown}')
    if not frame.health.isin([0, 1]).all():
        raise ValueError('health debe ser 0 o 1')
    frame['path'] = frame['path'].apply(lambda p: str(Path(root) / p))
    loader = crear_loader(frame, species_to_id, train=False, size=CONFIG['image_size'])
    metrics, pred = ejecutar_epoca(model, loader, loss_species, loss_health, len(species_to_id))
    print(json.dumps(metrics, indent=2))
    print(classification_report(pred[2], pred[3], labels=[0, 1], target_names=SALUD, zero_division=0))
    return metrics


# --- PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == '__main__':
    fijar_semilla(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Dispositivo:', DEVICE, '| PyTorch:', torch.__version__)

    TRAIN_DIR, VALID_DIR = localizar_splits(RAW_DATA_DIR)
    print('Train dir:', TRAIN_DIR)
    print('Valid dir:', VALID_DIR)

    # 1. Indexar train y valid oficiales de Kaggle
    train_df = indexar_carpeta(TRAIN_DIR, verificar=VERIFICAR_IMAGENES)
    valid_full_df = indexar_carpeta(VALID_DIR, verificar=VERIFICAR_IMAGENES)

    # 2. Eliminar de valid cualquier duplicado exacto que ya este en train
    duplicados_cruzados = valid_full_df.digest.isin(set(train_df.digest))
    if duplicados_cruzados.any():
        print('Duplicados eliminados de valid (presentes en train):', int(duplicados_cruzados.sum()))
        valid_full_df = valid_full_df[~duplicados_cruzados].reset_index(drop=True)

    # 3. Dividir valid en 50% validacion y 50% test de forma estratificada
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(splitter.split(valid_full_df, valid_full_df.joint))
    val_df = valid_full_df.iloc[val_idx].reset_index(drop=True)
    test_df = valid_full_df.iloc[test_idx].reset_index(drop=True)

    species_names = sorted(train_df.species.unique().tolist())
    species_to_id = {name: i for i, name in enumerate(species_names)}

    for name, part in [('train', train_df), ('val', val_df), ('test', test_df)]:
        part.to_csv(OUT_DIR / f'{name}.csv', index=False)
        print(f'{name}: {len(part)} imágenes')

    print('\nDistribución en train:')
    print(pd.crosstab(train_df.species, train_df.health).rename(columns={0: 'sana', 1: 'enferma'}))
    print('Categorías:', species_names)

    # 4. DataLoaders
    train_loader = crear_loader(train_df, species_to_id, train=True, size=CONFIG['image_size'])
    val_loader = crear_loader(val_df, species_to_id, train=False, size=CONFIG['image_size'])
    test_loader = crear_loader(test_df, species_to_id, train=False, size=CONFIG['image_size'])

    # 5. Modelo y perdidas
    model = CCTMultitarea(len(species_names), **CONFIG).to(DEVICE)
    print('Parámetros:', sum(p.numel() for p in model.parameters()))
    print('Tokens:', model.position.shape[1])

    loss_species = nn.CrossEntropyLoss(weight=pesos_clase(train_df.species.map(species_to_id), len(species_names)))
    loss_health = nn.CrossEntropyLoss(weight=pesos_clase(train_df.health, 2))

    # 6. Entrenamiento
    history = entrenar(model, train_loader, val_loader, loss_species, loss_health, species_names)
    history.plot(x='epoch', y=['train_loss', 'val_loss'], title='Pérdida durante el entrenamiento')
    plt.savefig(OUT_DIR / 'curva_perdida.png', bbox_inches='tight')
    plt.show()

    # 7. Evaluacion en test
    model, checkpoint = cargar_modelo(OUT_DIR / 'mejor_cct.pt')
    test_metrics, predictions = ejecutar_epoca(
        model, test_loader, loss_species, loss_health, len(species_names)
    )
    print(json.dumps(test_metrics, indent=2))
    (OUT_DIR / 'metricas_test.json').write_text(json.dumps(test_metrics, indent=2))

    ys, ps, yh, ph = predictions
    print(classification_report(ys, ps, labels=list(range(len(species_names))),
                                target_names=species_names, zero_division=0))
    print(classification_report(yh, ph, labels=[0, 1], target_names=SALUD, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ConfusionMatrixDisplay.from_predictions(
        ys, ps, labels=list(range(len(species_names))),
        display_labels=species_names, xticks_rotation=90, ax=axes[0], colorbar=False
    )
    ConfusionMatrixDisplay.from_predictions(
        yh, ph, labels=[0, 1], display_labels=SALUD, ax=axes[1], colorbar=False
    )
    axes[0].set_title('Especie / cultivo')
    axes[1].set_title('Estado de salud')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'matrices_confusion.png', bbox_inches='tight')
    plt.show()
