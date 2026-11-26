import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('resultados_cct_kaggle')
MODEL_PATH = OUT_DIR / 'mejor_cct.pt'
SVM_TEST_DIR = Path('PlantVillage-Dataset/data_distribution_for_SVM/test')
RAW_COLOR_DIR = Path('PlantVillage-Dataset/raw/color')
BATCH_SIZE = 64

# Orden alfabético estándar de las 38 clases de PlantVillage (0 a 37)
CLASES_PLANTVILLAGE = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


class CCTMultitarea(nn.Module):
    def __init__(self, num_species, image_size=128, embed_dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, embed_dim, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        )
        with torch.no_grad():
            token_count = self.tokenizer(torch.zeros(1, 3, image_size, image_size)).flatten(2).shape[2]
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))
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
        tokens = self.tokenizer(images).flatten(2).transpose(1, 2)
        tokens = self.dropout(tokens + self.position)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        weights = self.attention_pool(tokens).softmax(dim=1)
        pooled = (weights * tokens).sum(dim=1)
        return self.species_head(pooled), self.health_head(pooled)


def obtener_mapa_clases():
    """Usa las carpetas de raw/color si existen; si no, usa la lista estándar de 38 clases."""
    if RAW_COLOR_DIR.is_dir():
        carpetas = sorted(f.name for f in RAW_COLOR_DIR.iterdir() if f.is_dir() and '___' in f.name)
        if len(carpetas) == 38:
            return carpetas
    return CLASES_PLANTVILLAGE


def indexar_svm_test(test_dir, clases):
    if not test_dir.is_dir():
        raise FileNotFoundError(f'No existe la carpeta: {test_dir}')

    rows = []
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp'}
    carpetas = sorted(
        (f for f in test_dir.iterdir() if f.is_dir() and f.name.isdigit()),
        key=lambda f: int(f.name)
    )
    if not carpetas:
        raise ValueError(f'No se encontraron carpetas numéricas en {test_dir}')

    for folder in carpetas:
        idx = int(folder.name)
        if idx >= len(clases):
            raise ValueError(f'Carpeta {idx} fuera del rango de 38 clases.')
        species, condition = clases[idx].split('___', 1)
        health = int(condition.lower() != 'healthy')
        for img_path in sorted(folder.rglob('*')):
            if img_path.suffix.lower() in extensiones:
                rows.append({'path': str(img_path), 'species': species, 'health': health})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f'No se encontraron imágenes en {test_dir}')
    print(f'Imágenes encontradas en {test_dir}: {len(df)} (en {len(carpetas)} carpetas)')
    return df


class PlantasTestDataset(Dataset):
    def __init__(self, frame, mapping, transform):
        self.frame = frame.reset_index(drop=True)
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        r = self.frame.iloc[index]
        with Image.open(r.path) as im:
            x = self.transform(ImageOps.exif_transpose(im).convert('RGB'))
        return x, self.mapping[r.species], int(r.health)


def cargar_modelo(path):
    if not path.is_file():
        raise FileNotFoundError(f'No se encontró el modelo en: {path}')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    net = CCTMultitarea(len(ckpt['species_names']), **ckpt['config']).to(DEVICE)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net, ckpt


@torch.inference_mode()
def evaluar(model, loader, species_names):
    ys, yh, ps, ph = [], [], [], []
    for x, s, h in tqdm(loader, desc='Evaluando SVM test'):
        x = x.to(DEVICE)
        logits_s, logits_h = model(x)
        ys.extend(s.tolist())
        yh.extend(h.tolist())
        ps.extend(logits_s.argmax(1).cpu().tolist())
        ph.extend(logits_h.argmax(1).cpu().tolist())

    metrics = {
        'total_imagenes': len(ys),
        'accuracy_especie': round(accuracy_score(ys, ps), 4),
        'f1_especie_macro': round(f1_score(ys, ps, labels=list(range(len(species_names))), average='macro', zero_division=0), 4),
        'accuracy_salud': round(accuracy_score(yh, ph), 4),
        'f1_salud_macro': round(f1_score(yh, ph, labels=[0, 1], average='macro', zero_division=0), 4),
        'accuracy_conjunta': round(float(np.mean((np.array(ys) == ps) & (np.array(yh) == ph))), 4)
    }
    return metrics, (ys, ps, yh, ph)


if __name__ == '__main__':
    print('Dispositivo:', DEVICE)
    model, ckpt = cargar_modelo(MODEL_PATH)
    species_names = ckpt['species_names']
    health_names = ckpt['health_names']
    species_to_id = {name: i for i, name in enumerate(species_names)}

    clases = obtener_mapa_clases()
    test_df = indexar_svm_test(SVM_TEST_DIR, clases)

    transform = T.Compose([
        T.Resize((ckpt['config']['image_size'],) * 2),
        T.ToTensor(),
        T.Normalize(ckpt['mean'], ckpt['std'])
    ])

    ds = PlantasTestDataset(test_df, species_to_id, transform)
    workers = 2 if (os.name != 'nt' and DEVICE.type == 'cuda') else 0
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=DEVICE.type == 'cuda')

    metrics, (ys, ps, yh, ph) = evaluar(model, loader, species_names)

    print('\n=== RESUMEN DE MÉTRICAS (data_distribution_for_SVM/test) ===')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print('\n=== REPORTE POR ESPECIE / CULTIVO ===')
    print(classification_report(ys, ps, labels=list(range(len(species_names))), target_names=species_names, zero_division=0))

    print('\n=== REPORTE POR ESTADO DE SALUD ===')
    print(classification_report(yh, ph, labels=[0, 1], target_names=health_names, zero_division=0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'metricas_svm_test.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ConfusionMatrixDisplay.from_predictions(
        ys, ps, labels=list(range(len(species_names))),
        display_labels=species_names, xticks_rotation=90, ax=axes[0], colorbar=False
    )
    ConfusionMatrixDisplay.from_predictions(
        yh, ph, labels=[0, 1], display_labels=health_names, ax=axes[1], colorbar=False
    )
    axes[0].set_title('Especie / cultivo (SVM test)')
    axes[1].set_title('Estado de salud (SVM test)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'matrices_confusion_svm_test.png', bbox_inches='tight')
    print(f'\nGráfica guardada en: {OUT_DIR / "matrices_confusion_svm_test.png"}')
    plt.show()
