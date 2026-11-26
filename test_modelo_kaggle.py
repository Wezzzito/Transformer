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
TEST_CSV = OUT_DIR / 'test.csv'
BATCH_SIZE = 64


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
def evaluar(model, loader, species_names, health_names):
    ys, yh, ps, ph = [], [], [], []
    for x, s, h in tqdm(loader, desc='Evaluando test'):
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

    if not TEST_CSV.is_file():
        raise FileNotFoundError(f'No se encontró {TEST_CSV}. Ejecuta primero el entrenamiento.')

    test_df = pd.read_csv(TEST_CSV)
    species_names = ckpt['species_names']
    health_names = ckpt['health_names']
    species_to_id = {name: i for i, name in enumerate(species_names)}

    transform = T.Compose([
        T.Resize((ckpt['config']['image_size'],) * 2),
        T.ToTensor(),
        T.Normalize(ckpt['mean'], ckpt['std'])
    ])

    ds = PlantasTestDataset(test_df, species_to_id, transform)
    workers = 2 if (os.name != 'nt' and DEVICE.type == 'cuda') else 0
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=DEVICE.type == 'cuda')

    metrics, (ys, ps, yh, ph) = evaluar(model, loader, species_names, health_names)

    print('\n=== RESUMEN DE MÉTRICAS EN TEST ===')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print('\n=== REPORTE POR ESPECIE / CULTIVO ===')
    print(classification_report(ys, ps, labels=list(range(len(species_names))), target_names=species_names, zero_division=0))

    print('\n=== REPORTE POR ESTADO DE SALUD ===')
    print(classification_report(yh, ph, labels=[0, 1], target_names=health_names, zero_division=0))

    # Guardar métricas en JSON
    (OUT_DIR / 'metricas_test_eval.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Guardar matrices de confusión
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ConfusionMatrixDisplay.from_predictions(
        ys, ps, labels=list(range(len(species_names))),
        display_labels=species_names, xticks_rotation=90, ax=axes[0], colorbar=False
    )
    ConfusionMatrixDisplay.from_predictions(
        yh, ph, labels=[0, 1], display_labels=health_names, ax=axes[1], colorbar=False
    )
    axes[0].set_title('Especie / cultivo')
    axes[1].set_title('Estado de salud')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'matrices_confusion_eval.png', bbox_inches='tight')
    print(f'\nGráfica guardada en: {OUT_DIR / "matrices_confusion_eval.png"}')
    plt.show()
