"""
Generate HTML report showing predictions vs ground truth.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import pandas as pd
import json
import math
from tqdm import tqdm
from torchvision import transforms
import timm
import torch.nn as nn
import base64
from io import BytesIO


class TreeClassifier(nn.Module):
    def __init__(self, num_classes, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_features(self, x):
        return self.backbone(x)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def image_to_base64(img_path, max_size=300):
    """Convert image to base64 for embedding in HTML."""
    img = Image.open(img_path).convert('RGB')
    # Resize for display
    img.thumbnail((max_size, max_size))
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def generate_report(model_path, data_dir, image_base, embeddings_path, output_path,
                   sample_size=100, radius_m=30):
    """Generate HTML report with predictions - input vs predicted image side by side."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load embeddings with prototypes
    emb_data = torch.load(embeddings_path, map_location=device)
    tree_embeddings = {k: v.to(device) for k, v in emb_data['embeddings'].items()}
    tree_prototypes = None
    if 'prototypes' in emb_data:
        tree_prototypes = {k: [p.to(device) for p in protos]
                          for k, protos in emb_data['prototypes'].items()}

    # Load data
    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)

    df = pd.read_excel(data_dir / 'training_data_with_ground_truth.xlsx')

    # Get tree locations
    trees = df.groupby('key').agg({
        'gt_lat': 'first',
        'gt_lon': 'first'
    }).reset_index()
    tree_coords = {row['key']: (row['gt_lat'], row['gt_lon']) for _, row in trees.iterrows()}

    # Build lookup: tree_key -> list of image paths (for showing reference images)
    tree_images = df.groupby('key')['image_path'].apply(list).to_dict()

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')
    num_classes = len(encoder)

    model = TreeClassifier(num_classes, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Sample data
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    results = []
    correct_count = 0

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc='Processing'):
        img_path = image_base / row['image_path']
        true_key = row['key']
        photo_lat, photo_lon = row['photo_lat'], row['photo_lon']

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.extract_features(img_tensor)
                query_emb = F.normalize(features, p=2, dim=1).squeeze(0)

            # Find candidates
            candidates = []
            for key, (t_lat, t_lon) in tree_coords.items():
                dist = haversine(photo_lat, photo_lon, t_lat, t_lon)
                if dist <= radius_m:
                    # Compute prototype similarity
                    if tree_prototypes and key in tree_prototypes:
                        proto_sims = [torch.dot(query_emb, p).item() for p in tree_prototypes[key]]
                        similarity = max(proto_sims)
                    elif key in tree_embeddings:
                        similarity = torch.dot(query_emb, tree_embeddings[key]).item()
                    else:
                        similarity = 0.0

                    candidates.append({
                        'key': key,
                        'distance': dist,
                        'similarity': similarity
                    })

            if not candidates:
                continue

            # Sort by similarity (proto_only method)
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            pred_key = candidates[0]['key']
            is_correct = (pred_key == true_key)

            if is_correct:
                correct_count += 1

            # Get input image as base64
            input_b64 = image_to_base64(img_path)

            # Get reference image for predicted tree (different from input)
            pred_images = tree_images.get(pred_key, [])
            pred_ref_b64 = None
            for ref_path in pred_images:
                if ref_path != row['image_path']:
                    try:
                        pred_ref_b64 = image_to_base64(image_base / ref_path)
                        break
                    except:
                        continue
            if pred_ref_b64 is None and pred_images:
                try:
                    pred_ref_b64 = image_to_base64(image_base / pred_images[0])
                except:
                    pred_ref_b64 = None

            # Get reference image for true tree (if different from predicted)
            true_ref_b64 = None
            if not is_correct:
                true_images = tree_images.get(true_key, [])
                for ref_path in true_images:
                    if ref_path != row['image_path']:
                        try:
                            true_ref_b64 = image_to_base64(image_base / ref_path)
                            break
                        except:
                            continue

            results.append({
                'input_b64': input_b64,
                'pred_ref_b64': pred_ref_b64,
                'true_ref_b64': true_ref_b64,
                'image_path': str(row['image_path']),
                'true_key': true_key,
                'pred_key': pred_key,
                'is_correct': is_correct,
                'similarity': candidates[0]['similarity'],
                'distance': candidates[0]['distance'],
                'num_candidates': len(candidates),
                'top_candidates': candidates[:5]
            })

        except Exception as e:
            continue

    # Generate HTML
    accuracy = correct_count / len(results) * 100 if results else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tree ID Prediction Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .stats {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats h2 {{
            margin-top: 0;
            color: #2e7d32;
        }}
        .filters {{
            margin-bottom: 20px;
        }}
        .filters button {{
            padding: 10px 20px;
            margin-right: 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .filters button.active {{
            background: #1976d2;
            color: white;
        }}
        .filters button:not(.active) {{
            background: #e0e0e0;
        }}
        .cards {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card.correct {{
            border-left: 4px solid #4caf50;
        }}
        .card.wrong {{
            border-left: 4px solid #f44336;
        }}
        .images {{
            display: flex;
            gap: 10px;
            padding: 10px;
            background: #fafafa;
        }}
        .image-box {{
            flex: 1;
            text-align: center;
        }}
        .image-box img {{
            width: 100%;
            max-width: 300px;
            height: 220px;
            object-fit: cover;
            border-radius: 4px;
        }}
        .image-box .img-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            font-weight: bold;
        }}
        .image-box .img-label.correct {{
            color: #4caf50;
        }}
        .image-box .img-label.wrong {{
            color: #f44336;
        }}
        .image-box .img-sublabel {{
            font-size: 11px;
            color: #999;
            margin-top: 2px;
        }}
        .card-content {{
            padding: 15px;
        }}
        .card-content h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            word-break: break-all;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .info-row .label {{
            color: #999;
        }}
        .info-row .value {{
            font-weight: bold;
        }}
        .result-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }}
        .result-badge.correct {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .result-badge.wrong {{
            background: #ffebee;
            color: #c62828;
        }}
        .hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <h1>Tree Identification Results</h1>

    <div class="stats">
        <h2>Accuracy: {accuracy:.1f}%</h2>
        <p>Correct: {correct_count} / {len(results)}</p>
        <p>Method: Prototype similarity (max similarity to any cluster center)</p>
        <p>Search radius: {radius_m}m</p>
    </div>

    <div class="filters">
        <button class="active" onclick="filter('all')">All ({len(results)})</button>
        <button onclick="filter('correct')">Correct ({correct_count})</button>
        <button onclick="filter('wrong')">Wrong ({len(results) - correct_count})</button>
    </div>

    <div class="cards">
"""

    for i, r in enumerate(results):
        status = 'correct' if r['is_correct'] else 'wrong'
        status_text = 'CORRECT' if r['is_correct'] else 'WRONG'

        # Build images section
        images_html = f"""
            <div class="image-box">
                <img src="data:image/jpeg;base64,{r['input_b64']}" alt="Input">
                <div class="img-label">INPUT</div>
                <div class="img-sublabel">{r['true_key']}</div>
            </div>
        """

        if r['pred_ref_b64']:
            images_html += f"""
            <div class="image-box">
                <img src="data:image/jpeg;base64,{r['pred_ref_b64']}" alt="Predicted">
                <div class="img-label {status}">PREDICTED</div>
                <div class="img-sublabel">{r['pred_key']}</div>
            </div>
            """

        # Show true tree reference if wrong
        if not r['is_correct'] and r['true_ref_b64']:
            images_html += f"""
            <div class="image-box">
                <img src="data:image/jpeg;base64,{r['true_ref_b64']}" alt="True">
                <div class="img-label correct">ACTUAL</div>
                <div class="img-sublabel">{r['true_key']}</div>
            </div>
            """

        html += f"""
        <div class="card {status}" data-status="{status}">
            <div class="images">
                {images_html}
            </div>
            <div class="card-content">
                <div class="info-row">
                    <span class="label">File:</span>
                    <span>{r['image_path']}</span>
                </div>
                <div class="info-row">
                    <span class="label">Result:</span>
                    <span class="result-badge {status}">{status_text}</span>
                </div>
                <div class="info-row">
                    <span class="label">Similarity:</span>
                    <span class="value">{r['similarity']:.3f}</span>
                </div>
                <div class="info-row">
                    <span class="label">Distance:</span>
                    <span class="value">{r['distance']:.1f}m</span>
                </div>
                <div class="info-row">
                    <span class="label">Candidates in range:</span>
                    <span class="value">{r['num_candidates']}</span>
                </div>
            </div>
        </div>
"""

    html += """
    </div>

    <script>
        function filter(type) {
            document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            document.querySelectorAll('.card').forEach(card => {
                if (type === 'all') {
                    card.classList.remove('hidden');
                } else if (card.dataset.status === type) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport saved to: {output_path}")
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(results)})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    parser.add_argument('--embeddings', default=r'E:\tree_id_2.0\models\tree_embeddings_v2.pt')
    parser.add_argument('--output', default=r'E:\tree_id_2.0\prediction_report.html')
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--radius', type=float, default=30)
    args = parser.parse_args()

    generate_report(args.model, args.data, args.images, args.embeddings,
                   args.output, args.sample, args.radius)
