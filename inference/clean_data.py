"""
Data cleaning tool - uses model to find and fix mislabeled photos.

Reads metadata, runs predictions, flags errors, outputs cleaned version.
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
import numpy as np
from datetime import datetime


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


def clean_data(model_path, data_dir, image_base, embeddings_path,
               input_file, output_file, flagged_file, radius_m=30,
               confidence_threshold=0.7):
    """
    Run predictions on all data and flag/fix errors.

    Args:
        confidence_threshold: Min similarity to auto-correct (0-1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load embeddings with prototypes
    emb_data = torch.load(embeddings_path, map_location=device)
    tree_embeddings = {k: v.to(device) for k, v in emb_data['embeddings'].items()}
    tree_prototypes = None
    if 'prototypes' in emb_data:
        tree_prototypes = {k: [p.to(device) for p in protos]
                          for k, protos in emb_data['prototypes'].items()}
        print(f"Loaded prototypes for {len(tree_prototypes)} trees")

    # Load encoder
    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)
    num_classes = len(encoder)

    # Load working data
    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos from {input_file}")

    # Get tree locations
    trees = df.groupby('key').agg({
        'gt_lat': 'first',
        'gt_lon': 'first'
    }).reset_index()
    tree_coords = {row['key']: (row['gt_lat'], row['gt_lon']) for _, row in trees.iterrows()}

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')

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

    # Process all photos
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Analyzing'):
        img_path = image_base / row['image_path']
        current_key = row['key']
        photo_lat, photo_lon = row['photo_lat'], row['photo_lon']

        # Parse address from key (format: "address|tree_number")
        current_address = current_key.rsplit('|', 1)[0] if '|' in current_key else current_key
        current_tree_num = current_key.rsplit('|', 1)[1] if '|' in current_key else ''

        result = {
            'idx': idx,
            'image_path': row['image_path'],
            'address': current_address,
            'current_key': current_key,
            'current_tree_num': current_tree_num,
            'photo_lat': photo_lat,
            'photo_lon': photo_lon,
        }

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.extract_features(img_tensor)
                query_emb = F.normalize(features, p=2, dim=1).squeeze(0)

            # Find candidates - ONLY trees at the SAME ADDRESS
            # Key format is "address|tree_number"
            current_address = current_key.rsplit('|', 1)[0] if '|' in current_key else current_key

            candidates = []
            for key, (t_lat, t_lon) in tree_coords.items():
                # Only consider trees at the same address
                key_address = key.rsplit('|', 1)[0] if '|' in key else key
                if key_address != current_address:
                    continue

                dist = haversine(photo_lat, photo_lon, t_lat, t_lon)

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
                result['status'] = 'no_candidates'
                result['predicted_key'] = None
                result['confidence'] = 0
                result['action'] = 'review'
                results.append(result)
                continue

            # Sort by similarity
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            best = candidates[0]

            result['predicted_key'] = best['key']
            result['confidence'] = best['similarity']
            result['pred_distance'] = best['distance']
            result['num_candidates'] = len(candidates)

            # Check current label similarity
            current_sim = 0
            current_dist = 0
            for c in candidates:
                if c['key'] == current_key:
                    current_sim = c['similarity']
                    current_dist = c['distance']
                    break

            result['current_similarity'] = current_sim
            result['current_distance'] = current_dist

            # Determine action
            if best['key'] == current_key:
                result['status'] = 'correct'
                result['action'] = 'keep'
            elif best['similarity'] >= confidence_threshold:
                result['status'] = 'mislabeled'
                result['action'] = 'auto_correct'
            else:
                result['status'] = 'uncertain'
                result['action'] = 'review'

            # Add runner-up info
            if len(candidates) > 1:
                result['runner_up_key'] = candidates[1]['key']
                result['runner_up_sim'] = candidates[1]['similarity']

        except Exception as e:
            result['status'] = 'error'
            result['action'] = 'review'
            result['error'] = str(e)

        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Summary stats
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)

    status_counts = results_df['status'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    action_counts = results_df['action'].value_counts()
    print("\nActions:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")

    # Create flagged report (errors + uncertain)
    flagged = results_df[results_df['action'].isin(['auto_correct', 'review'])].copy()
    flagged = flagged.sort_values('confidence', ascending=False)

    if len(flagged) > 0:
        flagged.to_excel(data_dir / flagged_file, index=False)
        print(f"\nFlagged {len(flagged)} photos -> {flagged_file}")

    # Create cleaned dataset
    df_clean = df.copy()

    # Apply auto-corrections
    auto_corrected = 0
    for _, r in results_df[results_df['action'] == 'auto_correct'].iterrows():
        idx = r['idx']
        new_key = r['predicted_key']
        df_clean.at[idx, 'key'] = new_key
        df_clean.at[idx, 'cleaning_note'] = f"Auto-corrected from {r['current_key']} (conf: {r['confidence']:.3f})"
        auto_corrected += 1

    # Mark reviewed items
    for _, r in results_df[results_df['action'] == 'review'].iterrows():
        idx = r['idx']
        df_clean.at[idx, 'cleaning_note'] = f"Needs review: predicted {r.get('predicted_key', 'N/A')} (conf: {r.get('confidence', 0):.3f})"

    df_clean.to_excel(data_dir / output_file, index=False)
    print(f"Auto-corrected {auto_corrected} labels -> {output_file}")

    # Generate HTML report for flagged items
    generate_flagged_report(results_df, image_base, data_dir, tree_coords)

    return results_df


def generate_flagged_report(results_df, image_base, data_dir, tree_coords):
    """Generate HTML report for flagged items."""
    from io import BytesIO
    import base64

    def img_to_b64(path, max_size=200):
        try:
            img = Image.open(path).convert('RGB')
            img.thumbnail((max_size, max_size))
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=80)
            return base64.b64encode(buf.getvalue()).decode()
        except:
            return None

    flagged = results_df[results_df['action'].isin(['auto_correct', 'review'])]

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Data Cleaning Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f0f0f0; }
        h1 { color: #333; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .filters { margin-bottom: 20px; }
        .filters button { padding: 8px 16px; margin-right: 10px; border: none; border-radius: 4px; cursor: pointer; }
        .filters button.active { background: #1976d2; color: white; }
        .filters button:not(.active) { background: #ddd; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; position: sticky; top: 0; }
        img { max-width: 150px; max-height: 150px; border-radius: 4px; }
        .auto_correct { background: #fff3e0; }
        .review { background: #ffebee; }
        .confidence { font-weight: bold; }
        .high { color: #2e7d32; }
        .medium { color: #f57c00; }
        .low { color: #c62828; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <h1>Data Cleaning Report</h1>
    <div class="summary">
        <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        <p><strong>Flagged items:</strong> """ + str(len(flagged)) + """</p>
        <p><strong>Auto-correct:</strong> """ + str(len(flagged[flagged['action']=='auto_correct'])) + """</p>
        <p><strong>Needs review:</strong> """ + str(len(flagged[flagged['action']=='review'])) + """</p>
    </div>

    <div class="filters">
        <button class="active" onclick="filter('all')">All</button>
        <button onclick="filter('auto_correct')">Auto-correct</button>
        <button onclick="filter('review')">Needs Review</button>
    </div>

    <table>
        <tr>
            <th>Image</th>
            <th>Current Label</th>
            <th>Predicted</th>
            <th>Confidence</th>
            <th>Action</th>
            <th>Details</th>
        </tr>
"""

    for _, r in flagged.head(500).iterrows():  # Limit to 500 for performance
        img_b64 = img_to_b64(image_base / r['image_path'])
        img_html = f'<img src="data:image/jpeg;base64,{img_b64}">' if img_b64 else '[Error]'

        conf = r.get('confidence', 0)
        conf_class = 'high' if conf > 0.8 else 'medium' if conf > 0.6 else 'low'

        html += f"""
        <tr class="{r['action']}" data-action="{r['action']}">
            <td>{img_html}</td>
            <td>{r['current_key']}</td>
            <td>{r.get('predicted_key', 'N/A')}</td>
            <td class="confidence {conf_class}">{conf:.3f}</td>
            <td>{r['action']}</td>
            <td>
                Current sim: {r.get('current_similarity', 0):.3f}<br>
                Candidates: {r.get('num_candidates', 0)}
            </td>
        </tr>
"""

    html += """
    </table>
    <script>
        function filter(type) {
            document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelectorAll('tr[data-action]').forEach(row => {
                if (type === 'all' || row.dataset.action === type) {
                    row.classList.remove('hidden');
                } else {
                    row.classList.add('hidden');
                }
            });
        }
    </script>
</body>
</html>
"""

    report_path = data_dir / 'cleaning_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Visual report -> {report_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    parser.add_argument('--embeddings', default=r'E:\tree_id_2.0\models\tree_embeddings_v2.pt')
    parser.add_argument('--input', default='training_data_working.xlsx')
    parser.add_argument('--output', default='training_data_cleaned.xlsx')
    parser.add_argument('--flagged', default='flagged_for_review.xlsx')
    parser.add_argument('--radius', type=float, default=30)
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Min confidence to auto-correct')
    args = parser.parse_args()

    clean_data(args.model, args.data, args.images, args.embeddings,
               args.input, args.output, args.flagged, args.radius, args.threshold)
