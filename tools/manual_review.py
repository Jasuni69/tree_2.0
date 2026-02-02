"""
Manual review tool for uncertain tree photos.
Shows input image vs predicted tree reference. Accept or reject.
If reject, shows next candidate (up to top 3).

Controls:
  Y / Enter  - Accept current prediction
  N / Space  - Reject, show next candidate
  S          - Skip this image (mark unknown)
  Q          - Quit and save
  B          - Go back to previous image
"""

import pygame
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import pandas as pd
import json
from torchvision import transforms
import timm
import torch.nn as nn
import math


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


def load_image_pygame(path, max_size=500):
    """Load image and convert to pygame surface."""
    try:
        img = Image.open(path).convert('RGB')
        # Resize keeping aspect ratio
        img.thumbnail((max_size, max_size))
        mode = img.mode
        size = img.size
        data = img.tobytes()
        return pygame.image.fromstring(data, size, mode)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_candidates(query_emb, current_key, tree_coords, tree_prototypes, tree_embeddings, device):
    """Get all candidate trees at same address, ranked by similarity."""
    current_address = current_key.rsplit('|', 1)[0] if '|' in current_key else current_key

    candidates = []
    for key, (t_lat, t_lon) in tree_coords.items():
        key_address = key.rsplit('|', 1)[0] if '|' in key else key
        if key_address != current_address:
            continue

        # Compute similarity
        if tree_prototypes and key in tree_prototypes:
            proto_sims = [torch.dot(query_emb, p).item() for p in tree_prototypes[key]]
            similarity = max(proto_sims)
        elif key in tree_embeddings:
            similarity = torch.dot(query_emb, tree_embeddings[key]).item()
        else:
            similarity = 0.0

        candidates.append({
            'key': key,
            'similarity': similarity
        })

    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    return candidates


def main():
    # Paths
    data_dir = Path(r'E:\tree_id_2.0\data')
    image_base = Path(r'E:\tree_id_2.0\images')
    model_path = Path(r'E:\tree_id_2.0\models\model_seed42.pt')
    embeddings_path = Path(r'E:\tree_id_2.0\models\tree_embeddings_v2.pt')
    output_path = data_dir / 'manual_review_results.xlsx'

    # Load flagged data - only "review" items
    flagged = pd.read_excel(data_dir / 'flagged_for_review.xlsx')
    review_items = flagged[flagged['action'] == 'review'].reset_index(drop=True)
    print(f"Loaded {len(review_items)} items to review")

    # Load training data for tree locations and reference images
    df_train = pd.read_excel(data_dir / 'training_data_with_ground_truth.xlsx')

    # Get tree locations
    trees = df_train.groupby('key').agg({
        'gt_lat': 'first',
        'gt_lon': 'first'
    }).reset_index()
    tree_coords = {row['key']: (row['gt_lat'], row['gt_lon']) for _, row in trees.iterrows()}

    # Build tree -> images lookup
    tree_images = df_train.groupby('key')['image_path'].apply(list).to_dict()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)
    num_classes = len(encoder)

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')

    model = TreeClassifier(num_classes, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load embeddings
    emb_data = torch.load(embeddings_path, map_location=device)
    tree_embeddings = {k: v.to(device) for k, v in emb_data['embeddings'].items()}
    tree_prototypes = None
    if 'prototypes' in emb_data:
        tree_prototypes = {k: [p.to(device) for p in protos]
                          for k, protos in emb_data['prototypes'].items()}

    # Transform
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Init pygame
    pygame.init()
    screen_width, screen_height = 1600, 900
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Manual Review Tool")
    font = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (50, 200, 50)
    RED = (200, 50, 50)
    BLUE = (50, 100, 200)
    GRAY = (100, 100, 100)

    # State
    current_idx = 0
    current_candidate_idx = 0  # Which candidate (0=top1, 1=top2, 2=top3)
    results = []  # Store decisions
    candidates_cache = {}  # Cache computed candidates

    # Load existing results if any
    if output_path.exists():
        existing = pd.read_excel(output_path)
        results = existing.to_dict('records')
        # Find where to resume
        reviewed_paths = set(r['image_path'] for r in results)
        for i, row in review_items.iterrows():
            if row['image_path'] not in reviewed_paths:
                current_idx = i
                break
        else:
            current_idx = len(review_items)
        print(f"Resuming from item {current_idx}, {len(results)} already reviewed")

    running = True
    need_redraw = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in (pygame.K_y, pygame.K_RETURN):
                    # Accept current candidate
                    if current_idx < len(review_items):
                        row = review_items.iloc[current_idx]
                        candidates = candidates_cache.get(current_idx, [])
                        if candidates and current_candidate_idx < len(candidates):
                            chosen = candidates[current_candidate_idx]
                            results.append({
                                'image_path': row['image_path'],
                                'original_key': row['current_key'],
                                'chosen_key': chosen['key'],
                                'confidence': chosen['similarity'],
                                'candidate_rank': current_candidate_idx + 1
                            })
                        current_idx += 1
                        current_candidate_idx = 0
                        need_redraw = True
                elif event.key in (pygame.K_n, pygame.K_SPACE):
                    # Reject, show next candidate
                    candidates = candidates_cache.get(current_idx, [])
                    if current_candidate_idx < 2 and current_candidate_idx < len(candidates) - 1:
                        current_candidate_idx += 1
                    else:
                        # No more candidates, mark as unknown
                        if current_idx < len(review_items):
                            row = review_items.iloc[current_idx]
                            results.append({
                                'image_path': row['image_path'],
                                'original_key': row['current_key'],
                                'chosen_key': 'UNKNOWN',
                                'confidence': 0,
                                'candidate_rank': -1
                            })
                        current_idx += 1
                        current_candidate_idx = 0
                    need_redraw = True
                elif event.key == pygame.K_s:
                    # Skip
                    if current_idx < len(review_items):
                        row = review_items.iloc[current_idx]
                        results.append({
                            'image_path': row['image_path'],
                            'original_key': row['current_key'],
                            'chosen_key': 'SKIPPED',
                            'confidence': 0,
                            'candidate_rank': -1
                        })
                    current_idx += 1
                    current_candidate_idx = 0
                    need_redraw = True
                elif event.key == pygame.K_b:
                    # Go back - first through candidates, then to previous image
                    if current_candidate_idx > 0:
                        # Go back to previous candidate
                        current_candidate_idx -= 1
                        need_redraw = True
                    elif current_idx > 0:
                        # At candidate 0, go to previous image
                        current_idx -= 1
                        # Remove last result
                        if results:
                            results.pop()
                        # Set to last candidate viewed (top 3 = index 2)
                        candidates = candidates_cache.get(current_idx, [])
                        current_candidate_idx = min(2, len(candidates) - 1) if candidates else 0
                        need_redraw = True

        if not need_redraw:
            pygame.time.wait(50)
            continue

        need_redraw = False
        screen.fill(WHITE)

        # Check if done
        if current_idx >= len(review_items):
            text = font.render("All done! Press Q to quit and save.", True, GREEN)
            screen.blit(text, (screen_width//2 - text.get_width()//2, screen_height//2))
            pygame.display.flip()
            continue

        # Get current item
        row = review_items.iloc[current_idx]
        img_path = image_base / row['image_path']

        # Compute candidates if not cached
        if current_idx not in candidates_cache:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.extract_features(img_tensor)
                    query_emb = F.normalize(features, p=2, dim=1).squeeze(0)
                candidates = get_candidates(query_emb, row['current_key'], tree_coords,
                                          tree_prototypes, tree_embeddings, device)
                candidates_cache[current_idx] = candidates
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                candidates_cache[current_idx] = []

        candidates = candidates_cache[current_idx]

        # Draw header
        header = f"Review {current_idx + 1}/{len(review_items)} | Current label: {row['current_key']}"
        text = font.render(header, True, BLACK)
        screen.blit(text, (20, 20))

        # Draw instructions
        instructions = "Y/Enter=Accept  N/Space=Reject  S=Skip  B=Back  Q=Quit"
        text = font_small.render(instructions, True, GRAY)
        screen.blit(text, (20, screen_height - 30))

        # Load and draw input image
        input_surf = load_image_pygame(img_path, max_size=500)
        if input_surf:
            screen.blit(input_surf, (50, 80))
            label = font_small.render("INPUT IMAGE", True, BLUE)
            screen.blit(label, (50, 80 + input_surf.get_height() + 5))

            # Show current label info
            curr_label = font_small.render(f"Labeled as: {row['current_key']}", True, BLACK)
            screen.blit(curr_label, (50, 80 + input_surf.get_height() + 30))

        # Draw candidate info
        ref_x = 620  # Position for reference image (right of input)
        if candidates and current_candidate_idx < len(candidates):
            candidate = candidates[current_candidate_idx]

            # Draw candidate header
            cand_text = f"Candidate #{current_candidate_idx + 1}: {candidate['key']}"
            text = font.render(cand_text, True, BLACK)
            screen.blit(text, (ref_x, 80))

            sim_text = f"Similarity: {candidate['similarity']:.4f}"
            text = font_small.render(sim_text, True, BLACK)
            screen.blit(text, (ref_x, 115))

            # Load reference image for this candidate
            ref_images = tree_images.get(candidate['key'], [])
            ref_surf = None
            for ref_path in ref_images:
                if ref_path != row['image_path']:
                    ref_surf = load_image_pygame(image_base / ref_path, max_size=500)
                    if ref_surf:
                        break

            if ref_surf:
                screen.blit(ref_surf, (ref_x, 150))
                label = font_small.render("REFERENCE (from predicted tree)", True, GREEN)
                screen.blit(label, (ref_x, 150 + ref_surf.get_height() + 5))
            else:
                text = font_small.render("No reference image available", True, RED)
                screen.blit(text, (ref_x, 150))

            # Show other candidates preview
            preview_y = 700
            text = font_small.render("Other candidates:", True, GRAY)
            screen.blit(text, (ref_x, preview_y))

            for i, c in enumerate(candidates[:5]):
                if i == current_candidate_idx:
                    color = GREEN
                    prefix = "→ "
                else:
                    color = BLACK
                    prefix = "  "
                ctext = f"{prefix}{i+1}. {c['key']} ({c['similarity']:.3f})"
                text = font_small.render(ctext, True, color)
                screen.blit(text, (ref_x, preview_y + 25 + i * 22))
        else:
            text = font.render("No candidates found!", True, RED)
            screen.blit(text, (ref_x, 150))

        # Progress bar
        progress = current_idx / len(review_items)
        pygame.draw.rect(screen, GRAY, (20, screen_height - 60, screen_width - 40, 20))
        pygame.draw.rect(screen, GREEN, (20, screen_height - 60, int((screen_width - 40) * progress), 20))

        pygame.display.flip()

    # Save results
    pygame.quit()

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False)
        print(f"\nSaved {len(results)} decisions to {output_path}")

        # Summary
        chosen_counts = results_df['chosen_key'].value_counts()
        print("\nSummary:")
        unknown = len(results_df[results_df['chosen_key'] == 'UNKNOWN'])
        skipped = len(results_df[results_df['chosen_key'] == 'SKIPPED'])
        labeled = len(results_df) - unknown - skipped
        print(f"  Labeled: {labeled}")
        print(f"  Unknown: {unknown}")
        print(f"  Skipped: {skipped}")


if __name__ == '__main__':
    main()
