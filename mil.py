import torch, os, pandas as pd

ckpt_dir = r"C:\Users\milan\OneDrive\Documents\Project\ProtoPNet Crop Disease Research Workflow\Claude2\exp1 3\ablation_ckpts"

results = []
for fname in os.listdir(ckpt_dir):
    if not fname.endswith('.pth'):
        continue
    ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location='cpu')
    results.append({
        'filename':  fname,
        'test_acc':  round(ckpt.get('test_acc', 0) * 100, 2),
        'config':    ckpt.get('config', {}),
    })

df = pd.DataFrame(results).sort_values('filename')
print(df.to_string())
df.to_csv(
    r"C:\Users\milan\OneDrive\Documents\Project\ProtoPNet Crop Disease Research Workflow\Claude2\exp1 3\ablation_results_recovered.csv",
    index=False
)