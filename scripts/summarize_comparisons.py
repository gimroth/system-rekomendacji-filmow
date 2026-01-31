import json, os

def summarize(comp_dir):
    with open(os.path.join(comp_dir, 'comparison_report.json'), 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = []
    for ds in ['db','synthetic']:
        anfis = d[ds]['anfis']['summary'] if ds=='db' else d[ds]['anfis']['summary']
        anfis_rmse = d[ds]['anfis']['summary']['cv_rmse_mean']
        anfis_r2 = d[ds]['anfis']['summary']['cv_r2_mean']
        mlp_rmse = d[ds]['mlp']['rmse']
        mlp_r2 = d[ds]['mlp']['r2']
        delta = (mlp_rmse - anfis_rmse)
        better = 'MLP' if mlp_rmse < anfis_rmse else 'ANFIS'
        out.append(f"Dataset: {ds}\n  ANFIS RMSE={anfis_rmse:.4f}, R2={anfis_r2:.4f}\n  MLP   RMSE={mlp_rmse:.4f}, R2={mlp_r2:.4f}\n  Winner: {better} (delta RMSE={delta:.4f})\n")
    print('\n'.join(out))

if __name__=='__main__':
    import sys
    d = sys.argv[1] if len(sys.argv)>1 else 'comparison_anfis_mlp_multi_20260130_004610'
    summarize(d)
