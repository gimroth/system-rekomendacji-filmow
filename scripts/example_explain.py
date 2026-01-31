import os, sys
sys.path.insert(0, os.path.abspath('.'))
from app.ml import loader
import numpy as np

# Simple local runner for explain logic (bypasses FastAPI dependencies)

def explain_from_features(match1=0.6, match2=0.5, match3=0.7):
    raw = loader.get_raw_model()
    if raw is None:
        loader.load_model()
        raw = loader.get_raw_model()
    if raw is None:
        print('No model loaded')
        return

    features = {'match1': float(match1), 'match2': float(match2), 'match3': float(match3)}

    # Try PyTorchANFIS
    try:
        from app.ml.anfis_pytorch import PyTorchANFIS
        if isinstance(raw, PyTorchANFIS):
            import torch
            arr = np.array([[features['match1'], features['match2'], features['match3']]], dtype=np.float32)
            raw.eval()
            with torch.no_grad():
                t = torch.from_numpy(arr)
                out, info = raw.forward(t)
                firing = info.get('firing').cpu().numpy().tolist()[0]
                weights = info.get('weights').cpu().numpy().tolist()[0]
                consequents = raw.consequents.detach().cpu().numpy().tolist() if hasattr(raw, 'consequents') else None
                contrib = [w * c for w, c in zip(weights, consequents)] if consequents is not None else None

                centers = raw.centers.detach().cpu().numpy()
                sigmas = np.exp(raw.log_sigmas.detach().cpu().numpy())
                membership = {}
                for i, mname in enumerate(['match1','match2','match3']):
                    vals = []
                    x = features[mname]
                    for j in range(raw.n_mfs):
                        c = centers[i,j]
                        s = sigmas[i,j]
                        deg = float(np.exp(-0.5 * ((x - c) / (s + 1e-8)) ** 2))
                        vals.append({'term': f'mf{j}', 'degree': deg, 'center': float(c), 'sigma': float(s)})
                    membership[mname] = vals

                rule_labels = []
                for comb in raw.rule_combinations:
                    rule_labels.append(tuple([f'mf{idx}' for idx in comb]))

                explain = {
                    'features': features,
                    'model_type': type(raw).__name__,
                    'membership': membership,
                    'rule_labels': rule_labels,
                    'firing': firing,
                    'weights': weights,
                    'consequents': consequents,
                    'per_rule_contribution': contrib,
                    'prediction': float(out.cpu().numpy().ravel()[0])
                }
                return explain
    except Exception as e:
        print('PyTorch explain failed', e)

    # Fallback legacy
    try:
        from app.ml.anfis_model import ANFISRecommender
        if isinstance(raw, ANFISRecommender):
            membership = {}
            for name in raw.aspect_names:
                terms = list(raw.input_variables[name].terms)
                vals = []
                for term in terms:
                    deg = float(raw._sample_membership(name, term, features.get(name,0.5)))
                    vals.append({'term': term, 'degree': deg})
                membership[name] = vals
            F = raw._compute_firing_matrix([features])
            firing = F[0].tolist()
            s = sum(firing) if sum(firing) > 0 else 1e-8
            weights = (F[0] / s).tolist()
            consequents = raw.consequents.tolist() if raw.consequents is not None else None
            contrib = [(w * c) for w, c in zip(weights, consequents)] if consequents is not None else None
            prediction = float(np.dot(weights, consequents)) if consequents is not None else float(raw.predict(features))
            explain = {
                'features': features,
                'model_type': type(raw).__name__,
                'membership': membership,
                'rule_labels': raw.rule_labels,
                'firing': firing,
                'weights': weights,
                'consequents': consequents,
                'per_rule_contribution': contrib,
                'prediction': prediction
            }
            return explain
    except Exception as e:
        print('Legacy explain failed', e)


if __name__ == '__main__':
    e = explain_from_features(0.6,0.5,0.7)
    import json
    print(json.dumps(e, indent=2))
