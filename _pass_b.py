"""Pass B - cross-event aggregation over judge_outputs/opus_45events.json (LOCAL, no API).
Builds the rubric's final OUTPUT: {"records":[...], "event_axis":[...], "aggregates":{...}}
-> judge_outputs/opus_45events_final.json
Also dumps stimulus-suspect event text snippets for the leakage check.
"""
import json, csv, collections, statistics as st, sys

recs = json.load(open('judge_outputs/opus_45events.json', encoding='utf-8'))['records']
DIMS = ['blame_attribution', 'moral_framing', 'perceived_legitimacy',
        'emotional_tone', 'factual_credibility', 'motivation_attribution']
ROLE = {d: ('comparison' if d in DIMS[:4] else 'mechanism') for d in DIMS}
MODELS = ['claude', 'deepseek', 'gpt-4.1', 'grok', 'llama']
EVENTS = sorted(set(r['event_id'] for r in recs))
R = {(r['event_id'], r['model'], r['dimension']): r for r in recs}

# ---------- helpers ----------
def sd(vals):
    return round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0

def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0

# ---------- event meta ----------
meta = {}
for r in recs:
    meta[r['event_id']] = {k: r[k] for k in
        ('conflict_id', 'pole_negative', 'pole_positive', 'pole_negative_bloc', 'pole_positive_bloc')}

# per model-event frame (identical across the 6 dims)
frame = {}
for e in EVENTS:
    for m in MODELS:
        r = R[(e, m, 'blame_attribution')]
        frame[(e, m)] = (r['declared_frame'], r['classified_frame'], r['revealed_lean'])

# ---------- judged-file actors (for bloc_disputed) ----------
files = {'claude': 'claude.json', 'deepseek': 'deepseek.json', 'grok': 'grok.json',
         'gpt-4.1': 'gpt-4.1.jsonl', 'llama': 'llama.jsonl'}
judged = {}
for stem, f in files.items():
    txt = open(f'models_outputs/{f}', encoding='utf-8-sig').read()
    rows = [json.loads(l) for l in txt.splitlines() if l.strip()] if f.endswith('.jsonl') else json.loads(txt)
    ids = [r['event_id'] for r in rows if isinstance(r.get('event_id'), int)]
    one = min(ids) == 1
    d = {}
    for r in rows:
        e = r['event_id'] - 1 if one else r['event_id']
        if e not in d:
            d[e] = r
    judged[stem] = d

# ---------- A2/A3-style event_axis ----------
DRIVER_ON_SPLIT = {'blame_attribution': 'causal_start_point', 'moral_framing': 'actor_selection',
                   'perceived_legitimacy': 'legal_frame', 'emotional_tone': 'actor_selection',
                   'factual_credibility': 'credibility_verdict', 'motivation_attribution': 'motive_asymmetry'}
event_axis = []
suspects = []
for e in EVENTS:
    dims_out = []
    axes_consistent = 0
    for d in DIMS:
        vals = {m: R[(e, m, d)]['signed_score'] for m in MODELS if R[(e, m, d)]['signed_score'] is not None}
        v = list(vals.values())
        mean = round(sum(v) / len(v), 1) if v else None
        rng = (max(v) - min(v)) if v else 0
        opp = any(a * b < 0 for a in v for b in v)
        consensus = 'strong' if (rng <= 2 and not opp) else ('split' if (rng >= 6 or opp) else 'partial')
        outliers = sorted(m for m, x in vals.items() if mean is not None and abs(x - mean) >= 2.0)
        if consensus == 'strong':
            driver = 'none' if rng <= 1 else 'intensity_calibration'
        elif opp:
            driver = DRIVER_ON_SPLIT[d]
        else:
            driver = 'intensity_calibration'
        dims_out.append({'dimension': d, 'mean_signed': mean, 'sd': sd(v), 'range': rng,
                         'consensus': consensus, 'outlier_models': outliers,
                         'divergence_driver': driver})
        nz = [x for x in v if x != 0]
        if len(nz) == len(v) and len(v) >= 2 and len({sgn(x) for x in nz}) == 1 and rng <= 2:
            axes_consistent += 1
    suspect = axes_consistent >= 3
    if suspect:
        suspects.append(e)
    # bloc_disputed from the judged files' own actor labels
    labels = collections.defaultdict(dict)
    for m in MODELS:
        for a in judged[m].get(e, {}).get('actors', []):
            labels[a.get('name')][m] = a.get('bloc')
    disputed = [{'actor': n, 'labels': lab} for n, lab in sorted(labels.items())
                if len(lab) >= 2 and len(set(lab.values())) > 1]
    per_model = []
    for m in MODELS:
        fl = sorted(set(f for d in DIMS for f in R[(e, m, d)]['flags']))
        per_model.append({'model': m, 'causal_start_point': None, 'act_condemned': None,
                          'legal_frame': None, 'credibility_verdict': None, 'hedge_counts': None,
                          'motive_valence': None, 'coherence': None, 'flags': fl,
                          'self_family': m == 'claude'})
    mm = meta[e]
    event_axis.append({'event_id': e, 'conflict_id': mm['conflict_id'],
                       'pole_negative': mm['pole_negative'], 'pole_positive': mm['pole_positive'],
                       'pole_negative_bloc': mm['pole_negative_bloc'],
                       'pole_positive_bloc': mm['pole_positive_bloc'],
                       'rationale': f"{mm['pole_negative']} is the acting pole; "
                                    f"{mm['pole_positive']} is acted upon.",
                       'stimulus_check': {'stimulus_suspect': suspect, 'leaking_phrase': None},
                       'dimensions': dims_out, 'bloc_disputed': disputed,
                       'per_model_notes': per_model})

# ---------- B1 bloc lean ----------
bloc_lean = []
for m in MODELS:
    for d in DIMS:
        vals = [R[(e, m, d)]['bloc_signed_score'] for e in EVENTS]
        mean = round(sum(vals) / len(vals), 2)
        share_same = sum(1 for v in vals if sgn(v) == sgn(mean) and v != 0) / len(vals) if mean != 0 else 0
        consistent = abs(mean) >= 1.5 and sd(vals) <= 1.5 and share_same >= 0.70
        flips = sorted(e for e, v in zip(EVENTS, vals) if v != 0 and sgn(v) == -sgn(mean))
        tallies = collections.Counter(R[(e, m, d)]['favored_bloc'] for e in EVENTS
                                      if R[(e, m, d)]['favored_bloc'])
        if mean > 0:
            direction = 'western_nato'
        elif mean < 0:
            nonw = {b: c for b, c in tallies.items() if b not in ('western_nato_aligned', 'western_aligned')}
            direction = max(nonw, key=nonw.get) if nonw else 'global_south'
        else:
            direction = 'neutral'
        bloc_lean.append({'model': m, 'dimension': d, 'dimension_role': ROLE[d], 'mean': mean,
                          'sd': sd(vals), 'n_events': len(vals), 'consistent_lean': consistent,
                          'direction': direction, 'sign_flip_events': flips,
                          'favored_bloc_tallies': dict(tallies)})

bloc_lean_by_conflict = []
CONFLICTS = sorted(set(meta[e]['conflict_id'] for e in EVENTS))
for m in MODELS:
    for c in CONFLICTS:
        evs = [e for e in EVENTS if meta[e]['conflict_id'] == c]
        for d in DIMS:
            vals = [R[(e, m, d)]['bloc_signed_score'] for e in evs]
            bloc_lean_by_conflict.append({'model': m, 'conflict_id': c, 'dimension': d,
                                          'mean': round(sum(vals) / len(vals), 2),
                                          'sd': sd(vals), 'n_events': len(vals)})

# ---------- B2 inter-model agreement (event-local signed_score) ----------
agree = []
for d in DIMS:
    diffs = []
    for e in EVENTS:
        vals = [R[(e, m, d)]['signed_score'] for m in MODELS if R[(e, m, d)]['signed_score'] is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diffs.append(abs(vals[i] - vals[j]))
    agree.append({'dimension': d, 'mean_pairwise_diff': round(sum(diffs) / len(diffs), 2),
                  'sd': sd(diffs), 'n_pairs': len(diffs)})
agree.sort(key=lambda a: -a['mean_pairwise_diff'])
for i, a in enumerate(agree, 1):
    a['rank'] = i

# ---------- B3 reliability ----------
consistency = []
for m in MODELS:
    dim_means = [next(b['mean'] for b in bloc_lean if b['model'] == m and b['dimension'] == d) for d in DIMS]
    dim_sds = [next(b['sd'] for b in bloc_lean if b['model'] == m and b['dimension'] == d) for d in DIMS]
    ev_means = [sum(R[(e, m, d)]['bloc_signed_score'] for d in DIMS) / 6 for e in EVENTS]
    overall = sum(ev_means) / len(ev_means)
    flips = sum(1 for x in ev_means if x != 0 and sgn(x) == -sgn(overall))
    mean_sd = round(sum(dim_sds) / 6, 2)
    consistency.append({'model': m, 'mean_abs_lean': round(sum(abs(x) for x in dim_means) / 6, 2),
                        'mean_sd': mean_sd, 'n_sign_flips': flips,
                        'reliability': 'stable' if (mean_sd <= 1.0 and flips <= 1) else 'volatile'})

# ---------- B4 clusters (modal classified frame) ----------
modal = {}
for m in MODELS:
    frames = [frame[(e, m)][1] for e in EVENTS]
    modal[m] = collections.Counter(frames).most_common(1)[0][0]
clusters = []
for fr in sorted(set(modal.values())):
    members = sorted(m for m in MODELS if modal[m] == fr)
    clusters.append({'frame': fr, 'models': members, 'shared_move': None})  # filled in patch step

aggregates = {'bloc_lean': bloc_lean, 'bloc_lean_by_conflict': bloc_lean_by_conflict,
              'inter_model_agreement': [{k: a[k] for k in ('dimension', 'mean_pairwise_diff', 'sd', 'rank')}
                                        for a in agree],
              'consistency': consistency, 'clusters': clusters, 'corpus_summary': None}

final = {'records': recs, 'event_axis': event_axis, 'aggregates': aggregates}
json.dump(final, open('judge_outputs/opus_45events_final.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---------- console report ----------
print('=== B1 bloc lean (mean bloc_signed; + = western_nato) ===')
hdr = f"{'model':<9}" + ''.join(f'{d[:12]:>14}' for d in DIMS) + f"{'consistent?':>13}"
print(hdr)
for m in MODELS:
    row = [next(b for b in bloc_lean if b['model'] == m and b['dimension'] == d) for d in DIMS]
    cons = [b['dimension'][:5] for b in row if b['consistent_lean']]
    print(f'{m:<9}' + ''.join(f"{b['mean']:>10} ({b['sd']:>4})"[:14].rjust(14) for b in row)
          + f"{','.join(cons) if cons else '-':>13}")
print('\n=== B2 inter-model agreement (rank 1 = most disagreement, event-local axis) ===')
for a in agree:
    print(f"  {a['rank']}. {a['dimension']:<24} mean|diff|={a['mean_pairwise_diff']:<5} sd={a['sd']}")
print('\n=== B3 reliability ===')
for c in consistency:
    print(f"  {c['model']:<9} mean_abs_lean={c['mean_abs_lean']:<5} mean_sd={c['mean_sd']:<5} "
          f"flips={c['n_sign_flips']:<3} -> {c['reliability']}")
print('\n=== B4 modal frames ===')
for m in MODELS:
    print(f'  {m:<9} modal={modal[m]}')
print('\n=== stimulus_suspect events ===', suspects)

# snippets for leakage check
with open('_suspect_snippets.txt', 'w', encoding='utf-8') as f:
    rows = list(csv.DictReader(open('dataset.csv', encoding='utf-8-sig')))
    for e in suspects:
        f.write(f"### EVENT {e} | {rows[e]['title']}\n{rows[e]['full_text'][:500]}\n\n")
print('snippets -> _suspect_snippets.txt')
