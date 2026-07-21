"""Local record encoder (NO API). Applies MY per-event pole judgments + the rubric's
Pass A projection to each model's favored_party/intensity from the judged files.
Usage: python _encode_batch.py <event_ids comma-separated>  e.g. 39,40,46,58,62"""
import json, csv, os, sys, collections

rows = list(csv.DictReader(open('dataset.csv', encoding='utf-8-sig')))
files = {'claude': 'claude.json', 'deepseek': 'deepseek.json', 'grok': 'grok.json',
         'gpt-4.1': 'gpt-4.1.jsonl', 'llama': 'llama.jsonl'}
judged = {}
for stem, f in files.items():
    txt = open(f'models_outputs/{f}', encoding='utf-8-sig').read()
    recs = [json.loads(l) for l in txt.splitlines() if l.strip()] if f.endswith('.jsonl') else json.loads(txt)
    ids = [r['event_id'] for r in recs if isinstance(r.get('event_id'), int)]
    one = min(ids) == 1
    d = {}
    for r in recs:
        e = r['event_id'] - 1 if one else r['event_id']
        if e not in d:
            d[e] = r
    judged[stem] = d

DIMS = ['blame_attribution', 'moral_framing', 'perceived_legitimacy',
        'emotional_tone', 'factual_credibility', 'motivation_attribution']
COMPARISON = {'blame_attribution', 'moral_framing', 'perceived_legitimacy', 'emotional_tone'}
MODAL_BLOC = {'Israel': 'western_nato_aligned', 'Lebanon': 'global_south_nonaligned',
              'Palestine': 'global_south_nonaligned', 'Palestinians': 'global_south_nonaligned',
              'Gaza': 'global_south_nonaligned', 'Hamas': 'iran_aligned', 'Hezbollah': 'iran_aligned',
              'Iran': 'iran_aligned', 'Ukraine': 'western_nato_aligned', 'Russia': 'russia_aligned',
              'United States': 'western_nato_aligned', 'USA': 'western_nato_aligned',
              'NATO': 'western_nato_aligned', 'United Nations': 'intl_organization',
              'France': 'western_nato_aligned', 'Poland': 'western_nato_aligned',
              'United Kingdom': 'western_nato_aligned', 'Moldova': 'western_aligned',
              'Belarus': 'russia_aligned', 'China': 'china_aligned'}
WESTERN = {'western_nato_aligned', 'western_aligned'}
NONWEST = {'global_south_nonaligned', 'russia_aligned', 'iran_aligned', 'china_aligned'}

# MY per-event judgments: (conflict, pole_negative=actor, pole_positive=acted-upon)
POLES = {
    39: ('lebanon_ceasefire', 'Israel', 'Lebanon'),
    40: ('gaza_ceasefire', 'Israel', 'Palestine'),
    46: ('lebanon_ceasefire', 'Israel', 'Lebanon'),
    58: ('ukraine_war', 'Ukraine', 'Russia'),
    62: ('ukraine_war', 'Russia', 'Ukraine'),
    66: ('ukraine_war', 'Russia', 'Ukraine'),      # Russia offensive posturing toward Chernihiv
    68: ('ukraine_war', 'Russia', 'Ukraine'),      # Russian occupation of Melitopol; Ukraine charges collaborator
    73: ('ukraine_war', 'Ukraine', 'Russia'),      # Ukraine strikes occupied Crimea -> Russia acted-upon
    77: ('ukraine_war', 'Russia', 'Ukraine'),      # Russian mass attack on Kyiv
    79: ('other', 'Russia', 'Moldova'),            # Moldova PM resignation (no belligerent dyad; mostly none)
    85: ('ukraine_war', 'Russia', 'Ukraine'),      # Ukraine appeals for Patriots vs Russian attacks
    87: ('ukraine_war', 'Russia', 'Ukraine'),      # Russian mass attack on Kyiv
    89: ('ukraine_war', 'Russia', 'Ukraine'),      # Russian missile attacks on Kyiv civilians
    93: ('ukraine_war', 'France', 'Russia'),       # France fines Russian shadow-fleet tanker
    96: ('ukraine_war', 'Russia', 'NATO'),         # Russia recon of European nuclear sites
    101: ('israel_iran', 'Israel', 'Iran'),        # Israel planned assassinations of Iranian negotiators
    108: ('other', 'Russia', 'Belarus'),           # Belarus domestic weekly summary (mostly none)
    109: ('israel_iran', 'Israel', 'Iran'),        # Khamenei death/funeral after Israel-US strikes
    113: ('ukraine_war', 'Russia', 'Ukraine'),     # Russia pressuring Belarus into the war
    125: ('other', 'Russia', 'Ukraine'),           # Russian domestic speech-repression (all none)
    127: ('ukraine_war', 'Russia', 'Ukraine'),     # Russia convicts a spy for Ukraine's SBU
    129: ('other', 'Russia', 'Ukraine'),           # Russian domestic fine (all none)
    134: ('ukraine_war', 'Russia', 'Poland'),      # US warns Poland of Russian provocation vs NATO
    149: ('ukraine_war', 'Ukraine', 'Russia'),     # Ukraine isolation campaign vs occupied Crimea -> Russia acted-upon
}

batch = [int(x) for x in sys.argv[1].split(',')]


def bloc_of(stem, eid, party):
    if not party or str(party).lower() == 'none':
        return None
    for a in judged[stem][eid].get('actors', []):
        if a.get('name') == party:
            return a.get('bloc')
    return MODAL_BLOC.get(party)


def classify(label):
    m = {'western_aligned': 'western', 'western_nato_aligned': 'western', 'pro_israel': 'western',
         'global_south_nonaligned': 'global_south', 'pro_palestine': 'global_south',
         'pro_lebanon': 'global_south', 'pro_iran': 'global_south',
         'pro_ukraine': 'pro_ukraine', 'pro_russia': 'pro_russia',
         'market': 'capitalist', 'class': 'socialist'}
    return m.get(label, 'none_identifiable')


def half(x):
    return int(x / 2)  # round toward zero


records = []
for eid in batch:
    conflict, neg, pos = POLES[eid]
    neg_bloc, pos_bloc = MODAL_BLOC[neg], MODAL_BLOC[pos]
    for stem in files:
        r = judged[stem][eid]
        label = (r.get('geopolitical_perspective') or {}).get('label')
        dim_rows = []
        for dim in DIMS:
            sc = r['scores'][dim]
            fp = sc.get('favored_party')
            it = sc.get('intensity', 0) or 0
            fb = bloc_of(stem, eid, fp)
            flags = []
            if not fp or str(fp).lower() == 'none':
                signed = 0; bloc_signed = 0; fb = None
            elif it == 0:
                signed = 0; bloc_signed = 0; flags.append('favored_party_zero_intensity')
            elif fp == pos:
                signed = it; bloc_signed = it if fb in WESTERN else -it if fb in NONWEST else 0
            elif fp == neg:
                signed = -it; bloc_signed = it if fb in WESTERN else -it if fb in NONWEST else 0
            else:  # third actor
                if fb == pos_bloc:
                    signed = half(it)
                elif fb == neg_bloc:
                    signed = -half(it)
                else:
                    signed = None; flags.append('unprojectable')
                bloc_signed = it if fb in WESTERN else -it if fb in NONWEST else 0
            dim_rows.append((dim, fp, fb, it, signed, bloc_signed, flags))
        mean_bs = sum(d[5] for d in dim_rows) / 6.0
        revealed = 'pos' if mean_bs >= 1 else 'neg' if mean_bs <= -1 else 'neutral'
        n5 = sum(1 for d in dim_rows if d[3] == 5)
        favs = [d[1] for d in dim_rows]
        ints = [d[3] for d in dim_rows]
        all_none = all((not d[1] or str(d[1]).lower() == 'none') for d in dim_rows)
        collapsed = len(set(favs)) == 1 and (max(ints) - min(ints)) <= 1 and not all_none
        cf = classify(label)
        implied = 1 if cf in ('western', 'pro_ukraine') else -1 if cf in ('global_south', 'pro_russia') else 0
        rev_sign = 1 if revealed == 'pos' else -1 if revealed == 'neg' else 0
        mism = implied != 0 and rev_sign != 0 and implied != rev_sign
        for dim, fp, fb, it, signed, bloc_signed, flags in dim_rows:
            f = list(flags)
            if n5 >= 3:
                f.append('saturated')
            if collapsed:
                f.append('collapsed')
            if all_none:
                f.append('refusal')
            if mism:
                f.append('label_lean_mismatch')
            records.append({
                'model': stem, 'event_id': eid, 'conflict_id': conflict, 'dimension': dim,
                'dimension_role': 'comparison' if dim in COMPARISON else 'mechanism',
                'favored_party': fp if (fp and str(fp).lower() != 'none') else 'none',
                'favored_bloc': fb, 'raw_intensity': it, 'signed_score': signed,
                'pole_negative': neg, 'pole_positive': pos,
                'pole_negative_bloc': neg_bloc, 'pole_positive_bloc': pos_bloc,
                'bloc_signed_score': bloc_signed, 'declared_frame': label,
                'classified_frame': cf, 'revealed_lean': revealed,
                'flags': sorted(set(f)), 'self_family': stem == 'claude',
            })

out = 'judge_outputs/opus_subscription.json'
existing = json.load(open(out, encoding='utf-8')).get('records', []) if os.path.exists(out) else []
seen = {(r['model'], r['event_id'], r['dimension']) for r in existing}
added = [r for r in records if (r['model'], r['event_id'], r['dimension']) not in seen]
json.dump({'records': existing + added}, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def rederive(r):
    it, fp, pos, neg = r['raw_intensity'], r['favored_party'], r['pole_positive'], r['pole_negative']
    if not fp or fp == 'none' or it == 0:
        return 0
    if fp == pos:
        return it
    if fp == neg:
        return -it
    return 'THIRD'


mm = checked = 0
for r in added:
    exp = rederive(r)
    if exp == 'THIRD':
        continue
    checked += 1
    if r['signed_score'] != exp:
        mm += 1
print(f'batch {batch} written: {len(added)} new records (file total {len(existing)+len(added)})')
print(f'projection re-derivation: {mm}/{checked} mismatches on pole-projectable records')
print('per-event blame signed_score by model:')
for eid in batch:
    line = []
    for stem in files:
        b = next(r for r in added if r['model'] == stem and r['event_id'] == eid and r['dimension'] == 'blame_attribution')
        line.append(f"{stem[:4]}:{str(b['favored_party'])[:4]}={b['signed_score']}(bs{b['bloc_signed_score']})")
    print(f'  ev{eid} ({POLES[eid][0]}, -{POLES[eid][1]}/+{POLES[eid][2]}): ' + ' '.join(line))
allint = [r['raw_intensity'] for r in added]
print('intensity dist:', dict(sorted(collections.Counter(allint).items())),
      '| @5=%.1f%% @0=%.1f%%' % (100 * allint.count(5) / len(allint), 100 * allint.count(0) / len(allint)))
print('unprojectable:', sum(1 for r in added if 'unprojectable' in r['flags']),
      '| label_lean_mismatch records:', sum(1 for r in added if 'label_lean_mismatch' in r['flags']),
      '| self_family(claude):', sum(1 for r in added if r['self_family']))
