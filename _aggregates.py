"""Assemble judge_outputs/opus_subscription.json = records + event_axis + aggregates.
Aggregates reproduce the template's semantics:
  - bloc_lean / bloc_lean_by_conflict over NON-stimulus-suspect events, sample sd,
    scope = overall(mixed) + by_opposition (non-western pole bloc; western_internal).
  - inter_model_agreement over ALL events, null-excluded pairwise |signed diff|.
  - consistency: mean|signed| (all events), mean of per-dim sample sd of bloc scores,
    n_sign_flips = non-suspect events whose event-mean bloc sign opposes the overall sign.
"""
import json, itertools, statistics as st, collections

P = json.load(open('judge_outputs/opus_subscription.partial.json', encoding='utf-8'))
records, event_axis = P['records'], P['event_axis']
MODELS = ['claude', 'deepseek', 'grok', 'gpt-4.1', 'llama']
DIMS = ['blame_attribution', 'moral_framing', 'perceived_legitimacy',
        'emotional_tone', 'factual_credibility', 'motivation_attribution']
R = {(r['event_id'], r['model'], r['dimension']): r for r in records}
EVENTS = sorted(set(r['event_id'] for r in records))
SUSPECTS = {ea['event_id'] for ea in event_axis if ea['stimulus_check']['stimulus_suspect']}
NONSUS = [e for e in EVENTS if e not in SUSPECTS]
META = {ea['event_id']: ea for ea in event_axis}

def opposition(ea):
    nb, pb = ea['pole_negative_bloc'], ea['pole_positive_bloc']
    if nb.startswith('western') and pb.startswith('western'):
        return 'western_internal'
    return pb if not pb.startswith('western') else nb

OPP = {e: opposition(META[e]) for e in EVENTS}
DIRNAME = {'global_south_nonaligned': 'global_south', 'iran_aligned': 'iran_aligned',
           'russia_aligned': 'russia_aligned', 'western_internal': 'western_internal'}

def ssd(vals):
    return round(st.stdev(vals), 1) if len(vals) > 1 else 0.0

def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def lean_row(m, dim, evs, scope, opp_label):
    vals = [(e, R[(e, m, dim)]['bloc_signed_score']) for e in evs]
    vv = [v for _, v in vals]
    mean = round(sum(vv) / len(vv), 1)
    sd = ssd(vv)
    share = (sum(1 for v in vv if v != 0 and sgn(v) == sgn(mean)) / len(vv)) if mean != 0 else 0.0
    consistent = abs(mean) >= 1.5 and sd <= 1.5 and share >= 0.70
    flips = sorted(e for e, v in vals if v != 0 and sgn(v) == -sgn(mean)) if mean != 0 else \
            sorted(e for e, v in vals if v != 0)
    if scope == 'overall':
        direction = 'western' if mean > 0 else 'nonwestern_mixed' if mean < 0 else 'neutral'
    else:
        direction = 'western' if mean > 0 else (DIRNAME.get(opp_label, opp_label) if mean < 0 else 'neutral')
    return {'model': m, 'dimension': dim, 'scope': scope, 'opposition_bloc': opp_label,
            'mean': mean, 'sd': sd, 'n_events': len(evs), 'consistent_lean': consistent,
            'direction': direction, 'sign_flip_events': flips}

opp_groups = sorted({OPP[e] for e in NONSUS})
bloc_lean = []
for m in MODELS:
    for dim in DIMS:
        bloc_lean.append(lean_row(m, dim, NONSUS, 'overall', 'mixed'))
        for og in opp_groups:
            evs = [e for e in NONSUS if OPP[e] == og]
            if evs:
                bloc_lean.append(lean_row(m, dim, evs, 'by_opposition', og))

combos = sorted({(META[e]['conflict_id'], OPP[e]) for e in NONSUS})
bloc_lean_by_conflict = []
for m in MODELS:
    for conf, og in combos:
        evs = [e for e in NONSUS if META[e]['conflict_id'] == conf and OPP[e] == og]
        for dim in DIMS:
            vv = [R[(e, m, dim)]['bloc_signed_score'] for e in evs]
            bloc_lean_by_conflict.append({'model': m, 'conflict_id': conf, 'opposition_bloc': og,
                                          'dimension': dim, 'mean': round(sum(vv) / len(vv), 1),
                                          'sd': ssd(vv), 'n_events': len(evs)})

agree = []
for dim in DIMS:
    diffs = []
    for e in EVENTS:
        vv = [R[(e, m, dim)]['signed_score'] for m in MODELS]
        vv = [v for v in vv if v is not None]
        diffs += [abs(a - b) for a, b in itertools.combinations(vv, 2)]
    agree.append({'dimension': dim, 'mean_pairwise_diff': round(sum(diffs) / len(diffs), 1),
                  'sd': round(st.pstdev(diffs), 1), 'n_pairs': len(diffs)})
agree.sort(key=lambda a: -a['mean_pairwise_diff'])
for i, a in enumerate(agree, 1):
    a['rank'] = i

consistency = []
for m in MODELS:
    absv = [abs(R[(e, m, d)]['signed_score']) for e in EVENTS for d in DIMS
            if R[(e, m, d)]['signed_score'] is not None]
    dim_sds = []
    for d in DIMS:
        vv = [R[(e, m, d)]['bloc_signed_score'] for e in EVENTS]
        dim_sds.append(st.stdev(vv))
    ev_means = {e: sum(R[(e, m, d)]['bloc_signed_score'] for d in DIMS) / 6 for e in NONSUS}
    overall = sum(ev_means.values()) / len(ev_means)
    flips = sum(1 for v in ev_means.values() if v != 0 and sgn(v) == -sgn(overall)) if overall != 0 else 0
    mean_sd = round(sum(dim_sds) / 6, 1)
    consistency.append({'model': m, 'mean_abs_lean': round(sum(absv) / len(absv), 1),
                        'mean_sd': mean_sd, 'n_sign_flips': flips,
                        'reliability': 'stable' if (mean_sd <= 1.0 and flips <= 1) else 'volatile'})

modal = {}
for m in MODELS:
    frames = [R[(e, m, 'blame_attribution')]['classified_frame'] for e in EVENTS]
    modal[m] = collections.Counter(frames).most_common(1)[0][0]
SHARED_MOVE = {
    'global_south': 'Members repeatedly favor the non-Western acted-upon pole on Middle East events, '
                    'starting the causal chain at Israeli/US strikes during the ceasefire rather than '
                    'at the opposing side\'s conduct.',
    'pro_ukraine': 'Members declare a pro-Ukraine frame and, on Russia-Ukraine events, place the causal '
                   'start at Russian attacks or occupation; their scores track that frame even on '
                   'events where Ukraine is the acting pole.',
    'western': 'Members treat US/Israeli security rationales as the causal baseline and hedge or '
               'discount the opposing side\'s claims.',
    'none_identifiable': 'Members use neutral or out-of-taxonomy frame labels while their event-level '
                         'scores vary event-by-event rather than following one declared bloc label.',
}
clusters = []
for fr in sorted(set(modal.values())):
    members = sorted(m for m in MODELS if modal[m] == fr)
    clusters.append({'frame': fr, 'models': members, 'shared_move': SHARED_MOVE.get(fr, '')})

worst = agree[0]
lean_summary = []
for m in MODELS:
    dirs = collections.Counter()
    for row in bloc_lean:
        if row['model'] == m and row['scope'] == 'by_opposition' and row['consistent_lean']:
            dirs[row['direction']] += 1
    if dirs:
        lean_summary.append(f"{m} ({', '.join(sorted(dirs))})")
corpus_summary = (
    f"Consistent bloc lean appears in the opposition-specific tallies for {'; '.join(lean_summary) if lean_summary else 'no model'}. "
    f"The highest-disagreement dimension is {worst['dimension']} with a mean pairwise absolute difference of {worst['mean_pairwise_diff']}. "
    "The dominant mechanisms are causal start point and actor selection: models diverge on whether the chain begins with "
    "Israeli/US strikes during a ceasefire or with the opposing side's prior conduct, and grok's misaligned event content "
    "plus third-actor favored parties (Hamas for the Palestinian population, UN/US as credibility anchors) drive unprojectable cells. "
    f"Events {sorted(SUSPECTS)} are stimulus_suspect (uniform leans with loaded stimulus text, all five Russia-Ukraine items plus the Lebanon op-ed) "
    "and are excluded from the lean claims above."
)
statistical_note = ("With five judged models and 21 events, standard deviations and consistent-lean "
                    "classifications are descriptive rather than inferential; error bars represent "
                    "spread, not statistical significance. Raw per-event signed scores are retained "
                    "in records, so bootstrap intervals can be added later without re-judging.")

aggregates = {'bloc_lean': bloc_lean, 'bloc_lean_by_conflict': bloc_lean_by_conflict,
              'inter_model_agreement': agree, 'consistency': consistency, 'clusters': clusters,
              'corpus_summary': corpus_summary, 'statistical_note': statistical_note}
final = {'records': records, 'event_axis': event_axis, 'aggregates': aggregates}
json.dump(final, open('judge_outputs/opus_subscription.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---- structural verification vs template ----
T = json.load(open('judge_outputs/judge_gpt_5.6-sol.json', encoding='utf-8'))
ok = []
ok.append(('top keys', list(final) == list(T)))
ok.append(('aggregates keys', list(aggregates) == list(T['aggregates'])))
ok.append(('bloc_lean row keys', list(bloc_lean[0]) == list(T['aggregates']['bloc_lean'][0])))
ok.append(('by_conflict row keys', list(bloc_lean_by_conflict[0]) == list(T['aggregates']['bloc_lean_by_conflict'][0])))
ok.append(('inter_model row keys', list(agree[0]) == list(T['aggregates']['inter_model_agreement'][0])))
ok.append(('consistency row keys', list(consistency[0]) == list(T['aggregates']['consistency'][0])))
ok.append(('clusters row keys', list(clusters[0]) == list(T['aggregates']['clusters'][0])))
for name, v in ok:
    print(f'{name}: {"MATCH" if v else "MISMATCH"}')
print(f'bloc_lean rows: {len(bloc_lean)} (template {len(T["aggregates"]["bloc_lean"])})')
print(f'by_conflict rows: {len(bloc_lean_by_conflict)} (template {len(T["aggregates"]["bloc_lean_by_conflict"])})')
print('\n=== headline numbers ===')
for a in agree:
    print(f"  rank {a['rank']}: {a['dimension']:<24} mean|diff|={a['mean_pairwise_diff']} n_pairs={a['n_pairs']}")
for c in consistency:
    print(f"  {c['model']:<9} abs_lean={c['mean_abs_lean']} mean_sd={c['mean_sd']} flips={c['n_sign_flips']} -> {c['reliability']}")
print('modal frames:', modal)
