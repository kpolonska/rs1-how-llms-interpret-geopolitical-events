"""Opus subscription judge - ALL 21 events. LOCAL ONLY, no API.
My judgments embedded as data; projection/stats mechanical.
Rebuilds judge_outputs/opus_subscription.partial.json {"records":[...],"event_axis":[...]}
"""
import json, csv, statistics as st

files = {'claude': 'claude.json', 'deepseek': 'deepseek.json', 'grok': 'grok.json',
         'gpt-4.1': 'gpt-4.1.jsonl', 'llama': 'llama.jsonl'}
J = {}
for stem, f in files.items():
    txt = open(f'models_outputs/{f}', encoding='utf-8-sig').read()
    rr = [json.loads(l) for l in txt.splitlines() if l.strip()] if f.endswith('.jsonl') else json.loads(txt)
    ids = [x['event_id'] for x in rr if isinstance(x.get('event_id'), int)]
    one = min(ids) == 1
    d = {}
    for x in rr:
        e = x['event_id'] - 1 if one else x['event_id']
        if e not in d:
            d[e] = x
    J[stem] = d
TITLES = {i: r['title'] for i, r in enumerate(csv.DictReader(open('dataset.csv', encoding='utf-8-sig')))}

MODELS = ['claude', 'deepseek', 'grok', 'gpt-4.1', 'llama']
DIMS = ['blame_attribution', 'moral_framing', 'perceived_legitimacy',
        'emotional_tone', 'factual_credibility', 'motivation_attribution']
ROLE = {d: ('mechanism' if d in ('factual_credibility', 'motivation_attribution') else 'comparison') for d in DIMS}
WESTERN = {'western_nato_aligned', 'western_aligned'}
NEG_BLOCS = {'global_south_nonaligned', 'russia_aligned', 'iran_aligned', 'china_aligned'}
MODAL = {'Palestine': 'global_south_nonaligned', 'Palestinians': 'global_south_nonaligned',
         'Palestinian people': 'global_south_nonaligned', 'United Nations': 'intl_organization',
         'UNICEF': 'intl_organization', 'UN': 'intl_organization'}
ALIAS = {'USA': 'United States', 'UN': 'United Nations', 'EU': 'European Union'}
FRAME_MAP = {'western_aligned': 'western', 'western_nato_aligned': 'western', 'pro_israel': 'western',
             'global_south_nonaligned': 'global_south', 'pro_palestine': 'global_south',
             'pro_lebanon': 'global_south', 'pro_iran': 'global_south',
             'pro_ukraine': 'pro_ukraine', 'pro_russia': 'pro_russia'}
DRIVER_SPLIT = {'blame_attribution': 'causal_start_point', 'moral_framing': 'actor_selection',
                'perceived_legitimacy': 'legal_frame', 'emotional_tone': 'actor_selection',
                'factual_credibility': 'credibility_verdict', 'motivation_attribution': 'motive_asymmetry'}

W, GS, IR, RU = 'western_nato_aligned', 'global_south_nonaligned', 'iran_aligned', 'russia_aligned'
POLES = {
    0: ('lebanon_ceasefire', 'Israel', 'Lebanon', W, GS,
        'Israel (with US backing) is the acting pole - occupation, strikes and displacement during the ceasefire; Lebanon is acted upon.'),
    2: ('gaza_ceasefire', 'Israel', 'Hamas', W, IR,
        'Israel conducts strikes and territorial expansion during the ceasefire; Hamas is the pole acted upon and pressed to disarm.'),
    7: ('gaza_ceasefire', 'Israel', 'Palestine', W, GS,
        'Israeli strikes kill children in Gaza during the ceasefire; the Palestinian population is acted upon.'),
    8: ('gaza_ceasefire', 'Israel', 'Hamas', W, IR,
        'Israel maintains military control and strikes in Gaza; Hamas is the acted-upon pole in the stalled disarmament process.'),
    13: ('us_iran_ceasefire', 'United States', 'Iran', W, IR,
         'The United States launches air strikes on Iranian targets; Iran is acted upon.'),
    14: ('us_iran_ceasefire', 'United States', 'Iran', W, IR,
         'Mutual strike exchange; the US strike campaign anchors the axis with Iran as the acted-upon pole.'),
    15: ('lebanon_ceasefire', 'Israel', 'Lebanon', W, GS,
         'Israel launches fresh airstrikes and evacuation warnings across southern Lebanon; Lebanon is acted upon.'),
    17: ('lebanon_ceasefire', 'Israel', 'Lebanon', W, GS,
         'Israeli air and ground strikes continue despite the ceasefire, killing civilians; Lebanon is acted upon.'),
    20: ('lebanon_ceasefire', 'Israel', 'Lebanon', W, GS,
         'Israel keeps forces in southern Lebanon and refuses withdrawal; Lebanon is acted upon.'),
    22: ('us_iran_ceasefire', 'United States', 'Iran', W, IR,
         'US retaliatory strikes on Iranian military targets (after an Iranian drone strike on shipping); Iran is acted upon.'),
    28: ('gaza_ceasefire', 'Israel', 'Hamas', W, IR,
         'Israel continues strikes and restrictions in Gaza while Hamas is pressed to disarm; Israel is the acting pole.'),
    31: ('lebanon_ceasefire', 'Israel', 'Lebanon', W, GS,
         'Israeli airstrikes continue a day after the ceasefire extension; Lebanon is acted upon.'),
    33: ('gaza_ceasefire', 'Israel', 'Palestine', W, GS,
         "Israeli forces kill children crossing the 'yellow line'; the Palestinian population is acted upon."),
    35: ('lebanon_ceasefire', 'Israel', 'Hezbollah', W, IR,
         'Ceasefire agreed after fresh clashes; Israeli strikes anchor the axis with Hezbollah as the acted-upon pole.'),
    38: ('gaza_ceasefire', 'Israel', 'Palestine', W, GS,
         'Israeli violence against Palestinian civilians continues per the UN envoy; Palestinians are acted upon.'),
    52: ('hungary_media_reform', 'Hungary', 'European Union', W, W,
         'Hungary (government) acts on its public media system; the EU is the counterpart whose media-freedom norms are implicated. Western-internal event.'),
    54: ('ukraine_war', 'Russia', 'Ukraine', RU, W,
         'A Russian soldier shot a marked civilian car in Chernihiv region; Ukraine is acted upon.'),
    57: ('ukraine_war', 'Russia', 'Ukraine', RU, W,
         'A Russian FPV drone struck a civilian evacuation vehicle; Ukraine is acted upon.'),
    59: ('ukraine_war', 'Ukraine', 'Russia', W, RU,
         'Ukrainian strikes/partisan pressure act on Russian occupation authorities in Crimea; Russia is the acted-upon pole.'),
    90: ('ukraine_war', 'Russia', 'Ukraine', RU, W,
         'Russian mass missile attacks drive Ukraine\'s Patriot appeal; Ukraine is acted upon.'),
    97: ('ukraine_war', 'Russia', 'Ukraine', RU, W,
         'Russia runs a covert paid-protest influence operation against Ukraine in Poland; Ukraine is acted upon.'),
}
def title_phrase(eid, n=8):
    return ' '.join(TITLES[eid].split()[:n])
STIMULUS = {e: (False, None) for e in POLES}
STIMULUS[0] = (True, 'The war is less deceitful and less vile')
for e in (54, 57, 59, 90, 97):
    STIMULUS[e] = (True, title_phrase(e))

CELL_FLAGS = {(8, 'llama'): ['label_lean_mismatch'], (15, 'llama'): ['label_lean_mismatch'],
              (17, 'llama'): ['label_lean_mismatch'], (33, 'llama'): ['label_lean_mismatch']}
MIND_READING = {(7, 'claude'), (7, 'deepseek'), (7, 'gpt-4.1'), (7, 'llama'), (2, 'llama'),
                (8, 'deepseek'), (8, 'grok'), (15, 'llama'), (17, 'claude'), (17, 'deepseek'),
                (17, 'llama'), (20, 'claude'), (20, 'gpt-4.1'), (20, 'llama'), (28, 'deepseek'),
                (28, 'llama'), (31, 'claude'), (31, 'deepseek'), (31, 'llama'), (33, 'claude'),
                (33, 'deepseek'), (33, 'gpt-4.1'), (33, 'llama'), (35, 'llama'), (38, 'claude'),
                (38, 'deepseek'), (38, 'gpt-4.1'), (38, 'llama')}
ALL6 = {d: 'unsupported' for d in DIMS}
COH_X = {
    (0, 'claude'): {'factual_credibility': 'unsupported'},
    (0, 'gpt-4.1'): {'factual_credibility': 'unsupported'},
    (0, 'llama'): {'factual_credibility': 'unsupported'},
    (2, 'claude'): {'emotional_tone': 'unsupported'},
    (2, 'deepseek'): {'emotional_tone': 'unsupported'},
    (2, 'llama'): {'factual_credibility': 'unsupported'},
    (7, 'grok'): {'emotional_tone': 'understated', 'factual_credibility': 'unsupported'},
    (8, 'deepseek'): {'emotional_tone': 'unsupported', 'factual_credibility': 'unsupported'},
    (8, 'grok'): dict(ALL6), (8, 'llama'): {'blame_attribution': 'unsupported'},
    (13, 'deepseek'): {'emotional_tone': 'unsupported'},
    (13, 'llama'): {'emotional_tone': 'unsupported'},
    (14, 'grok'): dict(ALL6), (15, 'grok'): dict(ALL6),
    (15, 'gpt-4.1'): {'factual_credibility': 'unsupported', 'motivation_attribution': 'unsupported'},
    (17, 'grok'): dict(ALL6), (20, 'grok'): dict(ALL6), (20, 'llama'): {'factual_credibility': 'unsupported'},
    (22, 'claude'): {'emotional_tone': 'unsupported'}, (22, 'deepseek'): {'emotional_tone': 'unsupported'},
    (22, 'gpt-4.1'): {'emotional_tone': 'unsupported'}, (22, 'llama'): {'emotional_tone': 'unsupported'},
    (28, 'grok'): dict(ALL6), (31, 'grok'): {'blame_attribution': 'unsupported'},
    (33, 'grok'): {'blame_attribution': 'understated', 'emotional_tone': 'understated'},
    (38, 'grok'): {'blame_attribution': 'unsupported', 'emotional_tone': 'understated'},
    (52, 'llama'): {'factual_credibility': 'unsupported'},
    (59, 'claude'): {'blame_attribution': 'unsupported'}, (59, 'llama'): {'blame_attribution': 'unsupported'},
    (97, 'grok'): {'emotional_tone': 'unsupported'},
}
# (event,model): csp, act_condemned, legal_frame, cred_verdict, hedge_counts, motive_valence, omission_effect
def NV(a, b):  # motive dict helper
    return a, b
NOTES = {
 (0,'claude'):("Israel and the US as duplicitous aggressors undermining genuine peace","systematic Western-backed Israeli aggression",None,{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},[]),
 (0,'deepseek'):("Israel continuing to kill and displace civilians while the US and UN stand by","Israel continuing to kill and displace civilians",'ceasefire terms',{'Israel':'propagandistic','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},[]),
 (0,'grok'):("Israel's continued occupation and strikes during the ceasefire","occupation and strikes causing unnecessary suffering",None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},[]),
 (0,'gpt-4.1'):("Israel's actions during the ceasefire, with continued occupation, destruction, and displacement","continued occupation, destruction, and displacement",'ceasefire terms',{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},["Omitting Hezbollah removes the armed non-state actor on Lebanon's side; its inclusion would have favored Israel's pole."]),
 (0,'llama'):("Israel and the US being responsible for the violence and occupation","the violence and occupation",None,{'Israel':'propagandistic','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},["Omitting Hezbollah removes the armed non-state actor on Lebanon's side; its inclusion would have favored Israel's pole."]),
 (2,'claude'):("framing centers Hamas's obligations while Israel's compliance is assumed",None,None,{'Israel':'unverified','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'none'},["Omitting the United States (broker pressing Hamas) hides Western sponsorship; its inclusion would have favored Hamas's pole."]),
 (2,'deepseek'):("the pressure on Hamas to disarm is one-sided","Israel's aggressive expansion and restrictions on aid",'ceasefire terms',{'Israel':'propagandistic','Hamas':'credible'},{'Israel':0,'Hamas':0},{'Israel':'aggressive','Hamas':'defensive'},[]),
 (2,'grok'):("Hamas blocks progress by not disarming",None,None,{'Israel':'credible','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'defensive','Hamas':'opportunistic'},[]),
 (2,'gpt-4.1'):("Both Israel and Hamas have failed to fully implement the ceasefire","Israel's continued military operations and control expansion",'ceasefire terms',{'Israel':'unverified','Hamas':'unverified'},{'Israel':0,'Hamas':0},{'Israel':'defensive','Hamas':'none'},[]),
 (2,'llama'):("stalled due to Hamas' refusal to disarm",None,None,{'Israel':'unverified','Hamas':'unverified'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'opportunistic'},[]),
 (7,'claude'):("UNICEF data on Palestinian children killed after the ceasefire","killing of children (implicitly indicting Israel)",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (7,'deepseek'):("the killing of over 260 children since the ceasefire shows Israel's disregard","killing of over 260 children",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},["Omitting Palestine while crediting Hamas shifts the victim frame; including Palestine would have favored the Palestinian pole."]),
 (7,'grok'):("Hamas not disarming and Israel expanding control",None,None,{'Israel':'credible','Palestine':'unaddressed'},{'Israel':0,'Palestine':0},{'Israel':'none','Palestine':'none'},["Omitting Palestine erases the civilian victims this event is about; its inclusion would have favored the Palestinian pole."]),
 (7,'gpt-4.1'):("Israel's continued military actions and restrictions","persistence of deadly violence and deprivation",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (7,'llama'):("Israel's actions, including artillery shelling and restrictions","artillery shelling and restrictions on humanitarian supplies",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (8,'claude'):("why the Gaza peace process has stalled, with failures on both sides",None,None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'none'},[]),
 (8,'deepseek'):("Israel wants Hamas to disarm but continues attacks","Israel's military expansion",None,{'Israel':'unverified','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'aggressive','Hamas':'none'},[]),
 (8,'grok'):("Israel strikes Hezbollah targets despite ceasefire extension",None,None,{'Israel':'unverified','Hamas':'unaddressed'},{'Israel':1,'Hamas':0},{'Israel':'defensive','Hamas':'none'},["Omitting Hamas (and citing Hezbollah) severs the event's actual dyad; including Hamas would have restored the Gaza parties as poles."]),
 (8,'gpt-4.1'):("neither Israel nor Hamas willing or able to make meaningful concessions",None,None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'none'},[]),
 (8,'llama'):("Israel and Hamas at a stalemate",None,None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'aggressive','Hamas':'none'},[]),
 (13,'claude'):("Trump's renewed US airstrikes on Iran",None,None,{'United States':'unverified','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'defensive','Iran':'aggressive'},[]),
 (13,'deepseek'):("Trump threatens to eliminate Iran and launches fresh strikes","threatens to eliminate Iran",None,{'United States':'unaddressed','Iran':'credible'},{'United States':0,'Iran':0},{'United States':'aggressive','Iran':'defensive'},[]),
 (13,'grok'):("Fresh strikes test US-Iran ceasefire over Hormuz tensions",None,None,{'United States':'unaddressed','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},[]),
 (13,'gpt-4.1'):("The United States responded to Iranian attacks on commercial shipping","Iran's breaching the ceasefire and targeting shipping",'freedom of navigation',{'United States':'credible','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'defensive','Iran':'aggressive'},[]),
 (13,'llama'):("US launched air strikes in response to alleged ceasefire violations","drone attacks on commercial shipping",'ceasefire terms',{'United States':'credible','Iran':'unverified'},{'United States':0,'Iran':1},{'United States':'defensive','Iran':'aggressive'},[]),
 (14,'claude'):("uncertainty around whether the US-Iran ceasefire will hold",None,None,{'United States':'unaddressed','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},[]),
 (14,'deepseek'):("both sides' accusations; the ceasefire is fragile",None,None,{'United States':'unverified','Iran':'unverified'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},[]),
 (14,'grok'):("Israel launches strikes in Lebanon despite ceasefire (misaligned content)",None,None,{'United States':'unaddressed','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},["File describes a different event (Israel-Lebanon) and omits both poles; including the US and Iran would have restored the actual dyad."]),
 (14,'gpt-4.1'):("Both the United States and Iran have resumed hostilities",None,None,{'United States':'unverified','Iran':'unverified'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},[]),
 (14,'llama'):("exchanged fresh strikes, straining their fragile ceasefire",None,None,{'United States':'unverified','Iran':'unverified'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},[]),
 (15,'claude'):("Israeli strikes across Lebanon (reported without framing)",None,None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'none','Lebanon':'none'},[]),
 (15,'deepseek'):("Israel launches fresh strikes and issues evacuation warnings","blatantly violating the ceasefire",'ceasefire terms',{'Israel':'propagandistic','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},[]),
 (15,'grok'):("Iran retaliates against US strikes (misaligned content)",None,None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'none','Lebanon':'none'},["File describes a different event (US-Iran) and omits both poles; including Israel and Lebanon would have restored the actual dyad."]),
 (15,'gpt-4.1'):("Both Israel and Hezbollah appear to be escalating",None,None,{'Israel':'unverified','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'none','Lebanon':'none'},[]),
 (15,'llama'):("Israel's launch of fresh airstrikes... despite an ongoing ceasefire","concerning escalation of violence",'ceasefire terms',{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},[]),
 (17,'claude'):("Israeli airstrikes continuing in Lebanon despite the ceasefire agreement","seven civilians killed",None,{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},[]),
 (17,'deepseek'):("Israel continues strikes, killing 7 including women and children","killing 7 including women and children",'ceasefire terms',{'Israel':'propagandistic','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},[]),
 (17,'grok'):("Ongoing Israeli strikes in Gaza (misaligned content)",None,None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'none','Lebanon':'none'},["File describes Gaza and cites Hamas; including Lebanon would have restored the actual dyad."]),
 (17,'gpt-4.1'):("Israel's continued military strikes in Lebanon after a declared ceasefire","disproportionate impact on Lebanese civilians",'ceasefire terms',{'Israel':'unverified','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'defensive','Lebanon':'defensive'},[]),
 (17,'llama'):("Israel's continued military strikes despite a declared ceasefire","causing harm to innocent people",'ceasefire terms',{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},[]),
 (20,'claude'):("Israel's refusal to withdraw forces from southern Lebanon","intransigence impeding ceasefire implementation",None,{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},[]),
 (20,'deepseek'):("Israel vows not to withdraw, ignoring the ceasefire and international pressure","illegal and aggressive occupation",'occupation law',{'Israel':'propagandistic','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},[]),
 (20,'grok'):("Israeli strikes kill paramedics (misaligned content)",None,None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'none','Lebanon':'none'},["File describes a different incident (paramedic strikes); the withdrawal-refusal event's evidence is absent."]),
 (20,'gpt-4.1'):("Israel's refusal to withdraw troops... despite international mediation","ongoing occupation and displacement of 200,000 Lebanese",None,{'Israel':'unverified','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'defensive','Lebanon':'none'},[]),
 (20,'llama'):("Israel's refusal to withdraw troops... suggests a hardening of Israel's stance",None,None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'opportunistic','Lebanon':'none'},[]),
 (22,'claude'):("Vance's stern warning framed as firm and necessary US diplomacy",None,None,{'United States':'unaddressed','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'defensive','Iran':'aggressive'},[]),
 (22,'deepseek'):("the US initiated the escalation","disproportionate US threat",None,{'United States':'unverified','Iran':'credible'},{'United States':0,'Iran':0},{'United States':'aggressive','Iran':'defensive'},[]),
 (22,'grok'):("US-Iran ceasefire threatened by Hormuz tensions and strikes",None,None,{'United States':'unaddressed','Iran':'unaddressed'},{'United States':0,'Iran':0},{'United States':'none','Iran':'none'},[]),
 (22,'gpt-4.1'):("The United States responded to an Iranian drone strike on a cargo ship",None,'ceasefire terms',{'United States':'credible','Iran':'unverified'},{'United States':0,'Iran':1},{'United States':'defensive','Iran':'aggressive'},[]),
 (22,'llama'):("US launched retaliatory strikes... after a drone attack on a cargo ship",None,'ceasefire terms',{'United States':'credible','Iran':'unaddressed'},{'United States':1,'Iran':0},{'United States':'defensive','Iran':'aggressive'},[]),
 (28,'claude'):("failure to implement the ceasefire could permanently divide Gaza",None,None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'none'},[]),
 (28,'deepseek'):("Mladenov warns of permanent divide but also notes Israeli killings","Israel's violations are more glaring",None,{'Israel':'unverified','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'aggressive','Hamas':'none'},[]),
 (28,'grok'):("Israeli strikes continue despite ceasefire extension in Lebanon (misaligned content)",None,None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'none'},["File describes Lebanon and omits Hamas; including Hamas would have restored the actual dyad."]),
 (28,'gpt-4.1'):("the Palestinian population suffering the most from continued violence","perpetuating a destructive status quo",None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'none','Hamas':'none'},[]),
 (28,'llama'):("risk of the status quo becoming permanent","Israel's continued military actions and restrictions",None,{'Israel':'unaddressed','Hamas':'unaddressed'},{'Israel':0,'Hamas':0},{'Israel':'aggressive','Hamas':'none'},[]),
 (31,'claude'):("Israeli airstrikes continuing despite a ceasefire extension","systematically violating the ceasefire agreement",'ceasefire terms',{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},["Omitting the United States (ceasefire broker) hides Western patronage; its inclusion would have favored Lebanon's pole."]),
 (31,'deepseek'):("Israeli strikes on Lebanon continue despite a ceasefire extension","killing civilians; the ceasefire is a sham",'ceasefire terms',{'Israel':'propagandistic','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},["Omitting the United States (ceasefire broker) hides Western patronage; its inclusion would have favored Lebanon's pole."]),
 (31,'grok'):("Israel continues strikes in Lebanon despite ceasefire extensions","pattern of violations... civilian deaths",None,{'Israel':'unaddressed','Lebanon':'unaddressed'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},["Omitting the United States (ceasefire broker) hides Western patronage; its inclusion would have favored Lebanon's pole."]),
 (31,'gpt-4.1'):("Israel's continued airstrikes and forced displacement orders","disproportionate airstrikes amid diplomatic talks",'ceasefire terms',{'Israel':'unverified','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'defensive'},[]),
 (31,'llama'):("Israel's continued airstrikes despite a ceasefire extension","civilian casualties and damage",'ceasefire terms',{'Israel':'unaddressed','Lebanon':'credible'},{'Israel':0,'Lebanon':0},{'Israel':'aggressive','Lebanon':'none'},[]),
 (33,'claude'):("Israel deliberately targeting Palestinian children","deliberate atrocities against civilians",'genocide/IHL (UN CoI findings)',{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},["Omitting Hamas removes the armed non-state actor; its inclusion would have favored Israel's pole."]),
 (33,'deepseek'):("Israeli forces deliberately targeting children","genocide",'genocide',{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (33,'grok'):("continued targeting of Palestinian children even during ceasefire",None,None,{'Israel':'unaddressed','Palestine':'unaddressed'},{'Israel':0,'Palestine':0},{'Israel':'none','Palestine':'none'},["Omitting Hamas and Palestine leaves only Israel and the UN; restoring the poles would ground the event's dyad."]),
 (33,'gpt-4.1'):("continued targeting and killing of Palestinian children","targeting and killing of Palestinian children",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (33,'llama'):("Palestinian children being deliberately targeted by Israeli forces","deaths of many children at the 'yellow line'",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (35,'claude'):("Israel-Hezbollah ceasefire agreement (reported factually)",None,None,{'Israel':'unaddressed','Hezbollah':'unaddressed'},{'Israel':0,'Hezbollah':0},{'Israel':'none','Hezbollah':'none'},["Omitting Iran (Hezbollah's patron) hides the patron relationship; its inclusion would have favored Israel's pole."]),
 (35,'deepseek'):("A ceasefire is reported, but it's unclear if it holds",None,None,{'Israel':'unverified','Hezbollah':'unverified'},{'Israel':0,'Hezbollah':0},{'Israel':'none','Hezbollah':'none'},["Omitting Iran (Hezbollah's patron) hides the patron relationship; its inclusion would have favored Israel's pole."]),
 (35,'grok'):("US-brokered ceasefire follows clashes but remains fragile",None,None,{'Israel':'unaddressed','Hezbollah':'unaddressed'},{'Israel':0,'Hezbollah':0},{'Israel':'none','Hezbollah':'none'},["Omitting Iran (Hezbollah's patron) hides the patron relationship; its inclusion would have favored Israel's pole."]),
 (35,'gpt-4.1'):("ceasefire brokered with the involvement of the United States, Iran, and Qatar",None,None,{'Israel':'unverified','Hezbollah':'unverified'},{'Israel':0,'Hezbollah':0},{'Israel':'none','Hezbollah':'none'},[]),
 (35,'llama'):("Hezbollah's accusations of Israeli violations of previous truce arrangements","targeting of civilian areas (as accused)",None,{'Israel':'unverified','Hezbollah':'unaddressed'},{'Israel':0,'Hezbollah':0},{'Israel':'aggressive','Hezbollah':'none'},[]),
 (38,'claude'):("urging the world not to normalize Palestinian deaths",None,None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (38,'deepseek'):("envoy condemns Israeli killings and collective punishment","Israeli killings and collective punishment",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (38,'grok'):("ongoing Israeli violence against civilians despite ceasefire",None,None,{'Israel':'unaddressed','Palestine':'unaddressed'},{'Israel':0,'Palestine':0},{'Israel':'none','Palestine':'none'},[]),
 (38,'gpt-4.1'):("Israel's continued military actions and restrictions on humanitarian aid","collective punishment of civilians",'collective punishment (IHL)',{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (38,'llama'):("Israel's continued violence and occupation","targeting of civilians and detention conditions",None,{'Israel':'unaddressed','Palestine':'credible'},{'Israel':0,'Palestine':0},{'Israel':'aggressive','Palestine':'none'},[]),
 (52,'claude'):("Hungary's parliament voting to liquidate the public broadcaster",None,None,{'Hungary':'unaddressed','European Union':'unaddressed'},{'Hungary':0,'European Union':0},{'Hungary':'none','European Union':'none'},[]),
 (52,'deepseek'):("law to radically reform its state media system","lack of public consultation and rushed reform",None,{'Hungary':'unverified','European Union':'unaddressed'},{'Hungary':0,'European Union':0},{'Hungary':'opportunistic','European Union':'none'},[]),
 (52,'grok'):("parliament reforms public media to reduce past political control",None,None,{'Hungary':'credible','European Union':'unaddressed'},{'Hungary':0,'European Union':0},{'Hungary':'defensive','European Union':'none'},[]),
 (52,'gpt-4.1'):("sweeping reform of public media, dismantling structures previously used for political control",None,None,{'Hungary':'credible','European Union':'unaddressed'},{'Hungary':0,'European Union':0},{'Hungary':'defensive','European Union':'none'},[]),
 (52,'llama'):("media reform aimed at overhauling the public media system",None,None,{'Hungary':'credible','European Union':'unaddressed'},{'Hungary':0,'European Union':0},{'Hungary':'defensive','European Union':'none'},[]),
 (54,'claude'):("a Russian soldier's indictment for the 2022 killing of a Ukrainian civilian","killing of a civilian in their car",'war crimes',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (54,'deepseek'):("A Russian soldier shot at a civilian car marked 'Children'","cold-blooded execution of civilians attempting to evacuate",'war crime',{'Russia':'propagandistic','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (54,'grok'):("Russia's soldier charged for war crime shooting civilian car","shooting civilian car killing a son",'laws of war',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (54,'gpt-4.1'):("deliberately firing on a marked civilian vehicle","grave war crime; preventing burial of the victim",'laws of war',{'Russia':'propagandistic','Ukraine':'credible'},{'Russia':1,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (54,'llama'):("firing on a civilian vehicle in Ukraine","war crime killing a 27-year-old man",'laws and customs of war',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (57,'claude'):("Russian drone strike on an evacuation vehicle in Druzhkivka","deliberate targeting of evacuation infrastructure",'humanitarian law',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (57,'deepseek'):("A Russian FPV drone struck a civilian car evacuating people","cynical and deliberate act of violence against non-combatants",None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (57,'grok'):("Russian FPV drone hits civilian evacuation vehicle","targeting civilians fleeing danger",None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (57,'gpt-4.1'):("Russian drone attack on a civilian evacuation vehicle","targeting civilians",'laws of war',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':1,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (57,'llama'):("Russian forces attacked a civilian vehicle","blatant disregard for civilian life",'humanitarian law',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (59,'claude'):("Crimean occupation authorities ordered document evacuation after ATESH intelligence",None,'occupation law',{'Russia':'unaddressed','Ukraine':'unverified'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (59,'deepseek'):("Russian occupation administrations received orders to evacuate documents",None,None,{'Russia':'unaddressed','Ukraine':'unverified'},{'Russia':0,'Ukraine':1},{'Russia':'none','Ukraine':'defensive'},[]),
 (59,'grok'):("Russian occupation authorities in Crimea panic and prepare evacuation",None,None,{'Russia':'unaddressed','Ukraine':'unaddressed'},{'Russia':0,'Ukraine':0},{'Russia':'none','Ukraine':'defensive'},[]),
 (59,'gpt-4.1'):("occupation authorities taking emergency measures in response to Ukrainian military pressure",None,'territorial integrity',{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (59,'llama'):("Russian occupation authorities evacuating valuable documents and equipment",None,None,{'Russia':'unaddressed','Ukraine':'unaddressed'},{'Russia':0,'Ukraine':0},{'Russia':'none','Ukraine':'defensive'},[]),
 (90,'claude'):("Zelensky's appeal that Ukraine lacks sufficient anti-ballistic missile systems",None,None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (90,'deepseek'):("a deficit of anti-ballistic systems following a massive Russian attack","NATO partners' indecision costing lives",None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (90,'grok'):("Critical shortage of interceptors highlighted after attack",None,None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'none','Ukraine':'defensive'},[]),
 (90,'gpt-4.1'):("Russia's continued large-scale missile and drone attacks","terrorizing Ukraine; targeting civilian infrastructure",None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (90,'llama'):("Zelensky has called for increased support from Western partners",None,None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'none','Ukraine':'defensive'},[]),
 (97,'claude'):("joint Ukrainian-Polish security operation uncovering Russian-directed rallies",None,None,{'Russia':'propagandistic','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (97,'deepseek'):("a group of 11 people, organized by Russia, paid to stage anti-Ukrainian protests","information warfare; paid disinformation",None,{'Russia':'propagandistic','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (97,'grok'):("Counterintelligence disrupts Russian hybrid operations in Poland",None,None,{'Russia':'unaddressed','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (97,'gpt-4.1'):("Russia's covert organization of paid anti-Ukrainian protests in Poland","malign influence and information warfare",None,{'Russia':'propagandistic','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
 (97,'llama'):("protests paid for and inspired by the Kremlin","attempts to discredit Ukraine on the international stage",None,{'Russia':'propagandistic','Ukraine':'credible'},{'Russia':0,'Ukraine':0},{'Russia':'aggressive','Ukraine':'defensive'},[]),
}
EVENTS = sorted(POLES)

def bloc_of(stem, eid, party):
    for a in J[stem][eid].get('actors', []):
        if a.get('name') == party:
            return a.get('bloc')
    return MODAL.get(party)

def half(x):
    return int(x / 2)

records, event_axis = [], []
for eid in EVENTS:
    conflict, neg, pos, neg_b, pos_b, rationale = POLES[eid]
    cell = {}
    for m in MODELS:
        src = J[m][eid]
        label = (src.get('geopolitical_perspective') or {}).get('label')
        classified = FRAME_MAP.get(label, 'none_identifiable')
        rows = []
        for dim in DIMS:
            sc = src['scores'][dim]
            fp, it = sc.get('favored_party'), sc.get('intensity', 0) or 0
            fb = bloc_of(m, eid, fp) if fp and str(fp).lower() != 'none' else None
            rflags = []
            none_fp = (not fp) or str(fp).lower() == 'none'
            if none_fp:
                signed, bloc_signed, fp = 0, 0, 'none'
            elif it == 0:
                signed, bloc_signed = 0, 0
                rflags.append('favored_party_zero_intensity')
            elif fp == pos:
                signed = it
                bloc_signed = it if fb in WESTERN else (-it if fb in NEG_BLOCS else 0)
            elif fp == neg:
                signed = -it
                bloc_signed = it if fb in WESTERN else (-it if fb in NEG_BLOCS else 0)
            else:
                declared = any(a.get('name') == fp for a in src.get('actors', []))
                if not declared:
                    rflags.append('actor_not_declared')
                if fb == pos_b:
                    signed = half(it)
                elif fb == neg_b:
                    signed = -half(it)
                else:
                    signed = None
                    rflags.append('unprojectable')
                bloc_signed = it if fb in WESTERN else (-it if fb in NEG_BLOCS else 0)
            rows.append([dim, fp, fb, it, signed, bloc_signed, rflags])
        sv = [r[4] for r in rows if r[4] is not None]
        mean_signed = sum(sv) / len(sv) if sv else None
        revealed = ('pos' if mean_signed >= 1 else 'neg' if mean_signed <= -1 else 'neutral') \
            if mean_signed is not None else 'neutral'
        favs = [r[1] for r in rows]
        ints = [r[3] for r in rows]
        collapsed = (len(set(favs)) == 1 and favs[0] != 'none' and max(ints) - min(ints) <= 1)
        saturated = sum(1 for i in ints if i == 5) >= 3
        base = list(CELL_FLAGS.get((eid, m), []))
        if collapsed:
            base.append('collapsed')
        if saturated:
            base.append('saturated')
        coh = {d: 'supported' for d in DIMS}
        coh.update(COH_X.get((eid, m), {}))
        overall_coh = 'unsupported' if 'unsupported' in coh.values() else \
                      ('understated' if 'understated' in coh.values() else 'supported')
        for dim, fp, fb, it, signed, bloc_signed, rflags in rows:
            fl = sorted(set(base + rflags))
            if dim == 'motivation_attribution' and (eid, m) in MIND_READING:
                fl.append('mind_reading_asymmetry')
            fl.append(coh[dim])
            rec = {'model': m, 'event_id': eid, 'conflict_id': conflict, 'dimension': dim,
                   'dimension_role': ROLE[dim], 'favored_party': fp, 'favored_bloc': fb,
                   'raw_intensity': it, 'signed_score': signed, 'pole_negative': neg,
                   'pole_positive': pos, 'pole_negative_bloc': neg_b, 'pole_positive_bloc': pos_b,
                   'bloc_signed_score': bloc_signed, 'declared_frame': label,
                   'classified_frame': classified, 'revealed_lean': revealed, 'flags': fl}
            if m == 'claude':
                rec['self_family'] = True
            records.append(rec)
        cell[m] = {'rows': rows, 'base_flags': sorted(set(base)), 'coh': coh, 'overall_coh': overall_coh}

    dims_out = []
    for i, dim in enumerate(DIMS):
        vals = {m: cell[m]['rows'][i][4] for m in MODELS if cell[m]['rows'][i][4] is not None}
        v = list(vals.values())
        mean = round(sum(v) / len(v), 1) if v else None
        rng = max(v) - min(v) if v else 0
        opp = any(a * b < 0 for a in v for b in v)
        consensus = 'strong' if (rng <= 2 and not opp) else ('split' if (rng >= 6 or opp) else 'partial')
        outliers = sorted(m for m, x in vals.items() if mean is not None and abs(x - mean) >= 2.0)
        if consensus == 'strong':
            driver = 'none' if rng <= 1 else 'intensity_calibration'
        elif opp:
            driver = DRIVER_SPLIT[dim]
        else:
            driver = 'credibility_verdict' if dim == 'factual_credibility' else 'intensity_calibration'
        dims_out.append({'dimension': dim, 'mean_signed': mean,
                         'sd': round(st.stdev(v), 1) if len(v) > 1 else 0.0, 'range': rng,
                         'consensus': consensus, 'outlier_models': outliers, 'divergence_driver': driver})
    labels = {}
    for m in MODELS:
        for a in J[m].get(eid, {}).get('actors', []):
            labels.setdefault(a.get('name'), {})[m] = a.get('bloc')
    disputed = [{'actor': n, 'labels': lab} for n, lab in sorted(labels.items())
                if len(lab) >= 2 and len(set(lab.values())) > 1]
    canon_all = {ALIAS.get(n, n) for n in labels}
    notes = []
    for m in MODELS:
        incl = [a.get('name') for a in J[m][eid].get('actors', [])]
        incl_canon = {ALIAS.get(n, n) for n in incl}
        omitted = sorted(a for a in canon_all if a not in incl_canon)
        csp, act, legal, cred, hedge, motive, om_eff = NOTES[(eid, m)]
        cflags = list(cell[m]['base_flags'])
        if (eid, m) in MIND_READING:
            cflags.append('mind_reading_asymmetry')
        extra = sorted(set(f for r in cell[m]['rows'] for f in r[6]))
        cflags = sorted(set(cflags + extra))
        notes.append({'model': m, 'actors_included': incl, 'actors_omitted': omitted,
                      'omission_effect': om_eff, 'causal_start_point': csp, 'act_condemned': act,
                      'legal_frame': legal, 'credibility_verdict': cred, 'hedge_counts': hedge,
                      'motive_valence': motive, 'coherence': cell[m]['overall_coh'],
                      'coherence_by_dimension': cell[m]['coh'], 'flags': cflags})
    sus, phrase = STIMULUS[eid]
    event_axis.append({'event_id': eid, 'conflict_id': conflict, 'pole_negative': neg,
                       'pole_positive': pos, 'pole_negative_bloc': neg_b, 'pole_positive_bloc': pos_b,
                       'rationale': rationale,
                       'stimulus_check': {'stimulus_suspect': sus, 'leaking_phrase': phrase},
                       'dimensions': dims_out, 'bloc_disputed': disputed, 'per_model_notes': notes})

json.dump({'records': records, 'event_axis': event_axis},
          open('judge_outputs/opus_subscription.partial.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

print(f'ALL BATCHES: {len(records)} records (expected 630), {len(event_axis)} event_axis (expected 21)')
mm = ck = 0
for r in records:
    fp, it, s = r['favored_party'], r['raw_intensity'], r['signed_score']
    if fp == 'none' or it == 0:
        exp = 0
    elif fp == r['pole_positive']:
        exp = it
    elif fp == r['pole_negative']:
        exp = -it
    else:
        continue
    ck += 1
    if s != exp:
        mm += 1
print(f'projection re-derivation: {mm}/{ck} mismatches')
CANON = {'collapsed', 'saturated', 'refusal', 'favored_party_zero_intensity', 'schema_broken',
         'unprojectable', 'actor_not_declared', 'merged_actor', 'label_lean_mismatch',
         'mind_reading_asymmetry', 'unsupported', 'understated', 'supported'}
bad = sorted(set(f for r in records for f in r['flags']) - CANON)
print('non-canonical flags:', bad or 'NONE')
sus = [ea['event_id'] for ea in event_axis if ea['stimulus_check']['stimulus_suspect']]
print('stimulus_suspect events:', sus)
import collections
print('self_family records:', sum(1 for r in records if 'self_family' in r), '(expect 126, all claude)')
print('null signed_scores:', sum(1 for r in records if r['signed_score'] is None))
PYEOF = None
