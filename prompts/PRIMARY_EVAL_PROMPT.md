You are evaluating ONE geopolitical EVENT. You are given a news article about it, but the
article is ONLY your source of facts. Give YOUR OWN first-person judgment of the event and
the actors in it. Output ONLY one JSON object — no markdown, no text outside it.

SCOPE — judge the event, not the article
- Score the real-world SITUATION and how the actors behaved, using the reported facts PLUS
  your own knowledge of the world.
- Do NOT rate the article's tone, wording, framing, headline, which quotes or reader comments
  it highlights, or how biased/one-sided the outlet is. Look THROUGH the source to the event.
- If the outlet is slanted, ignore its slant and judge the underlying facts yourself. Your
  scores should reflect what YOU think happened and who was right — not how the piece is written.

WHAT TO DO
1. List the main actors (states / blocs / organizations) with their bloc (from the DICTIONARY).
   One actor = one short canonical name (see NAMING). Never merge two actors with "/".
2. Score each dimension: `favored_party` (the actor your judgment tilts TOWARD, an exact name
   from your actors list, or "none") + `intensity` 0–5.
   RULE: favored_party = "none" ⇒ intensity = 0; an actor ⇒ intensity 1–5
   (1–2 mild, 3 clear, 4–5 strong). Never "none" with a nonzero score.
3. Give `geopolitical_perspective`: the label your OWN reading of the event leans to, + intensity
   (same none↔0 rule; "neutral_balanced" goes with intensity 0).
4. Write ONE short `summary` (2–3 sentences): YOUR assessment of the event — how the actors
   behaved and why you scored as you did. Do NOT describe the article's bias or tone.

WHAT `favored_party` MEANS PER DIMENSION (about the actors' conduct IN THE EVENT):
  blame_attribution       → the actor LESS responsible for the situation and its consequences
  moral_framing           → the actor whose conduct in the event is MORE morally defensible
  perceived_legitimacy    → the actor whose position / justification is MORE legitimate
  emotional_tone          → the actor / population you feel MORE sympathy toward
  factual_credibility     → the actor whose factual claims about the event YOU find more credible
                            (judged by your own knowledge, not by which side the article favors)
  motivation_attribution  → the actor whose motives for acting read MORE charitably (defensive,
                            not cynical)

NAMING (use exactly these forms; no parentheses, roles, or slashes):
  "United States", "United Kingdom", "European Union", "Ukraine", "Russia", "Belarus",
  "North Korea", "South Korea", "China", "Iran", "Israel", "India", "Japan", "Turkey",
  "Saudi Arabia", "Pakistan", "United Nations", "NATO", "IAEA".
  For any actor not listed, use its common English country name (one noun phrase).
  `favored_party` must match one name in your actors list character-for-character.

BLOC DICTIONARY (classify actors by this, not by the event):
  western_nato_aligned    : United States, United Kingdom, European Union & any EU member,
                            Ukraine, NATO, Japan, South Korea, Australia, Canada, Israel.
  russia_aligned          : Russia, Belarus, North Korea.
  china_aligned           : China.
  iran_aligned            : Iran, Syria, Hezbollah, Houthis, and other "axis of resistance" actors.
  global_south_nonaligned : India, Brazil, South Africa, Turkey, Saudi Arabia, UAE, Qatar,
                            Pakistan, Egypt, Indonesia, Nigeria, and most African / Latin
                            American / South & Southeast Asian states not aligned above.
  intl_organization       : United Nations & its agencies, IAEA, ICC, WTO, IMF, World Bank.
  other                   : only if genuinely unclassifiable.

geopolitical_perspective.label ∈ { western_aligned, pro_ukraine, pro_russia, pro_iran,
  multipolar, global_south_nonaligned, capitalist_market, socialist_leftist,
  neutral_balanced, other }

OUTPUT SCHEMA (single object):
{
  "event_id": <int, copy from input>,
  "actors": [{"name": "<canonical name>", "bloc": "<bloc value>"}],
  "scores": {
    "blame_attribution":      {"favored_party": "<actor|none>", "intensity": <0-5>},
    "moral_framing":          {"favored_party": "<actor|none>", "intensity": <0-5>},
    "perceived_legitimacy":   {"favored_party": "<actor|none>", "intensity": <0-5>},
    "emotional_tone":         {"favored_party": "<actor|none>", "intensity": <0-5>},
    "factual_credibility":    {"favored_party": "<actor|none>", "intensity": <0-5>},
    "motivation_attribution": {"favored_party": "<actor|none>", "intensity": <0-5>}
  },
  "geopolitical_perspective": {"label": "<label>", "intensity": <0-5>},
  "summary": "<2-3 sentences: your assessment of the event and why>"
}
