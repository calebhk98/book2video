SCENE_BOUNDARY = """
Find dramatic scene boundaries in the supplied chapter. Return line ranges using the original 1-based line numbers.
A new scene may be suggested by a meaningful location/time change, but avoid splitting only because a paragraph changes.
Preserve the source order and cover the chapter without overlaps.
""".strip()

SCENE_CRITIC = """
Review this one proposed scene and decide whether it is actually one dramatic scene or should be split into two or more scenes. Look for meaningful location/time breaks or separate dramatic units. If splitting seems useful, return the absolute source line numbers after which a split should occur. Do not split merely because a paragraph changes.
""".strip()

BEAT_BREAKDOWN = """
Turn this single scene into an ordered beat breakdown. A beat is an observable story event, line of dialogue, reaction, discovery, or meaningful hold. Preserve chronology and avoid inventing connective action merely to make the scene smoother.
""".strip()

BEAT_CRITIC = """
Compare the proposed beats against the exact scene text and supplied character/location context. Identify missing, invented, reordered, or character-inconsistent beats. Return a revised beat plan only when revision seems warranted.
""".strip()

TIMING = """
Estimate screen duration for each beat. Include dialogue speaking time, visible action time, and any intentional hold. Allow offscreen dialogue to overlap visible action when the source supports it. These are planning estimates, not frame-accurate commitments.
""".strip()

SHOT_PLAN = """
Convert the timed beats into a cinematic shot list. A shot can cover several beats when continuous framing makes sense. Keep shots distinct from generator clip limits; target_duration_sec may exceed the current video model maximum. Prefer ordinary cuts unless the story calls for a different transition.
""".strip()

GENERATION_INTENT = """
Compile this physical generation clip into a provider-neutral video intent. Describe what should be visible and what should move. Preserve named character identity, location, wardrobe, movement style, and continuity notes from context. Avoid adding events not present in the shot.
""".strip()

VIDEO_JUDGE = """
Evaluate the candidate takes against the required shot, beat list, character references, and continuity context. Rank story fidelity and identity above tiny cosmetic defects. Recommend regeneration only for a meaningful first-pass failure such as wrong identity, missing critical action, major location/wardrobe continuity error, or unusable artifact.
""".strip()
