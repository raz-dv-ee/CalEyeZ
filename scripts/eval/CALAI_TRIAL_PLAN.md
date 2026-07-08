# Cal.ai trial — 2-day capture plan (utilise the free window)

Goal: collect the data that proves **recognition ≠ measurement** with the fewest, highest-value
Cal.ai queries. Priorities are ordered — do P1 first; it is the strongest result and needs the
least data. Everything is logged the instant you read Cal.ai's number, or it is worthless.

## Golden rules (make the data defensible)
0. **No numbers in Cal.ai's frame.** Cal.ai is a vision-language model and *reads text/digits*. If the
   scale display, a nutrition label, or any written weight is visible, it may OCR it and the "photo
   apps can't measure portion" claim is contaminated. Shoot the food on a neutral surface, nothing
   numeric in view. (The scale still feeds CalEyeZ over Bluetooth, not through the camera, so CalEyeZ
   is unaffected. For packaged foods, turn the label away or crop it out.) Result-screen screenshots
   for your evidence folder are separate and may include the scale.
1. **Weigh every food on our BLE scale first** and write down the grams + the label's kcal/100 g.
   That is ground truth. Without it a Cal.ai number proves nothing.
2. **One session per food** — do not eat, add, or let the food shrink/cool between shots. The food
   must be physically identical; only the *camera* changes.
3. **Change exactly one variable at a time** and note it (angle, distance, layout, plate, lighting).
4. **Log Cal.ai's food name too**, not just kcal — if the identity also flips, that's a second finding.
5. Take a phone photo of *each* Cal.ai result screen (with the scale in frame if you can). These
   screenshots are the evidence for the panel; name them so they map to the `shot_id`.

---

## P1 — Viewpoint repeatability (THE result)  →  `repeatability.csv`
Same food, same grams, **6–8 photos varying only the camera/layout.** This proves the app is not
even *reproducible*: identical input, different calories. Convincing at small n because ground truth
never moves.

Pick **3 foods that stress different failure modes:**
- **A loose, scatterable food** — french fries / grapes / nuts. Layout changes hugely → biggest
  variance. (Fries at 53 g is pre-seeded in the sheet.)
- **A piled / height-ambiguous food** — rice, hummus mound, salad. Top-down hides the mass.
- **A packaged food with a printed label** — canned tuna, a protein bar. Ground truth is exact.

For each food, shoot this fixed set of 8 (duplicate/adjust as needed):
| shot | what to change |
|---|---|
| 1 | top-down, centered, spread out |
| 2 | top-down, piled/stacked into a mound |
| 3 | 45° angle, from the side |
| 4 | low angle, almost eye-level |
| 5 | close-up (fills frame) |
| 6 | far (lots of plate/background) |
| 7 | different plate or surface colour |
| 8 | rotated 90° / dimmer or brighter light |

**~24 queries.** This is the priority — if you only get P1 done, the project still wins.

## P2 — Portion monotonicity  →  `portion_sweep.csv`
One or two foods weighed at **4–5 increasing masses** (e.g. fries 20 / 50 / 100 / 150 g), one Cal.ai
photo each. Proves their kcal barely tracks the actual grams — the flat-line you already saw with
tuna. **~8–10 queries.**

## P4 — OCR probe (optional, 1 food, ~5 queries) — turns the confound into a finding
Test whether Cal.ai secretly reads text instead of estimating portion. Same food, same grams:
- **A · number hidden** (2–3 photos): food alone, nothing numeric in frame → honest portion guess.
- **B · number shown** (2–3 photos): same food, but a visible weight in frame (scale display, or a
  hand-written "NN g" card) → does Cal.ai's kcal snap toward the true weight?
If B tracks the shown number and A doesn't, Cal.ai has no real portion measurement — it only gets
close when handed a number. Log both conditions (note A/B in the `angle_layout` field of a copy of
`repeatability.csv`, or just jot them down and tell me).

## P3 — Grow the accuracy pilot (only if time left)
A handful of **new packaged foods with labels** (yogurt, protein bar, cereal, a boxed meal), one
photo each, logged into the existing `calorie_comparison.xlsx`. Every labelled meal moves the
head-to-head n toward significance. Bonus, not required.

---

## Time budget (2 days is plenty)
- Day 1: P1 foods A + B (16 queries) + weigh/label logging.
- Day 2: P1 food C (8) + P2 sweep (10) + any P3.
Total ≈ 34–44 Cal.ai queries. Log each in the matching CSV **as you go**.

When the sheets are filled, send them back (or just paste the numbers) and I'll generate the
repeatability + monotonicity charts and write Experiment E into the App-vs-Market tab and the book.
