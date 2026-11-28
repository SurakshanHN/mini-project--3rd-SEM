# Community Interaction Guide (Round 1 & Round 2)

**Status:** **PENDING REAL HOUSEHOLD INTERVIEWS**  
`field/interaction_log.csv` currently contains **zero logged records**. No household responses, quotes, or participant counts have been fabricated.

---

## 1. Round 1 (Problem & Alert Format Discovery)
- Use `field/outreach_message.txt` to invite 6–8 local households (family, neighbours, hostel/PG residents).
- Ask the 10 questions in `field/round1_questions.md`.
- Record each real conversation immediately as a new row in `field/interaction_log.csv`:

```csv
date,household_label,consent_given,what_i_showed,what_they_said,what_i_changed_in_project
2026-11-28,HH-01 (Neighbour - verbal consent),yes,"Explained project idea and 3 alert options","Prefers simple evening high-usage window warning over raw Wh numbers","Prioritized high-usage window banner above numeric Wh cards in app.py"
```
*(Example format only — do not copy this placeholder row into `field/interaction_log.csv`.)*

---

## 2. Round 2 (Prototype App Demonstration)

### What to Show in `app.py` During Round 2
1. **Horizon Selector:** Toggle between **10-minute** and **1-hour** forecasts (`1 hour` is typically easier for scheduling appliances like washing machines or water heaters).
2. **High-Usage Window Banner:** Show the traffic-light style status banner (`Normal Usage Window` vs `High-Usage Window Warning`).
3. **Metric Cards & Chart:** Show the predicted Wh vs persistence baseline vs actual retrospective usage on a sample test day (e.g., May 9, 2016 evening).

### App Design Decisions That Need Real Household Feedback
1. **Alert format preference:** Do households act more readily on a qualitative *"High-Usage Window Warning"* banner or on exact *Wh* numbers?
2. **Lead time:** Is 10 minutes useful for immediate awareness, or is 1 hour (or earlier evening notice) required to shift real chores?
3. **Threshold clarity:** Should the warning threshold be tuned to a specific percentile (e.g., 90th percentile vs 300 Wh spike threshold) or phrased in plain language (e.g., *"Heavy Load"* vs *"Wh"*)?
4. **Actionability:** Which specific appliances (washing machine, iron, geyser, pump) would participants realistically delay when warned?
