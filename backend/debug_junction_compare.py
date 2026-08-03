"""
debug_junction_compare.py
--------------------------
Runs the OLD junction_control logic and the NEW one on the exact same
route steps, side by side, so you can see exactly how many segments
flipped and why your risk scores changed.

Usage:
    python debug_junction_compare.py "Hebbal, Bangalore" "Varthur, Bangalore"
"""
import sys
import requests

GMAPS_DIRECTIONS = 'https://maps.googleapis.com/maps/api/directions/json'
API_KEY = 'AIzaSyAD3HhUgB4XRAMII2_8NMcrTofMdHmmLtU'   # same one from your .env / config


def old_junction_control(html_instr, maneuver):
    if any(x in html_instr for x in ['signal', 'traffic light']):
        return 2
    elif 'roundabout' in html_instr or 'roundabout' in maneuver:
        return 3
    elif any(x in maneuver for x in ['turn', 'merge', 'fork']):
        return 1
    else:
        return 0


def new_junction_control(html_instr, maneuver):
    if 'roundabout' in maneuver or 'roundabout' in html_instr:
        return 3
    elif any(x in html_instr for x in ['signal', 'traffic light']):
        return 2
    elif maneuver in ('merge', 'fork-left', 'fork-right', 'ramp-left', 'ramp-right') \
            or maneuver.startswith('uturn'):
        return 1
    else:
        return 0


def main():
    origin = sys.argv[1] if len(sys.argv) > 1 else "Hebbal, Bangalore"
    destination = sys.argv[2] if len(sys.argv) > 2 else "Varthur, Bangalore"

    params = {'origin': origin, 'destination': destination,
              'alternatives': 'true', 'key': API_KEY}
    resp = requests.get(GMAPS_DIRECTIONS, params=params, timeout=10).json()

    if resp.get('status') != 'OK':
        print("Google error:", resp.get('status'), resp.get('error_message'))
        return

    for r_idx, route in enumerate(resp['routes']):
        print(f"\n{'='*60}\nROUTE {r_idx + 1}: {route.get('summary', '')}\n{'='*60}")

        old_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        new_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        flipped = 0
        total_steps = 0

        for leg in route['legs']:
            for step in leg['steps']:
                total_steps += 1
                maneuver = step.get('maneuver', '')
                html_instr = step.get('html_instructions', '').lower()

                old_val = old_junction_control(html_instr, maneuver)
                new_val = new_junction_control(html_instr, maneuver)

                old_counts[old_val] += 1
                new_counts[new_val] += 1

                if old_val != new_val:
                    flipped += 1
                    instr_preview = step.get('html_instructions', '')[:45]
                    print(f"  CHANGED: \"{instr_preview}...\" "
                          f"maneuver='{maneuver}' | old={old_val} -> new={new_val}")

        print(f"\n  Total steps: {total_steps}")
        print(f"  Old distribution: not_junction={old_counts[0]} "
              f"merge/turn={old_counts[1]} signal={old_counts[2]} roundabout={old_counts[3]}")
        print(f"  New distribution: not_junction={new_counts[0]} "
              f"merge/turn={new_counts[1]} signal={new_counts[2]} roundabout={new_counts[3]}")
        print(f"  Steps that flipped: {flipped} / {total_steps} "
              f"({round(100*flipped/total_steps, 1)}%)")
        old_junction_pct = round(100 * (old_counts[1]+old_counts[2]+old_counts[3]) / total_steps, 1)
        new_junction_pct = round(100 * (new_counts[1]+new_counts[2]+new_counts[3]) / total_steps, 1)
        print(f"  % steps flagged as 'junction-like' — OLD: {old_junction_pct}% → NEW: {new_junction_pct}%")


if __name__ == "__main__":
    main()