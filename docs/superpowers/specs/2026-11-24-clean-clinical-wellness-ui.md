# Clean Clinical Wellness UI Redesign

## Goal

Redesign the complete Streamlit dashboard with a clean clinical wellness visual style while preserving every existing safety and monitoring behavior.

## Visual Direction

- Background: warm off-white.
- Surfaces: white cards with soft borders and restrained shadows.
- Primary accent: calm blue.
- Success accent: mint green.
- Warning accent: amber.
- Critical accent: red.
- Typography: modern sans-serif, strong hierarchy, and compact but readable metadata.

## Application Layout

- Keep the Streamlit sidebar, but restyle it as a slim clinical navigation panel.
- Replace emoji-heavy page branding with simple iconography and concise labels.
- Keep the live drowsiness monitor first and give the camera feed visual priority.
- Group monitoring settings under clear sidebar sections.
- Apply consistent page headers and bordered content panels across all four pages.

## Live Monitor

- Use a large camera panel on the left and telemetry panel on the right.
- Keep the status badge prominent above the monitoring content.
- Present EAR, MAR, PERCLOS, and yawns as compact metric cards.
- Keep calibration, countdown, and monitoring stopped states highly visible.
- Preserve the live chart and session summary behavior.

## Other Pages

- Sleep advisor: stronger metric cards, clearly separated risk factors and action plan, and compact sleep logging controls.
- Emergency page: clear configuration, location, and contact sections with distinct primary and destructive actions.
- Assistant page: conversational layout with reduced surrounding chrome and readable suggested prompts.

## Implementation Scope

- Update the injected CSS in `src/app.py`.
- Update Streamlit theme colors in `.streamlit/config.toml`.
- Make small presentational markup improvements in `src/app.py`.
- Do not change detection algorithms, analytics thresholds, emergency escalation, camera capture, session logging, sleep scoring, contact handling, or chatbot behavior.

## Verification

- Start the Streamlit app and confirm the app loads without Python or template errors.
- Confirm the Live Monitor, Sleep Advisor, Emergency, and Assistant pages render.
- Confirm no core controls or session state behavior are removed.
