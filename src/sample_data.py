"""
A tiny INVENTED show used by the web app's "try the built-in sample" button.

Same screenplay format as the real input pack (speaker + delivery note, start/end
timestamps), but every character, line and plot point is made up here. It exists so
the app can be tried without anyone uploading a pack. No Kuku text is in this repo.
"""

from __future__ import annotations

import os
from typing import List, Tuple

# (tag, text). Tag is "SPEAKER (Delivery, Note)" for dialogue, or CAMERA / ACTION / SFX.
Beat = Tuple[str, str]

EPISODES: List[Tuple[str, List[Beat]]] = [
    ("Ashes at the Gala", [
        ("CAMERA", "Wide shot of a glittering ballroom. Guests turn toward the staircase."),
        ("JULIAN (Cold, Mocking)", "Everyone, meet the woman who thinks she owns this house."),
        ("MARA (Shaking, Humiliated)", "Julian, please. Not in front of everyone."),
        ("JULIAN (Venomous, Loud)", "You are nothing without your father's name, and he is dead!"),
        ("ACTION", "Julian slaps Mara. The room goes silent."),
        ("SFX", "Glass shatters on marble."),
        ("MARA (Sobbing, Broken)", "Eleven years. I gave you eleven years."),
        ("JULIAN (Cold, Dismissive)", "And I gave you a roof. Security, take her out."),
        ("GUARD (Gruff, Hesitant)", "Miss Voss, you have to leave."),
        ("MARA (Whispering, Desperate)", "Please, somebody. Anybody."),
        ("CAMERA", "A tall man in a plain driver's coat steps out of the crowd."),
        ("DORIAN (Calm, Dangerous)", "Take your hand off her."),
        ("JULIAN (Furious, Stunned)", "Who let the driver in?"),
        ("DORIAN (Low, Commanding)", "Your father-in-law's driver. Twenty years. I drove him to sign the will."),
        ("JULIAN (Panicked, Shouting)", "Shut your mouth!"),
        ("DORIAN (Icy, Threatening)", "Ask your bride who really signed it, Julian."),
        ("MARA (Trembling, Shocked)", "What does he mean?"),
        ("JULIAN (Whispering, Terrified)", "He means nothing. Nothing."),
        ("SFX", "A phone buzzes. Julian's face drains of color."),
        ("DORIAN (Soft, Ominous)", "Answer it. It is the lawyer. He is outside."),
    ]),
    ("The Forged Name", [
        ("CAMERA", "A quiet study. Rain on the windows. Mara sits alone with a cup of tea."),
        ("MARA (Calm, Gentle)", "I used to read here with Father on Sundays."),
        ("DORIAN (Soft, Reassuring)", "He talked about you every single day."),
        ("MARA (Tender, Sad)", "Thank you for coming back, Dorian."),
        ("DORIAN (Calm, Warm)", "I never left. I was waiting until it was safe."),
        ("MARA (Curious, Quiet)", "Safe from what?"),
        ("ACTION", "Dorian lays a yellowed envelope on the desk."),
        ("DORIAN (Grave, Low)", "The will you were shown is a forgery. This is the real one."),
        ("MARA (Gasping, Shocked)", "Julian told me Father left everything to him."),
        ("DORIAN (Cold, Serious)", "Julian's signature is on the forged page. And on the check that paid for the fire."),
        ("MARA (Horrified, Shaking)", "The fire that killed my father?"),
        ("SFX", "A door slams somewhere in the house."),
        ("JULIAN (Furious, Menacing)", "Mara! Where are you?"),
        ("DORIAN (Urgent, Whispering)", "Take the envelope. Go through the library. Do not look back."),
        ("MARA (Terrified, Desperate)", "Come with me!"),
        ("DORIAN (Calm, Resolute)", "Someone has to hold the door."),
        ("JULIAN (Screaming, Furious)", "You will never leave this house alive!"),
    ]),
    ("A Quiet Morning", [
        ("CAMERA", "Garden at sunrise. Birds. Mara waters roses in a plain dress."),
        ("MARA (Calm, Peaceful)", "It is strange to feel safe again."),
        ("HELEN (Gentle, Warm)", "You earned it, dear. Come, breakfast is ready."),
        ("MARA (Soft, Content)", "I will be right there."),
        ("HELEN (Kind, Tender)", "The lawyer called. Everything is signed. The house is yours."),
        ("MARA (Relieved, Quiet)", "Then it is finally over."),
        ("HELEN (Smiling, Gentle)", "Yes. It is over."),
        ("ACTION", "They walk inside together and close the garden door."),
    ]),
]


def _stamp(x: float) -> str:
    return f"{int(x // 60):02d}:{x - 60 * int(x // 60):05.2f}"


def _render(title: str, beats: List[Beat], ep_no: int) -> str:
    lines = [
        f"EPISODE NUMBER: Episode {ep_no}",
        f"EPISODE LOGLINE: {title} (invented sample episode).",
        "CONTINUITY NOTES: Invented sample.",
        "",
        "--- SCRIPT START ---",
        "",
        "SCENE 01",
        "Scene heading: INT. MANOR - DAY",
        "Scene start: 00:00.00",
    ]
    t = 0.0
    body = []
    for tag, text in beats:
        words = len(text.split())
        dur = round(max(2.0, 0.38 * words + 1.0), 2)
        start, end = t, t + dur
        t = end + 0.8
        shown = f'"{text}"' if "(" in tag else text
        body += [f"**{tag}:**", f"Start: ({_stamp(start)}) | End: ({_stamp(end)})", shown, ""]
    lines += [f"Scene end: {_stamp(t)}", "", "Actions and dialogue:", ""] + body
    lines.append("--- SCRIPT END ---")
    return "\n".join(lines)


def write_sample_pack(dest_dir: str) -> str:
    """Write the invented episodes as .txt files into dest_dir and return it."""
    os.makedirs(dest_dir, exist_ok=True)
    for i, (title, beats) in enumerate(EPISODES, 1):
        path = os.path.join(dest_dir, f"episode-{i:03d}-{title}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_render(title, beats, i))
    return dest_dir
