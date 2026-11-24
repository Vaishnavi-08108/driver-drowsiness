"""
contacts.py
Emergency contact storage for the Driver Drowsiness Detection System.

A contact is a name plus a Telegram chat ID. ContactBook keeps up to
MAX_CONTACTS of them in a small JSON file under data/, so they survive
restarts.

Privacy notes:
    * The file stays on this computer. data/ is listed in .gitignore, so it
      is never uploaded to GitHub.
    * Chat IDs are masked when shown on screen (see mask_chat_id).

Telegram chat IDs are whole numbers. A private chat ID is positive, and a
group ID is negative, so a leading minus sign is allowed.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from src import config

MAX_CONTACTS = 5
MAX_NAME_LENGTH = 50
MAX_RELATION_LENGTH = 30

# Optional minus sign, then 5 to 20 digits
_CHAT_ID_PATTERN = re.compile(r"^-?\d{5,20}$")


class ContactError(ValueError):
    """Raised when a contact is invalid or cannot be added."""


@dataclass(frozen=True)
class Contact:
    """One emergency contact."""

    name: str
    chat_id: str            # stored as text so a leading minus sign is kept
    relation: str = ""      # for example "Mother" or "Friend"

    @property
    def relationship(self) -> str:
        return self.relation


def mask_chat_id(chat_id: str) -> str:
    """Hide all but the last 3 digits, for example '******789'."""
    digits = str(chat_id)
    if len(digits) <= 3:
        return "*" * len(digits)
    return "*" * (len(digits) - 3) + digits[-3:]


class ContactBook:
    """Loads, validates, and saves the emergency contact list."""

    def __init__(self, path: Union[str, Path] = config.CONTACTS_FILE) -> None:
        self.path = Path(path)
        self._contacts: List[Contact] = []
        # Set when the file could not be read, so the dashboard can show it
        self.load_warning: Optional[str] = None
        self.load()

    # -----------------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------------
    @property
    def contacts(self) -> List[Contact]:
        """A copy of the contact list, safe for the caller to modify."""
        return list(self._contacts)

    def chat_ids(self) -> List[str]:
        return [c.chat_id for c in self._contacts]

    def __len__(self) -> int:
        return len(self._contacts)

    # -----------------------------------------------------------------------
    # Changing
    # -----------------------------------------------------------------------
    def add(
        self,
        name: str,
        chat_id: str,
        relation: str = "",
        relationship: Optional[str] = None,
    ) -> Contact:
        """
        Validate, add, and save a contact.

        Raises:
            ContactError: bad name or chat ID, duplicate ID, or list is full.
        """
        rel = relationship if relationship is not None else relation
        contact = self._validated(name, chat_id, rel)
        if len(self._contacts) >= MAX_CONTACTS:
            raise ContactError(f"You can save at most {MAX_CONTACTS} contacts.")
        if contact.chat_id in self.chat_ids():
            raise ContactError("A contact with this chat ID already exists.")
        self._contacts.append(contact)
        self.save()
        return contact

    def remove(self, identifier: str) -> bool:
        """Remove the contact with this chat ID or name. Returns True if one was removed."""
        val = str(identifier).strip()
        kept = [c for c in self._contacts if c.chat_id != val and c.name != val]
        if len(kept) == len(self._contacts):
            return False
        self._contacts = kept
        self.save()
        return True

    # -----------------------------------------------------------------------
    # Disk
    # -----------------------------------------------------------------------
    def load(self) -> None:
        """
        Read the contact file. A missing file means an empty list. A damaged
        file is renamed to *.bad and the list starts empty, so one bad edit
        can never stop the app from starting.
        """
        self._contacts = []
        self.load_warning = None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("the file must hold a list of contacts")
            loaded = [
                self._validated(item["name"], item["chat_id"], item.get("relation", ""))
                for item in raw
            ]
            self._contacts = loaded[:MAX_CONTACTS]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self._contacts = []
            self._set_aside_bad_file()
            self.load_warning = (
                f"The contacts file could not be read ({exc}). "
                "It was renamed to contacts.json.bad and the list is empty."
            )

    def save(self) -> None:
        """Write the list to disk. The file is replaced in one step, so a
        crash while saving cannot leave a half-written file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"name": c.name, "chat_id": c.chat_id, "relation": c.relation}
            for c in self._contacts
        ]
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                suffix=".tmp", delete=False,
            ) as tmp:
                tmp_name = tmp.name
                json.dump(data, tmp, indent=2, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, self.path)
        except OSError:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    @staticmethod
    def _validated(name: str, chat_id: str, relation: str = "") -> Contact:
        name = str(name).strip()
        chat_id = str(chat_id).strip()
        relation = str(relation or "").strip()

        if not name:
            raise ContactError("Please enter a name.")
        if len(name) > MAX_NAME_LENGTH:
            raise ContactError(f"The name can have at most {MAX_NAME_LENGTH} characters.")
        if not _CHAT_ID_PATTERN.match(chat_id):
            raise ContactError(
                "The chat ID must be a number with 5 to 20 digits, "
                "for example 123456789."
            )
        return Contact(name=name, chat_id=chat_id, relation=relation[:MAX_RELATION_LENGTH])

    def _set_aside_bad_file(self) -> None:
        bad = self.path.with_name(self.path.name + ".bad")
        try:
            os.replace(self.path, bad)
        except OSError:
            pass  # if even this fails, the app still starts with an empty list