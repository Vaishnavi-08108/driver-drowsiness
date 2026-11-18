import json
import tempfile
from pathlib import Path

from src.contacts import MAX_CONTACTS, ContactBook, ContactError, mask_chat_id


def raises(fn, *args):
    try:
        fn(*args)
    except ContactError:
        return True
    return False


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "data" / "contacts.json"

    # 1. A missing file gives an empty book and creates nothing
    book = ContactBook(path)
    assert len(book) == 0 and book.load_warning is None and not path.exists()

    # 2. Adding saves to disk, and a new ContactBook reads it back
    c = book.add("  Asha  ", " 123456789 ", "Mother")
    assert c.name == "Asha" and c.chat_id == "123456789"   # whitespace trimmed
    again = ContactBook(path)
    assert [x.name for x in again.contacts] == ["Asha"]
    assert again.contacts[0].relation == "Mother"

    # 3. Invalid input is rejected
    assert raises(book.add, "", "123456789")                # empty name
    assert raises(book.add, "X" * 51, "123456789")          # name too long
    assert raises(book.add, "Ravi", "abc12345")             # not a number
    assert raises(book.add, "Ravi", "1234")                 # too short
    assert raises(book.add, "Ravi", "12 345 678")           # spaces inside
    assert len(book) == 1

    # 4. Duplicate chat IDs are rejected
    assert raises(book.add, "Asha again", "123456789")

    # 5. Group IDs with a leading minus sign are accepted
    book.add("Family group", "-1001234567890")
    assert "-1001234567890" in book.chat_ids()

    # 6. The list size is limited
    for i in range(MAX_CONTACTS - len(book)):
        book.add(f"Friend {i}", f"90000000{i}")
    assert len(book) == MAX_CONTACTS
    assert raises(book.add, "One too many", "555555555")

    # 7. Removing works and is saved
    assert book.remove("123456789") is True
    assert book.remove("123456789") is False                 # already gone
    assert "123456789" not in ContactBook(path).chat_ids()

    # 8. The returned list is a copy
    book.contacts.clear()
    assert len(book) == MAX_CONTACTS - 1

    # 9. A damaged file is set aside and the book starts empty
    path.write_text("{ this is not json", encoding="utf-8")
    broken = ContactBook(path)
    assert len(broken) == 0 and broken.load_warning is not None
    assert path.with_name("contacts.json.bad").exists()

    # 10. A file with a wrong structure is also handled
    path.write_text(json.dumps({"name": "not a list"}), encoding="utf-8")
    assert len(ContactBook(path)) == 0

    # 11. Masking hides most digits
    assert mask_chat_id("123456789") == "******789"
    assert mask_chat_id("12") == "**"

print("All contacts tests passed")