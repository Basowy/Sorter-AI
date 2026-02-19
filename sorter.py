import os
import json
import shutil
import re
import requests
import zipfile
import tempfile
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
from ebooklib import epub
from collections import defaultdict
from bs4 import BeautifulSoup

INPUT_DIR = "input_books"
OUTPUT_DIR = "output_sorted"
USED_BOOKS_DIR = "used books"

SUPPORTED_EXTS = (".epub", ".pdf", ".txt", ".mobi", ".azw3", ".html", ".htm", ".rtf", ".zip")


def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t


def clean_folder_name(name: str) -> str:
    # bezpieczna nazwa folderu w Windows
    name = clean_text(name)
    name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    name = re.sub(r"\.+$", "", name)  # usuń kropki na końcu
    return name[:80] or "Unknown Author"
def guess_author_from_filename(path_or_name: str) -> str:
    name = os.path.splitext(os.path.basename(path_or_name))[0]

    # typowe separatory spotykane w ebookach
    seps = [" - ", " — ", " – ", "_", "|", "—", "–"]

    for sep in seps:
        if sep in name:
            left = name.split(sep, 1)[0].strip()

            # odfiltruj śmieci
            if len(left) < 3:
                continue
            if left.isdigit():
                continue
            if len(left.split()) == 1 and len(left) < 4:
                continue

            return clean_folder_name(left)

    return ""

def parse_author_title_from_folder(folder_name: str):
    """
    Przykład:
    'Alan Dean Foster - krzywe zwierciadło'
    -> ('Alan Dean Foster', 'krzywe zwierciadło')
    """
    name = folder_name.strip()

    # typowy separator
    if " - " in name:
        parts = name.split(" - ", 1)
        author = parts[0].strip()
        title = parts[1].strip()
        return author, title

    # fallback: nie wiemy
    return "", name

def extract_from_epub(path, max_chars=6000):
    book = epub.read_epub(path)
    title = ""
    author = ""
    desc = ""

    # metadane
    t = book.get_metadata("DC", "title")
    if t and len(t[0]) > 0:
        title = t[0][0]

    a = book.get_metadata("DC", "creator")
    if a and len(a[0]) > 0:
        author = a[0][0]

    d = book.get_metadata("DC", "description")
    if d and len(d[0]) > 0:
        desc = d[0][0]

    # jeśli brak opisu, bierz fragment treści
    text = ""
    if not desc:
        for item in book.get_items():
            if item.get_type() == 9:  # DOCUMENT
                raw = item.get_content().decode("utf-8", errors="ignore")
                raw = re.sub("<[^<]+?>", " ", raw)
                text += " " + raw
                if len(text) > max_chars:
                    break

    snippet = clean_text(desc or text)[:max_chars]
    return clean_text(title), clean_text(author), snippet


def extract_from_pdf(path, max_chars=6000):
    reader = PdfReader(path)
    title = ""
    author = ""
    try:
        meta = reader.metadata
        if meta:
            title = meta.title or ""
            author = meta.author or ""
    except:
        pass

    text = ""
    for i in range(min(3, len(reader.pages))):
        try:
            text += " " + (reader.pages[i].extract_text() or "")
        except:
            pass

    snippet = clean_text(text)[:max_chars]
    return clean_text(title), clean_text(author), snippet


def extract_text_from_rtf(path, max_chars=8000):
    """
    Prosty parser RTF -> tekst.
    Bez dodatkowych bibliotek.
    """
    # RTF bywa w cp1250 / iso-8859-2
    raw = ""
    for enc in ("utf-8", "cp1250", "iso-8859-2", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                raw = f.read(max_chars * 5)
            if raw:
                break
        except:
            pass

    if not raw:
        return ""

    # usuń grupy RTF typu {\*\...}
    raw = re.sub(r"{\\\*[^{}]*}", " ", raw)

    # usuń kontrolki typu \par, \b0, \fs24, \lang1045 itd.
    raw = re.sub(r"\\[a-zA-Z]+\d* ?", " ", raw)

    # usuń encje hex \'f3 itd.
    def _hex_replace(m):
        try:
            return bytes.fromhex(m.group(1)).decode("cp1250", errors="ignore")
        except:
            return " "

    raw = re.sub(r"\\'([0-9a-fA-F]{2})", _hex_replace, raw)

    # usuń nawiasy klamrowe RTF
    raw = raw.replace("{", " ").replace("}", " ")

    # normalizacja
    text = clean_text(raw)

    # filtr: jeśli nadal wygląda jak RTF (za dużo backslashy), utnij
    if text.count("\\") > 20:
        text = text.replace("\\", " ")

    return text[:max_chars]

def extract_text_from_html(path, max_chars=8000):
    # HTML w starych ebookach często ma iso-8859-2
    for enc in ("utf-8", "iso-8859-2", "cp1250", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            text = clean_text(text)
            if text:
                return text[:max_chars]
        except:
            pass
    return ""


def extract_from_txt_like(path, max_chars=6000):
    ext = os.path.splitext(path)[1].lower()

    # RTF -> parser
    if ext == ".rtf":
        snippet = extract_text_from_rtf(path, max_chars=max_chars)
        return "", "", snippet

    # TXT / HTML fallback (gdyby coś weszło tu omyłkowo)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read(max_chars * 2)
    except:
        raw = ""

    raw = re.sub(r"<[^<]+?>", " ", raw)
    snippet = clean_text(raw)[:max_chars]
    return "", "", snippet


def classify_with_ollama(model, genres, title, author, snippet, max_chars=6000):
    prompt = f"""
Jesteś bibliotekarzem. Masz sklasyfikować książkę do JEDNEGO gatunku z listy.

DOZWOLONE GATUNKI:
{", ".join(genres)}

DANE:
Tytuł: {title or "brak"}
Autor: {author or "brak"}
Opis / fragment:
{(snippet or "")[:max_chars]}

Zwróć wynik jako JSON w formacie:
{{
  "genre": "...",
  "confidence": 0.0,
  "reason": "..."
}}

Zasady:
- genre musi być dokładnie jednym z dozwolonych gatunków.
- confidence 0.0–1.0.
- Jeśli nie wiesz, wybierz "Inne" z niską pewnością.
""".strip()

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    r.raise_for_status()
    raw = r.json()["response"]

    # proste wydobycie JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return "Inne", 0.0, "Brak poprawnego JSON"

    try:
        data = json.loads(m.group(0))
        genre = data.get("genre", "Inne")
        conf = float(data.get("confidence", 0.0))
        reason = data.get("reason", "")
    except:
        return "Inne", 0.0, "Błąd parsowania JSON"

    if genre not in genres:
        genre = "Inne"

    return genre, conf, reason

def extract_author_with_llm(model, title, snippet, max_chars=6000):
    if not (title or "").strip() and not (snippet or "").strip():
        return ""

    prompt = f"""
Wyciągnij autora książki.

DANE:
Tytuł: {title or "brak"}
Fragment:
{(snippet or "")[:max_chars]}

Zwróć TYLKO JSON:
{{
  "author": "..."
}}

Zasady:
- Jeśli nie da się ustalić autora, zwróć pusty string: "".
- Nie zgaduj na siłę.
""".strip()

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        r.raise_for_status()
        raw = r.json()["response"]

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return ""

        data = json.loads(m.group(0))
        author = clean_text(data.get("author", ""))

        # filtr na typowe śmieci
        if not author:
            return ""
        if author.lower() in ("brak", "unknown", "nieznany", "n/a"):
            return ""
        if len(author) < 3:
            return ""

        return author
    except:
        return ""



def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    genres = cfg["genres"]
    model = cfg.get("model", "llama3.1:8b")
    threshold = float(cfg.get("confidence_threshold", 0.55))
    max_chars = int(cfg.get("max_chars_from_book", 6000))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(USED_BOOKS_DIR, exist_ok=True)

    rows = []

    SUPPORTED_EXT = (".epub", ".pdf", ".txt", ".mobi", ".azw3", ".html", ".htm", ".rtf")

    items = []
    folder_groups = defaultdict(list)
    used_zip_files = set()

        # ==========================================
    # 1) SKAN INPUT_DIR
    # - pliki luzem -> ("file", path)
    # - foldery z rozdziałami -> ("folder_book", folder_name, [files])
    # - zip -> ("zip_file", zip_path, inner_path)
    # - zip folder -> ("zip_folder_book", zip_path, folder_name, [inner_files])
    # ==========================================

    for root, dirs, filenames in os.walk(INPUT_DIR):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower().strip()
            full_path = os.path.join(root, fn)

            # ------------------------------------------
            # ZIP
            # ------------------------------------------
            if ext == ".zip":
                try:
                    with zipfile.ZipFile(full_path, "r") as z:
                        inner_files = []

                        for inner in z.namelist():
                            # pomijamy foldery
                            if inner.endswith("/"):
                                continue

                            inner_ext = os.path.splitext(inner)[1]
                            inner_ext = (inner_ext or "").lower().strip()

                            # czasem w zip trafiają się dziwne końcówki typu ".RTF " albo ".rtf\t"
                            inner_ext = inner_ext.replace(" ", "").replace("\t", "")

                            if inner_ext in SUPPORTED_EXT:
                                inner_files.append(inner)

                        # grupowanie po top-folderze
                        zip_folder_groups = defaultdict(list)
                        zip_root_files = []

                        for inner in inner_files:
                            parts = inner.split("/")
                            if len(parts) >= 2:
                                zip_folder_groups[parts[0]].append(inner)
                            else:
                                zip_root_files.append(inner)

                        # pliki luzem w zip
                        for inner in zip_root_files:
                            items.append(("zip_file", full_path, inner))

                        # foldery książek w zip
                        for folder_name, inner_list in zip_folder_groups.items():
                            items.append(("zip_folder_book", full_path, folder_name, inner_list))

                        # DEBUG (możesz usunąć później)
                        # print(f"[ZIP] {os.path.basename(full_path)} -> {len(inner_files)} plików obsługiwanych")

                except Exception as e:
                    rows.append({
                        "file": full_path,
                        "title": "",
                        "author": "",
                        "genre": "Inne",
                        "format": "ZIP",
                        "confidence": 0.0,
                        "reason": f"Błąd czytania ZIP: {e}"
                    })

                continue

            # ------------------------------------------
            # Normalne pliki (nie zip)
            # ------------------------------------------
            if ext not in SUPPORTED_EXT:
                continue

            rel = os.path.relpath(full_path, INPUT_DIR)
            parts = rel.split(os.sep)

            if len(parts) >= 2:
                # plik jest w folderze -> książka folderowa
                top_folder = parts[0]
                folder_groups[top_folder].append(full_path)
            else:
                # pojedynczy plik wrzucony luzem
                items.append(("file", full_path))

    # foldery dodajemy jako osobne "książki"
    for folder_name, files_in_folder in folder_groups.items():
        items.append(("folder_book", folder_name, files_in_folder))

    print(f"Znaleziono elementów do przetworzenia: {len(items)}")

    # ==========================================
    # 2) PRZETWARZANIE
    # ==========================================
    for item in tqdm(items, desc="Sortowanie"):
        try:
            # ==========================================
            # A) Pojedynczy plik
            # ==========================================
            if item[0] == "file":
                path = item[1]
                ext = os.path.splitext(path)[1].lower()

                title, author, snippet = "", "", ""

                if ext == ".epub":
                    title, author, snippet = extract_from_epub(path, max_chars)
                elif ext == ".pdf":
                    title, author, snippet = extract_from_pdf(path, max_chars)
                elif ext in (".html", ".htm"):
                    title = os.path.splitext(os.path.basename(path))[0]
                    snippet = extract_text_from_html(path, max_chars)
                elif ext in (".txt", ".rtf"):
                    title = os.path.splitext(os.path.basename(path))[0]
                    snippet = extract_from_txt_like(path, max_chars)[2]
                else:
                    title = os.path.splitext(os.path.basename(path))[0]
                    snippet = ""

                # fallback autora
                if not author.strip():
                    author = guess_author_from_filename(path)
                if not author.strip():
                    author = extract_author_with_llm(model, title, snippet, max_chars=max_chars)

                genre, conf, reason = classify_with_ollama(model, genres, title, author, snippet, max_chars=max_chars)
                if conf < threshold:
                    genre = "Inne"

                fmt = ext.replace(".", "").upper() or "UNKNOWN"
                author_dir = clean_folder_name(author)

                out_dir = os.path.join(OUTPUT_DIR, genre, fmt, author_dir)
                os.makedirs(out_dir, exist_ok=True)

                dest = os.path.join(out_dir, os.path.basename(path))
                if not os.path.exists(dest):
                    shutil.move(path, dest)

                rows.append({
                    "file": path,
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "format": fmt,
                    "confidence": conf,
                    "reason": reason
                })

            # ==========================================
            # B1) Plik z ZIP (pojedynczy plik)
            # ==========================================
            elif item[0] == "zip_file":
                zip_path, inner_path = item[1], item[2]
                used_zip_files.add(zip_path)
                ext = os.path.splitext(inner_path)[1].lower()

                with zipfile.ZipFile(zip_path, "r") as z:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        extracted = z.extract(inner_path, tmpdir)

                        title, author, snippet = "", "", ""

                        if ext == ".epub":
                            title, author, snippet = extract_from_epub(extracted, max_chars)
                        elif ext == ".pdf":
                            title, author, snippet = extract_from_pdf(extracted, max_chars)
                        elif ext in (".html", ".htm"):
                            title = os.path.splitext(os.path.basename(inner_path))[0]
                            snippet = extract_text_from_html(extracted, max_chars)
                        elif ext in (".txt", ".rtf"):
                            title = os.path.splitext(os.path.basename(inner_path))[0]
                            snippet = extract_from_txt_like(extracted, max_chars)[2]
                        else:
                            title = os.path.splitext(os.path.basename(inner_path))[0]
                            snippet = ""

                        # fallback autora
                        if not author.strip():
                            author = guess_author_from_filename(inner_path)
                        if not author.strip():
                            author = extract_author_with_llm(model, title, snippet, max_chars=max_chars)

                        genre, conf, reason = classify_with_ollama(model, genres, title, author, snippet, max_chars=max_chars)
                        if conf < threshold:
                            genre = "Inne"

                        fmt = ext.replace(".", "").upper() or "UNKNOWN"
                        author_dir = clean_folder_name(author)

                        out_dir = os.path.join(OUTPUT_DIR, genre, fmt, author_dir)
                        os.makedirs(out_dir, exist_ok=True)

                        dest = os.path.join(out_dir, os.path.basename(inner_path))
                        if not os.path.exists(dest):
                            shutil.copy2(extracted, dest)

                        rows.append({
                            "file": f"{zip_path}::{inner_path}",
                            "title": title,
                            "author": author,
                            "genre": genre,
                            "format": fmt,
                            "confidence": conf,
                            "reason": reason
                        })


            # ==========================================
            # B2) Folder-książka z ZIP (rozdziały)
            # ==========================================
            elif item[0] == "zip_folder_book":
                zip_path = item[1]
                used_zip_files.add(zip_path)
                folder_name = item[2]
                inner_list = sorted(item[3])

                # autor/tytuł z nazwy folderu
                author_guess, title_guess = parse_author_title_from_folder(folder_name)

                # budujemy snippet z 1-3 pierwszych plików
                snippet_parts = []

                with zipfile.ZipFile(zip_path, "r") as z:
                    with tempfile.TemporaryDirectory() as tmpdir:

                        for inner in inner_list[:3]:
                            ext = os.path.splitext(inner)[1].lower()
                            extracted = z.extract(inner, tmpdir)

                            if ext in (".html", ".htm"):
                                snippet_parts.append(extract_text_from_html(extracted, max_chars))
                            elif ext in (".txt", ".rtf"):
                                snippet_parts.append(extract_from_txt_like(extracted, max_chars)[2])

                        snippet = clean_text(" ".join(snippet_parts))[:max_chars]

                        # fallback autora jeśli folder nie ma autora
                        if not author_guess.strip():
                            author_guess = extract_author_with_llm(model, title_guess, snippet, max_chars=max_chars)

                        genre, conf, reason = classify_with_ollama(model, genres, title_guess, author_guess, snippet, max_chars=max_chars)
                        if conf < threshold:
                            genre = "Inne"

                        fmt = "FOLDER_BOOK"
                        author_dir = clean_folder_name(author_guess)
                        title_dir = clean_folder_name(title_guess)

                        out_dir = os.path.join(OUTPUT_DIR, genre, fmt, author_dir, title_dir)
                        os.makedirs(out_dir, exist_ok=True)

                        # kopiujemy WSZYSTKIE pliki z tego folderu w zip
                        for inner in inner_list:
                            extracted = z.extract(inner, tmpdir)

                            # ścieżka wewnątrz folderu książki
                            rel_inside = inner.split("/", 1)[1] if "/" in inner else os.path.basename(inner)
                            dest_path = os.path.join(out_dir, rel_inside)

                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            if not os.path.exists(dest_path):
                                shutil.copy2(extracted, dest_path)

                        rows.append({
                            "file": f"{zip_path}::{folder_name}/",
                            "title": title_guess,
                            "author": author_guess,
                            "genre": genre,
                            "format": fmt,
                            "confidence": conf,
                            "reason": reason
                        })

            # ==========================================
            # C) Książka-folder (rozdziały)
            # ==========================================
            elif item[0] == "folder_book":
                folder_name = item[1]
                files_in_folder = sorted(item[2])

                author_guess, title_guess = parse_author_title_from_folder(folder_name)

                # budujemy snippet z 1-3 pierwszych plików
                snippet_parts = []
                for fp in files_in_folder[:3]:
                    ext = os.path.splitext(fp)[1].lower()

                    if ext in (".html", ".htm"):
                        snippet_parts.append(extract_text_from_html(fp, max_chars))
                    elif ext in (".txt", ".rtf"):
                        snippet_parts.append(extract_from_txt_like(fp, max_chars)[2])

                snippet = clean_text(" ".join(snippet_parts))[:max_chars]

                # fallback autora jeśli folder nie ma autora
                if not author_guess.strip():
                    author_guess = extract_author_with_llm(model, title_guess, snippet, max_chars=max_chars)

                genre, conf, reason = classify_with_ollama(model, genres, title_guess, author_guess, snippet, max_chars=max_chars)
                if conf < threshold:
                    genre = "Inne"

                fmt = "FOLDER_BOOK"
                author_dir = clean_folder_name(author_guess)
                title_dir = clean_folder_name(title_guess)

                out_dir = os.path.join(OUTPUT_DIR, genre, fmt, author_dir, title_dir)
                os.makedirs(out_dir, exist_ok=True)

                src_folder_path = os.path.join(INPUT_DIR, folder_name)

                # kopiujemy pliki rozdziałów
                for fp in files_in_folder:
                    rel_inside = os.path.relpath(fp, src_folder_path)
                    dest_path = os.path.join(out_dir, rel_inside)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    if not os.path.exists(dest_path):
                        shutil.move(fp, dest_path)

                rows.append({
                    "file": src_folder_path,
                    "title": title_guess,
                    "author": author_guess,
                    "genre": genre,
                    "format": fmt,
                    "confidence": conf,
                    "reason": reason
                })

        except Exception as e:
            rows.append({
                "file": str(item),
                "title": "",
                "author": "",
                "genre": "Inne",
                "format": "",
                "confidence": 0.0,
                "reason": f"Błąd: {e}"
            })

            # ==========================================
            # 3) PRZENIESIENIE UŻYTYCH ZIP DO used books/
            # ==========================================
            for zp in sorted(used_zip_files):
                try:
                    if not os.path.exists(zp):
                        continue

                    dest = os.path.join(USED_BOOKS_DIR, os.path.basename(zp))

                    # jeśli już istnieje (np. ta sama nazwa) -> dopisz suffix
                    if os.path.exists(dest):
                        base, ext = os.path.splitext(os.path.basename(zp))
                        i = 2
                        while True:
                            dest = os.path.join(USED_BOOKS_DIR, f"{base} ({i}){ext}")
                            if not os.path.exists(dest):
                                break
                            i += 1

                    shutil.move(zp, dest)

                except Exception as e:
                    rows.append({
                        "file": zp,
                        "title": "",
                        "author": "",
                        "genre": "Inne",
                        "format": "ZIP",
                        "confidence": 0.0,
                        "reason": f"Nie udało się przenieść ZIP do used books: {e}"
        })

    df = pd.DataFrame(rows)
    df.to_csv("report.csv", index=False, encoding="utf-8")
    print("Gotowe. Wynik w output_sorted/ oraz report.csv")

if __name__ == "__main__":
    main()