import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

from paddleocr import PPStructure, save_structure_res

# ============ Utils ============

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
PDF_EXT = {".pdf"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_inputs(input_dir: Path, max_files: Optional[int]=None) -> List[Path]:
    files = []
    for ext in list(IMG_EXT | PDF_EXT):
        files.extend(sorted(input_dir.rglob(f"*{ext}")))
    if max_files is not None:
        files = files[:max_files]
    return files

def load_pdf_pages(pdf_path: Path) -> List[np.ndarray]:
    # Lazy import to avoid poppler requirement if не нужен
    from pdf2image import convert_from_path
    images = convert_from_path(str(pdf_path), dpi=200)
    return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

def preprocess(img: np.ndarray) -> np.ndarray:
    # Лёгкий препроцесс: авто-контраст для слабых скриншотов
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def bbox_crop(img: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1]-1, int(x2)), min(img.shape[0]-1, int(y2))
    return img[y1:y2, x1:x2]

def normalize_sheet_name(name: str) -> str:
    # Excel sheet name max 31 chars, remove forbidden chars
    bad = '[]:*?/\\'
    for ch in bad:
        name = name.replace(ch, "_")
    return name[:31] if len(name) > 31 else name

# ============ Core ============

def run_ocr(
    input_dir: Path,
    output_xlsx: Path,
    langs: str = "ru,en",
    sheet_mode: str = "one-per-table",
    max_files: Optional[int] = None,
    runs_dir: Optional[Path] = None,
):
    ensure_dir(output_xlsx.parent)
    if runs_dir is None:
        runs_dir = output_xlsx.parent / "runs"
    ensure_dir(runs_dir)

    # PPStructure: det(text) + table layout + rec
    ocr_engine = PPStructure(
        layout=True,
        show_log=False,
        lang=langs,
        # в runs_dir будут падать html/vis
        save_dir=str(runs_dir),
    )

    inputs = list_inputs(input_dir, max_files=max_files)
    if not inputs:
        print("В папке input нет подходящих файлов (png/jpg/jpeg/bmp/tif/pdf).", file=sys.stderr)
        sys.exit(2)

    writer = pd.ExcelWriter(output_xlsx, engine="openpyxl")
    appended_rows = []

    for path in tqdm(inputs, desc="Обработка файлов"):
        pages: List[Tuple[np.ndarray, str]] = []
        if path.suffix.lower() in PDF_EXT:
            imgs = load_pdf_pages(path)
            for i, img in enumerate(imgs):
                pages.append((img, f"{path.stem}_p{i+1}"))
        else:
            img = cv2.imread(str(path))
            pages.append((img, path.stem))

        for img, page_name in pages:
            if img is None:
                continue
            proc = preprocess(img)

            # Запуск PPStructure
            result = ocr_engine(proc)

            # Сохраним html/визуализации на диск для контроля качества
            save_structure_res(result, os.path.join(str(runs_dir), page_name), to_pil(proc))

            # Соберём таблицы
            table_counter = 0
            any_table = False

            for res in result:
                clz = res.get("type")
                if clz == "table":
                    any_table = True
                    table_counter += 1

                    # Попробуем получить HTML/CSV
                    html = res.get("res", {}).get("html")
                    if html:
                        # Парсим HTML табличной структурой
                        try:
                            dfs = pd.read_html(html)
                        except ValueError:
                            dfs = []
                    else:
                        dfs = []

                    if not dfs:
                        # Фолбэк: OCR по ячейкам (если разметка не вернулась)
                        txts = []
                        cells = res.get("res", {}).get("cells", [])
                        for cell in cells:
                            txts.append(cell.get("text", ""))
                        if txts:
                            dfs = [pd.DataFrame({"text": txts})]

                    if not dfs:
                        # Совсем фолбэк: вырезать bbox и сделать OCR как одну колонку
                        bbox = res.get("bbox")
                        if bbox is not None:
                            crop = bbox_crop(proc, bbox)
                            # Одной строкой:
                            txt = "\n".join([x[1][0] for x in ocr_engine.ocr.ocr(crop, cls=False) or []])
                            dfs = [pd.DataFrame({"text": txt.splitlines()})]

                    sheet_base = normalize_sheet_name(f"{page_name}_tbl{table_counter}")
                    for idx, df in enumerate(dfs):
                        if sheet_mode == "append":
                            df2 = df.copy()
                            df2.insert(0, "source", f"{page_name}#{table_counter}")
                            appended_rows.append(df2)
                        else:
                            sheet_name = sheet_base if idx == 0 else normalize_sheet_name(f"{sheet_base}_{idx+1}")
                            # пустые df Excel не любит – заменим хотя бы одной пустой строкой
                            if df.empty:
                                df = pd.DataFrame({"empty": [""]})
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

            if not any_table:
                # Снимок без таблиц — сохраним распознанный текст
                lines = []
                ocr_lines = ocr_engine.ocr.ocr(proc, cls=False) or []
                for line in ocr_lines:
                    for _, (text, prob) in line:
                        lines.append(text)
                df = pd.DataFrame({"text": lines if lines else ["(ничего не распознано)"]})
                sheet_name = normalize_sheet_name(f"{page_name}_text")
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    if sheet_mode == "append" and appended_rows:
        big = pd.concat(appended_rows, ignore_index=True)
        big.to_excel(writer, sheet_name="all_tables", index=False)

    writer.close()
    print(f"Готово → {output_xlsx}")

def parse_args():
    p = argparse.ArgumentParser(description="Извлечение таблиц со скриншотов в Excel")
    p.add_argument("--input", type=Path, default=Path("input"), help="Папка с изображениями/PDF")
    p.add_argument("--output", type=Path, default=Path("output.xlsx"), help="Путь к Excel")
    p.add_argument("--langs", type=str, default="ru,en", help="Языки OCR через запятую (напр. ru,en или en)")
    p.add_argument("--sheet-mode", type=str, default="one-per-table", choices=["one-per-table", "append"], help="one-per-table: каждый фрагмент в свой лист; append: всё в один лист")
    p.add_argument("--max-files", type=int, default=None, help="Ограничить число обрабатываемых файлов")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_ocr(
        input_dir=args.input,
        output_xlsx=args.output,
        langs=args.langs,
        sheet_mode=args.sheet_mode,
        max_files=args.max_files,
    )
