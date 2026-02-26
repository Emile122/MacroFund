from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.normalize.validation import ValidationError, validate_bloomberg_excel_template


def load_export(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Load manually exported Bloomberg workbook and enforce template rules."""
    try:
        import openpyxl
    except Exception as exc:
        raise RuntimeError("openpyxl is required for Bloomberg Excel ingestion") from exc

    wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
    ws = wb[sheet_name] if isinstance(sheet_name, str) else wb.worksheets[sheet_name]

    if ws.merged_cells.ranges:
        raise ValidationError("Merged cells are not allowed in export template")
    for dim in ws.column_dimensions.values():
        if dim.hidden:
            raise ValidationError("Hidden columns are not allowed in export template")

    data = ws.values
    rows = list(data)
    if not rows:
        raise ValidationError("Workbook is empty")
    columns = [str(c) for c in rows[0]]
    frame = pd.DataFrame(rows[1:], columns=columns)
    validate_bloomberg_excel_template(frame, first_col_expected="Date")
    return frame
